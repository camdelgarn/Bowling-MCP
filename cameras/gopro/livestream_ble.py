"""
Start GoPro RTMP livestreams via BLE for all three cameras.

Connects to each GoPro over BLE, provisions WiFi, starts the RTMP livestream,
then disconnects BLE. Each GoPro streams independently over WiFi — no ongoing
BLE connection needed after setup.

Usage:
  # Start all 3 cameras:
  python -m cameras.gopro.livestream_ble

  # Start only one camera by ID:
  python -m cameras.gopro.livestream_ble --only 1298

  # Scan for GoPro BLE devices:
  python -m cameras.gopro.livestream_ble --scan-gopros

  # Scan WiFi networks via a specific GoPro:
  python -m cameras.gopro.livestream_ble --scan-wifi --only 1298

  # Stop all streaming GoPros:
  python -m cameras.gopro.livestream_ble --stop
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from open_gopro import WirelessGoPro
from open_gopro.models import proto


# ---------------------------------------------------------------------------
# Version guard — keep in sync with rest of this module
# ---------------------------------------------------------------------------

def _require_python_312() -> None:
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "Python 3.13+ is not supported for this GoPro workflow. Use Python 3.12.x."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RTMP_HOST_IP = "10.0.0.57"
RTMP_PORT = 1935
RTMP_APP = "live"

# WiFi network the GoPros connect to so they can push RTMP to this machine
WIFI_SSID = "grassville"
WIFI_PASSWORD = "2537973312"

# Camera definitions — id = last 4 of serial, mac = BLE address (no colons)
CAMERAS: list[dict] = [
    {"id": "1298", "mac": "C3BD00F1C3D3", "stream_key": "gopro1"},
    {"id": "3404", "mac": "EAC313813BD5", "stream_key": "gopro2"},
    {"id": "5497", "mac": "E86DDA5629DA", "stream_key": "gopro3"},
]

# Stream quality defaults
DEFAULT_RESOLUTION = 1080
DEFAULT_MIN_BITRATE = 800
DEFAULT_MAX_BITRATE = 8000
DEFAULT_START_BITRATE = 4000

# Seconds between sequential BLE connections to let the adapter settle
INTER_CAMERA_DELAY = 8

LOG_FILE = Path(__file__).parent / "gopro_livestream.log"

_CLEANUP_ATTEMPTED_MACS: set[str] = set()
_GLOBAL_WIFI_GUARD_INSTALLED = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rtmp_url_for(stream_key: str) -> str:
    return f"rtmp://{RTMP_HOST_IP}:{RTMP_PORT}/{RTMP_APP}/{stream_key}"


def cleanup_stale_ble_devices(macs: list[str] | None = None) -> None:
    """Remove stale GoPro BLE device entries from Windows."""
    global _CLEANUP_ATTEMPTED_MACS

    if macs is None:
        macs = [cam["mac"] for cam in CAMERAS]

    normalized_macs = [m.strip().lower() for m in macs if m and m.strip()]
    pending_macs = [m for m in normalized_macs if m not in _CLEANUP_ATTEMPTED_MACS]
    if not pending_macs:
        print("Skipping stale BLE cleanup (already attempted for these camera MACs).")
        return

    print("Checking for stale GoPro BLE devices...")
    try:
        result = subprocess.run(
            ["pnputil", "/enum-devices", "/class", "Bluetooth"],
            capture_output=True, text=True, timeout=10,
        )
        all_device_ids: list[str] = []
        for mac_lower in pending_macs:
            # Use strict Bluetooth LE instance matching to avoid over-matching unrelated devices.
            ids = re.findall(
                r"Instance ID:\s+(BTHLE\\Dev_" + mac_lower + r".+?)$",
                result.stdout,
                re.MULTILINE | re.IGNORECASE,
            )
            all_device_ids.extend(ids)

        all_device_ids = list(set(d.strip() for d in all_device_ids))
        if not all_device_ids:
            print("  No stale BLE devices found.")
            _CLEANUP_ATTEMPTED_MACS.update(pending_macs)
            return

        print(f"  Found {len(all_device_ids)} stale BLE device(s), removing...")
        removed = 0
        failed = 0
        for dev_id in all_device_ids:
            proc = subprocess.run(
                ["pnputil", "/remove-device", dev_id],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0:
                removed += 1
            else:
                failed += 1

        print(f"  Stale BLE cleanup results: removed={removed}, failed={failed}")
        _CLEANUP_ATTEMPTED_MACS.update(pending_macs)
    except Exception as exc:
        print(f"  Warning: BLE cleanup failed ({exc}), continuing anyway...")
        _CLEANUP_ATTEMPTED_MACS.update(pending_macs)


def _install_global_wifi_disconnect_guard() -> None:
    """Patch WirelessGoPro class teardown hooks so new instances keep host WiFi connected."""
    global _GLOBAL_WIFI_GUARD_INSTALLED
    if _GLOBAL_WIFI_GUARD_INSTALLED:
        return

    async def _noop_class(self, *args, **kwargs) -> None:  # noqa: ANN001
        logging.getLogger(__name__).info("Skipping host WiFi disconnect (global patch)")

    patched: list[str] = []
    for attr in ("_close_wifi", "_close_wifi_ap", "_disconnect_wifi"):
        if hasattr(WirelessGoPro, attr):
            setattr(WirelessGoPro, attr, _noop_class)
            patched.append(attr)

    if patched:
        logging.getLogger(__name__).info(
            "Applied GLOBAL host WiFi disconnect patch on: %s", ", ".join(patched)
        )
    else:
        logging.getLogger(__name__).warning(
            "No known WiFi teardown hooks found on WirelessGoPro class."
        )

    _GLOBAL_WIFI_GUARD_INSTALLED = True


def _current_wifi_ssid() -> str | None:
    """Return currently connected WiFi SSID on Windows, if detectable."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return None

    for line in result.stdout.splitlines():
        if "BSSID" in line:
            continue
        match = re.match(r"\s*SSID\s*:\s*(.+)", line)
        if match:
            value = match.group(1).strip()
            if value and value.lower() != "n/a":
                return value
    return None


def ensure_host_wifi_connected(target_ssid: str) -> None:
    """Best-effort reconnect to host WiFi profile after BLE workflow exits."""
    if not target_ssid:
        return

    current = _current_wifi_ssid()
    if current and current.lower() == target_ssid.lower():
        return

    print(f"Ensuring host WiFi is connected to '{target_ssid}'...")
    try:
        subprocess.run(
            ["netsh", "wlan", "connect", f"name={target_ssid}"],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
    except Exception as exc:
        print(f"  Warning: WiFi reconnect command failed ({exc})")
        return

    time.sleep(2)
    current = _current_wifi_ssid()
    if current and current.lower() == target_ssid.lower():
        print(f"  Host WiFi connected: {current}")
    else:
        print(
            "  Warning: host WiFi is not on the target SSID yet. "
            "Reconnect manually if needed."
        )


def patch_gopro_no_wifi_disconnect(gopro: WirelessGoPro) -> None:
    """Prevent the SDK from disconnecting the host PC's WiFi on exit."""
    _install_global_wifi_disconnect_guard()

    async def _noop() -> None:
        logging.getLogger(__name__).info("Skipping host WiFi disconnect (patched)")

    patched: list[str] = []

    # open_gopro internals have changed names across versions.
    # Patch any known teardown hook that can drop the host WiFi interface.
    for attr in ("_close_wifi", "_close_wifi_ap", "_disconnect_wifi"):
        if hasattr(gopro, attr):
            setattr(gopro, attr, _noop)
            patched.append(attr)

    if patched:
        logging.getLogger(__name__).info(
            "Applied host WiFi disconnect patch on: %s", ", ".join(patched)
        )
    else:
        logging.getLogger(__name__).warning(
            "No known WiFi teardown hooks found on WirelessGoPro; "
            "host WiFi may still be affected during BLE disconnect."
        )


def _livestream_status_tuple(status_resp: object) -> tuple[str, str]:
    """Return (state, error) from an open_gopro livestream status response."""
    data = getattr(status_resp, "data", None)
    state = str(getattr(data, "live_stream_status", "UNKNOWN"))
    error = str(getattr(data, "live_stream_error", "UNKNOWN"))
    return state, error


# ---------------------------------------------------------------------------
# BLE scan helpers
# ---------------------------------------------------------------------------

async def scan_gopros() -> None:
    """Print all GoPro BLE devices currently in range."""
    from bleak import BleakScanner
    print("Scanning for GoPro BLE devices (10 s)...")
    devices = await BleakScanner.discover(timeout=10)
    gopros = [d for d in devices if d.name and "GoPro" in d.name]
    if not gopros:
        print("No GoPro devices found.")
        return
    print(f"Found {len(gopros)} GoPro(s):")
    for d in gopros:
        mac_clean = d.address.replace(":", "")
        configured = any(c["mac"].upper() == mac_clean.upper() for c in CAMERAS)
        tag = " [configured]" if configured else " [NOT in CAMERAS list]"
        print(f"  {d.name}  MAC: {d.address}{tag}")


# ---------------------------------------------------------------------------
# Per-camera stream start
# ---------------------------------------------------------------------------

async def start_camera_stream(
    cam: dict,
    wifi_ssid: str,
    wifi_password: str,
    resolution: int,
    min_bitrate: int,
    max_bitrate: int,
    start_bitrate: int,
    encode: bool,
) -> None:
    identifier = cam["id"]
    url = rtmp_url_for(cam["stream_key"])

    window_size_map = {
        480: proto.EnumWindowSize.WINDOW_SIZE_480,
        720: proto.EnumWindowSize.WINDOW_SIZE_720,
        1080: proto.EnumWindowSize.WINDOW_SIZE_1080,
    }
    window_size = window_size_map.get(resolution, proto.EnumWindowSize.WINDOW_SIZE_1080)

    print(f"\n{'='*60}")
    print(f"  Starting GoPro {identifier}  ->  {url}")
    print(f"{'='*60}")

    gopro = WirelessGoPro(
        target=identifier,
        interfaces={WirelessGoPro.Interface.BLE},
        maintain_state=False,
    )
    patch_gopro_no_wifi_disconnect(gopro)
    # Override close() to only do BLE teardown — skips streaming.close() →
    # stop_active_stream() and any WiFi teardown that could drop host WiFi.
    async def _ble_only_close() -> None:
        try:
            await gopro._close_ble()  # type: ignore[attr-defined]
        except Exception:
            pass
        gopro._open = False  # type: ignore[attr-defined]
    gopro.close = _ble_only_close  # type: ignore[method-assign]

    async with gopro:
        print(f"  [{identifier}] Connected via BLE")

        print(f"  [{identifier}] Joining WiFi: {wifi_ssid}")
        wifi_resp = await gopro.ble_command.request_wifi_connect_new(
            ssid=wifi_ssid, password=wifi_password
        )
        print(f"  [{identifier}] WiFi response: {wifi_resp.status}")

        print(f"  [{identifier}] Waiting 5 s for WiFi to establish...")
        await asyncio.sleep(5)

        initial_status = await gopro.ble_command.register_livestream_status(
            register=[
                proto.EnumRegisterLiveStreamStatus.REGISTER_LIVE_STREAM_STATUS_STATUS,
                proto.EnumRegisterLiveStreamStatus.REGISTER_LIVE_STREAM_STATUS_ERROR,
            ]
        )
        pre_state, pre_error = _livestream_status_tuple(initial_status)
        print(f"  [{identifier}] Initial live status: state={pre_state}, error={pre_error}")

        print(f"  [{identifier}] Setting livestream mode ({resolution}p, {start_bitrate} kbps)")
        resp = await gopro.ble_command.set_livestream_mode(
            url=url,
            minimum_bitrate=min_bitrate,
            maximum_bitrate=max_bitrate,
            starting_bitrate=start_bitrate,
            encode=encode,
            window_size=window_size,
            lens=proto.EnumLens.LENS_LINEAR,
        )
        print(f"  [{identifier}] Livestream mode response: {resp.status}")

        await asyncio.sleep(3)

        final_state = "UNKNOWN"
        final_error = "UNKNOWN"
        for attempt in range(1, 3):
            print(f"  [{identifier}] Starting shutter (attempt {attempt}/2)...")
            await gopro.ble_command.set_shutter(shutter=True)
            await asyncio.sleep(5)

            final_status = await gopro.ble_command.register_livestream_status(
                register=[
                    proto.EnumRegisterLiveStreamStatus.REGISTER_LIVE_STREAM_STATUS_STATUS,
                    proto.EnumRegisterLiveStreamStatus.REGISTER_LIVE_STREAM_STATUS_ERROR,
                ]
            )
            final_state, final_error = _livestream_status_tuple(final_status)
            print(f"  [{identifier}] Final live status: state={final_state}, error={final_error}")

            if "STREAMING" in final_state and "ERROR_NONE" in final_error:
                break

            if attempt < 2:
                print(f"  [{identifier}] Not streaming yet, retrying livestream start...")
                await gopro.ble_command.set_shutter(shutter=False)
                await asyncio.sleep(2)
                resp = await gopro.ble_command.set_livestream_mode(
                    url=url,
                    minimum_bitrate=min_bitrate,
                    maximum_bitrate=max_bitrate,
                    starting_bitrate=start_bitrate,
                    encode=encode,
                    window_size=window_size,
                    lens=proto.EnumLens.LENS_LINEAR,
                )
                print(f"  [{identifier}] Retry livestream mode response: {resp.status}")
                await asyncio.sleep(2)

        if "STREAMING" not in final_state or "ERROR_NONE" not in final_error:
            raise RuntimeError(
                f"Camera did not reach STREAMING state (state={final_state}, error={final_error})"
            )

        print(f"  [{identifier}] LIVE -> {url}")

    print(f"  [{identifier}] BLE disconnected — camera streams independently over WiFi.")


async def stop_camera_stream(cam: dict) -> None:
    """Connect via BLE and send shutter-off to stop streaming."""
    identifier = cam["id"]
    print(f"  Stopping GoPro {identifier}...")
    gopro = WirelessGoPro(
        target=identifier,
        interfaces={WirelessGoPro.Interface.BLE},
        maintain_state=False,
    )
    patch_gopro_no_wifi_disconnect(gopro)
    # Override close() to only do BLE teardown — skips WiFi/streaming feature
    # teardown that can drop host WiFi even when using BLE-only interface.
    async def _ble_only_close() -> None:
        try:
            await gopro._close_ble()  # type: ignore[attr-defined]
        except Exception:
            pass
        gopro._open = False  # type: ignore[attr-defined]
    gopro.close = _ble_only_close  # type: ignore[method-assign]
    try:
        async with gopro:
            await gopro.ble_command.set_shutter(shutter=False)
            print(f"  [{identifier}] Stream stopped.")
    except Exception as exc:
        print(f"  [{identifier}] Stop failed: {exc}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def start_all_streams(
    cameras: list[dict],
    wifi_ssid: str,
    wifi_password: str,
    scan_wifi: bool = False,
    resolution: int = DEFAULT_RESOLUTION,
    min_bitrate: int = DEFAULT_MIN_BITRATE,
    max_bitrate: int = DEFAULT_MAX_BITRATE,
    start_bitrate: int = DEFAULT_START_BITRATE,
    encode: bool = True,
    cleanup_between_cameras: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ],
    )

    cleanup_stale_ble_devices()

    if scan_wifi:
        cam = cameras[0]
        gopro = WirelessGoPro(
            target=cam["id"],
            interfaces={WirelessGoPro.Interface.BLE},
            maintain_state=False,
        )
        patch_gopro_no_wifi_disconnect(gopro)
        async with gopro:
            print(f"Scanning WiFi via GoPro {cam['id']}...")
            scan_resp = await gopro.ble_command.scan_wifi_networks()
            scan_id = scan_resp.data.scanning_id
            await asyncio.sleep(5)
            entries = await gopro.ble_command.get_ap_entries(scan_id=scan_id)
            print(f"\nFound {len(entries.data.entries)} WiFi network(s):")
            for entry in entries.data.entries:
                signal = getattr(entry, "signal_strength_bars", "?")
                print(f"  {entry.ssid}  (signal bars: {signal})")
        return

    print()
    print("=" * 60)
    print("FIRST-TIME PAIRING: each GoPro must be in pairing mode.")
    print("On the camera: Preferences > Connections > Connect Device")
    print("=" * 60)

    started: list[dict] = []
    failed: list[dict] = []

    for i, cam in enumerate(cameras):
        if i > 0:
            print(f"\n  Waiting {INTER_CAMERA_DELAY} s for BLE adapter cooldown...")
            if cleanup_between_cameras:
                cleanup_stale_ble_devices(macs=[cameras[i - 1]["mac"]])
            await asyncio.sleep(INTER_CAMERA_DELAY)

        try:
            await start_camera_stream(
                cam, wifi_ssid, wifi_password,
                resolution, min_bitrate, max_bitrate, start_bitrate, encode,
            )
            started.append(cam)
        except Exception as exc:
            print(f"  [{cam['id']}] FAILED: {exc}")
            failed.append(cam)

    print(f"\n{'='*60}")
    print("STREAM SUMMARY")
    print(f"{'='*60}")
    for cam in started:
        print(f"  OK  GoPro {cam['id']}  ->  {rtmp_url_for(cam['stream_key'])}")
    for cam in failed:
        print(f"  FAIL  GoPro {cam['id']}")
    print()
    print(f"NGINX stats: http://{RTMP_HOST_IP}:8080/stat")


async def stop_all_streams(cameras: list[dict]) -> None:
    for i, cam in enumerate(cameras):
        if i > 0:
            cleanup_stale_ble_devices(macs=[cameras[i - 1]["mac"]])
            await asyncio.sleep(INTER_CAMERA_DELAY)
        await stop_camera_stream(cam)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    _require_python_312()

    parser = argparse.ArgumentParser(
        description="Control GoPro RTMP livestreams via BLE"
    )
    parser.add_argument("--only", metavar="ID",
                        help="Only act on this camera (last 4 of serial, e.g. 1298)")
    parser.add_argument("--scan-gopros", action="store_true",
                        help="Scan for GoPro BLE devices and exit")
    parser.add_argument("--scan-wifi", action="store_true",
                        help="Scan WiFi networks via first camera and exit")
    parser.add_argument("--stop", action="store_true",
                        help="Stop streaming on all (or --only) cameras")
    parser.add_argument(
        "--cleanup-between-cameras",
        action="store_true",
        help="Also remove stale BLE entries between cameras (can be slow/noisy)",
    )

    wifi = parser.add_argument_group("WiFi")
    wifi.add_argument("--wifi-ssid", default=WIFI_SSID)
    wifi.add_argument("--wifi-password", default=WIFI_PASSWORD)

    stream = parser.add_argument_group("Stream quality")
    stream.add_argument("--resolution", type=int, choices=[480, 720, 1080],
                        default=DEFAULT_RESOLUTION)
    stream.add_argument("--min-bitrate", type=int, default=DEFAULT_MIN_BITRATE)
    stream.add_argument("--max-bitrate", type=int, default=DEFAULT_MAX_BITRATE)
    stream.add_argument("--start-bitrate", type=int, default=DEFAULT_START_BITRATE)
    stream.add_argument("--no-encode", action="store_true",
                        help="Don't save to SD card while streaming")

    args = parser.parse_args()

    try:
        if args.scan_gopros:
            asyncio.run(scan_gopros())
            return 0

        cameras = CAMERAS
        if args.only:
            cameras = [c for c in CAMERAS if c["id"] == args.only]
            if not cameras:
                ids = [c["id"] for c in CAMERAS]
                print(f"Camera '{args.only}' not found. Available: {ids}")
                return 1

        if args.stop:
            asyncio.run(stop_all_streams(cameras))
            # Stop path never opens WiFi — skip the reconnect check.
            return 0

        asyncio.run(
            start_all_streams(
                cameras=cameras,
                wifi_ssid=args.wifi_ssid,
                wifi_password=args.wifi_password,
                scan_wifi=args.scan_wifi,
                resolution=args.resolution,
                min_bitrate=args.min_bitrate,
                max_bitrate=args.max_bitrate,
                start_bitrate=args.start_bitrate,
                encode=not args.no_encode,
                cleanup_between_cameras=args.cleanup_between_cameras,
            )
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl-C).")
        return 130
    finally:
        # Defensively restore host connectivity in case BLE teardown dropped WiFi.
        # Skip on --stop: that path never opens WiFi so nothing to restore.
        if not args.stop:
            ensure_host_wifi_connected(args.wifi_ssid)


if __name__ == "__main__":
    raise SystemExit(main())
