#!/usr/bin/env python3
"""
Start GoPro RTMP livestreams via BLE for one or more cameras.

Connects to each GoPro over BLE, provisions WiFi, starts the livestream,
then disconnects BLE and moves to the next camera. Each GoPro streams
independently over WiFi — no ongoing BLE connection needed.

Usage examples:
  # Start all configured cameras:
  python gopro_livestream_ble.py

  # Start only one specific camera:
  python gopro_livestream_ble.py --only 1298

  # Scan for GoPro devices in range:
  python gopro_livestream_ble.py --scan-gopros

  # Scan WiFi networks via a specific GoPro:
  python gopro_livestream_ble.py --scan-wifi --only 1298
"""

import asyncio
import argparse
import logging
import subprocess
import re
import time
from open_gopro import WirelessGoPro
from open_gopro.models import proto

# ============================================================
# Configuration — update these for your location / network
# ============================================================
WIFI_SSID = "northbow-public-5g-2"
WIFI_PASSWORD = "northbowl"
RTMP_HOST_IP = "192.168.1.7"  # Your PC's LAN IP on the WiFi network

# Camera definitions: identifier (last 4 of serial), BLE MAC, RTMP stream key
CAMERAS = [
    {"id": "1298", "mac": "C3BD00F1C3D3", "stream_key": "gopro1298"},
    {"id": "3404", "mac": "EAC313813BD5", "stream_key": "gopro3404"},
    {"id": "5497", "mac": "E86DDA5629DA", "stream_key": "gopro5497"},
]

# Stream defaults
DEFAULT_RESOLUTION = 1080
DEFAULT_MIN_BITRATE = 800
DEFAULT_MAX_BITRATE = 8000
DEFAULT_START_BITRATE = 4000

# Delay (seconds) between camera connections to let Windows BLE adapter reset
INTER_CAMERA_DELAY = 8


def cleanup_stale_ble_devices(macs: list[str] | None = None):
    """Remove stale GoPro BLE device entries from Windows.

    Args:
        macs: List of MAC addresses (no colons) to clean up.
              Defaults to all configured cameras.
    """
    if macs is None:
        macs = [cam["mac"] for cam in CAMERAS]
    print("Checking for stale GoPro BLE devices...")
    try:
        result = subprocess.run(
            ["pnputil", "/enum-devices", "/class", "Bluetooth"],
            capture_output=True, text=True, timeout=10
        )
        all_device_ids = []
        for mac in macs:
            mac_lower = mac.lower()
            ids = re.findall(
                r"Instance ID:\s+(.+?" + mac_lower + r".+?)$",
                result.stdout, re.MULTILINE | re.IGNORECASE
            )
            ids += re.findall(
                r"Instance ID:\s+(BTHLE\\Dev_" + mac_lower + r".+?)$",
                result.stdout, re.MULTILINE | re.IGNORECASE
            )
            all_device_ids.extend(ids)

        all_device_ids = list(set(did.strip() for did in all_device_ids))

        if not all_device_ids:
            print("  No stale BLE devices found.")
            return

        print(f"  Found {len(all_device_ids)} stale BLE device(s), removing...")
        remove_cmds = "; ".join(
            f'pnputil /remove-device "{did}"' for did in all_device_ids
        )
        subprocess.run(
            ["powershell", "-Command",
             f"Start-Process powershell -Verb RunAs -Wait -ArgumentList "
             f"'-Command {remove_cmds}'"],
            timeout=30
        )
        print("  Stale BLE devices removed.")
    except Exception as e:
        print(f"  Warning: BLE cleanup failed ({e}), continuing anyway...")


def patch_gopro_no_wifi_disconnect(gopro: WirelessGoPro):
    """Prevent the SDK from disconnecting the host PC's WiFi on exit."""
    async def _noop_close_wifi() -> None:
        logging.getLogger(__name__).info("Skipping host WiFi disconnect (BLE-only mode)")
    gopro._close_wifi = _noop_close_wifi


async def scan_gopros():
    """Scan for GoPro BLE devices in range."""
    from bleak import BleakScanner
    print("Scanning for GoPro BLE devices (10 seconds)...")
    devices = await BleakScanner.discover(timeout=10)
    gopros = [d for d in devices if d.name and "GoPro" in d.name]
    if not gopros:
        print("No GoPro devices found.")
    else:
        print(f"Found {len(gopros)} GoPro(s):")
        for d in gopros:
            mac_no_colons = d.address.replace(":", "")
            configured = any(c["mac"].upper() == mac_no_colons.upper() for c in CAMERAS)
            status = " [configured]" if configured else " [NOT in CAMERAS list]"
            print(f"  {d.name}  MAC: {d.address}{status}")


async def start_camera_stream(
    cam: dict,
    wifi_ssid: str,
    wifi_password: str,
    resolution: int,
    min_bitrate: int,
    max_bitrate: int,
    start_bitrate: int,
    encode: bool,
):
    """Connect to one GoPro, provision WiFi, start livestream, then disconnect BLE."""
    identifier = cam["id"]
    stream_key = cam["stream_key"]
    rtmp_url = f"rtmp://{RTMP_HOST_IP}:1935/live/{stream_key}"

    window_size_map = {
        480: proto.EnumWindowSize.WINDOW_SIZE_480,
        720: proto.EnumWindowSize.WINDOW_SIZE_720,
        1080: proto.EnumWindowSize.WINDOW_SIZE_1080,
    }
    window_size = window_size_map.get(resolution, proto.EnumWindowSize.WINDOW_SIZE_1080)

    print(f"\n{'='*60}")
    print(f"Starting GoPro {identifier} → {rtmp_url}")
    print(f"{'='*60}")

    gopro = WirelessGoPro(
        target=identifier,
        interfaces={WirelessGoPro.Interface.BLE},
        maintain_state=False,
    )
    patch_gopro_no_wifi_disconnect(gopro)

    async with gopro:
        print(f"  [{identifier}] Connected via BLE")

        # Provision WiFi
        print(f"  [{identifier}] Connecting to WiFi: {wifi_ssid}")
        wifi_resp = await gopro.ble_command.request_wifi_connect_new(ssid=wifi_ssid, password=wifi_password)
        print(f"  [{identifier}] WiFi connect response: {wifi_resp.status}")

        # Wait for WiFi to establish and verify via livestream status
        print(f"  [{identifier}] Waiting for WiFi connection to establish...")
        await asyncio.sleep(5)

        # Register for livestream status to monitor connection
        livestream_status = await gopro.ble_command.register_livestream_status(
            register=[proto.EnumRegisterLiveStreamStatus.REGISTER_LIVE_STREAM_STATUS_STATUS,
                      proto.EnumRegisterLiveStreamStatus.REGISTER_LIVE_STREAM_STATUS_ERROR]
        )
        print(f"  [{identifier}] Livestream status registration: {livestream_status.status}")

        # Configure livestream
        print(f"  [{identifier}] Configuring livestream ({resolution}p, {start_bitrate} kbps)")
        print(f"  [{identifier}] RTMP URL: {rtmp_url}")
        resp = await gopro.ble_command.set_livestream_mode(
            url=rtmp_url,
            minimum_bitrate=min_bitrate,
            maximum_bitrate=max_bitrate,
            starting_bitrate=start_bitrate,
            encode=encode,
            window_size=window_size,
            lens=proto.EnumLens.LENS_LINEAR,
        )
        print(f"  [{identifier}] Livestream mode set: {resp.status}")

        # Wait for livestream to be ready before starting shutter
        print(f"  [{identifier}] Waiting for livestream to be ready...")
        await asyncio.sleep(3)

        # Start streaming
        print(f"  [{identifier}] Starting livestream (shutter on)...")
        await gopro.ble_command.set_shutter(shutter=True)

        # Wait and check if stream actually started
        print(f"  [{identifier}] Waiting for stream to establish...")
        await asyncio.sleep(5)
        print(f"  [{identifier}] LIVE! Stream: {rtmp_url}")

    # BLE disconnects here — GoPro keeps streaming over WiFi
    print(f"  [{identifier}] BLE disconnected. Camera continues streaming independently.")


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
    log: str = "gopro_livestream.log",
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if log:
        fh = logging.FileHandler(log)
        fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(fh)

    cleanup_stale_ble_devices()

    # If scanning WiFi, just use the first camera to scan
    if scan_wifi:
        cam = cameras[0]
        gopro = WirelessGoPro(target=cam["id"], interfaces={WirelessGoPro.Interface.BLE}, maintain_state=False)
        patch_gopro_no_wifi_disconnect(gopro)
        async with gopro:
            print(f"Scanning WiFi via GoPro {cam['id']}...")
            scan_resp = await gopro.ble_command.scan_wifi_networks()
            scan_id = scan_resp.data.scanning_id
            await asyncio.sleep(5)
            entries_resp = await gopro.ble_command.get_ap_entries(scan_id=scan_id)
            print(f"\nFound {len(entries_resp.data.entries)} WiFi network(s):")
            for i, entry in enumerate(entries_resp.data.entries):
                signal = getattr(entry, "signal_strength_bars", "?")
                freq = getattr(entry, "signal_frequency_mhz", "?")
                print(f"  {i+1}. {entry.ssid}  (signal: {signal}, freq: {freq} MHz)")
        return

    # Start each camera sequentially (BLE can only pair one at a time)
    print("\n" + "="*60)
    print("IMPORTANT: Each GoPro must be in pairing mode the FIRST time.")
    print("On each camera: Preferences > Connections > Connect Device")
    print("="*60)

    started = []
    failed = []
    for i, cam in enumerate(cameras):
        if i > 0:
            # Clean up the BLE bond from the previous camera and give adapter time to settle
            print(f"\n  Waiting {INTER_CAMERA_DELAY}s for BLE adapter cooldown...")
            cleanup_stale_ble_devices(macs=[cameras[i-1]["mac"]])
            await asyncio.sleep(INTER_CAMERA_DELAY)

        try:
            await start_camera_stream(
                cam, wifi_ssid, wifi_password,
                resolution, min_bitrate, max_bitrate, start_bitrate, encode,
            )
            started.append(cam)
        except Exception as e:
            print(f"  [{cam['id']}] FAILED: {e}")
            failed.append(cam)

    # Summary
    print(f"\n{'='*60}")
    print("STREAM SUMMARY")
    print(f"{'='*60}")
    for cam in started:
        url = f"rtmp://{RTMP_HOST_IP}:1935/live/{cam['stream_key']}"
        print(f"  ✓ GoPro {cam['id']} → {url}")
    for cam in failed:
        print(f"  ✗ GoPro {cam['id']} — failed to start")
    print()
    print("All cameras are streaming independently over WiFi.")
    print("View streams with:  python view_livestream.py rtmp://...")
    print("Check NGINX stats:  http://localhost:8080/stat")


def main():
    parser = argparse.ArgumentParser(
        description="Start GoPro RTMP livestreams via BLE (multi-camera)"
    )
    # Camera selection
    parser.add_argument("--only", help="Only start this camera (last 4 digits of serial, e.g. 1298)")
    parser.add_argument("--scan-gopros", action="store_true", help="Scan for GoPro BLE devices and exit")
    parser.add_argument("--scan-wifi", action="store_true", help="Scan WiFi networks via first camera and exit")

    # WiFi network options
    wifi_group = parser.add_argument_group("WiFi network (configured over BLE)")
    wifi_group.add_argument("--wifi-ssid", default=WIFI_SSID, help="WiFi SSID")
    wifi_group.add_argument("--wifi-password", default=WIFI_PASSWORD, help="WiFi password")

    # Stream options
    stream_group = parser.add_argument_group("Livestream settings")
    stream_group.add_argument("--resolution", type=int, choices=[480, 720, 1080], default=DEFAULT_RESOLUTION)
    stream_group.add_argument("--min-bitrate", type=int, default=DEFAULT_MIN_BITRATE)
    stream_group.add_argument("--max-bitrate", type=int, default=DEFAULT_MAX_BITRATE)
    stream_group.add_argument("--start-bitrate", type=int, default=DEFAULT_START_BITRATE)
    stream_group.add_argument("--no-encode", action="store_true", help="Don't save to SD card while streaming")

    # General
    parser.add_argument("--log", default="gopro_livestream.log")
    args = parser.parse_args()

    if args.scan_gopros:
        asyncio.run(scan_gopros())
        return

    cameras = CAMERAS
    if args.only:
        cameras = [c for c in CAMERAS if c["id"] == args.only]
        if not cameras:
            print(f"Camera '{args.only}' not found in CAMERAS config. Available: {[c['id'] for c in CAMERAS]}")
            return

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
            log=args.log,
        )
    )


if __name__ == "__main__":
    main()
