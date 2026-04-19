# gopro.py

"""
GoPro-specific connection and stream handling.

This module provides:
- RTMP nginx start/stop helpers for local relay
- BLE discovery and connectivity checks for GoPro cameras
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cameras.camera import CameraBase

try:
    _bleak = importlib.import_module("bleak")
    BleakClient = _bleak.BleakClient
    BleakScanner = _bleak.BleakScanner
except ImportError:  # pragma: no cover - optional dependency
    BleakClient = None
    BleakScanner = None


@dataclass
class GoProConfig:
    """Runtime config for a GoPro camera and local RTMP endpoint."""

    ble_address: str | None = None
    stream_name: str = "gopro"
    rtmp_host: str = "10.0.0.57"
    rtmp_port: int = 1935
    rtmp_app: str = "live"

    @property
    def rtmp_url(self) -> str:
        return f"rtmp://{self.rtmp_host}:{self.rtmp_port}/{self.rtmp_app}/{self.stream_name}"


class GoProCamera(CameraBase):
    """GoPro camera helper for BLE connectivity and RTMP server control."""

    def __init__(self, config: GoProConfig | None = None, project_root: str | Path | None = None):
        self.config = config or GoProConfig()

        if project_root is None:
            # cameras/gopro/gopro.py -> project root is two parents up from cameras
            project_root = Path(__file__).resolve().parents[2]

        self.project_root = Path(project_root)
        self.rtmp_dir = self.project_root / "rtmp"
        self.nginx_exe = self.rtmp_dir / "nginx.exe"
        self.nginx_conf = self.rtmp_dir / "nginx_rtmp.conf"
        self.stop_script = self.rtmp_dir / "stop.bat"

    def _require_bleak(self) -> None:
        if BleakScanner is None or BleakClient is None:
            raise RuntimeError(
                "BLE features require the 'bleak' package. Install with: pip install bleak"
            )

    async def scan_ble_devices(self, timeout: float = 8.0, name_contains: str = "GoPro"):
        """Scan nearby BLE devices and return GoPro-like matches."""
        self._require_bleak()

        discovered = await BleakScanner.discover(timeout=timeout)
        results = []
        name_filter = name_contains.lower()

        for device in discovered:
            name = (device.name or "").strip()
            if not name:
                continue
            if name_filter in name.lower():
                results.append(
                    {
                        "name": name,
                        "address": device.address,
                        "rssi": getattr(device, "rssi", None),
                    }
                )
        return results

    async def connect(self, timeout: float = 10.0) -> bool:
        """Connect to a GoPro camera over BLE to verify reachability."""
        self._require_bleak()

        address = self.config.ble_address
        if not address:
            devices = await self.scan_ble_devices(timeout=8.0)
            if not devices:
                return False
            address = devices[0]["address"]
            self.config.ble_address = address

        async with BleakClient(address, timeout=timeout) as client:
            return bool(client.is_connected)

    def start_nginx(self, force_reload: bool = False) -> None:
        """Start local RTMP nginx server using the workspace rtmp folder."""
        if not self.nginx_exe.exists():
            raise FileNotFoundError(f"nginx.exe not found: {self.nginx_exe}")
        if not self.nginx_conf.exists():
            raise FileNotFoundError(f"nginx config not found: {self.nginx_conf}")

        command = [str(self.nginx_exe), "-c", str(self.nginx_conf)]
        if force_reload:
            command.extend(["-s", "reload"])

        subprocess.run(command, cwd=self.rtmp_dir, check=False)

    def stop_nginx(self) -> None:
        """Stop local RTMP nginx server."""
        if self.stop_script.exists():
            subprocess.run([str(self.stop_script)], cwd=self.rtmp_dir, check=False, shell=True)
            return

        if self.nginx_exe.exists():
            subprocess.run([str(self.nginx_exe), "-s", "stop"], cwd=self.rtmp_dir, check=False)

    def start_stream(self) -> str:
        """Return the RTMP publish URL that the GoPro should use."""
        return self.config.rtmp_url


def ensure_supported_python() -> None:
    """Fail fast on Python versions known to be unstable for this workflow."""
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "Python 3.13+ is not recommended for this GoPro workflow. "
            "Use Python 3.12.x instead."
        )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GoPro BLE + RTMP helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("start-nginx", help="Start local RTMP nginx")
    sub.add_parser("stop-nginx", help="Stop local RTMP nginx")

    scan_cmd = sub.add_parser("scan", help="Scan BLE for GoPro devices")
    scan_cmd.add_argument("--timeout", type=float, default=8.0)

    connect_cmd = sub.add_parser("connect", help="Connect BLE to a GoPro")
    connect_cmd.add_argument("--address", type=str, default=None)
    connect_cmd.add_argument("--timeout", type=float, default=10.0)

    sub.add_parser("stream-url", help="Show GoPro RTMP publish URL")

    stream_cmd = sub.add_parser("stream", help="Start all 3 GoPros streaming RTMP via BLE")
    stream_cmd.add_argument("--only", metavar="ID", default=None,
                            help="Only start this camera (last 4 of serial, e.g. 1298)")
    stream_cmd.add_argument("--resolution", type=int, choices=[480, 720, 1080], default=1080)
    stream_cmd.add_argument("--no-encode", action="store_true")

    stop_cmd = sub.add_parser("stop-stream", help="Stop all (or one) GoPro RTMP streams")
    stop_cmd.add_argument("--only", metavar="ID", default=None)

    return parser


def main() -> int:
    ensure_supported_python()
    parser = _build_cli()
    args = parser.parse_args()

    camera = GoProCamera(config=GoProConfig(ble_address=getattr(args, "address", None)))

    if args.command == "start-nginx":
        camera.start_nginx()
        print("nginx start requested")
        return 0

    if args.command == "stop-nginx":
        camera.stop_nginx()
        print("nginx stop requested")
        return 0

    if args.command == "scan":
        devices = asyncio.run(camera.scan_ble_devices(timeout=args.timeout))
        if not devices:
            print("No GoPro BLE devices found")
            return 1
        for device in devices:
            print(f"{device['name']} | {device['address']} | RSSI={device['rssi']}")
        return 0

    if args.command == "connect":
        connected = asyncio.run(camera.connect(timeout=args.timeout))
        print(f"connected={connected}, address={camera.config.ble_address}")
        return 0 if connected else 1

    if args.command == "stream-url":
        print(camera.start_stream())
        return 0

    if args.command in ("stream", "stop-stream"):
        from cameras.gopro import livestream_ble as _ls
        cameras_list = _ls.CAMERAS
        only = getattr(args, "only", None)
        if only:
            cameras_list = [c for c in cameras_list if c["id"] == only]
            if not cameras_list:
                print(f"Camera '{only}' not found. Available: {[c['id'] for c in _ls.CAMERAS]}")
                return 1
        if args.command == "stop-stream":
            asyncio.run(_ls.stop_all_streams(cameras_list))
        else:
            asyncio.run(
                _ls.start_all_streams(
                    cameras=cameras_list,
                    wifi_ssid=_ls.WIFI_SSID,
                    wifi_password=_ls.WIFI_PASSWORD,
                    resolution=getattr(args, "resolution", 1080),
                    encode=not getattr(args, "no_encode", False),
                )
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
