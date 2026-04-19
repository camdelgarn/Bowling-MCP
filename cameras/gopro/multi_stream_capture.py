"""
Capture three GoPro RTMP streams concurrently, display live windows,
and record synchronized timestamped MP4 files with per-frame timestamp CSVs.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StreamConfig:
    label: str
    stream_name: str
    url: str


class StreamReader(threading.Thread):
    """Continuously reads frames for one RTMP stream on its own thread."""

    def __init__(self, stream: StreamConfig, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.stream = stream
        self.stop_event = stop_event
        self.capture = cv2.VideoCapture(stream.url)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts_ns = 0
        self.read_frames = 0
        self.opened = self.capture.isOpened()
        self.error: str | None = None

        if self.opened:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def run(self) -> None:
        if not self.opened:
            self.error = f"Failed to open stream: {self.stream.url}"
            return

        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.02)
                continue

            ts_ns = time.time_ns()
            with self.lock:
                self.latest_frame = frame
                self.latest_ts_ns = ts_ns
                self.read_frames += 1

        self.capture.release()

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0
            return self.latest_frame.copy(), self.latest_ts_ns


def default_streams(host: str, port: int, app: str) -> list[StreamConfig]:
    names = ["gopro1", "gopro2", "gopro3"]
    streams = []
    for idx, name in enumerate(names, start=1):
        streams.append(
            StreamConfig(
                label=f"GoPro-{idx}",
                stream_name=name,
                url=f"rtmp://{host}:{port}/{app}/{name}",
            )
        )
    return streams


def load_streams(streams_file: Path) -> list[StreamConfig]:
    payload = json.loads(streams_file.read_text(encoding="utf-8"))
    streams = []
    for item in payload["streams"]:
        streams.append(
            StreamConfig(
                label=item["label"],
                stream_name=item["stream_name"],
                url=item["url"],
            )
        )
    return streams


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and record three GoPro RTMP streams with synchronized timestamps"
    )
    parser.add_argument("--host", default="10.0.0.57", help="NGINX RTMP host")
    parser.add_argument("--port", type=int, default=1935, help="NGINX RTMP port")
    parser.add_argument("--app", default="live", help="RTMP application name")
    parser.add_argument("--fps", type=float, default=30.0, help="Output MP4 FPS")
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID for output folder (default: UTC timestamp)",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/outputs/gopro_sessions",
        help="Root output directory",
    )
    parser.add_argument(
        "--streams-file",
        default=None,
        help="Optional JSON file with stream definitions",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=None,
        help="Width scale multiplier for each panel (default: auto-fit to screen height)",
    )
    return parser


def ensure_supported_python() -> None:
    """Fail fast on Python versions known to be unstable for this workflow."""
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "Python 3.13+ is not recommended for synchronized GoPro capture. "
            "Use Python 3.12.x instead."
        )


def open_writer(mp4_path: Path, fps: float, frame_shape):
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")
    return writer


def to_iso_utc(ts_ns: int) -> str:
    dt = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat()


def run_capture(args: argparse.Namespace) -> int:
    stop_event = threading.Event()

    def _stop_handler(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop_handler)

    if args.streams_file:
        streams = load_streams(Path(args.streams_file))
    else:
        streams = default_streams(args.host, args.port, args.app)

    if len(streams) != 3:
        raise ValueError("This workflow expects exactly 3 streams.")

    session_id = args.session or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_root = Path(args.output_dir) / session_id
    session_root.mkdir(parents=True, exist_ok=True)

    session_start_wall_ns = time.time_ns()
    session_start_perf_ns = time.perf_counter_ns()

    readers = [StreamReader(stream=s, stop_event=stop_event) for s in streams]
    for reader in readers:
        reader.start()

    for reader in readers:
        if not reader.opened:
            stop_event.set()
            print(reader.error or f"Failed to open {reader.stream.label}")
            return 1

    writers = {}
    timestamp_files = {}
    timestamp_writers = {}
    last_written_ts = {reader.stream.stream_name: 0 for reader in readers}
    frame_count = {reader.stream.stream_name: 0 for reader in readers}

    for reader in readers:
        stream = reader.stream
        csv_path = session_root / f"{stream.stream_name}_{session_id}_timestamps.csv"
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "capture_time_utc", "elapsed_ms"])
        timestamp_files[stream.stream_name] = csv_file
        timestamp_writers[stream.stream_name] = writer

    manifest = {
        "session_id": session_id,
        "session_start_utc": to_iso_utc(session_start_wall_ns),
        "session_start_wall_ns": session_start_wall_ns,
        "session_start_perf_ns": session_start_perf_ns,
        "status": "recording",
        "streams": [
            {
                "label": s.label,
                "stream_name": s.stream_name,
                "url": s.url,
                "mp4": str((session_root / f"{s.stream_name}_{session_id}.mp4")),
                "timestamps_csv": str(
                    session_root / f"{s.stream_name}_{session_id}_timestamps.csv"
                ),
            }
            for s in streams
        ],
    }
    manifest_path = session_root / "session_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Session started")
    print(f"Session ID: {session_id}")
    print(f"Output: {session_root}")
    print("Press q in the preview window to stop")

    # Detect screen dimensions to auto-size the window (Windows; falls back to 1080p).
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SetProcessDPIAware()
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except Exception:
        screen_w, screen_h = 1920, 1080

    TASKBAR_PX = 48  # reserve space for taskbar
    N_STREAMS = len(streams)
    panel_h = (screen_h - TASKBAR_PX) // N_STREAMS
    # panel_w is derived from actual frame AR once the first frame arrives;
    # seed with 16:9 so the window can be created immediately.
    panel_w = panel_h * 16 // 9
    window_sized = False

    # Create a resizable window so dragging the corner scales the content.
    cv2.namedWindow("GoPro Streams", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GoPro Streams", panel_w, panel_h * N_STREAMS)

    # Per-stream latest RAW frame (scaled at composite time to exact panel dims).
    display_frames: dict[str, object] = {r.stream.stream_name: None for r in readers}

    try:
        while not stop_event.is_set():
            wrote_any = False
            for reader in readers:
                frame, ts_ns = reader.get_latest()
                stream = reader.stream

                if frame is None or ts_ns <= last_written_ts[stream.stream_name]:
                    continue

                if stream.stream_name not in writers:
                    mp4_path = session_root / f"{stream.stream_name}_{session_id}.mp4"
                    writers[stream.stream_name] = open_writer(mp4_path, args.fps, frame.shape)

                writers[stream.stream_name].write(frame)
                frame_count[stream.stream_name] += 1
                elapsed_ms = int((ts_ns - session_start_wall_ns) / 1_000_000)
                timestamp_writers[stream.stream_name].writerow(
                    [frame_count[stream.stream_name], to_iso_utc(ts_ns), elapsed_ms]
                )
                last_written_ts[stream.stream_name] = ts_ns
                wrote_any = True

                display_frames[stream.stream_name] = frame  # store raw frame

            # Composite all available frames into one vertically stacked window.
            panels = [
                display_frames[r.stream.stream_name]
                for r in readers
                if display_frames[r.stream.stream_name] is not None
            ]
            if panels:
                # On first real frame, derive exact panel_w from actual aspect ratio.
                if not window_sized:
                    fh, fw = panels[0].shape[:2]
                    panel_w = int(panel_h * fw / fh)
                    cv2.resizeWindow("GoPro Streams", panel_w, panel_h * N_STREAMS)
                    window_sized = True

                # Resize each panel to exact pixel dims — no blurry upscaling.
                resized = [
                    cv2.resize(p, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
                    for p in panels
                ]
                composite = np.vstack(resized)
                cv2.imshow("GoPro Streams", composite)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()

            if not wrote_any:
                time.sleep(0.005)

    finally:
        stop_event.set()

        for reader in readers:
            reader.join(timeout=2.0)

        for writer in writers.values():
            writer.release()

        for csv_file in timestamp_files.values():
            csv_file.close()

        cv2.destroyAllWindows()

    print("Session stopped")
    for stream_name, count in frame_count.items():
        print(f"  {stream_name}: {count} frames")

    # Finalize manifest with actual stats.
    session_end_wall_ns = time.time_ns()
    duration_ms = int((session_end_wall_ns - session_start_wall_ns) / 1_000_000)
    for stream_entry in manifest["streams"]:
        sname = stream_entry["stream_name"]
        stream_entry["frames_recorded"] = frame_count.get(sname, 0)
        mp4_path = Path(stream_entry["mp4"])
        stream_entry["file_size_bytes"] = mp4_path.stat().st_size if mp4_path.exists() else 0
    manifest["status"] = "complete"
    manifest["session_end_utc"] = to_iso_utc(session_end_wall_ns)
    manifest["duration_ms"] = duration_ms
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSession manifest: {manifest_path}")
    print(f"Duration: {duration_ms / 1000:.1f}s")
    for s in manifest["streams"]:
        size_mb = s["file_size_bytes"] / 1_048_576
        print(f"  {s['stream_name']}: {s['frames_recorded']} frames  {size_mb:.1f} MB  {s['mp4']}")

    return 0


def main() -> int:
    ensure_supported_python()
    parser = make_parser()
    args = parser.parse_args()
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
