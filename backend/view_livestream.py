#!/usr/bin/env python3
"""
View one or more RTMP livestreams in tiled windows using OpenCV.

Usage:
  # View all 3 GoPro streams in a tiled grid:
  python view_livestream.py

  # View a single stream:
  python view_livestream.py rtmp://10.0.0.57:1935/live/gopro1298

  # View specific cameras by key:
  python view_livestream.py gopro1298 gopro3404
"""

import cv2
import sys
import numpy as np

RTMP_HOST_IP = "10.0.0.57"

DEFAULT_STREAMS = {
    "gopro1298": f"rtmp://{RTMP_HOST_IP}:1935/live/gopro1298",
    "gopro3404": f"rtmp://{RTMP_HOST_IP}:1935/live/gopro3404",
    "gopro5497": f"rtmp://{RTMP_HOST_IP}:1935/live/gopro5497",
}


def view_single(url: str):
    """View a single RTMP stream."""
    print(f"Opening stream: {url}")
    print("Press 'q' to quit, 's' to save a screenshot")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"ERROR: Could not open stream at {url}")
        sys.exit(1)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or lost connection.")
            break
        cv2.imshow("GoPro Livestream", frame)
        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()


def view_multi(streams: dict[str, str]):
    """View multiple RTMP streams in a tiled grid."""
    names = list(streams.keys())
    urls = list(streams.values())
    caps = {}

    print(f"Opening {len(urls)} stream(s)...")
    for name, url in zip(names, urls):
        print(f"  {name}: {url}")
        cap = cv2.VideoCapture(url)
        caps[name] = cap

    print("Press 'q' to quit, 's' to save screenshots of all streams")

    # Grid layout: 1 stream = 1x1, 2 = 1x2, 3-4 = 2x2, etc.
    n = len(names)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    tile_w, tile_h = 640, 360

    frame_count = 0
    while True:
        tiles = []
        for name in names:
            ret, frame = caps[name].read()
            if ret:
                tile = cv2.resize(frame, (tile_w, tile_h))
            else:
                # Black tile with label for disconnected streams
                tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(tile, f"{name} - no signal", (20, tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Add label
            cv2.putText(tile, name, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            tiles.append(tile)

        # Pad to fill the grid
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

        # Assemble grid
        grid_rows = []
        for r in range(rows):
            row_tiles = tiles[r * cols:(r + 1) * cols]
            grid_rows.append(np.hstack(row_tiles))
        grid = np.vstack(grid_rows)

        cv2.imshow("GoPro Multi-View", grid)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            for name in names:
                ret, frame = caps[name].read()
                if ret:
                    fn = f"screenshot_{name}_{frame_count}.jpg"
                    cv2.imwrite(fn, frame)
                    print(f"Saved {fn}")

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


def main():
    args = sys.argv[1:]

    if not args:
        # No args: open all default streams
        view_multi(DEFAULT_STREAMS)
    elif len(args) == 1 and args[0].startswith("rtmp://"):
        # Single full URL
        view_single(args[0])
    else:
        # Stream keys or full URLs
        streams = {}
        for arg in args:
            if arg.startswith("rtmp://"):
                streams[arg.split("/")[-1]] = arg
            elif arg in DEFAULT_STREAMS:
                streams[arg] = DEFAULT_STREAMS[arg]
            else:
                print(f"Unknown stream key: {arg}. Available: {list(DEFAULT_STREAMS.keys())}")
                sys.exit(1)
        if len(streams) == 1:
            view_single(list(streams.values())[0])
        else:
            view_multi(streams)


if __name__ == "__main__":
    main()
