# Behind-the-Bowler Video Analysis

Analyze bowling approach walk-ups from a behind-the-bowler camera angle. Tracks which board the bowler starts on, ends on, and how much they drift during their approach.

## Prerequisites

```
pip install opencv-python numpy scipy matplotlib
```

Video files go in `../video/behind/` (e.g. GoPro footage at 5312×2988, 30fps).

## Quick Start

```bash
cd behind/

# Track bowler walk-up (auto-detects lane edges and empty frame)
python track_bowler.py ../video/behind/1.MP4

# Use existing calibration (faster, skips lane detection)
python track_bowler.py ../video/behind/1.MP4 --calibration lane_calibration.json
```

Output: `bowler_tracking.json` with per-frame foot position, board number, and shot summary.

## Scripts

### track_bowler.py

Main analysis script. Detects lane edges, subtracts background to find the bowler, tracks foot position frame-by-frame, converts to board numbers, and identifies the approach walk-up.

```
python track_bowler.py <video> [options]

positional arguments:
  video                   Path to the video file

options:
  --empty-frame N         Frame index with no bowler (auto-detected if omitted)
  --foul-y N              Foul line Y coordinate in pixels (default: 1350)
  --calibration FILE      Path to lane_calibration.json to skip calibration
  --output FILE           Output JSON path (default: bowler_tracking.json)
  --scale FACTOR          Downscale factor for processing (default: 0.25)
  --skip N                Process every Nth frame (default: 3)
```

By default, frames are downscaled to 25% and only every 3rd frame is processed, making it ~48x faster than full resolution. Use `--scale 1.0 --skip 1` for maximum accuracy.

**Example output:**
```
Shot 1:
  Frames: 408-476 (69 frames, 2.3s)
  Start: board 17.2 at y=2842 (t=13.61s)
  End:   board 18.1 at y=1487 (t=15.88s)
  Drift: +0.9 boards (straight)
```

### lane_calibration.py

Calibrates lane edge positions from a video frame. Saves results to `lane_calibration.json` for reuse by other scripts.

### detect_lane_track.py

Detects and tracks lane edges across multiple videos, starting from the foul line and searching upward. Useful for comparing lane visibility across different recordings.

### generate_overlay.py

Generates a composite lane overlay visualization showing lane edges and approach area.

### draw_boards.py

Draws board markers (0–40) on the approach with dot positions for visual reference.

## Data Files

- **lane_calibration.json** — Saved lane edge coefficients (`x = slope*y + intercept`), foul line Y, frame dimensions, and empty frame index.
- **bowler_tracking.json** — Per-shot tracking data including start/end board, drift, and frame-by-frame foot coordinates.

## Board Numbering

- Board 0 = right gutter edge
- Board 20 = center of lane
- Board 40 = left gutter edge
- Positive drift = moving left; negative = moving right
