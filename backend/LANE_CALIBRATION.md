# Bowling Lane Calibration — Detection Results

## Camera Setup
- **Camera**: GoPro Hero (MAC suffix 1298)
- **Lens**: Linear
- **Position**: Raised, behind the approach looking down the lane
- **Resolution**: 1920×1080 @ 30fps
- **Stream**: RTMP to `rtmp://192.168.1.7:1935/live/stream`
- **Mean brightness**: ~65.6

## Bowling Lane Dimensions
- **Lane width**: 39 boards
- **Board numbering**: Left to right, board 0 (left gutter edge) to board 39 (right gutter edge)
- **Center board**: Board 20

## Approach Dot Layout
This bowling alley has **5 dots per row**, spaced **5 boards apart**, centered on **board 20**.

| Dot | Board |
|-----|-------|
| 1 (leftmost) | 10 |
| 2 | 15 |
| 3 (center) | 20 |
| 4 | 25 |
| 5 (rightmost) | 30 |

> Other alleys may have 7 dots per row (boards 5–35).

### Row Positions
| Row | Distance from Foul Line | Mean Y (px) | Dots Found | px/board |
|-----|------------------------|-------------|------------|----------|
| Row 1 (farther from camera) | ~12 ft | ~403 | 5 | 10.48 |
| Row 2 (closer to camera) | ~15 ft | ~629 | 5 | 15.56 |

### Detected Dot Coordinates
**Row 1** (y ≈ 388–418):

| Board | x | y | size | contrast |
|-------|---|---|------|----------|
| 10 | 877 | 418 | 5.0 | 58.7 |
| 15 | 929 | 411 | 4.4 | 49.9 |
| 20 | 981 | 404 | 4.7 | 53.2 |
| 25 | 1035 | 396 | 5.1 | 49.0 |
| 30 | 1086 | 388 | 4.7 | 44.7 |

**Row 2** (y ≈ 605–652):

| Board | x | y | size | contrast |
|-------|---|---|------|----------|
| 10 | 907 | 652 | 6.6 | 98.2 |
| 15 | 984 | 641 | 5.9 | 85.3 |
| 20 | 1064 | 629 | 8.4 | 109.4 |
| 25 | 1140 | 618 | 6.3 | 87.0 |
| 30 | 1218 | 605 | 6.8 | 103.3 |

## Lane Edge Calibration
Computed from 2-row dot calibration using linear fit through each board's position at both rows.

| Edge | Equation (x = slope × y + intercept) |
|------|--------------------------------------|
| Left (board 0) | x = −0.0850 × y + 808.8 |
| Right (board 39) | x = 0.8468 × y + 863.3 |

**Vanishing point**: (814, −58) — above the frame

### Lane Width at Key Y Positions
| Position | y | Left x | Right x | Width (px) | Boards |
|----------|---|--------|---------|-----------|--------|
| Top of visible area | 103 | 800 | 950 | 150 | ~40.5 |
| Middle | 516 | 765 | 1300 | 535 | ~41.1 |
| Bottom | 929 | 730 | 1650 | 920 | ~41.2 |

## Detection Method

### Board Assignment
- **Middle dot = center** (board 20). For 5 dots, the middle (3rd) dot is the center.
- Board numbers assigned symmetrically: `board = 20 + (i − mid_index) × 5`
- This approach is simpler and more reliable than blob-size or perspective-matching heuristics, since dot sizes at the farther row are too similar to distinguish center by size.

### Dot Detection Pipeline
1. **Blob detection** (OpenCV `SimpleBlobDetector`): dark blobs, min area 8, circularity ≥ 0.20, convexity ≥ 0.30
2. **Contrast filter**: neighborhood mean − pixel value > 25 (real dots have contrast 45–110)
3. **Y-clustering**: group blobs within 60px of each other vertically (wide tolerance for perspective tilt)
4. **Collinear run finding** (`_find_best_dot_run`): for each cluster, find the longest evenly-spaced collinear sequence; score by `length × avg_contrast` to prefer real dots over screw cover artifacts

### Filtering Screw Cover Dots
Random dots from screw covers appear on the approach (contrast ~40–50, scattered at y ≈ 475–500). They are filtered out because:
- They don't form evenly-spaced collinear runs
- The `length × avg_contrast` scoring prefers the real dot rows which have more dots and higher contrast

### Board Line Drawing
Each board line is a single straight line from top to bottom, computed by finding the board's x-position at both calibrated dot rows and fitting a line through those two points. No per-row interpolation — lines are truly straight, converging toward the vanishing point.
