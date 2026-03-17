#!/usr/bin/env python3
"""
Now that we know the foul line is at y≈1330, detect the lane edges
precisely at various y positions to build the lane outline trapezoid.
Also detect the approach edges (wider than lane).
"""

import cv2
import numpy as np

frame = cv2.imread("empty_frame.png")
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame: {w}x{h}")

FOUL_LINE_Y = 1330

# Strategy: at each row in the lane area (y=200 to y=1330),
# find the left and right edges of the lane surface.
# The lane is a bright strip with darker gutters on each side.
# 
# Approach: find the intensity profile, and locate the edges
# where we transition from lane to gutter (brightness drop).

# First, let's establish what the "lane brightness" is at different y levels.
# At y=700, the lane center was around x=2950-3050 with intensity ~178.

# Let's scan more carefully. For each y, we'll:
# 1. Take the smoothed intensity profile
# 2. Find the widest contiguous region above a local brightness threshold
#    (approach/lane surface should be the widest bright region)
# 3. Use adaptive threshold based on each row's own statistics

print("=== Lane edge detection ===")
print("Scanning for the widest bright contiguous region at each y level")

lane_edges = []  # (y, left_x, right_x)

for y in range(100, h - 50, 5):
    row = gray[y, :].astype(float)
    
    # Smooth to reduce noise
    k = 31
    smooth = np.convolve(row, np.ones(k)/k, mode='same')
    
    # Use local statistics for threshold
    # The lane/approach surface should be brighter than the gutters and surroundings
    # Only look in the relevant x range (1500-4800 based on earlier analysis)
    seg = smooth[1500:4800]
    
    if np.max(seg) < 30:  # Too dark, skip
        continue
    
    # Adaptive threshold: use the 60th percentile of this segment
    thresh = np.percentile(seg, 60)
    if thresh < 30:
        thresh = 30
    
    # Find contiguous bright regions
    bright = seg > thresh
    
    # Find runs of True (bright pixels)
    runs = []
    in_run = False
    start = 0
    for i in range(len(bright)):
        if bright[i]:
            if not in_run:
                in_run = True
                start = i
        else:
            if in_run:
                runs.append((start + 1500, i + 1500, i - start))
                in_run = False
    if in_run:
        runs.append((start + 1500, len(bright) + 1500, len(bright) - start))
    
    if runs:
        # For the lane area (y < foul_line), take the widest run
        # For the approach area (y >= foul_line), same
        widest = max(runs, key=lambda r: r[2])
        if widest[2] > 200:  # minimum width filter
            lane_edges.append((y, widest[0], widest[1]))

# Now analyze the edges
print(f"\nFound {len(lane_edges)} rows with detected bright strip")

# Separate into lane (above foul line) and approach (below)
lane_part = [(y, lx, rx) for y, lx, rx in lane_edges if y < FOUL_LINE_Y]
approach_part = [(y, lx, rx) for y, lx, rx in lane_edges if y >= FOUL_LINE_Y]

print(f"\nLane rows: {len(lane_part)} (y={lane_part[0][0] if lane_part else '?'} to y={lane_part[-1][0] if lane_part else '?'})")
print(f"Approach rows: {len(approach_part)} (y={approach_part[0][0] if approach_part else '?'} to y={approach_part[-1][0] if approach_part else '?'})")

# Print lane edges
print("\nLane edges:")
print("  y   | left_x | right_x | width  | center")
print("-" * 55)
for y, lx, rx in lane_part[::max(1, len(lane_part)//25)]:
    w_ = rx - lx
    cx = (lx + rx) // 2
    print(f"  {y:>4} | {lx:>6} | {rx:>7} | {w_:>6} | {cx:>6}")

print("\nApproach edges:")
print("  y   | left_x | right_x | width  | center")
print("-" * 55)
for y, lx, rx in approach_part[::max(1, len(approach_part)//25)]:
    w_ = rx - lx
    cx = (lx + rx) // 2
    print(f"  {y:>4} | {lx:>6} | {rx:>7} | {w_:>6} | {cx:>6}")

# Fit lines to the lane edges to get clean trapezoid corners
# The lane edges should be roughly linear (perspective lines converging toward pins)
if lane_part:
    lane_ys = np.array([y for y, _, _ in lane_part])
    lane_lefts = np.array([lx for _, lx, _ in lane_part])
    lane_rights = np.array([rx for _, _, rx in lane_part])
    
    # Use robust fitting (median of nearby points) rather than linear regression
    # to handle noise
    
    # Group by y bands
    bands = []
    band_size = 50
    for y_start in range(int(lane_ys.min()), int(lane_ys.max()), band_size):
        mask = (lane_ys >= y_start) & (lane_ys < y_start + band_size)
        if np.sum(mask) >= 3:
            bands.append({
                'y': y_start + band_size // 2,
                'left': np.median(lane_lefts[mask]),
                'right': np.median(lane_rights[mask]),
                'n': np.sum(mask)
            })
    
    print("\n\nLane edges (median by band):")
    print("  y_center | left  | right | width | n")
    for b in bands:
        print(f"  {b['y']:>8} | {b['left']:>5.0f} | {b['right']:>5.0f} | {b['right']-b['left']:>5.0f} | {b['n']}")
    
    # Fit linear model to the banded data
    band_ys = np.array([b['y'] for b in bands])
    band_lefts = np.array([b['left'] for b in bands])
    band_rights = np.array([b['right'] for b in bands])
    
    # Linear fit: x = a*y + b
    left_fit = np.polyfit(band_ys, band_lefts, 1)
    right_fit = np.polyfit(band_ys, band_rights, 1)
    
    print(f"\nLinear fits:")
    print(f"  Left edge:  x = {left_fit[0]:.3f}*y + {left_fit[1]:.1f}")
    print(f"  Right edge: x = {right_fit[0]:.3f}*y + {right_fit[1]:.1f}")
    
    # Compute lane corners
    # Top of lane (smallest y where we have data)
    lane_top_y = int(band_ys[0])
    lane_top_left = int(left_fit[0] * lane_top_y + left_fit[1])
    lane_top_right = int(right_fit[0] * lane_top_y + right_fit[1])
    
    # Bottom of lane (foul line)
    lane_bot_y = FOUL_LINE_Y
    lane_bot_left = int(left_fit[0] * lane_bot_y + left_fit[1])
    lane_bot_right = int(right_fit[0] * lane_bot_y + right_fit[1])
    
    print(f"\nLane trapezoid corners:")
    print(f"  Top (pins):      ({lane_top_left}, {lane_top_y}) - ({lane_top_right}, {lane_top_y})")
    print(f"  Bottom (foul):   ({lane_bot_left}, {lane_bot_y}) - ({lane_bot_right}, {lane_bot_y})")
    print(f"  Top width: {lane_top_right - lane_top_left}")
    print(f"  Bottom width: {lane_bot_right - lane_bot_left}")

# Same for approach
if approach_part:
    app_ys = np.array([y for y, _, _ in approach_part])
    app_lefts = np.array([lx for _, lx, _ in approach_part])
    app_rights = np.array([rx for _, _, rx in approach_part])
    
    # Group by y bands  
    app_bands = []
    for y_start in range(int(app_ys.min()), int(app_ys.max()), 100):
        mask = (app_ys >= y_start) & (app_ys < y_start + 100)
        if np.sum(mask) >= 3:
            app_bands.append({
                'y': y_start + 50,
                'left': np.median(app_lefts[mask]),
                'right': np.median(app_rights[mask]),
                'n': np.sum(mask)
            })
    
    print("\n\nApproach edges (median by band):")
    print("  y_center | left  | right | width | n")
    for b in app_bands:
        print(f"  {b['y']:>8} | {b['left']:>5.0f} | {b['right']:>5.0f} | {b['right']-b['left']:>5.0f} | {b['n']}")
    
    app_band_ys = np.array([b['y'] for b in app_bands])
    app_band_lefts = np.array([b['left'] for b in app_bands])
    app_band_rights = np.array([b['right'] for b in app_bands])
    
    app_left_fit = np.polyfit(app_band_ys, app_band_lefts, 1)
    app_right_fit = np.polyfit(app_band_ys, app_band_rights, 1)
    
    print(f"\nLinear fits:")
    print(f"  Left edge:  x = {app_left_fit[0]:.3f}*y + {app_left_fit[1]:.1f}")
    print(f"  Right edge: x = {app_right_fit[0]:.3f}*y + {app_right_fit[1]:.1f}")
    
    # Approach corners
    app_top_y = FOUL_LINE_Y
    app_top_left = int(app_left_fit[0] * app_top_y + app_left_fit[1])
    app_top_right = int(app_right_fit[0] * app_top_y + app_right_fit[1])
    
    app_bot_y = h - 1
    app_bot_left = int(app_left_fit[0] * app_bot_y + app_left_fit[1])
    app_bot_right = int(app_right_fit[0] * app_bot_y + app_right_fit[1])
    
    print(f"\nApproach trapezoid corners:")
    print(f"  Top (foul):      ({app_top_left}, {app_top_y}) - ({app_top_right}, {app_top_y})")
    print(f"  Bottom (camera): ({app_bot_left}, {app_bot_y}) - ({app_bot_right}, {app_bot_y})")
    print(f"  Top width: {app_top_right - app_top_left}")
    print(f"  Bottom width: {app_bot_right - app_bot_left}")

# Save visualization
debug = frame.copy()

# Draw all detected edges
for y, lx, rx in lane_edges[::3]:
    color = (0, 255, 0) if y < FOUL_LINE_Y else (255, 150, 0)
    cv2.circle(debug, (lx, y), 2, color, -1)
    cv2.circle(debug, (rx, y), 2, color, -1)

if lane_part and len(bands) > 1:
    # Draw lane trapezoid
    cv2.line(debug, (lane_top_left, lane_top_y), (lane_top_right, lane_top_y), (0, 255, 0), 3)
    cv2.line(debug, (lane_bot_left, lane_bot_y), (lane_bot_right, lane_bot_y), (0, 255, 0), 3)
    cv2.line(debug, (lane_top_left, lane_top_y), (lane_bot_left, lane_bot_y), (0, 255, 0), 3)
    cv2.line(debug, (lane_top_right, lane_top_y), (lane_bot_right, lane_bot_y), (0, 255, 0), 3)
    cv2.putText(debug, "LANE", ((lane_top_left+lane_top_right)//2-50, (lane_top_y+lane_bot_y)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

# Foul line
cv2.line(debug, (1500, FOUL_LINE_Y), (4800, FOUL_LINE_Y), (0, 0, 255), 4)
cv2.putText(debug, "FOUL LINE", (1500, FOUL_LINE_Y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

if approach_part and len(app_bands) > 1:
    # Draw approach trapezoid
    cv2.line(debug, (app_top_left, app_top_y), (app_top_right, app_top_y), (255, 150, 0), 3)
    cv2.line(debug, (app_bot_left, app_bot_y), (app_bot_right, app_bot_y), (255, 150, 0), 3)
    cv2.line(debug, (app_top_left, app_top_y), (app_bot_left, app_bot_y), (255, 150, 0), 3)
    cv2.line(debug, (app_top_right, app_top_y), (app_bot_right, app_bot_y), (255, 150, 0), 3)
    cv2.putText(debug, "APPROACH", ((app_top_left+app_top_right)//2-80, (app_top_y+app_bot_y)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 150, 0), 3)

cv2.imwrite("lane_outline_on_video.png", debug)
print(f"\nSaved lane_outline_on_video.png")
