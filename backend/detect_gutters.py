#!/usr/bin/env python3
"""
Detect where the gutters start in the video frame.
Gutters are the narrow channels on either side of the lane surface.
They only exist on the lane (not the approach), so where they start = foul line.

Strategy:
- The lane surface is a lighter wood color
- The gutters are darker channels alongside the lane
- Look for the vertical dark lines (gutter edges) and find where they start/end
"""

import cv2
import numpy as np

# Load the empty frame
frame = cv2.imread("empty_frame.png")
if frame is None:
    print("No empty_frame.png found, extracting from video...")
    cap = cv2.VideoCapture("../video/behind/1.MP4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 730)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read frame")
        exit(1)
    cv2.imwrite("empty_frame.png", frame)

h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# We know from template matching the lane is roughly in the center-right area
# Lane center is around x=2800-3200 based on previous analysis
# Let's examine a vertical strip through the center of where we expect the lane

# First, let's look at the intensity profile along vertical columns
# to find where the gutter (dark channel) appears

# Scan horizontal line profiles at various y positions to see the lane cross-section
print("\n=== Horizontal intensity profiles ===")
print("Looking for gutter signatures (dark dips alongside bright lane surface)")

# Sample several y positions from top to bottom
y_samples = list(range(200, h, 200))

# For each y, look at the intensity profile across the expected lane area
# Previous analysis showed lane between roughly x=2300 and x=3800
x_start = 1800
x_end = 4200

profiles = {}
for y in y_samples:
    if y >= h:
        continue
    row = gray[y, x_start:x_end].astype(float)
    # Smooth to reduce noise
    kernel_size = 15
    smoothed = np.convolve(row, np.ones(kernel_size)/kernel_size, mode='same')
    profiles[y] = smoothed
    
    # Find local minima (potential gutter locations)
    grad = np.gradient(smoothed)
    # Look for significant dips
    mean_val = np.mean(smoothed)
    dips = []
    in_dip = False
    dip_start = 0
    for i in range(len(smoothed)):
        if smoothed[i] < mean_val * 0.8:
            if not in_dip:
                in_dip = True
                dip_start = i
        else:
            if in_dip:
                dip_center = (dip_start + i) // 2
                dip_depth = mean_val - smoothed[dip_center]
                dips.append((dip_center + x_start, dip_depth, smoothed[dip_center]))
                in_dip = False
    
    if dips:
        # Sort by depth (deepest first)
        dips.sort(key=lambda d: -d[1])
        top_dips = dips[:6]
        dip_str = ", ".join([f"x={d[0]} (depth={d[1]:.0f}, val={d[2]:.0f})" for d in top_dips])
        print(f"  y={y}: {len(dips)} dips. Top: {dip_str}")
    else:
        print(f"  y={y}: no significant dips (mean={mean_val:.0f})")

# Now let's do a more targeted analysis:
# Look for the gutter channels specifically by examining edge detection
print("\n=== Edge-based gutter detection ===")

# Use Canny edge detection
edges = cv2.Canny(gray, 30, 100)

# Look at vertical edge density in columns
# Gutters should show as consistent vertical edges
print("\nVertical edge density by column (in lane region):")
for x in range(2200, 4000, 50):
    col_edges = edges[:, x]
    # Count edge pixels in different vertical bands
    top_count = np.sum(col_edges[200:1000] > 0)
    mid_count = np.sum(col_edges[1000:2000] > 0)
    bot_count = np.sum(col_edges[2000:2800] > 0)
    total = top_count + mid_count + bot_count
    if total > 50:
        print(f"  x={x}: top={top_count}, mid={mid_count}, bot={bot_count}, total={total}")

# Most promising approach: look for where dark vertical channels appear
# by examining the gradient in the x-direction
print("\n=== Detecting gutter channels by horizontal gradient ===")

# Compute horizontal gradient (dx)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

# For each row, find strong positive-then-negative gradient pairs
# (indicating a dark channel: bright->dark = +gradient, dark->bright = -gradient)
gutter_left_xs = {}  # y -> x position of left gutter
gutter_right_xs = {}  # y -> x position of right gutter

for y in range(100, h - 100, 5):
    row = sobel_x[y, x_start:x_end]
    
    # Find strong gradient peaks
    threshold = 30
    pos_peaks = []
    neg_peaks = []
    
    # Simple peak finding
    for i in range(10, len(row) - 10):
        if row[i] > threshold and row[i] > row[i-5] and row[i] > row[i+5]:
            pos_peaks.append(i + x_start)
        elif row[i] < -threshold and row[i] < row[i-5] and row[i] < row[i+5]:
            neg_peaks.append(i + x_start)
    
    # Look for pairs: positive peak followed closely by negative peak = left gutter edge
    # Or negative peak followed by positive peak = right gutter edge
    # The gutter is narrow (~30-60 pixels wide probably)
    
    for pp in pos_peaks:
        for np_ in neg_peaks:
            gap = np_ - pp
            if 10 < gap < 80:  # Narrow dark channel
                # This could be a left gutter (bright lane, dark gutter, then ?)
                if y not in gutter_left_xs or pp < gutter_left_xs[y]:
                    # Check if this is in the expected lane area
                    if 2200 < pp < 3500:
                        gutter_left_xs[y] = pp
    
    for np_ in neg_peaks:
        for pp in pos_peaks:
            gap = pp - np_
            if 10 < gap < 80:
                if y not in gutter_right_xs or pp > gutter_right_xs[y]:
                    if 2800 < pp < 4200:
                        gutter_right_xs[y] = pp

# Find where gutters start and end
if gutter_left_xs:
    left_ys = sorted(gutter_left_xs.keys())
    print(f"\nLeft gutter detected from y={left_ys[0]} to y={left_ys[-1]}")
    step = max(1, len(left_ys) // 15)
    for i in range(0, len(left_ys), step):
        y = left_ys[i]
        print(f"  y={y}: gutter at x={gutter_left_xs[y]}")
    
    # The foul line is approximately where the gutter starts (highest y = closest to camera)
    print(f"\n  => Left gutter starts at y={left_ys[-1]} (foul line candidate)")
    print(f"  => Left gutter ends at y={left_ys[0]} (toward pins)")

if gutter_right_xs:
    right_ys = sorted(gutter_right_xs.keys())
    print(f"\nRight gutter detected from y={right_ys[0]} to y={right_ys[-1]}")
    step = max(1, len(right_ys) // 15)
    for i in range(0, len(right_ys), step):
        y = right_ys[i]
        print(f"  y={y}: gutter at x={gutter_right_xs[y]}")
    
    print(f"\n  => Right gutter starts at y={right_ys[-1]} (foul line candidate)")
    print(f"  => Right gutter ends at y={right_ys[0]} (toward pins)")

# Alternative: direct intensity difference approach
# Look at the difference between the lane center and nearby columns
print("\n=== Lane center vs side intensity difference ===")
# Estimate lane center x from template matching data
lane_center_x = 3050  # rough center from previous analysis

for y in range(200, h, 100):
    center_val = float(np.mean(gray[y, lane_center_x-20:lane_center_x+20]))
    left_val = float(np.mean(gray[y, lane_center_x-300:lane_center_x-250]))
    right_val = float(np.mean(gray[y, lane_center_x+250:lane_center_x+300]))
    far_left = float(np.mean(gray[y, lane_center_x-500:lane_center_x-450]))
    far_right = float(np.mean(gray[y, lane_center_x+450:lane_center_x+500]))
    
    # If center is bright and sides show a dip then recovery, that's a gutter
    left_dip = center_val - left_val
    right_dip = center_val - right_val
    
    if abs(left_dip) > 15 or abs(right_dip) > 15:
        print(f"  y={y}: center={center_val:.0f}, L-300={left_val:.0f}(diff={left_dip:.0f}), R+300={right_val:.0f}(diff={right_dip:.0f}), farL={far_left:.0f}, farR={far_right:.0f}")

# Save a debug visualization
debug = frame.copy()

# Draw detected gutter positions
for y, x in gutter_left_xs.items():
    cv2.circle(debug, (x, y), 2, (0, 0, 255), -1)  # Red dots for left gutter

for y, x in gutter_right_xs.items(): 
    cv2.circle(debug, (x, y), 2, (255, 0, 0), -1)  # Blue dots for right gutter

cv2.imwrite("debug_gutter_detection.png", debug)
print(f"\nSaved debug_gutter_detection.png with gutter positions marked")
print(f"  Red = left gutter edge, Blue = right gutter edge")
