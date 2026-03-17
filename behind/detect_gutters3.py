#!/usr/bin/env python3
"""
Detect the foul line and lane boundaries more robustly.

Strategy:
1. Find the lane center by looking for a vertically consistent bright strip
2. Use the lane center to precisely scan for gutter edges on each side
3. Find where the gutter edges disappear (= foul line)
4. Also try to detect the foul line directly as a horizontal dark line
"""

import cv2
import numpy as np

frame = cv2.imread("empty_frame.png")
if frame is None:
    cap = cv2.VideoCapture("../video/behind/1.MP4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 730)
    ret, frame = cap.read()
    cap.release()
    cv2.imwrite("empty_frame.png", frame)

h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame: {w}x{h}")

# Step 1: Find the lane center and width at various y positions
# by looking at the pixel intensities in vertical strips.
# Build a "column brightness map" - average brightness of each x column
# across different y ranges.

print("\n=== Step 1: Lane center detection via column brightness ===")

# Average brightness of each column in various y bands
y_bands = [
    (200, 400, "top_quarter"),
    (400, 800, "upper_mid"),
    (800, 1200, "mid"),
    (1200, 1600, "lower_mid"),
    (1600, 2000, "upper_lower"),
    (2000, 2400, "lower"),
    (2400, 2800, "bottom"),
    (2800, h-50, "very_bottom"),
]

lane_center_estimates = []

for y_lo, y_hi, label in y_bands:
    # Average each column's brightness in this band
    band = gray[y_lo:y_hi, :].astype(float)
    col_avg = np.mean(band, axis=0)
    
    # Smooth to find broad bright region
    k = 201
    col_smooth = np.convolve(col_avg, np.ones(k)/k, mode='same')
    
    # Find peak (lane center candidate)
    peak_x = np.argmax(col_smooth[500:-500]) + 500  # avoid edges
    peak_val = col_smooth[peak_x]
    
    # Find where the brightness drops significantly on each side
    threshold = peak_val * 0.7
    left_x = peak_x
    while left_x > 100 and col_smooth[left_x] > threshold:
        left_x -= 1
    right_x = peak_x
    while right_x < w - 100 and col_smooth[right_x] > threshold:
        right_x += 1
    
    bright_width = right_x - left_x
    center_x = (left_x + right_x) // 2
    
    print(f"  {label:>15} (y={y_lo}-{y_hi}): center_x={center_x}, width={bright_width}, brightness={peak_val:.0f}, range=({left_x}-{right_x})")
    lane_center_estimates.append((center_x, y_lo, y_hi, left_x, right_x, bright_width))

# Look for where the bright strip width changes significantly
print("\n=== Width transitions (approach is wider, lane is narrower) ===")
for i in range(1, len(lane_center_estimates)):
    prev = lane_center_estimates[i-1]
    curr = lane_center_estimates[i]
    width_change = curr[5] - prev[5]
    if abs(width_change) > 100:
        print(f"  Width change at y≈{curr[1]}: {prev[5]} -> {curr[5]} (change={width_change})")

# Step 2: More precise lane edge detection
# Now that we have rough lane positions, scan more carefully
print("\n=== Step 2: Precise lane edge scanning ===")

# Use a finer y step and focus on finding consistent edges
# For each y, look at the derivative of brightness (gradient) to find edges

# Compute horizontal Sobel
sobel_x = cv2.Sobel(gray.astype(float), cv2.CV_64F, 1, 0, ksize=5)

# For each row, find the strongest left-to-dark and dark-to-right transitions
# near the expected lane area
# Use the column brightness analysis to define the search range

lane_edges = []  # (y, left_x, right_x)

for y in range(100, h - 50, 10):
    # Get the sobel row
    row_grad = sobel_x[y, :]
    
    # Smooth the gradient  
    k = 11
    smooth_grad = np.convolve(row_grad, np.ones(k)/k, mode='same')
    
    # Find the strongest positive gradient (bright -> dark going right = left edge of dark feature)
    # and strongest negative gradient (dark -> bright going right = right edge of dark feature)
    
    # Focus on the plausible lane region (x=2000 to x=4500 based on earlier results)
    search_lo, search_hi = 2000, 4500
    seg = smooth_grad[search_lo:search_hi]
    
    if len(seg) < 100:
        continue
    
    # The lane edges should be:
    # LEFT EDGE: going from left to right, you hit a bright-to-less-bright transition
    # This would be a negative gradient (going from high intensity to lower intensity)
    # 
    # Actually, let's think about this from scratch:
    # Looking at a row from left to right across the lane:
    # [darker area] [gutter edge: dark-to-bright = neg to pos gradient peak] [bright lane] [gutter edge: bright-to-dark = pos to neg gradient] [darker area]
    #
    # The gutter itself is narrow, so we're looking for:
    # On the left side: a sharp rise in brightness (entering the lane)
    # On the right side: a sharp drop in brightness (leaving the lane)
    
    # Find all significant positive peaks (entering bright zone)
    pos_threshold = 15
    neg_threshold = -15
    
    # Simple: find where brightness goes from low to high (left edge of lane)
    # and high to low (right edge of lane)
    row_brightness = gray[y, search_lo:search_hi].astype(float)
    smooth_brightness = np.convolve(row_brightness, np.ones(31)/31, mode='same')
    
    # Find the lane as the widest bright region
    bright_threshold = np.percentile(smooth_brightness, 60)
    
    # Find contiguous bright regions
    bright_runs = []
    in_bright = False
    run_start = 0
    for i in range(len(smooth_brightness)):
        if smooth_brightness[i] > bright_threshold:
            if not in_bright:
                in_bright = True
                run_start = i
        else:
            if in_bright:
                bright_runs.append((run_start + search_lo, i + search_lo))
                in_bright = False
    if in_bright:
        bright_runs.append((run_start + search_lo, len(smooth_brightness) + search_lo))
    
    # Find the widest bright run
    if bright_runs:
        widest = max(bright_runs, key=lambda r: r[1] - r[0])
        lane_width = widest[1] - widest[0]
        if lane_width > 100:  # reasonable minimum lane width in pixels
            lane_edges.append((y, widest[0], widest[1]))

# Print the lane edges, focusing on width changes
print("\nLane boundaries (widest bright region per row):")
print("y     | left_x | right_x | width  | center_x")
print("-" * 56)
prev_width = None
for y, lx, rx in lane_edges[::10]:  # every 10th
    width = rx - lx
    cx = (lx + rx) // 2
    marker = ""
    if prev_width and abs(width - prev_width) > 50:
        marker = " <-- WIDTH CHANGE"
    print(f"{y:>5} | {lx:>6} | {rx:>6} | {width:>6} | {cx:>8}{marker}")
    prev_width = width

# Step 3: Find where the width changes dramatically
# Going from bottom to top, the approach should be wide, then suddenly narrow 
# at the foul line where the gutters begin
print("\n=== Step 3: Width change analysis (looking for foul line) ===")

widths = [(y, rx - lx, lx, rx) for y, lx, rx in lane_edges]
# Smooth the widths
smooth_w = 5
width_vals = [w for _, w, _, _ in widths]
width_ys = [y for y, _, _, _ in widths]

if len(width_vals) > smooth_w * 2:
    smoothed_widths = np.convolve(width_vals, np.ones(smooth_w)/smooth_w, mode='valid')
    smoothed_ys = width_ys[smooth_w//2:smooth_w//2+len(smoothed_widths)]
    
    # Find the biggest width derivative (where width changes fastest)
    width_deriv = np.gradient(smoothed_widths)
    
    # Find the row where width drops most dramatically (going from bottom/approach to top/lane)
    # Since approach is wider and lane is narrower, going up (decreasing y) 
    # we should see a sudden decrease in width at the foul line
    
    # Actually, going from top to bottom: the lane starts narrow (pins) and gets wider
    # toward the foul line. Then at the foul line, it should get even wider suddenly 
    # (approach has no gutter constraints).
    
    # Find the biggest positive derivative (width increasing = entering approach going down)
    max_deriv_idx = np.argmax(np.abs(width_deriv))
    print(f"Biggest width change at y≈{smoothed_ys[max_deriv_idx]}, derivative={width_deriv[max_deriv_idx]:.1f}")
    
    # Find all significant width jumps
    for i in range(len(width_deriv)):
        if abs(width_deriv[i]) > 20:
            print(f"  Significant width change at y≈{smoothed_ys[i]}: d(width)/dy = {width_deriv[i]:.1f}, width={smoothed_widths[i]:.0f}")

# Step 4: Try to detect the foul line directly
# The foul line is a physical dark horizontal line on the floor
print("\n=== Step 4: Direct foul line detection (horizontal dark line) ===")

# Compute horizontal Sobel (dy) to find horizontal edges
sobel_y = cv2.Sobel(gray.astype(float), cv2.CV_64F, 0, 1, ksize=5)

# For each row, sum the absolute horizontal gradient across the lane region
# Strong horizontal edges at any row would show up as a high sum
# Use the lane edges we found to define the x range at each y level

h_edge_strength = []
for y, lx, rx in lane_edges:
    cx = (lx + rx) // 2
    margin = 100
    # Look across the full lane width plus a bit
    row_seg = np.abs(sobel_y[y, max(0,lx-margin):min(w,rx+margin)])
    avg_strength = np.mean(row_seg)
    max_strength = np.max(row_seg)
    h_edge_strength.append((y, avg_strength, max_strength))

# Find peaks in horizontal edge strength
print("\nHorizontal edge strength by row (potential foul line locations):")
strengths = [s for _, s, _ in h_edge_strength]
threshold = np.percentile(strengths, 90) if strengths else 0

for y, avg_s, max_s in h_edge_strength:
    if avg_s > threshold:
        print(f"  y={y}: avg={avg_s:.1f}, max={max_s:.1f}  ** STRONG **")

# Save debug visualization
debug = frame.copy()

# Draw detected lane edges
for y, lx, rx in lane_edges[::5]:
    cv2.circle(debug, (lx, y), 2, (0, 255, 0), -1)  # Left edge green
    cv2.circle(debug, (rx, y), 2, (0, 0, 255), -1)   # Right edge red

# Mark band centers
for cx, y_lo, y_hi, lx, rx, bw in lane_center_estimates:
    mid_y = (y_lo + y_hi) // 2
    cv2.line(debug, (lx, mid_y), (rx, mid_y), (255, 255, 0), 2)
    cv2.circle(debug, (cx, mid_y), 5, (0, 255, 255), -1)
    cv2.putText(debug, f"w={bw}", (rx+10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

cv2.imwrite("debug_lane_detection.png", debug)
print(f"\nSaved debug_lane_detection.png")
print("  Green dots = left edge, Red dots = right edge")
print("  Yellow lines = band-average widths")
