#!/usr/bin/env python3
"""
Trace the bowler's lane by following the specific bright strip near x≈2800.
From the intensity analysis:
  - At y=1325: bright strip at x=2550-3050 (the bowler's lane)
  - Dark gap at x=3050-3275 (ball return)
  - Another bright strip at x=3300-3825 (adjacent lane)

Strategy: At each y level in the lane area, find the bright region
closest to x≈2800 (lane center estimate) and track its boundaries.
"""

import cv2
import numpy as np

frame = cv2.imread("empty_frame.png")
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame: {w}x{h}")

FOUL_LINE_Y = 1330
LANE_CENTER_ESTIMATE = 2800  # Approximate center from intensity analysis

# At each y from 200 to foul line, find the bright region containing x≈2800
print("=== Tracing bowler's lane ===")
print("Finding the bright region nearest to x≈2800 at each y level\n")

lane_trace = []  # (y, left_x, right_x, center_x, brightness)

for y in range(100, FOUL_LINE_Y + 50, 5):
    row = gray[y, :].astype(float)
    
    # Smooth
    k = 21
    smooth = np.convolve(row, np.ones(k)/k, mode='same')
    
    # Use the local mean around the expected lane center as threshold
    local_region = smooth[max(0, LANE_CENTER_ESTIMATE-400):min(w, LANE_CENTER_ESTIMATE+400)]
    if len(local_region) == 0:
        continue
    
    lane_brightness = np.mean(local_region)
    
    if lane_brightness < 20:  # Too dark, skip
        continue
    
    # Threshold: 50% of local brightness
    thresh = max(lane_brightness * 0.5, 30)
    
    # Find contiguous bright regions
    bright = smooth > thresh
    
    # Find runs of True (bright pixels)
    runs = []
    in_run = False
    start = 0
    for i in range(max(0, LANE_CENTER_ESTIMATE-600), min(w, LANE_CENTER_ESTIMATE+600)):
        if bright[i]:
            if not in_run:
                in_run = True
                start = i
        else:
            if in_run:
                runs.append((start, i))
                in_run = False
    if in_run:
        runs.append((start, min(w, LANE_CENTER_ESTIMATE+600)))
    
    # Find the run that contains or is closest to LANE_CENTER_ESTIMATE
    best_run = None
    best_dist = float('inf')
    for start, end in runs:
        center = (start + end) // 2
        dist = abs(center - LANE_CENTER_ESTIMATE)
        # Also require minimum width
        if end - start > 50 and dist < best_dist:
            best_dist = dist
            best_run = (start, end)
    
    if best_run:
        left, right = best_run
        center = (left + right) // 2
        width = right - left
        brightness = np.mean(smooth[left:right])
        lane_trace.append((y, left, right, center, brightness))

# Print the trace
print(f"Found {len(lane_trace)} rows tracing the lane")
print()
print("  y   | left  | right | width | center | brightness")
print("-" * 60)
for y, lx, rx, cx, br in lane_trace[::max(1, len(lane_trace)//30)]:
    print(f"  {y:>4} | {lx:>5} | {rx:>5} | {rx-lx:>5} | {cx:>6} | {br:>10.1f}")

# Clean up the trace using median filtering
# Group into bands and take median
band_size = 30
bands = []
for y_start in range(100, FOUL_LINE_Y + 50, band_size):
    band_data = [(y, lx, rx) for y, lx, rx, _, _ in lane_trace 
                 if y_start <= y < y_start + band_size]
    if len(band_data) >= 3:
        lefts = [lx for _, lx, _ in band_data]
        rights = [rx for _, _, rx in band_data]
        bands.append({
            'y': y_start + band_size // 2,
            'left': np.median(lefts),
            'right': np.median(rights),
            'width': np.median(rights) - np.median(lefts),
            'n': len(band_data)
        })

print("\n\nMedian-filtered lane edges by band:")
print("  y_mid | left  | right | width | n")
print("-" * 50)
for b in bands:
    print(f"  {b['y']:>5} | {b['left']:>5.0f} | {b['right']:>5.0f} | {b['width']:>5.0f} | {b['n']}")

# Fit perspective lines to the clean band data
band_ys = np.array([b['y'] for b in bands])
band_lefts = np.array([b['left'] for b in bands])
band_rights = np.array([b['right'] for b in bands])
band_widths = np.array([b['width'] for b in bands])

# Use RANSAC-like approach: fit to inliers only
# First rough fit, then remove outliers, then refit
if len(bands) >= 5:
    # Full fit
    left_fit = np.polyfit(band_ys, band_lefts, 1)
    right_fit = np.polyfit(band_ys, band_rights, 1)
    
    # Compute residuals
    left_pred = np.polyval(left_fit, band_ys)
    right_pred = np.polyval(right_fit, band_ys)
    left_resid = np.abs(band_lefts - left_pred)
    right_resid = np.abs(band_rights - right_pred)
    
    # Keep only inliers (residual < 2 * median)
    left_inlier = left_resid < 2 * np.median(left_resid) + 50
    right_inlier = right_resid < 2 * np.median(right_resid) + 50
    inlier = left_inlier & right_inlier
    
    if np.sum(inlier) >= 5:
        left_fit = np.polyfit(band_ys[inlier], band_lefts[inlier], 1)
        right_fit = np.polyfit(band_ys[inlier], band_rights[inlier], 1)
        
        print(f"\nRobust linear fits ({np.sum(inlier)} inlier bands):")
        print(f"  Left edge:  x = {left_fit[0]:.3f}*y + {left_fit[1]:.1f}")
        print(f"  Right edge: x = {right_fit[0]:.3f}*y + {right_fit[1]:.1f}")
        
        # Compute lane corners
        # Top of visible lane (exclude very dark rows)
        # Use y where the width starts being reasonable
        min_visible_y = band_ys[inlier].min()
        
        lane_top_y = int(min_visible_y)
        lane_top_left = int(np.polyval(left_fit, lane_top_y))
        lane_top_right = int(np.polyval(right_fit, lane_top_y))
        
        lane_bot_y = FOUL_LINE_Y
        lane_bot_left = int(np.polyval(left_fit, lane_bot_y))
        lane_bot_right = int(np.polyval(right_fit, lane_bot_y))
        
        print(f"\n  Lane trapezoid (pins to foul line):")
        print(f"    Top (pins):     ({lane_top_left}, {lane_top_y}) - ({lane_top_right}, {lane_top_y})")
        print(f"    Bottom (foul):  ({lane_bot_left}, {lane_bot_y}) - ({lane_bot_right}, {lane_bot_y})")
        print(f"    Top width:    {lane_top_right - lane_top_left}")
        print(f"    Bottom width: {lane_bot_right - lane_bot_left}")
        
        # Vanishing point (where left and right lines meet)
        # left_fit[0]*y + left_fit[1] = right_fit[0]*y + right_fit[1]
        # y * (left_fit[0] - right_fit[0]) = right_fit[1] - left_fit[1]
        if abs(left_fit[0] - right_fit[0]) > 0.001:
            vanish_y = (right_fit[1] - left_fit[1]) / (left_fit[0] - right_fit[0])
            vanish_x = int(np.polyval(left_fit, vanish_y))
            print(f"    Vanishing point: ({vanish_x}, {int(vanish_y)})")
        
        # Approach: extrapolate perspective lines below foul line
        # The approach is wider than the lane (no gutters)
        # Gutter width is about 9.25 inches, lane is 41.5 inches
        # So approach width ≈ lane width + 2 * gutter width ≈ lane * (41.5 + 18.5) / 41.5 ≈ lane * 1.45
        gutter_ratio = 1.0  # Keep same width for now, can adjust
        
        # Extend lines below foul line
        app_top_y = FOUL_LINE_Y
        app_bot_y = h - 1
        
        app_top_left = lane_bot_left
        app_top_right = lane_bot_right
        app_bot_left = int(np.polyval(left_fit, app_bot_y))
        app_bot_right = int(np.polyval(right_fit, app_bot_y))
        
        # Widen the approach slightly (add gutter width on each side)
        lane_width_at_foul = lane_bot_right - lane_bot_left
        gutter_width_pixels = int(lane_width_at_foul * 9.25 / 41.5)  # Scale gutter to lane width
        
        print(f"\n  Approach (extrapolated):")
        print(f"    Top (foul):     ({app_top_left}, {app_top_y}) - ({app_top_right}, {app_top_y})")
        print(f"    Bottom (camera): ({app_bot_left}, {app_bot_y}) - ({app_bot_right}, {app_bot_y})")
        print(f"    Top width:    {app_top_right - app_top_left}")
        print(f"    Bottom width: {app_bot_right - app_bot_left}")
        print(f"    Gutter width estimate: {gutter_width_pixels}px each side")
        
        # Save visualization
        debug = frame.copy()
        
        # Draw lane trapezoid in green
        pts_lane = np.array([
            [lane_top_left, lane_top_y],
            [lane_top_right, lane_top_y],
            [lane_bot_right, lane_bot_y],
            [lane_bot_left, lane_bot_y]
        ], dtype=np.int32)
        cv2.polylines(debug, [pts_lane], True, (0, 255, 0), 3)
        
        # Fill with semi-transparent green
        overlay = debug.copy()
        cv2.fillPoly(overlay, [pts_lane], (0, 255, 0))
        debug = cv2.addWeighted(debug, 0.85, overlay, 0.15, 0)
        
        mid_x = (lane_top_left + lane_top_right + lane_bot_left + lane_bot_right) // 4
        mid_y = (lane_top_y + lane_bot_y) // 2
        cv2.putText(debug, "LANE", (mid_x - 50, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Draw foul line in red
        cv2.line(debug, (lane_bot_left - 100, FOUL_LINE_Y), 
                (lane_bot_right + 100, FOUL_LINE_Y), (0, 0, 255), 4)
        cv2.putText(debug, "FOUL LINE", 
                    (lane_bot_right + 20, FOUL_LINE_Y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Draw approach trapezoid in blue/orange
        pts_app = np.array([
            [app_top_left, app_top_y],
            [app_top_right, app_top_y],
            [app_bot_right, app_bot_y],
            [app_bot_left, app_bot_y]
        ], dtype=np.int32)
        cv2.polylines(debug, [pts_app], True, (255, 150, 0), 3)
        
        overlay2 = debug.copy()
        cv2.fillPoly(overlay2, [pts_app], (255, 150, 0))
        debug = cv2.addWeighted(debug, 0.85, overlay2, 0.15, 0)
        
        a_mid_x = (app_top_left + app_top_right + app_bot_left + app_bot_right) // 4
        a_mid_y = (app_top_y + app_bot_y) // 2
        cv2.putText(debug, "APPROACH", (a_mid_x - 100, a_mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 150, 0), 3)
        
        # Draw the detected lane edge dots
        for y, lx, rx, cx, br in lane_trace[::2]:
            cv2.circle(debug, (lx, y), 2, (0, 255, 0), -1)
            cv2.circle(debug, (rx, y), 2, (0, 0, 255), -1)
        
        # Draw the fitted lines
        for y_draw in range(lane_top_y, app_bot_y, 50):
            lx = int(np.polyval(left_fit, y_draw))
            rx = int(np.polyval(right_fit, y_draw))
            cv2.circle(debug, (lx, y_draw), 3, (0, 255, 255), -1)
            cv2.circle(debug, (rx, y_draw), 3, (0, 255, 255), -1)
        
        cv2.imwrite("lane_outline_on_video.png", debug)
        print(f"\nSaved lane_outline_on_video.png")
        
    else:
        print(f"Not enough inliers: {np.sum(inlier)}")
else:
    print("Not enough band data")
