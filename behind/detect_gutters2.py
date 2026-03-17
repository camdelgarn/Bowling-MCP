#!/usr/bin/env python3
"""
More targeted gutter detection using cross-section intensity profiles.
The gutters are narrow dark channels (~9 inches wide) alongside the lane surface.
We need to find them and determine where they start (= foul line).

The approach area (no gutters) vs lane area (has gutters) difference should be visible.
"""

import cv2
import numpy as np

frame = cv2.imread("empty_frame.png")
if frame is None:
    print("No empty_frame.png")
    exit(1)

h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame: {w}x{h}")

# From the layout analysis, we know the lane is roughly centered around x=3050
# Let's look at detailed cross-sections to understand what's really there

# Create a visualization of cross-sections
debug = frame.copy()

# Print detailed row profiles at many y positions
# Focus on the bottom half of the frame where the foul line should be
print("\n=== Detailed cross-section profiles ===")
print("Looking for the gutter pattern: bright surface, then narrow dark channel, then bright again")

# For a systematic approach: at each y-row, find the brightest central region
# and look for dark dips on either side that could be gutters

results = []

for y in range(100, h - 50, 25):
    row = gray[y, :].astype(float)
    
    # Smooth 
    k = 21
    smoothed = np.convolve(row, np.ones(k)/k, mode='same')
    
    # Only look at the relevant x range (1800-4500)
    x_lo, x_hi = 1800, 4500
    seg = smoothed[x_lo:x_hi]
    
    # Find the brightest region (lane surface)
    # Use a wider kernel to find the broad bright area
    wide_k = 201
    wide_smooth = np.convolve(seg, np.ones(wide_k)/wide_k, mode='same')
    
    # Find the peak of the wide-smoothed signal
    peak_idx = np.argmax(wide_smooth[100:-100]) + 100
    peak_x = peak_idx + x_lo
    peak_val = wide_smooth[peak_idx]
    
    if peak_val < 50:  # Too dark, no clear lane
        continue
    
    # Now search left and right of the peak for gutter-like dips
    # A gutter would be: going left from lane center, intensity drops into a narrow dip
    # then recovers (to the adjacent lane or approach)
    
    # Search left
    left_gutter_x = None
    left_gutter_depth = 0
    search_seg = seg[max(0,peak_idx-800):peak_idx]
    if len(search_seg) > 50:
        # Look for minimum that's significantly below the lane surface
        for i in range(len(search_seg)-1, 9, -1):  # scan from center outward
            val = search_seg[i]
            if val < peak_val * 0.65:  # significant drop
                # Check if it recovers after
                after = search_seg[max(0,i-100):max(0,i-20)]
                if len(after) > 0 and np.max(after) > val + 20:
                    left_gutter_x = (peak_idx - 800 + i) + x_lo if peak_idx >= 800 else i + x_lo
                    left_gutter_depth = peak_val - val
                    break
    
    # Search right
    right_gutter_x = None
    right_gutter_depth = 0
    search_seg = seg[peak_idx:min(len(seg), peak_idx+800)]
    if len(search_seg) > 50:
        for i in range(0, len(search_seg)-10):
            val = search_seg[i]
            if val < peak_val * 0.65:
                after = search_seg[min(len(search_seg),i+20):min(len(search_seg),i+100)]
                if len(after) > 0 and np.max(after) > val + 20:
                    right_gutter_x = peak_idx + i + x_lo
                    right_gutter_depth = peak_val - val
                    break
    
    lane_width = None
    if left_gutter_x and right_gutter_x:
        lane_width = right_gutter_x - left_gutter_x
    
    results.append({
        'y': y, 
        'peak_x': peak_x, 
        'peak_val': peak_val,
        'left_gutter': left_gutter_x,
        'left_depth': left_gutter_depth,
        'right_gutter': right_gutter_x,
        'right_depth': right_gutter_depth,
        'lane_width': lane_width
    })

# Print results, focusing on where gutters appear/disappear
print("\ny     | peak_x | peak_val | L_gutter | L_depth | R_gutter | R_depth | lane_w")
print("-" * 90)
for r in results:
    lg = f"{r['left_gutter']:>6}" if r['left_gutter'] else "  None"
    ld = f"{r['left_depth']:>6.0f}" if r['left_gutter'] else "     -"
    rg = f"{r['right_gutter']:>6}" if r['right_gutter'] else "  None"
    rd = f"{r['right_depth']:>6.0f}" if r['right_gutter'] else "     -"
    lw = f"{r['lane_width']:>6}" if r['lane_width'] else "     -"
    print(f"{r['y']:>5} | {r['peak_x']:>6} | {r['peak_val']:>8.1f} | {lg} | {ld} | {rg} | {rd} | {lw}")

# Identify the transition zone where gutters appear/disappear
print("\n=== Gutter transition analysis ===")
has_both_gutters = [(r['y'], r['lane_width']) for r in results if r['left_gutter'] and r['right_gutter']]
has_neither = [r['y'] for r in results if not r['left_gutter'] and not r['right_gutter']]

if has_both_gutters:
    print(f"Rows with both gutters visible: y={has_both_gutters[0][0]} to y={has_both_gutters[-1][0]}")
    print(f"  Lane widths range: {min(w for _,w in has_both_gutters)} to {max(w for _,w in has_both_gutters)}")
if has_neither:
    print(f"Rows with NO gutters visible: includes y={has_neither[:5]}...{has_neither[-5:]}")

# Find the boundary - where do gutters stop appearing?
# Going from top to bottom, gutters should be present in the lane area
# and absent in the approach area
print("\n=== Looking for foul line (gutter start/end) ===")
prev_had_gutters = False
transitions = []
for r in results:
    has = r['left_gutter'] is not None or r['right_gutter'] is not None
    if has != prev_had_gutters:
        transitions.append((r['y'], 'gutters_start' if has else 'gutters_end'))
    prev_had_gutters = has

for y, kind in transitions:
    print(f"  y={y}: {kind}")

# Draw cross-section lines and gutter positions on debug image
for r in results[::4]:  # every 4th result
    y = r['y']
    if r['left_gutter']:
        cv2.circle(debug, (r['left_gutter'], y), 4, (0, 255, 0), -1)
    if r['right_gutter']:
        cv2.circle(debug, (r['right_gutter'], y), 4, (0, 0, 255), -1)
    # Mark the lane peak
    cv2.circle(debug, (r['peak_x'], y), 3, (255, 255, 0), -1)

# Draw transition lines
for y, kind in transitions:
    color = (0, 255, 255) if 'start' in kind else (0, 0, 255)
    cv2.line(debug, (1800, y), (4500, y), color, 2)
    cv2.putText(debug, kind, (4520, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imwrite("debug_gutter_detection2.png", debug)
print(f"\nSaved debug_gutter_detection2.png")
print("  Green dots = left gutter, Red dots = right gutter, Cyan dots = lane peak")
print("  Yellow lines = gutter transitions")

# Also save a few cross-section profiles as images for visual inspection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
sample_ys = [300, 600, 1000, 1500, 2000, 2400, 2700, 2850]

for idx, y_sample in enumerate(sample_ys):
    ax = axes[idx // 2][idx % 2]
    row = gray[y_sample, 1800:4500].astype(float)
    k = 21
    smoothed = np.convolve(row, np.ones(k)/k, mode='same')
    xs = np.arange(1800, 1800+len(row))
    ax.plot(xs, smoothed, 'b-', linewidth=0.5)
    ax.set_title(f'y={y_sample}')
    ax.set_ylabel('Intensity')
    ax.set_ylim(0, 255)
    ax.axhline(y=128, color='gray', linestyle='--', alpha=0.3)
    
    # Mark detected gutters
    r = next((r for r in results if r['y'] == y_sample), None)
    if r is None:
        r = next((r for r in results if abs(r['y'] - y_sample) <= 25), None)
    if r:
        if r['left_gutter']:
            ax.axvline(x=r['left_gutter'], color='green', linestyle='--', label='L gutter')
        if r['right_gutter']:
            ax.axvline(x=r['right_gutter'], color='red', linestyle='--', label='R gutter')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("debug_crosssections.png", dpi=100)
print("Saved debug_crosssections.png with intensity profiles")
