#!/usr/bin/env python3
"""
Detailed intensity slices at specific y positions to identify lane vs approach vs gutter.
Focus on finding where gutters appear/disappear to locate the foul line.
"""

import cv2
import numpy as np

frame = cv2.imread("empty_frame.png")
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame: {w}x{h}")

# From the morphological analysis, candidate gutter positions are around x=2550-2750 (left)
# and x=3300-3400 (right). Let's examine these areas very carefully.

# Let's look at narrow horizontal slices at various y positions
# and print the intensity profile around the gutter candidates

print("\n=== Intensity profiles around gutter candidates ===")
print("Comparing x=2400-2800 (left gutter area) and x=3200-3500 (right gutter area)")

# Print detailed intensity data at key y positions
slice_ys = list(range(500, 2900, 100))

print("\nLeft gutter area (x=2400-2800):")
print("  y  |  min |  max | mean | dip locations (x where val < 0.7*mean)")
for y in slice_ys:
    seg = gray[y, 2400:2800].astype(float)
    smooth = np.convolve(seg, np.ones(11)/11, mode='same')
    mean_val = np.mean(smooth)
    min_val = np.min(smooth)
    max_val = np.max(smooth)
    
    # Find dip positions
    threshold = mean_val * 0.7
    dips = np.where(smooth < threshold)[0]
    if len(dips) > 0:
        # Find contiguous dip runs
        runs = []
        start = dips[0]
        for i in range(1, len(dips)):
            if dips[i] - dips[i-1] > 3:
                runs.append((start + 2400, dips[i-1] + 2400, dips[i-1] - start + 1))
                start = dips[i]
        runs.append((start + 2400, dips[-1] + 2400, dips[-1] - start + 1))
        dip_str = ", ".join([f"x={s}-{e}(w={w})" for s, e, w in runs if w > 3])
    else:
        dip_str = "none"
    print(f"  {y:>4} | {min_val:>4.0f} | {max_val:>4.0f} | {mean_val:>4.0f} | {dip_str}")

print("\nRight gutter area (x=3200-3600):")
print("  y  |  min |  max | mean | dip locations")
for y in slice_ys:
    seg = gray[y, 3200:3600].astype(float)
    smooth = np.convolve(seg, np.ones(11)/11, mode='same')
    mean_val = np.mean(smooth)
    min_val = np.min(smooth)
    max_val = np.max(smooth)
    
    threshold = mean_val * 0.7
    dips = np.where(smooth < threshold)[0]
    if len(dips) > 0:
        runs = []
        start = dips[0]
        for i in range(1, len(dips)):
            if dips[i] - dips[i-1] > 3:
                runs.append((start + 3200, dips[i-1] + 3200, dips[i-1] - start + 1))
                start = dips[i]
        runs.append((start + 3200, dips[-1] + 3200, dips[-1] - start + 1))
        dip_str = ", ".join([f"x={s}-{e}(w={w})" for s, e, w in runs if w > 3])
    else:
        dip_str = "none"
    print(f"  {y:>4} | {min_val:>4.0f} | {max_val:>4.0f} | {mean_val:>4.0f} | {dip_str}")

# Also look at the full lane cross-section at several critical y values
# to understand the overall structure
print("\n\n=== Full cross-section intensity (x=2000-4200, sampled every 50px) ===")
critical_ys = [500, 700, 800, 900, 950, 1000, 1050, 1100, 1200, 1500, 2000, 2500, 2800]

for y in critical_ys:
    print(f"\ny={y}:")
    vals = []
    for x in range(2000, 4200, 50):
        val = int(np.mean(gray[y, x:x+50]))
        vals.append((x, val))
    
    # Print as a bar chart
    for x, val in vals:
        bar = '#' * (val // 5)  # Scale to fit
        print(f"  x={x:>4}: {val:>3} |{bar}")

# Now look specifically at whether there's a horizontal feature (foul line)
# Scan at finer granularity around y=700-1100
print("\n\n=== Fine-grained horizontal scan y=700-1100 ===")
print("Looking for a row where the cross-section pattern changes distinctly")

# For each y, compute a lane "signature" - the intensities at x=2500-3500
# Then correlate consecutive rows. Where the correlation drops = transition
prev_sig = None
for y in range(700, 1100, 5):
    sig = gray[y, 2500:3500].astype(float)
    sig_smooth = np.convolve(sig, np.ones(21)/21, mode='same')[10:-10]
    
    if prev_sig is not None:
        # Correlation between this row and previous
        if len(sig_smooth) == len(prev_sig):
            corr = np.corrcoef(sig_smooth, prev_sig)[0, 1]
            mean_diff = np.mean(np.abs(sig_smooth - prev_sig))
            if corr < 0.9 or mean_diff > 10:
                print(f"  y={y}: correlation={corr:.3f}, mean_diff={mean_diff:.1f}  ** TRANSITION **")
    
    prev_sig = sig_smooth

# Also check y=2800-2988 (bottom of frame) to see if the foul line might be there
print("\n=== Fine-grained horizontal scan y=2700-2988 ===")
prev_sig = None
for y in range(2700, h-10, 5):
    # Use wider x range for approach area
    sig = gray[y, 2200:4000].astype(float)
    sig_smooth = np.convolve(sig, np.ones(21)/21, mode='same')[10:-10]
    
    if prev_sig is not None and len(sig_smooth) == len(prev_sig):
        corr = np.corrcoef(sig_smooth, prev_sig)[0, 1]
        mean_diff = np.mean(np.abs(sig_smooth - prev_sig))
        if corr < 0.9 or mean_diff > 10:
            print(f"  y={y}: correlation={corr:.3f}, mean_diff={mean_diff:.1f}  ** TRANSITION **")
    
    prev_sig = sig_smooth
