#!/usr/bin/env python3
"""
Find the foul line by measuring cross-section variance at each y.
The approach is a flat uniform surface (low variance).
The lane has gutters, shadows, lighting differences (high variance).
The foul line is where variance drops sharply.
"""

import cv2
import numpy as np

frame = cv2.imread("empty_frame.png")
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame: {w}x{h}")

# Compute row-by-row variance in a consistent x range
# Use a wide range that covers the full lane + approach area
x_lo, x_hi = 2000, 4200

# Also compute mean brightness per row
print("=== Row variance and mean brightness ===")
print("(High variance = lane/gutter features; Low variance = uniform approach surface)")

row_variance = np.zeros(h)
row_mean = np.zeros(h)
row_range = np.zeros(h)

for y in range(h):
    seg = gray[y, x_lo:x_hi].astype(float)
    # Smooth slightly to reduce pixel noise
    k = 11
    smooth = np.convolve(seg, np.ones(k)/k, mode='same')
    row_variance[y] = np.var(smooth)
    row_mean[y] = np.mean(smooth)
    row_range[y] = np.max(smooth) - np.min(smooth)

# Smooth the variance curve to find the transition
var_smooth_k = 31
var_smoothed = np.convolve(row_variance, np.ones(var_smooth_k)/var_smooth_k, mode='same')
mean_smoothed = np.convolve(row_mean, np.ones(var_smooth_k)/var_smooth_k, mode='same')

# Print every 50 rows
print("\n  y  | variance | smoothed_var | mean_brightness | range")
print("-" * 65)
for y in range(0, h, 50):
    print(f"{y:>5} | {row_variance[y]:>8.0f} | {var_smoothed[y]:>12.0f} | {mean_smoothed[y]:>15.0f} | {row_range[y]:>5.0f}")

# Find where variance drops significantly (foul line)
print("\n=== Variance transitions ===")
# Look for the y where smoothed variance drops below a threshold
# and stays low (approach area)

# Find the median variance of the bottom half (approach)
approach_var = np.median(var_smoothed[h//2:])
lane_var = np.median(var_smoothed[200:h//2])
print(f"Median variance - top half (lane): {lane_var:.0f}")
print(f"Median variance - bottom half (approach): {approach_var:.0f}")

# Threshold: midpoint between lane and approach variance
threshold = (lane_var + approach_var) / 2

# Scan from bottom up - find where variance exceeds the threshold
foul_line_y = h // 2  # default
for y in range(h - 100, 100, -1):
    if var_smoothed[y] > threshold:
        foul_line_y = y
        break

print(f"\nVariance threshold: {threshold:.0f}")
print(f"Foul line estimate (variance method): y={foul_line_y}")

# Also look for the specific y range where:
# 1. Brightness increases (approach is closer to camera = brighter)
# 2. Variance drops
# 3. Range (max-min in row) drops

# Fine-grained look around the estimated foul line
print(f"\n=== Detail around foul line estimate (y={foul_line_y-200} to y={foul_line_y+200}) ===")
for y in range(max(100, foul_line_y-200), min(h-50, foul_line_y+200), 10):
    print(f"  y={y}: var={var_smoothed[y]:>8.0f}, mean={mean_smoothed[y]:>5.0f}, range={row_range[y]:>5.0f}")

# Also compute some additional metrics around the transition
# The gradient of variance should be steepest at the foul line
var_grad = np.gradient(var_smoothed)
var_grad_smooth = np.convolve(var_grad, np.ones(21)/21, mode='same')

print(f"\n=== Variance gradient (steepest descent = foul line) ===")
print(f"Looking for the biggest negative gradient (variance dropping)")

# Find the most negative gradient (sharpest drop from lane to approach)
min_grad_idx = np.argmin(var_grad_smooth[200:h-200]) + 200
print(f"Steepest variance drop at y={min_grad_idx}, gradient={var_grad_smooth[min_grad_idx]:.1f}")

# Top 5 steepest drops
steep_drops = []
for y in range(200, h-200):
    if var_grad_smooth[y] < -50:  # significant negative gradient
        steep_drops.append((y, var_grad_smooth[y]))

# Cluster nearby drops
if steep_drops:
    clusters = []
    current_cluster = [steep_drops[0]]
    for i in range(1, len(steep_drops)):
        if steep_drops[i][0] - steep_drops[i-1][0] < 30:
            current_cluster.append(steep_drops[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [steep_drops[i]]
    clusters.append(current_cluster)
    
    print("\nVariance drop clusters:")
    for cluster in clusters:
        center_y = int(np.mean([c[0] for c in cluster]))
        max_drop = min(c[1] for c in cluster)
        span = cluster[-1][0] - cluster[0][0]
        print(f"  y≈{center_y} (span={span}): max_gradient={max_drop:.0f}")

# Save a visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

ax1.plot(range(h), var_smoothed, 'b-', linewidth=1)
ax1.set_title('Row Variance (smoothed)')
ax1.set_ylabel('Variance')
ax1.axvline(x=foul_line_y, color='red', linestyle='--', label=f'Foul line estimate y={foul_line_y}')
ax1.axhline(y=threshold, color='green', linestyle=':', label=f'Threshold={threshold:.0f}')
ax1.legend()

ax2.plot(range(h), mean_smoothed, 'g-', linewidth=1)
ax2.set_title('Mean Brightness (smoothed)')
ax2.set_ylabel('Mean Intensity')
ax2.axvline(x=foul_line_y, color='red', linestyle='--')

ax3.plot(range(h), var_grad_smooth, 'r-', linewidth=1)
ax3.set_title('Variance Gradient (steepest drop = foul line)')
ax3.set_ylabel('d(Variance)/dy')
ax3.set_xlabel('y (pixels)')
ax3.axvline(x=foul_line_y, color='red', linestyle='--')
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig("debug_foul_line_variance.png", dpi=100)
print("\nSaved debug_foul_line_variance.png")

# Final: draw the detected foul line on the frame
debug = frame.copy()
cv2.line(debug, (x_lo, foul_line_y), (x_hi, foul_line_y), (0, 0, 255), 4)
cv2.putText(debug, f"DETECTED FOUL LINE y={foul_line_y}", (x_lo, foul_line_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
cv2.imwrite("debug_foul_line_detected.png", debug)
print(f"Saved debug_foul_line_detected.png with foul line at y={foul_line_y}")
