#!/usr/bin/env python3
"""
Detect gutters and foul line directly in the video frame.
No layout PNG dependency.

Key insight: Gutters are thin dark channels running along the lane.
They exist only on the lane side of the foul line, not on the approach.

Approach:
1. Look at vertical intensity profiles through the frame center
2. Look for thin dark vertical features (gutters) using morphological operations
3. Find where these features start/end vertically
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

frame = cv2.imread("empty_frame.png")
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame: {w}x{h}")

# ============================================================================
# PART A: Vertical profile analysis
# Take vertical strips and look at how brightness changes top to bottom
# ============================================================================
print("\n=== Part A: Vertical brightness profiles ===")

# Take several vertical profiles at different x positions
x_positions = [2500, 2800, 3000, 3200, 3400, 3600, 3800, 4000]

fig, axes = plt.subplots(len(x_positions), 1, figsize=(14, 3*len(x_positions)))

for idx, x_pos in enumerate(x_positions):
    # Average a strip of 30 pixels wide centered at x_pos
    strip = gray[:, x_pos-15:x_pos+15].astype(float)
    profile = np.mean(strip, axis=1)
    
    # Smooth
    k = 31
    smooth_profile = np.convolve(profile, np.ones(k)/k, mode='same')
    
    axes[idx].plot(range(h), smooth_profile, 'b-', linewidth=0.5)
    axes[idx].set_title(f'Vertical profile at x={x_pos}')
    axes[idx].set_ylabel('Intensity')
    axes[idx].set_xlabel('y (top=pins, bottom=camera)')
    axes[idx].set_ylim(0, 255)
    
    # Find significant transitions (brightness jumps)
    grad = np.gradient(smooth_profile)
    # Mark where gradient is strong (potential foul line / boundary)
    for i in range(50, len(grad)-50):
        if abs(grad[i]) > 2:  # Strong vertical gradient
            axes[idx].axvline(x=i, color='red', alpha=0.1, linewidth=0.5)

plt.tight_layout()
plt.savefig("debug_vertical_profiles.png", dpi=100)
print("Saved debug_vertical_profiles.png")

# ============================================================================
# PART B: Detect thin vertical dark features (gutter channels)
# Use morphological operations to isolate narrow dark vertical features
# ============================================================================
print("\n=== Part B: Morphological gutter channel detection ===")

# Create a vertically-oriented morphological kernel
# The gutter is a narrow dark channel: dark pixels surrounded by brighter pixels horizontally
# Use a horizontal top-hat transform: close (fill) narrow dark gaps, then subtract original

# Black top hat = closing - original : highlights narrow dark features
# Use a wide horizontal kernel to bridge the gutter width
kernel_width = 61  # Should be wider than the gutter (~30-50 px at most)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
dark_features = cv2.subtract(closed, gray)

# Threshold the dark features
_, gutter_mask = cv2.threshold(dark_features, 15, 255, cv2.THRESH_BINARY)

# Now apply a vertical morphological operation to keep only vertically continuous features
# (gutters run vertically in the image, noise doesn't)
vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 51))
gutter_mask = cv2.morphologyEx(gutter_mask, cv2.MORPH_OPEN, vert_kernel)

cv2.imwrite("debug_gutter_mask.png", gutter_mask)
print("Saved debug_gutter_mask.png (white = detected gutter-like features)")

# Analyze the gutter mask
# For each row, find the x positions of detected gutter pixels
gutter_pixels_by_row = {}
for y in range(0, h, 5):
    row = gutter_mask[y, :]
    gutter_xs = np.where(row > 0)[0]
    if len(gutter_xs) > 0:
        gutter_pixels_by_row[y] = gutter_xs

# Find clusters of gutter pixels (each cluster is a gutter channel)
print(f"\nRows with gutter-like features: {len(gutter_pixels_by_row)} out of {h//5}")

# For rows with gutter features, find the distinct clusters
gutter_features = []
for y in sorted(gutter_pixels_by_row.keys()):
    xs = gutter_pixels_by_row[y]
    # Cluster the x positions (gaps > 50 px = different cluster)
    clusters = []
    cluster_start = xs[0]
    prev_x = xs[0]
    for x in xs[1:]:
        if x - prev_x > 30:
            clusters.append((cluster_start, prev_x))
            cluster_start = x
        prev_x = x
    clusters.append((cluster_start, prev_x))
    
    for cl, cr in clusters:
        width = cr - cl
        if 5 < width < 80:  # gutter-like width
            gutter_features.append((y, cl, cr, width))

# Group these features by their x position to find continuous gutter lines
if gutter_features:
    # Sort by x center
    features_by_x = {}
    for y, cl, cr, w in gutter_features:
        cx = (cl + cr) // 2
        # Bin by x (50 pixel bins)
        bin_x = cx // 50 * 50
        if bin_x not in features_by_x:
            features_by_x[bin_x] = []
        features_by_x[bin_x].append((y, cl, cr, w))
    
    print(f"\nGutter feature clusters by x position:")
    for bin_x in sorted(features_by_x.keys()):
        features = features_by_x[bin_x]
        if len(features) >= 5:  # Need enough vertical continuity
            ys = [f[0] for f in features]
            y_range = max(ys) - min(ys)
            avg_x = int(np.mean([(f[1]+f[2])//2 for f in features]))
            avg_w = np.mean([f[3] for f in features])
            print(f"  x≈{avg_x} (bin {bin_x}): {len(features)} rows, y={min(ys)}-{max(ys)}, span={y_range}, avg_width={avg_w:.0f}")

# ============================================================================
# PART C: Try horizontal edge detection to find the foul line directly
# The foul line is a physical line on the floor - should show as a horizontal edge
# ============================================================================
print("\n=== Part C: Horizontal line detection (foul line) ===")

# Compute vertical Sobel (horizontal edges)
sobel_y = cv2.Sobel(gray.astype(float), cv2.CV_64F, 0, 1, ksize=5)

# For each row, compute the average horizontal edge strength across the central region
# The foul line should span the full lane width
lane_x_range = (2200, 4200)  # approximate from earlier analysis

row_h_edge_strength = np.zeros(h)
for y in range(h):
    row_seg = np.abs(sobel_y[y, lane_x_range[0]:lane_x_range[1]])
    row_h_edge_strength[y] = np.mean(row_seg)

# Smooth
k = 21
smooth_h_edge = np.convolve(row_h_edge_strength, np.ones(k)/k, mode='same')

# Find peaks (potential horizontal line locations)
from scipy import signal
peaks, properties = signal.find_peaks(smooth_h_edge, height=np.percentile(smooth_h_edge, 95), distance=20)

print(f"\nTop horizontal edge peaks (potential foul line candidates):")
peak_strengths = [(y, smooth_h_edge[y]) for y in peaks]
peak_strengths.sort(key=lambda p: -p[1])
for y, strength in peak_strengths[:20]:
    # Check if this is a continuous horizontal edge (not just noise)
    row = np.abs(sobel_y[y, lane_x_range[0]:lane_x_range[1]])
    # Count how many pixels in this row have strong horizontal edge
    strong_count = np.sum(row > 30)
    total = lane_x_range[1] - lane_x_range[0]
    coverage = strong_count / total * 100
    print(f"  y={y}: strength={strength:.1f}, edge_coverage={coverage:.0f}%")

# ============================================================================
# PART D: Color-based analysis 
# The approach and lane may have different wood colors
# ============================================================================  
print("\n=== Part D: Color difference between lane vs approach ===")

# Convert to HSV to analyze hue/saturation
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Sample horizontal bands at different y positions
# Compare color characteristics
print("\nColor analysis at different y levels (center of frame):")
print("y     | B    | G    | R    | H    | S    | V")
print("-" * 60)
center_x = w // 2
sample_w = 200
for y in range(200, h, 100):
    b_avg = np.mean(frame[y, center_x-sample_w:center_x+sample_w, 0])
    g_avg = np.mean(frame[y, center_x-sample_w:center_x+sample_w, 1])
    r_avg = np.mean(frame[y, center_x-sample_w:center_x+sample_w, 2])
    h_avg = np.mean(hsv[y, center_x-sample_w:center_x+sample_w, 0])
    s_avg = np.mean(hsv[y, center_x-sample_w:center_x+sample_w, 1])
    v_avg = np.mean(hsv[y, center_x-sample_w:center_x+sample_w, 2])
    print(f"{y:>5} | {b_avg:>4.0f} | {g_avg:>4.0f} | {r_avg:>4.0f} | {h_avg:>4.0f} | {s_avg:>4.0f} | {v_avg:>4.0f}")

# ============================================================================
# PART E: Create a comprehensive debug image
# ============================================================================
debug = frame.copy()

# Overlay gutter mask in red
gutter_overlay = np.zeros_like(frame)
gutter_overlay[:,:,2] = gutter_mask  # Red channel
debug = cv2.addWeighted(debug, 1.0, gutter_overlay, 0.5, 0)

# Draw horizontal edge peaks
for y, strength in peak_strengths[:10]:
    cv2.line(debug, (lane_x_range[0], y), (lane_x_range[1], y), (0, 255, 255), 2)
    cv2.putText(debug, f"h_edge y={y}", (lane_x_range[1]+10, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

cv2.imwrite("debug_comprehensive.png", debug)
print(f"\nSaved debug_comprehensive.png")
print("  Red overlay = gutter-like features")
print("  Yellow lines = horizontal edge peaks (foul line candidates)")
