#!/usr/bin/env python3
"""
Focused lane edge detection in the brighter part of the lane near the foul line.
Then extrapolate perspective lines to pins and approach.

Strategy:
1. Focus on y=1000-1330 where the lane is visible
2. Use the Sobel-x gradient to find strong vertical edges (gutter walls)  
3. Group consistent edges into left and right gutter lines
4. Fit perspective lines and extrapolate
"""

import cv2
import numpy as np

frame = cv2.imread("empty_frame.png")
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame: {w}x{h}")

FOUL_LINE_Y = 1330

# Step 1: Print raw intensity values at a few key y positions
# to manually identify where the gutter edges are
print("\n=== Raw intensity scan near foul line ===")
print("These are the rows closest to the foul line where the lane structure should be clearest")

for y_sample in [1100, 1150, 1200, 1250, 1300, 1325]:
    row = gray[y_sample, :].astype(float)
    # Smooth slightly
    smooth = np.convolve(row, np.ones(15)/15, mode='same')
    
    # Print intensity at every 25 pixels in the lane region 
    print(f"\ny={y_sample}:")
    vals = []
    for x in range(2000, 4500, 25):
        v = int(smooth[x])
        vals.append((x, v))
    
    # Show as text bar chart (compact)
    for x, v in vals:
        bar = '#' * (v // 8)
        print(f"  {x:>4}: {v:>3} |{bar}")

# Step 2: Look at the intensity derivative to find edges
print("\n\n=== Gradient-based edge detection near foul line ===")

# Compute horizontal Sobel
sobel_x = cv2.Sobel(gray.astype(float), cv2.CV_64F, 1, 0, ksize=5)

# At each row near foul line, find the strongest edges
print("\nAt each y, find the strongest positive and negative gradient peaks")
print("Positive = left edge of bright region, Negative = right edge of bright region")

edge_data = []  # (y, edge_type, x, strength)

for y in range(800, FOUL_LINE_Y + 50, 5):
    row_grad = sobel_x[y, :]
    smooth_grad = np.convolve(row_grad, np.ones(11)/11, mode='same')
    
    # In the lane x range
    search_lo, search_hi = 2000, 4500
    seg = smooth_grad[search_lo:search_hi]
    
    # Find local maxima (positive gradient = dark-to-bright = left edge of bright feature)
    pos_peaks = []
    neg_peaks = []
    
    for i in range(20, len(seg) - 20):
        val = seg[i]
        # Check if it's a local maximum/minimum
        if val > 20 and val == max(seg[max(0,i-10):i+11]):
            pos_peaks.append((i + search_lo, val))
        elif val < -20 and val == min(seg[max(0,i-10):i+11]):
            neg_peaks.append((i + search_lo, val))
    
    # Store all peaks
    for x, v in pos_peaks:
        edge_data.append((y, 'pos', x, v))
    for x, v in neg_peaks:
        edge_data.append((y, 'neg', x, abs(v)))

# Cluster edges by x position to find consistent vertical features
print("\nClustering edges by x position...")

# Bin edges by x (25px bins)
from collections import defaultdict
bins = defaultdict(list)
for y, etype, x, strength in edge_data:
    bin_x = x // 25 * 25
    bins[bin_x].append((y, etype, x, strength))

# Find bins with many detections (consistent vertical features)
consistent_features = []
for bin_x in sorted(bins.keys()):
    entries = bins[bin_x]
    if len(entries) >= 15:  # Need enough vertical points
        ys = [e[0] for e in entries]
        avg_x = np.mean([e[2] for e in entries])
        avg_str = np.mean([e[3] for e in entries])
        y_range = max(ys) - min(ys)
        # Check what type (mostly positive or negative gradients?)
        pos_count = sum(1 for e in entries if e[1] == 'pos')
        neg_count = sum(1 for e in entries if e[1] == 'neg')
        dominant = 'pos' if pos_count > neg_count else 'neg'
        consistent_features.append({
            'bin_x': bin_x,
            'avg_x': avg_x,
            'avg_strength': avg_str,
            'count': len(entries),
            'y_min': min(ys), 
            'y_max': max(ys),
            'y_range': y_range,
            'dominant': dominant,
            'pos_count': pos_count,
            'neg_count': neg_count
        })

print(f"\nConsistent vertical features (at least 15 detections):")
print(f"{'bin_x':>6} | {'avg_x':>6} | {'count':>5} | {'y_range':>7} | {'strength':>8} | {'type':>4} | pos/neg")
print("-" * 75)
for f in consistent_features:
    print(f"  {f['bin_x']:>4} | {f['avg_x']:>6.0f} | {f['count']:>5} | {f['y_min']:>4}-{f['y_max']:<4} | {f['avg_strength']:>8.1f} | {f['dominant']:>4} | {f['pos_count']}/{f['neg_count']}")

# Step 3: The gutter edges should be:
# Left gutter: a 'neg' edge (bright lane → dark gutter) followed by a 'pos' edge (dark gutter → bright adjacent)
# Right gutter: a 'pos' edge (bright adjacent → dark gutter) followed by a 'neg' edge (dark gutter → bright lane)
# 
# Actually:
# LEFT side of lane (looking down):
#   [adjacent area/approach extension] [left gutter (dark)] [LANE (bright)]
#   So going left-to-right: bright→dark(neg edge) then dark→bright(pos edge=left edge of lane)
# 
# RIGHT side of lane:
#   [LANE (bright)] [right gutter (dark)] [adjacent area]  
#   Going left-to-right: bright→dark(neg=right edge of lane) then dark→bright(pos)
#
# So the lane surface is bounded by:
#   LEFT boundary: a positive gradient peak (entering the bright lane from left gutter)
#   RIGHT boundary: a negative gradient peak (leaving the bright lane into right gutter)

# Find pairs of edges that could be gutters (pos then neg or neg then pos within ~50px)
print("\n=== Looking for gutter channel pairs ===")

# Among consistent features, find features that are:
# 1. Close together in x (~20-80 px apart)
# 2. One is 'pos' and the other is 'neg'
for i, f1 in enumerate(consistent_features):
    for f2 in consistent_features[i+1:]:
        gap = f2['avg_x'] - f1['avg_x']
        if 15 < gap < 100:
            if f1['dominant'] != f2['dominant']:
                # Could be a gutter pair
                if f1['dominant'] == 'neg' and f2['dominant'] == 'pos':
                    desc = "LEFT gutter (bright→dark→bright = left side of lane)"
                else:
                    desc = "RIGHT gutter (bright→dark→bright = right side of lane)"
                
                overlap_y = max(f1['y_min'], f2['y_min']), min(f1['y_max'], f2['y_max'])
                if overlap_y[1] > overlap_y[0]:
                    print(f"  PAIR: x={f1['avg_x']:.0f}({f1['dominant']}) -- {gap:.0f}px gap -- x={f2['avg_x']:.0f}({f2['dominant']})")
                    print(f"    y overlap: {overlap_y[0]}-{overlap_y[1]}, {desc}")

# Step 4: Also try Hough line detection on the lane area
print("\n=== Hough line detection on lane area ===")
lane_region = gray[600:FOUL_LINE_Y+50, 2000:4500]

# Edge detection
edges = cv2.Canny(lane_region, 50, 150)

# Detect lines (looking for near-vertical lines = gutter edges)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=20)

if lines is not None:
    # Filter for near-vertical lines (angle close to 90 degrees from horizontal)
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1:
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        
        # Near-vertical: angle between 75 and 105 degrees
        if 70 < angle < 110:
            # Convert back to full frame coordinates
            x1_full = x1 + 2000
            y1_full = y1 + 600
            x2_full = x2 + 2000
            y2_full = y2 + 600
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            avg_x = (x1_full + x2_full) // 2
            vertical_lines.append((avg_x, y1_full, y2_full, length, angle))
    
    # Sort by x position
    vertical_lines.sort(key=lambda l: l[0])
    
    print(f"Found {len(vertical_lines)} near-vertical lines:")
    for avg_x, y1, y2, length, angle in vertical_lines:
        if length > 50:
            print(f"  x≈{avg_x}, y={y1}-{y2}, length={length:.0f}, angle={angle:.1f}°")
else:
    print("No lines detected")

# Step 5: Save visualization
debug = frame.copy()

# Draw edge data as colored dots
for y, etype, x, strength in edge_data:
    if strength > 30:
        color = (0, 255, 0) if etype == 'pos' else (0, 0, 255)
        cv2.circle(debug, (x, y), 2, color, -1)

# Draw consistent features as vertical lines
for f in consistent_features:
    color = (0, 255, 0) if f['dominant'] == 'pos' else (0, 0, 255)
    x = int(f['avg_x'])
    cv2.line(debug, (x, f['y_min']), (x, f['y_max']), color, 2)
    cv2.putText(debug, f"x={x}", (x+5, f['y_min']), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# Foul line
cv2.line(debug, (2000, FOUL_LINE_Y), (4500, FOUL_LINE_Y), (0, 0, 255), 4)

cv2.imwrite("debug_edge_analysis.png", debug)
print(f"\nSaved debug_edge_analysis.png")
