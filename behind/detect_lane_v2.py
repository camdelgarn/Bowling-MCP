"""
Detect lane edges using smoothed gradient approach.
Focus on the clear zone (y=900-1330) where lane is well-lit.
Use heavy smoothing to ignore board joints, find edges via gradient peaks.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load empty frame
cap = cv2.VideoCapture('../video/behind/1.MP4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 730)
ret, frame = cap.read()
cap.release()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"Frame: {w}x{h}")

# Known: foul line at y≈1330, lane center near x≈2800
FOUL_LINE_Y = 1330
LANE_CENTER_APPROX = 2800
SEARCH_LEFT = 2100   # search window
SEARCH_RIGHT = 3600

# For each y level, find left and right lane edges using smoothed gradient
y_levels = list(range(850, FOUL_LINE_Y + 1, 5))
left_edges = []
right_edges = []
valid_y = []

print("\n=== Smoothed gradient edge detection ===")
print(f"{'y':>5} | {'left':>5} | {'right':>5} | {'width':>5}")
print("-" * 35)

for y in y_levels:
    row = gray[y, SEARCH_LEFT:SEARCH_RIGHT].astype(np.float64)
    
    # Heavy Gaussian smoothing to blur out board joints (sigma=30px)
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(row, sigma=30)
    
    # Compute gradient
    gradient = np.gradient(smoothed)
    
    # The lane center in the search window
    center_idx = LANE_CENTER_APPROX - SEARCH_LEFT
    
    # Left edge: strongest positive gradient (dark→bright) to the left of center
    left_region = gradient[:center_idx]
    if len(left_region) == 0:
        continue
    
    # Right edge: strongest negative gradient (bright→dark) to the right of center
    right_region = gradient[center_idx:]
    if len(right_region) == 0:
        continue
    
    # Find the peak positive gradient (left edge of lane)
    # Look for the CLOSEST significant peak to the lane center
    from scipy.signal import find_peaks
    
    # Left edge: peaks in positive gradient, search from center leftward
    left_peaks, left_props = find_peaks(left_region, height=0.3, distance=20)
    if len(left_peaks) == 0:
        continue
    
    # Take the rightmost peak (closest to center) as the left lane edge
    left_edge_idx = left_peaks[-1]
    left_edge_x = left_edge_idx + SEARCH_LEFT
    
    # Right edge: peaks in negative gradient (bright→dark) 
    neg_gradient = -gradient[center_idx:]
    right_peaks, right_props = find_peaks(neg_gradient, height=0.3, distance=20)
    if len(right_peaks) == 0:
        continue
    
    # Take the leftmost peak (closest to center) as the right lane edge
    right_edge_idx = right_peaks[0] + center_idx
    right_edge_x = right_edge_idx + SEARCH_LEFT
    
    width = right_edge_x - left_edge_x
    
    # Sanity check: lane should be between 150-700px wide in this range
    if 150 < width < 700:
        left_edges.append(left_edge_x)
        right_edges.append(right_edge_x)
        valid_y.append(y)
        if y % 50 == 0:
            print(f"{y:5d} | {left_edge_x:5d} | {right_edge_x:5d} | {width:5d}")

print(f"\nValid detections: {len(valid_y)} / {len(y_levels)}")

# Convert to arrays
valid_y = np.array(valid_y)
left_edges = np.array(left_edges)
right_edges = np.array(right_edges)

# Fit lines with RANSAC-like approach (iterative outlier removal)
def robust_line_fit(y_vals, x_vals, max_iter=5, threshold=20):
    mask = np.ones(len(y_vals), dtype=bool)
    for iteration in range(max_iter):
        y_fit = y_vals[mask]
        x_fit = x_vals[mask]
        if len(y_fit) < 5:
            break
        coeffs = np.polyfit(y_fit, x_fit, 1)
        predicted = np.polyval(coeffs, y_vals)
        residuals = np.abs(x_vals - predicted)
        mask = residuals < threshold
    return coeffs, mask

left_coeffs, left_mask = robust_line_fit(valid_y, left_edges)
right_coeffs, right_mask = robust_line_fit(valid_y, right_edges)

print(f"\nRobust line fits:")
print(f"  Left edge:  x = {left_coeffs[0]:.3f}*y + {left_coeffs[1]:.1f}")
print(f"  Right edge: x = {right_coeffs[0]:.3f}*y + {right_coeffs[1]:.1f}")

# Check perspective direction
print(f"\n  At y=850 (upper lane):  left={np.polyval(left_coeffs, 850):.0f}, right={np.polyval(right_coeffs, 850):.0f}, width={np.polyval(right_coeffs, 850)-np.polyval(left_coeffs, 850):.0f}")
print(f"  At y=1330 (foul line):  left={np.polyval(left_coeffs, 1330):.0f}, right={np.polyval(right_coeffs, 1330):.0f}, width={np.polyval(right_coeffs, 1330)-np.polyval(left_coeffs, 1330):.0f}")
print(f"  At y=2988 (bottom):     left={np.polyval(left_coeffs, 2988):.0f}, right={np.polyval(right_coeffs, 2988):.0f}, width={np.polyval(right_coeffs, 2988)-np.polyval(left_coeffs, 2988):.0f}")

# Sanity: lane should widen as y increases (toward camera)
width_top = np.polyval(right_coeffs, 850) - np.polyval(left_coeffs, 850)
width_bottom = np.polyval(right_coeffs, 1330) - np.polyval(left_coeffs, 1330)
if width_bottom > width_top:
    print("\n  ✓ Lane widens toward camera (correct perspective)")
else:
    print("\n  ✗ Lane narrows toward camera (WRONG - may need to flip or recheck)")

# ---- Visualization ----
fig, axes = plt.subplots(1, 3, figsize=(24, 10))

# Plot 1: Raw detections and fits
ax1 = axes[0]
ax1.scatter(left_edges[left_mask], valid_y[left_mask], c='green', s=3, label='Left inliers')
ax1.scatter(left_edges[~left_mask], valid_y[~left_mask], c='red', s=3, alpha=0.3, label='Left outliers')
ax1.scatter(right_edges[right_mask], valid_y[right_mask], c='blue', s=3, label='Right inliers')
ax1.scatter(right_edges[~right_mask], valid_y[~right_mask], c='orange', s=3, alpha=0.3, label='Right outliers')

y_line = np.array([100, 2988])
ax1.plot(np.polyval(left_coeffs, y_line), y_line, 'g--', linewidth=2, label='Left fit')
ax1.plot(np.polyval(right_coeffs, y_line), y_line, 'b--', linewidth=2, label='Right fit')
ax1.axhline(y=FOUL_LINE_Y, color='red', linewidth=2, label='Foul line')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Lane Edge Detections')
ax1.legend(fontsize=8)
ax1.invert_yaxis()

# Plot 2: Sample smoothed profiles
ax2 = axes[1]
sample_ys = [900, 1000, 1100, 1200, 1300]
for sy in sample_ys:
    row = gray[sy, SEARCH_LEFT:SEARCH_RIGHT].astype(np.float64)
    smoothed = gaussian_filter1d(row, sigma=30)
    x_coords = np.arange(SEARCH_LEFT, SEARCH_RIGHT)
    ax2.plot(x_coords, smoothed, label=f'y={sy}')
ax2.axvline(x=LANE_CENTER_APPROX, color='gray', linestyle=':', label='Center')
ax2.set_xlabel('x')
ax2.set_ylabel('Intensity')
ax2.set_title('Smoothed intensity profiles')
ax2.legend()

# Plot 3: Lane outline on video
ax3 = axes[2]
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
ax3.imshow(frame_rgb)

# Draw lane trapezoid
lane_y_top = 200
lane_y_bot = FOUL_LINE_Y
lt_x = np.polyval(left_coeffs, lane_y_top)
rt_x = np.polyval(right_coeffs, lane_y_top)
lb_x = np.polyval(left_coeffs, lane_y_bot)
rb_x = np.polyval(right_coeffs, lane_y_bot)

lane_pts = np.array([[lt_x, lane_y_top], [rt_x, lane_y_top], 
                      [rb_x, lane_y_bot], [lb_x, lane_y_bot]])
from matplotlib.patches import Polygon
lane_poly = Polygon(lane_pts, closed=True, fill=False, edgecolor='lime', linewidth=2, label='Lane')
ax3.add_patch(lane_poly)

# Draw approach (extrapolate below foul line)
approach_y_bot = h - 1
al_x = np.polyval(left_coeffs, approach_y_bot)
ar_x = np.polyval(right_coeffs, approach_y_bot)

# The approach area is wider - add gutter width
# Bowling: lane=41.5", each gutter=9.25" → ratio = 9.25/41.5 = 0.223
lane_width_at_foul = rb_x - lb_x
gutter_px = lane_width_at_foul * 0.223
lane_width_at_bottom = ar_x - al_x
gutter_px_bottom = lane_width_at_bottom * 0.223

approach_pts = np.array([
    [lb_x - gutter_px, lane_y_bot],
    [rb_x + gutter_px, lane_y_bot],
    [ar_x + gutter_px_bottom, approach_y_bot],
    [al_x - gutter_px_bottom, approach_y_bot]
])
approach_poly = Polygon(approach_pts, closed=True, fill=False, edgecolor='cyan', linewidth=2, linestyle='--', label='Approach')
ax3.add_patch(approach_poly)

# Draw foul line
ax3.plot([lb_x - gutter_px, rb_x + gutter_px], [FOUL_LINE_Y, FOUL_LINE_Y], 'r-', linewidth=2, label='Foul line')

ax3.set_xlim(1500, 4500)
ax3.set_ylim(h, 0)
ax3.legend(loc='upper right')
ax3.set_title('Lane outline on video')

plt.tight_layout()
plt.savefig('debug_lane_v2.png', dpi=150)
print("\nSaved debug_lane_v2.png")

# Also save a full-frame annotated version
frame_annotated = frame.copy()

# Draw on the frame
def draw_line_on_frame(img, x1, y1, x2, y2, color, thickness=3):
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

# Lane outline (green)
draw_line_on_frame(frame_annotated, lt_x, lane_y_top, rt_x, lane_y_top, (0, 255, 0))
draw_line_on_frame(frame_annotated, rt_x, lane_y_top, rb_x, lane_y_bot, (0, 255, 0))
draw_line_on_frame(frame_annotated, rb_x, lane_y_bot, lb_x, lane_y_bot, (0, 255, 0))
draw_line_on_frame(frame_annotated, lb_x, lane_y_bot, lt_x, lane_y_top, (0, 255, 0))

# Foul line (red)
draw_line_on_frame(frame_annotated, lb_x - gutter_px, FOUL_LINE_Y, rb_x + gutter_px, FOUL_LINE_Y, (0, 0, 255), 4)

# Approach outline (cyan)
draw_line_on_frame(frame_annotated, lb_x - gutter_px, FOUL_LINE_Y, al_x - gutter_px_bottom, approach_y_bot, (255, 255, 0))
draw_line_on_frame(frame_annotated, rb_x + gutter_px, FOUL_LINE_Y, ar_x + gutter_px_bottom, approach_y_bot, (255, 255, 0))
draw_line_on_frame(frame_annotated, al_x - gutter_px_bottom, approach_y_bot, ar_x + gutter_px_bottom, approach_y_bot, (255, 255, 0))

cv2.imwrite('lane_outline_v2.png', frame_annotated)
print("Saved lane_outline_v2.png")
