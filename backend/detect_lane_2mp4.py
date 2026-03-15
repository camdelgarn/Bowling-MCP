"""
Detect lane edges on 2.MP4 using smoothed gradient in the clear zone (y=900-1350).
Focus on finding consistent gutter channel edges (dark-bright transitions).
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Load empty frame from 2.MP4
cap = cv2.VideoCapture('../video/behind/2.MP4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 691)
ret, frame = cap.read()
cap.release()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"Frame: {w}x{h}")

FOUL_LINE_Y = 1353

# ============================================================
# Step 1: Find the bowler's lane at the foul line
# ============================================================
# At y=1323, bowler's lane was at x=2551-3018 (width 467)
# Let's confirm by looking at the raw intensity

print("\n=== Raw intensity at y=1320 ===")
row = gray[1320, :].astype(np.float64)
smoothed = gaussian_filter1d(row, sigma=40)

# Find all bright regions
threshold = np.mean(smoothed) + 5
bright = smoothed > threshold

# Find contiguous bright regions
regions = []
in_reg = False
for x in range(w):
    if bright[x] and not in_reg:
        in_reg = True
        start = x
    elif not bright[x] and in_reg:
        in_reg = False
        if x - start > 100:  # minimum width
            regions.append((start, x, np.mean(smoothed[start:x])))

if in_reg and w - start > 100:
    regions.append((start, w, np.mean(smoothed[start:w])))

print("Bright regions (width>100):")
for s, e, b in regions:
    print(f"  x={s}-{e}, width={e-s}, brightness={b:.0f}")

# ============================================================
# Step 2: Trace lane edges from y=800 to y=1350
# ============================================================
# Strategy: at each y, compute smoothed gradient and find the
# left edge (positive gradient) and right edge (negative gradient)
# of the bright region nearest to the expected lane center

# Initial lane center estimate from foul line data
LANE_CENTER = 2785  # midpoint of 2551-3018
LANE_HALF_WIDTH = 250  # half of expected width

print(f"\n=== Tracing lane edges (y=800 to {FOUL_LINE_Y}) ===")

y_levels = list(range(800, FOUL_LINE_Y + 1, 3))
left_edges = []
right_edges = []
valid_y = []

for y in y_levels:
    row = gray[y, :].astype(np.float64)
    smoothed = gaussian_filter1d(row, sigma=25)
    grad = np.gradient(smoothed)
    
    # Expected lane center at this y (start with constant, will adaptive later)
    expected_cx = LANE_CENTER
    
    # Search for left edge: positive gradient peak in range [cx-400, cx]
    search_left = max(0, expected_cx - 500)
    search_right = min(w, expected_cx + 500)
    
    # Left edge: strongest positive gradient peak near expected left edge
    expected_left = expected_cx - LANE_HALF_WIDTH
    left_search_start = max(0, expected_left - 200)
    left_search_end = expected_cx
    
    left_grad = grad[left_search_start:left_search_end]
    if len(left_grad) == 0:
        continue
    
    peaks, props = find_peaks(left_grad, height=0.2, distance=20)
    if len(peaks) == 0:
        continue
    
    # Pick the peak closest to expected left edge position
    peak_positions = peaks + left_search_start
    distances = np.abs(peak_positions - expected_left)
    best_left = peak_positions[np.argmin(distances)]
    
    # Right edge: strongest negative gradient peak
    expected_right = expected_cx + LANE_HALF_WIDTH
    right_search_start = expected_cx
    right_search_end = min(w, expected_right + 200)
    
    right_grad = -grad[right_search_start:right_search_end]
    if len(right_grad) == 0:
        continue
    
    peaks, props = find_peaks(right_grad, height=0.2, distance=20)
    if len(peaks) == 0:
        continue
    
    peak_positions = peaks + right_search_start
    distances = np.abs(peak_positions - expected_right)
    best_right = peak_positions[np.argmin(distances)]
    
    # Validate
    width = best_right - best_left
    if 150 < width < 700:
        left_edges.append(best_left)
        right_edges.append(best_right)
        valid_y.append(y)

print(f"Valid detections: {len(valid_y)} / {len(y_levels)}")

valid_y = np.array(valid_y)
left_edges = np.array(left_edges)
right_edges = np.array(right_edges)

# ============================================================
# Step 3: Robust linear fit with iterative outlier removal
# ============================================================
def robust_line_fit(y_vals, x_vals, max_iter=10, threshold=15):
    mask = np.ones(len(y_vals), dtype=bool)
    for _ in range(max_iter):
        y_fit = y_vals[mask]
        x_fit = x_vals[mask]
        if len(y_fit) < 10:
            break
        coeffs = np.polyfit(y_fit, x_fit, 1)
        predicted = np.polyval(coeffs, y_vals)
        residuals = np.abs(x_vals - predicted)
        new_mask = residuals < threshold
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
    return coeffs, mask

left_coeffs, left_mask = robust_line_fit(valid_y, left_edges)
right_coeffs, right_mask = robust_line_fit(valid_y, right_edges)

n_left_inliers = np.sum(left_mask)
n_right_inliers = np.sum(right_mask)

print(f"\nRobust line fits:")
print(f"  Left edge:  x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}  ({n_left_inliers} inliers)")
print(f"  Right edge: x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}  ({n_right_inliers} inliers)")

# Key positions
for y_check in [100, 500, 800, 1000, 1200, FOUL_LINE_Y, 2000, 2988]:
    lx = np.polyval(left_coeffs, y_check)
    rx = np.polyval(right_coeffs, y_check)
    print(f"  y={y_check:4d}: left={lx:.0f}, right={rx:.0f}, width={rx-lx:.0f}")

# Perspective check
w_top = np.polyval(right_coeffs, 800) - np.polyval(left_coeffs, 800)
w_bot = np.polyval(right_coeffs, FOUL_LINE_Y) - np.polyval(left_coeffs, FOUL_LINE_Y)
print(f"\n  Width at y=800: {w_top:.0f}px")
print(f"  Width at foul line: {w_bot:.0f}px")
if w_bot > w_top:
    print("  ✓ Lane widens toward camera (correct)")
else:
    print("  ✗ WRONG perspective")

# Print edge data for inspection
print(f"\n  Sample left edges (every 50y):")
for j in range(0, len(valid_y), 15):
    y = valid_y[j]
    x = left_edges[j]
    pred = np.polyval(left_coeffs, y)
    tag = "✓" if left_mask[j] else "✗"
    print(f"    y={y:4d}: x={x:4d} (pred={pred:.0f}) {tag}")

print(f"\n  Sample right edges (every 50y):")
for j in range(0, len(valid_y), 15):
    y = valid_y[j]
    x = right_edges[j]
    pred = np.polyval(right_coeffs, y)
    tag = "✓" if right_mask[j] else "✗"
    print(f"    y={y:4d}: x={x:4d} (pred={pred:.0f}) {tag}")

# ============================================================
# Step 4: Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(24, 16))

# Plot 1: Edge detections
ax = axes[0, 0]
ax.scatter(left_edges[left_mask], valid_y[left_mask], c='green', s=5, label=f'Left inliers ({n_left_inliers})')
ax.scatter(left_edges[~left_mask], valid_y[~left_mask], c='red', s=2, alpha=0.3, label='Left outliers')
ax.scatter(right_edges[right_mask], valid_y[right_mask], c='blue', s=5, label=f'Right inliers ({n_right_inliers})')
ax.scatter(right_edges[~right_mask], valid_y[~right_mask], c='orange', s=2, alpha=0.3, label='Right outliers')
y_line = np.array([100, 2988])
ax.plot(np.polyval(left_coeffs, y_line), y_line, 'g--', linewidth=2)
ax.plot(np.polyval(right_coeffs, y_line), y_line, 'b--', linewidth=2)
ax.axhline(y=FOUL_LINE_Y, color='red', linewidth=2, label='Foul line')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Lane Edge Detections (2.MP4)')
ax.legend()
ax.invert_yaxis()
ax.set_xlim(2000, 3500)

# Plot 2: Width vs y
ax = axes[0, 1]
widths = right_edges - left_edges
ax.scatter(valid_y[left_mask & right_mask], widths[left_mask & right_mask], c='green', s=5)
y_range = np.linspace(800, FOUL_LINE_Y, 100)
fit_widths = np.polyval(right_coeffs, y_range) - np.polyval(left_coeffs, y_range)
ax.plot(y_range, fit_widths, 'r-', linewidth=2, label='Fit width')
ax.set_xlabel('y')
ax.set_ylabel('Lane width (px)')
ax.set_title('Lane Width vs Y')
ax.legend()

# Plot 3: Intensity profiles at key y levels
ax = axes[1, 0]
for check_y in [900, 1000, 1100, 1200, 1300]:
    row = gray[check_y, 2200:3400].astype(np.float64)
    smoothed = gaussian_filter1d(row, sigma=25)
    ax.plot(np.arange(2200, 3400), smoothed, label=f'y={check_y}', linewidth=0.8)
    
    # Mark fitted edges
    lx = int(np.polyval(left_coeffs, check_y))
    rx = int(np.polyval(right_coeffs, check_y))
    if 2200 <= lx <= 3400:
        ax.axvline(x=lx, color='green', alpha=0.3, linewidth=0.5)
    if 2200 <= rx <= 3400:
        ax.axvline(x=rx, color='blue', alpha=0.3, linewidth=0.5)

ax.set_xlabel('x')
ax.set_ylabel('Intensity')
ax.set_title('Smoothed Intensity Profiles with Fitted Edges')
ax.legend(fontsize=8)

# Plot 4: Lane outline on frame
ax = axes[1, 1]
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
ax.imshow(frame_rgb)

# Lane trapezoid
lt_x = np.polyval(left_coeffs, 100)
rt_x = np.polyval(right_coeffs, 100)
lb_x = np.polyval(left_coeffs, FOUL_LINE_Y)
rb_x = np.polyval(right_coeffs, FOUL_LINE_Y)

from matplotlib.patches import Polygon
lane_pts = np.array([[lt_x, 100], [rt_x, 100], [rb_x, FOUL_LINE_Y], [lb_x, FOUL_LINE_Y]])
lane_poly = Polygon(lane_pts, closed=True, fill=False, edgecolor='lime', linewidth=2, label='Lane')
ax.add_patch(lane_poly)

# Foul line
ax.plot([lb_x, rb_x], [FOUL_LINE_Y, FOUL_LINE_Y], 'r-', linewidth=3, label='Foul line')

# Approach extrapolation
al_x = np.polyval(left_coeffs, h-1)
ar_x = np.polyval(right_coeffs, h-1)
# Approach is wider (includes gutter width) - bowling ratio: gutter=9.25"/lane=41.5"
lane_w_foul = rb_x - lb_x
gutter_foul = lane_w_foul * 9.25 / 41.5
lane_w_bot = ar_x - al_x
gutter_bot = lane_w_bot * 9.25 / 41.5

approach_pts = np.array([
    [lb_x - gutter_foul, FOUL_LINE_Y],
    [rb_x + gutter_foul, FOUL_LINE_Y],
    [ar_x + gutter_bot, h-1],
    [al_x - gutter_bot, h-1]
])
approach_poly = Polygon(approach_pts, closed=True, fill=False, edgecolor='cyan', linewidth=2, linestyle='--', label='Approach')
ax.add_patch(approach_poly)

ax.set_xlim(1500, 4500)
ax.set_ylim(h, 0)
ax.legend(loc='upper right')
ax.set_title('Lane Outline on 2.MP4')

plt.tight_layout()
plt.savefig('debug_lane_2mp4.png', dpi=150)
print("\nSaved debug_lane_2mp4.png")

# Also save OpenCV annotated full frame
frame_out = frame.copy()
pts_lane = np.array([[int(lt_x), 100], [int(rt_x), 100], [int(rb_x), FOUL_LINE_Y], [int(lb_x), FOUL_LINE_Y]], np.int32)
cv2.polylines(frame_out, [pts_lane], True, (0, 255, 0), 3)
cv2.line(frame_out, (int(lb_x), FOUL_LINE_Y), (int(rb_x), FOUL_LINE_Y), (0, 0, 255), 4)

pts_approach = np.array([
    [int(lb_x - gutter_foul), FOUL_LINE_Y],
    [int(rb_x + gutter_foul), FOUL_LINE_Y],
    [int(ar_x + gutter_bot), h-1],
    [int(al_x - gutter_bot), h-1]
], np.int32)
cv2.polylines(frame_out, [pts_approach], True, (255, 255, 0), 3)

cv2.imwrite('lane_outline_2mp4.png', frame_out)
print("Saved lane_outline_2mp4.png")

print(f"\nFinal detection for 2.MP4:")
print(f"  Foul line: y={FOUL_LINE_Y}")
print(f"  Lane at foul: left={lb_x:.0f}, right={rb_x:.0f}, width={rb_x-lb_x:.0f}px")
print(f"  Lane at pins:  left={lt_x:.0f}, right={rt_x:.0f}, width={rt_x-lt_x:.0f}px")
print(f"  Left edge:  x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}")
print(f"  Right edge: x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}")
