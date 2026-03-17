"""
Detect lane edges by walking outward from the known lane center.
At each y level, start at x≈2780 (lane center) and walk left/right 
until brightness drops below a threshold (gutter channel = dark).
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Try all 4 videos
VIDEO_DIR = '../video/behind/'
videos_info = {
    '1.MP4': {'frame': 770, 'foul_y': 1350},
    '2.MP4': {'frame': 691, 'foul_y': 1353},
    '3.MP4': {'frame': 525, 'foul_y': 1420},
    '4.MP4': {'frame': 619, 'foul_y': 1504},
}

fig, axes = plt.subplots(4, 2, figsize=(24, 28))

all_results = {}

for vi, (vname, vinfo) in enumerate(videos_info.items()):
    cap = cv2.VideoCapture(VIDEO_DIR + vname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, vinfo['frame'])
    ret, frame = cap.read()
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    foul_y = vinfo['foul_y']
    
    print(f"\n{'='*60}")
    print(f"=== {vname} (foul line y={foul_y}) ===")
    print(f"{'='*60}")
    
    # Step 1: Find lane center at foul line by finding the brightest plateau
    # in the mid-region of the frame
    row_foul = gaussian_filter1d(gray[foul_y - 30, :].astype(np.float64), sigma=40)
    
    # Find all bright regions
    threshold = np.percentile(row_foul, 60)
    bright = row_foul > threshold
    
    # Find contiguous bright regions
    regions = []
    in_reg = False
    for x in range(w):
        if bright[x] and not in_reg:
            in_reg = True
            start = x
        elif not bright[x] and in_reg:
            in_reg = False
            width_r = x - start
            if width_r > 200:  # lane must be at least 200px wide
                center = (start + x) // 2
                brightness = np.mean(row_foul[start:x])
                regions.append((start, x, width_r, center, brightness))
    
    print(f"  Bright regions at y={foul_y-30} (width>200):")
    for s, e, wd, c, b in regions:
        print(f"    x={s}-{e}, width={wd}, center={c}, brightness={b:.0f}")
    
    # Pick the region closest to frame center
    frame_cx = w // 2
    if regions:
        # Sort by distance to frame center
        regions.sort(key=lambda r: abs(r[3] - frame_cx))
        lane_region = regions[0]
        lane_cx_estimate = lane_region[3]
        print(f"  → Using lane center estimate: x={lane_cx_estimate}")
    else:
        lane_cx_estimate = frame_cx
        print(f"  → No clear region found, using frame center: {frame_cx}")
    
    # Step 2: Walk outward from lane center at each y
    y_start = max(600, foul_y - 800)  # up to 800px above foul line
    y_end = foul_y
    y_levels = list(range(y_start, y_end + 1, 3))
    
    left_edges = []
    right_edges = []
    valid_y = []
    
    for y in y_levels:
        # Smooth the row heavily to eliminate board joints
        row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=20)
        
        # Lane brightness: median of a 100px window around center
        cx = lane_cx_estimate
        lane_window = row[max(0,cx-50):min(w,cx+50)]
        lane_brightness = np.median(lane_window)
        
        if lane_brightness < 40:  # too dark (far end of lane)
            continue
        
        # Edge threshold: 70% of lane brightness
        edge_thresh = lane_brightness * 0.65
        
        # Walk left from center
        left_x = cx
        for x in range(cx, max(0, cx - 500), -1):
            if row[x] < edge_thresh:
                left_x = x
                break
        
        # Walk right from center
        right_x = cx
        for x in range(cx, min(w, cx + 500)):
            if row[x] < edge_thresh:
                right_x = x
                break
        
        width_det = right_x - left_x
        
        # Sanity: lane should be 150-600px wide
        if 150 < width_det < 600:
            left_edges.append(left_x)
            right_edges.append(right_x)
            valid_y.append(y)
    
    print(f"  Valid walk detections: {len(valid_y)} / {len(y_levels)}")
    
    if len(valid_y) < 20:
        print("  Too few detections, skipping")
        all_results[vname] = None
        continue
    
    valid_y = np.array(valid_y)
    left_edges = np.array(left_edges)
    right_edges = np.array(right_edges)
    
    # Step 3: Robust line fit
    def robust_fit(y_vals, x_vals, max_iter=10, threshold=12):
        mask = np.ones(len(y_vals), dtype=bool)
        for _ in range(max_iter):
            yf = y_vals[mask]
            xf = x_vals[mask]
            if len(yf) < 10:
                break
            coeffs = np.polyfit(yf, xf, 1)
            pred = np.polyval(coeffs, y_vals)
            res = np.abs(x_vals - pred)
            new_mask = res < threshold
            if np.array_equal(mask, new_mask):
                break
            mask = new_mask
        return coeffs, mask
    
    lc, lm = robust_fit(valid_y, left_edges)
    rc, rm = robust_fit(valid_y, right_edges)
    
    li = np.sum(lm)
    ri = np.sum(rm)
    
    print(f"  Left fit:  x = {lc[0]:.4f}*y + {lc[1]:.1f}  ({li}/{len(valid_y)} inliers)")
    print(f"  Right fit: x = {rc[0]:.4f}*y + {rc[1]:.1f}  ({ri}/{len(valid_y)} inliers)")
    
    # Check perspective
    w_top = np.polyval(rc, y_start) - np.polyval(lc, y_start)
    w_foul = np.polyval(rc, foul_y) - np.polyval(lc, foul_y)
    w_bottom = np.polyval(rc, h-1) - np.polyval(lc, h-1)
    
    print(f"  Width at y={y_start}: {w_top:.0f}")
    print(f"  Width at foul (y={foul_y}): {w_foul:.0f}")
    print(f"  Width at bottom (y={h-1}): {w_bottom:.0f}")
    
    inlier_pct = (li + ri) / (2 * len(valid_y)) * 100
    
    # Compute consistency: std of residuals for inliers
    left_residuals = np.abs(left_edges[lm] - np.polyval(lc, valid_y[lm]))
    right_residuals = np.abs(right_edges[rm] - np.polyval(rc, valid_y[rm]))
    consistency = np.mean(left_residuals) + np.mean(right_residuals)
    
    print(f"  Inlier %: {inlier_pct:.0f}%")
    print(f"  Mean residual: left={np.mean(left_residuals):.1f}, right={np.mean(right_residuals):.1f}")
    print(f"  Consistency score: {consistency:.1f} (lower=better)")
    
    all_results[vname] = {
        'left_coeffs': lc, 'right_coeffs': rc,
        'left_inliers': li, 'right_inliers': ri,
        'total': len(valid_y),
        'inlier_pct': inlier_pct,
        'consistency': consistency,
        'foul_y': foul_y,
        'w_foul': w_foul,
        'frame': frame,
    }
    
    # Plot 1: Edge scatter + fits
    ax = axes[vi, 0]
    ax.scatter(left_edges[lm], valid_y[lm], c='green', s=3, label=f'Left ({li})')
    ax.scatter(left_edges[~lm], valid_y[~lm], c='red', s=1, alpha=0.3)
    ax.scatter(right_edges[rm], valid_y[rm], c='blue', s=3, label=f'Right ({ri})')
    ax.scatter(right_edges[~rm], valid_y[~rm], c='orange', s=1, alpha=0.3)
    y_line = np.array([100, h-1])
    ax.plot(np.polyval(lc, y_line), y_line, 'g--', lw=2)
    ax.plot(np.polyval(rc, y_line), y_line, 'b--', lw=2)
    ax.axhline(y=foul_y, color='red', lw=2)
    ax.set_title(f'{vname} - Edges (inlier {inlier_pct:.0f}%, consistency {consistency:.1f})')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(2000, 3600)
    
    # Plot 2: Lane overlay
    ax = axes[vi, 1]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    
    # Lane trapezoid
    from matplotlib.patches import Polygon as MPoly
    lt = np.polyval(lc, 100)
    rt = np.polyval(rc, 100)
    lb = np.polyval(lc, foul_y)
    rb = np.polyval(rc, foul_y)
    pts = np.array([[lt, 100], [rt, 100], [rb, foul_y], [lb, foul_y]])
    poly = MPoly(pts, closed=True, fill=False, edgecolor='lime', lw=2)
    ax.add_patch(poly)
    
    # Foul line
    ax.plot([lb, rb], [foul_y, foul_y], 'r-', lw=3)
    
    # Approach
    ab_l = np.polyval(lc, h-1)
    ab_r = np.polyval(rc, h-1)
    gutter_f = (rb - lb) * 9.25 / 41.5
    gutter_b = (ab_r - ab_l) * 9.25 / 41.5
    app_pts = np.array([
        [lb - gutter_f, foul_y], [rb + gutter_f, foul_y],
        [ab_r + gutter_b, h-1], [ab_l - gutter_b, h-1]
    ])
    app_poly = MPoly(app_pts, closed=True, fill=False, edgecolor='cyan', lw=2, linestyle='--')
    ax.add_patch(app_poly)
    
    ax.set_xlim(1500, 4500)
    ax.set_ylim(h, 0)
    ax.set_title(f'{vname} - Width at foul: {w_foul:.0f}px')

plt.tight_layout()
plt.savefig('debug_walk_all.png', dpi=120)
print("\n\nSaved debug_walk_all.png")

# Final comparison
print("\n" + "="*60)
print("FINAL COMPARISON - Walk-out Method")
print("="*60)
print(f"{'Video':>6} | {'Inlier%':>8} | {'Consist':>8} | {'W@Foul':>8} | {'L_inl':>6} | {'R_inl':>6}")
print("-" * 55)
for vname, r in all_results.items():
    if r is None:
        print(f"{vname:>6} | {'FAILED':>8}")
    else:
        print(f"{vname:>6} | {r['inlier_pct']:>7.0f}% | {r['consistency']:>8.1f} | {r['w_foul']:>8.0f} | {r['left_inliers']:>6} | {r['right_inliers']:>6}")

# Pick the best
valid_results = {k: v for k, v in all_results.items() if v is not None}
if valid_results:
    best_name = min(valid_results, key=lambda k: valid_results[k]['consistency'])
    best = valid_results[best_name]
    print(f"\n→ Best video: {best_name} (consistency={best['consistency']:.1f}, {best['inlier_pct']:.0f}% inliers)")
