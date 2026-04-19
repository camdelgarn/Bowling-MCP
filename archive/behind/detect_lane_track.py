"""
Track lane edges starting from the foul line (where edges are clearest)
and tracking upward with a small search window. Test on all 4 videos.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

VIDEO_DIR = '../video/behind/'
videos_info = {
    '1.MP4': {'frame': 770, 'foul_y': 1350},
    '2.MP4': {'frame': 691, 'foul_y': 1353},
    '3.MP4': {'frame': 525, 'foul_y': 1420},
    '4.MP4': {'frame': 619, 'foul_y': 1504},
}

fig, axes = plt.subplots(4, 3, figsize=(30, 28))

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
    print(f"=== {vname} (foul y={foul_y}) ===")
    print(f"{'='*60}")
    
    # Step 1: Find lane edges at foul line using gradient
    row = gaussian_filter1d(gray[foul_y - 20, :].astype(np.float64), sigma=30)
    grad = np.gradient(row)
    
    # Find all dark→bright transitions (positive gradient)
    pos_peaks, pos_props = find_peaks(grad, height=0.3, distance=40)
    neg_peaks, neg_props = find_peaks(-grad, height=0.3, distance=40)
    
    print(f"  Dark→Bright edges at y={foul_y-20}: {pos_peaks}")
    print(f"  Bright→Dark edges at y={foul_y-20}: {neg_peaks}")
    
    # Find the bowler's lane: bright region closest to frame center
    # Pair each pos peak with the nearest following neg peak
    lane_candidates = []
    for pp in pos_peaks:
        matching_neg = neg_peaks[neg_peaks > pp]
        if len(matching_neg) == 0:
            continue
        np_close = matching_neg[0]
        width = np_close - pp
        if 200 < width < 700:
            center = (pp + np_close) // 2
            brightness = np.mean(row[pp:np_close])
            lane_candidates.append((pp, np_close, width, center, brightness))
    
    print(f"  Lane candidates (200-700px wide):")
    for lc_item in lane_candidates:
        dist = abs(lc_item[3] - w // 2)
        print(f"    x={lc_item[0]}-{lc_item[1]}, w={lc_item[2]}, center={lc_item[3]}, dist_from_center={dist}")
    
    if not lane_candidates:
        print("  No lane candidates found!")
        all_results[vname] = None
        continue
    
    # Pick closest to center
    lane_candidates.sort(key=lambda c: abs(c[3] - w // 2))
    lane = lane_candidates[0]
    left_start = lane[0]
    right_start = lane[1]
    
    print(f"  → Bowler's lane at foul line: x={left_start}-{right_start}, width={lane[2]}")
    
    # Step 2: Track edges upward from foul line
    # At each y level, search for the edge within ±30px of the previous position
    SEARCH_RADIUS = 40
    STEP = 3  # y step
    
    left_track = [left_start]
    right_track = [right_start]
    y_track = [foul_y - 20]
    
    prev_left = left_start
    prev_right = right_start
    
    for y in range(foul_y - 20 - STEP, 400, -STEP):
        row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=20)
        grad = np.gradient(row)
        
        # Lane center (rough)
        lane_center = (prev_left + prev_right) // 2
        lane_brightness = np.mean(row[max(0,lane_center-50):min(w,lane_center+50)])
        
        if lane_brightness < 30:
            # Too dark, stop tracking
            break
        
        # Find left edge: positive gradient near prev_left
        search_l = max(0, prev_left - SEARCH_RADIUS)
        search_r = min(w, prev_left + SEARCH_RADIUS)
        local_grad = grad[search_l:search_r]
        
        peaks, _ = find_peaks(local_grad, height=0.15, distance=10)
        if len(peaks) > 0:
            # Pick the peak with strongest gradient
            peak_heights = local_grad[peaks]
            best_peak = peaks[np.argmax(peak_heights)]
            new_left = best_peak + search_l
        else:
            new_left = prev_left
        
        # Find right edge: negative gradient near prev_right
        search_l = max(0, prev_right - SEARCH_RADIUS)
        search_r = min(w, prev_right + SEARCH_RADIUS)
        local_grad = -grad[search_l:search_r]
        
        peaks, _ = find_peaks(local_grad, height=0.15, distance=10)
        if len(peaks) > 0:
            peak_heights = local_grad[peaks]
            best_peak = peaks[np.argmax(peak_heights)]
            new_right = best_peak + search_l
        else:
            new_right = prev_right
        
        # Validate: lane should be getting narrower as we go up
        new_width = new_right - new_left
        prev_width = prev_right - prev_left
        
        if new_width < 100 or new_width > 700:
            break
        
        # Don't allow sudden jumps > 20px
        if abs(new_left - prev_left) > 25 or abs(new_right - prev_right) > 25:
            # Use linear prediction instead
            if len(left_track) >= 2:
                new_left = 2 * left_track[-1] - left_track[-2]
                new_right = 2 * right_track[-1] - right_track[-2]
            else:
                continue
        
        left_track.append(new_left)
        right_track.append(new_right)
        y_track.append(y)
        prev_left = new_left
        prev_right = new_right
    
    print(f"  Tracked {len(y_track)} y levels (y={min(y_track)} to y={max(y_track)})")
    
    y_arr = np.array(y_track)
    left_arr = np.array(left_track)
    right_arr = np.array(right_track)
    
    # Step 3: Robust line fit
    def robust_fit(y_vals, x_vals, max_iter=10, threshold=10):
        mask = np.ones(len(y_vals), dtype=bool)
        for _ in range(max_iter):
            yf, xf = y_vals[mask], x_vals[mask]
            if len(yf) < 10:
                break
            c = np.polyfit(yf, xf, 1)
            res = np.abs(x_vals - np.polyval(c, y_vals))
            new_mask = res < threshold
            if np.array_equal(mask, new_mask):
                break
            mask = new_mask
        return c, mask
    
    lc, lm = robust_fit(y_arr, left_arr)
    rc, rm = robust_fit(y_arr, right_arr)
    
    li = np.sum(lm)
    ri = np.sum(rm)
    
    print(f"  Left fit:  x = {lc[0]:.4f}*y + {lc[1]:.1f}  ({li}/{len(y_arr)} inliers)")
    print(f"  Right fit: x = {rc[0]:.4f}*y + {rc[1]:.1f}  ({ri}/{len(y_arr)} inliers)")
    
    # Key widths
    for y_check in [200, 500, 800, foul_y, 2000, h-1]:
        lx = np.polyval(lc, y_check)
        rx = np.polyval(rc, y_check)
        print(f"  y={y_check:4d}: left={lx:.0f}, right={rx:.0f}, width={rx-lx:.0f}")
    
    # Perspective convergence point (vanishing point)
    # left: x = lc[0]*y + lc[1]
    # right: x = rc[0]*y + rc[1]
    # Intersection: lc[0]*y + lc[1] = rc[0]*y + rc[1]
    # y = (lc[1] - rc[1]) / (rc[0] - lc[0])
    if abs(rc[0] - lc[0]) > 0.001:
        vp_y = (lc[1] - rc[1]) / (rc[0] - lc[0])
        vp_x = np.polyval(lc, vp_y)
        print(f"  Vanishing point: ({vp_x:.0f}, {vp_y:.0f})")
    
    # Consistency
    left_res = np.abs(left_arr[lm] - np.polyval(lc, y_arr[lm]))
    right_res = np.abs(right_arr[rm] - np.polyval(rc, y_arr[rm]))
    consistency = np.mean(left_res) + np.mean(right_res)
    inlier_pct = (li + ri) / (2 * len(y_arr)) * 100
    
    print(f"  Inlier: {inlier_pct:.0f}%, Consistency: {consistency:.1f}")
    
    all_results[vname] = {
        'lc': lc, 'rc': rc, 'li': li, 'ri': ri,
        'total': len(y_arr), 'inlier_pct': inlier_pct,
        'consistency': consistency, 'foul_y': foul_y,
        'frame': frame,
        'w_foul': np.polyval(rc, foul_y) - np.polyval(lc, foul_y),
        'left_start': left_start, 'right_start': right_start,
        'y_min': min(y_track), 'y_max': max(y_track),
    }
    
    # === Plots ===
    # Plot 1: Tracked edges
    ax = axes[vi, 0]
    ax.scatter(left_arr[lm], y_arr[lm], c='green', s=3, label=f'Left ({li})')
    ax.scatter(left_arr[~lm], y_arr[~lm], c='red', s=1, alpha=0.3)
    ax.scatter(right_arr[rm], y_arr[rm], c='blue', s=3, label=f'Right ({ri})')
    ax.scatter(right_arr[~rm], y_arr[~rm], c='orange', s=1, alpha=0.3)
    y_ext = np.array([200, h-1])
    ax.plot(np.polyval(lc, y_ext), y_ext, 'g--', lw=2)
    ax.plot(np.polyval(rc, y_ext), y_ext, 'b--', lw=2)
    ax.axhline(y=foul_y, color='red', lw=2)
    ax.set_title(f'{vname} - Track ({inlier_pct:.0f}%, res={consistency:.1f})')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    
    # Plot 2: Width evolution
    ax = axes[vi, 1]
    widths = right_arr - left_arr
    ax.plot(y_arr, widths, 'k-', lw=0.5, alpha=0.5)
    y_fit = np.linspace(min(y_track), foul_y, 100)
    fit_widths = np.polyval(rc, y_fit) - np.polyval(lc, y_fit)
    ax.plot(y_fit, fit_widths, 'r-', lw=2, label='Fit')
    ax.axvline(x=foul_y, color='red', lw=1)
    ax.set_xlabel('y')
    ax.set_ylabel('Width (px)')
    ax.set_title(f'{vname} - Width: {np.polyval(rc, foul_y)-np.polyval(lc, foul_y):.0f}px at foul')
    ax.legend()
    
    # Plot 3: Lane overlay
    ax = axes[vi, 2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    
    from matplotlib.patches import Polygon as MPoly
    lt = np.polyval(lc, 200)
    rt = np.polyval(rc, 200)
    lb = np.polyval(lc, foul_y)
    rb = np.polyval(rc, foul_y)
    pts = np.array([[lt, 200], [rt, 200], [rb, foul_y], [lb, foul_y]])
    poly = MPoly(pts, closed=True, fill=False, edgecolor='lime', lw=2)
    ax.add_patch(poly)
    ax.plot([lb, rb], [foul_y, foul_y], 'r-', lw=3)
    
    # Approach
    al = np.polyval(lc, h-1)
    ar = np.polyval(rc, h-1)
    gf = (rb - lb) * 9.25 / 41.5
    gb = (ar - al) * 9.25 / 41.5
    app = np.array([[lb-gf, foul_y], [rb+gf, foul_y], [ar+gb, h-1], [al-gb, h-1]])
    apoly = MPoly(app, closed=True, fill=False, edgecolor='cyan', lw=2, ls='--')
    ax.add_patch(apoly)
    
    ax.set_xlim(1500, 4500)
    ax.set_ylim(h, 0)
    ax.set_title(f'{vname} - Overlay')

plt.tight_layout()
plt.savefig('debug_track_all.png', dpi=120)
print("\n\nSaved debug_track_all.png")

# Summary
print("\n" + "="*60)
print("TRACKING COMPARISON")
print("="*60)
print(f"{'Video':>6} | {'Y range':>12} | {'Inlier%':>8} | {'Consist':>8} | {'W@Foul':>8} | {'Slope_L':>8} | {'Slope_R':>8}")
print("-" * 75)
for vname, r in all_results.items():
    if r is None:
        print(f"{vname:>6} | FAILED")
    else:
        yr = f"{r['y_min']}-{r['y_max']}"
        print(f"{vname:>6} | {yr:>12} | {r['inlier_pct']:>7.0f}% | {r['consistency']:>8.1f} | {r['w_foul']:>8.0f} | {r['lc'][0]:>8.4f} | {r['rc'][0]:>8.4f}")

best_name = min((k for k,v in all_results.items() if v), key=lambda k: all_results[k]['consistency'])
print(f"\n→ Best: {best_name}")
