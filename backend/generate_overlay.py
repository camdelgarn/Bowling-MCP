"""
Generate final lane + approach overlay for videos 1 and 2.
Uses the proven tracking method from detect_lane_track.py.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

VIDEO_DIR = '../video/behind/'

def find_lane_edges(gray, foul_y):
    """Find bowler's lane edges at foul line and track upward."""
    h, w = gray.shape
    
    # Find lane edges at foul line using gradient
    row = gaussian_filter1d(gray[foul_y - 20, :].astype(np.float64), sigma=30)
    grad = np.gradient(row)
    
    pos_peaks, _ = find_peaks(grad, height=0.3, distance=40)
    neg_peaks, _ = find_peaks(-grad, height=0.3, distance=40)
    
    # Pair positive peaks with following negative peaks to find bright regions
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
    
    # Pick brightest wide region nearest to frame center
    lane_candidates.sort(key=lambda c: abs(c[3] - w // 2))
    if not lane_candidates:
        return None
    
    lane = lane_candidates[0]
    left_start, right_start = lane[0], lane[1]
    
    # Track edges upward from foul line
    SEARCH_RADIUS = 40
    STEP = 3
    
    left_track = [left_start]
    right_track = [right_start]
    y_track = [foul_y - 20]
    prev_left, prev_right = left_start, right_start
    
    for y in range(foul_y - 20 - STEP, 400, -STEP):
        row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=20)
        grad = np.gradient(row)
        
        lane_center = (prev_left + prev_right) // 2
        lane_brightness = np.median(row[max(0, lane_center-50):min(w, lane_center+50)])
        if lane_brightness < 30:
            break
        
        # Left edge
        sl = max(0, prev_left - SEARCH_RADIUS)
        sr = min(w, prev_left + SEARCH_RADIUS)
        local_grad = grad[sl:sr]
        peaks, _ = find_peaks(local_grad, height=0.15, distance=10)
        if len(peaks) > 0:
            best = peaks[np.argmax(local_grad[peaks])]
            new_left = best + sl
        else:
            new_left = prev_left
        
        # Right edge
        sl = max(0, prev_right - SEARCH_RADIUS)
        sr = min(w, prev_right + SEARCH_RADIUS)
        local_grad = -grad[sl:sr]
        peaks, _ = find_peaks(local_grad, height=0.15, distance=10)
        if len(peaks) > 0:
            best = peaks[np.argmax(local_grad[peaks])]
            new_right = best + sl
        else:
            new_right = prev_right
        
        width = new_right - new_left
        if width < 100 or width > 700:
            break
        
        if abs(new_left - prev_left) > 25 or abs(new_right - prev_right) > 25:
            if len(left_track) >= 2:
                new_left = 2 * left_track[-1] - left_track[-2]
                new_right = 2 * right_track[-1] - right_track[-2]
            else:
                continue
        
        left_track.append(new_left)
        right_track.append(new_right)
        y_track.append(y)
        prev_left, prev_right = new_left, new_right
    
    # Robust line fit
    y_arr = np.array(y_track)
    left_arr = np.array(left_track)
    right_arr = np.array(right_track)
    
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
    
    return lc, rc, np.sum(lm), np.sum(rm), len(y_arr)


def draw_overlay(frame, lc, rc, foul_y, output_path):
    """Draw lane and approach outlines on the frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    
    # Find the y where lane width = ~30px (near pin area)
    # width = rc[0]*y + rc[1] - lc[0]*y - lc[1] = (rc[0]-lc[0])*y + (rc[1]-lc[1])
    slope_diff = rc[0] - lc[0]
    intercept_diff = rc[1] - lc[1]
    
    if slope_diff > 0:
        # Width = slope_diff * y + intercept_diff
        # Width = 30 → y = (30 - intercept_diff) / slope_diff
        pin_y = max(0, int((30 - intercept_diff) / slope_diff))
    else:
        pin_y = 0
    
    # Clamp to reasonable range
    pin_y = max(50, pin_y)
    
    # Lane outline
    lt_x = int(np.polyval(lc, pin_y))
    rt_x = int(np.polyval(rc, pin_y))
    lb_x = int(np.polyval(lc, foul_y))
    rb_x = int(np.polyval(rc, foul_y))
    
    lane_pts = np.array([[lt_x, pin_y], [rt_x, pin_y], [rb_x, foul_y], [lb_x, foul_y]], np.int32)
    
    # Semi-transparent fill
    overlay = out.copy()
    cv2.fillPoly(overlay, [lane_pts], (0, 180, 0))  # green fill
    cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
    
    # Lane outline (green)
    cv2.polylines(out, [lane_pts], True, (0, 255, 0), 3)
    
    # Foul line (red, thicker)
    # Approach is wider: lane + 2*gutter = lane * (41.5 + 2*9.25) / 41.5 = lane * 1.446
    lane_w_foul = rb_x - lb_x
    approach_extra = lane_w_foul * 9.25 / 41.5  # gutter width in pixels at foul line
    
    foul_left = int(lb_x - approach_extra)
    foul_right = int(rb_x + approach_extra)
    cv2.line(out, (foul_left, foul_y), (foul_right, foul_y), (0, 0, 255), 4)
    
    # Approach outline (cyan)
    # Extrapolate lane edges to bottom, then widen by gutter proportion
    al_x = np.polyval(lc, h - 1)
    ar_x = np.polyval(rc, h - 1)
    lane_w_bottom = ar_x - al_x
    approach_extra_bottom = lane_w_bottom * 9.25 / 41.5
    
    approach_pts = np.array([
        [foul_left, foul_y],
        [foul_right, foul_y],
        [int(ar_x + approach_extra_bottom), h - 1],
        [int(al_x - approach_extra_bottom), h - 1]
    ], np.int32)
    
    # Semi-transparent blue fill
    overlay = out.copy()
    cv2.fillPoly(overlay, [approach_pts], (200, 150, 0))  # cyan fill
    cv2.addWeighted(overlay, 0.1, out, 0.9, 0, out)
    
    cv2.polylines(out, [approach_pts], True, (255, 255, 0), 3)
    
    # Labels
    cv2.putText(out, 'LANE', (int((lt_x + rt_x) / 2) - 40, pin_y + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(out, 'APPROACH', (int((foul_left + foul_right) / 2) - 100, foul_y + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    cv2.putText(out, 'FOUL LINE', (foul_right + 20, foul_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, out)
    return pin_y, lane_w_foul


# Process videos 1 and 2
for vname, frame_idx, foul_y in [('1.MP4', 770, 1350), ('2.MP4', 691, 1353)]:
    print(f"\n{'='*60}")
    print(f"=== {vname} ===")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(VIDEO_DIR + vname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    result = find_lane_edges(gray, foul_y)
    if result is None:
        print("  Failed to find lane edges!")
        continue
    
    lc, rc, li, ri, total = result
    
    print(f"  Left fit:  x = {lc[0]:.4f}*y + {lc[1]:.1f}  ({li}/{total} inliers)")
    print(f"  Right fit: x = {rc[0]:.4f}*y + {rc[1]:.1f}  ({ri}/{total} inliers)")
    
    # Key positions
    for y_check in [200, 500, 800, foul_y, 2000, h-1]:
        lx = np.polyval(lc, y_check)
        rx = np.polyval(rc, y_check)
        print(f"  y={y_check:4d}: left={lx:.0f}, right={rx:.0f}, width={rx-lx:.0f}")
    
    # Draw overlay
    tag = vname.replace('.MP4', '')
    out_path = f'lane_overlay_{tag}.png'
    pin_y, lane_w = draw_overlay(frame, lc, rc, foul_y, out_path)
    print(f"\n  Output: {out_path}")
    print(f"  Pin area y: {pin_y}")
    print(f"  Lane width at foul: {lane_w}px")
    
    # Also save zoomed version (approach + foul area)
    out_zoom = frame.copy()
    # Draw on zoomed version
    _, _ = draw_overlay(frame, lc, rc, foul_y, f'lane_overlay_{tag}_full.png')

print("\nDone! Check lane_overlay_1.png and lane_overlay_2.png")
