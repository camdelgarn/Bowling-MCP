"""
Draw board lines (0-40) on the approach area with dot markers.
Board mapping:
  - Board 20 = center (big dot)
  - Heading right (camera right, larger x): board numbers decrease (-5 per dot)
  - Heading left (camera left, smaller x): board numbers increase (+5 per dot)
  - Board 0 = far right edge, Board 40 = far left edge
  - Dots at boards: 10, 15, 20 (big), 25, 30

Coordinate mapping:
  - Lane left edge (smaller x in image) = board 40 (bowler's left)
  - Lane right edge (larger x in image) = board 0 (bowler's right)
  - Board n at y: x = x_right(y) + (x_left(y) - x_right(y)) * n / 40
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

VIDEO_DIR = '../video/behind/'

def find_lane_edges(gray, foul_y):
    """Find bowler's lane edges at foul line and track upward."""
    h, w = gray.shape
    row = gaussian_filter1d(gray[foul_y - 20, :].astype(np.float64), sigma=30)
    grad = np.gradient(row)
    
    pos_peaks, _ = find_peaks(grad, height=0.3, distance=40)
    neg_peaks, _ = find_peaks(-grad, height=0.3, distance=40)
    
    lane_candidates = []
    for pp in pos_peaks:
        matching_neg = neg_peaks[neg_peaks > pp]
        if len(matching_neg) == 0:
            continue
        np_close = matching_neg[0]
        width = np_close - pp
        if 200 < width < 700:
            center = (pp + np_close) // 2
            lane_candidates.append((pp, np_close, width, center))
    
    lane_candidates.sort(key=lambda c: abs(c[3] - w // 2))
    if not lane_candidates:
        return None
    
    lane = lane_candidates[0]
    left_start, right_start = lane[0], lane[1]
    
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
        
        sl = max(0, prev_left - SEARCH_RADIUS)
        sr = min(w, prev_left + SEARCH_RADIUS)
        local_grad = grad[sl:sr]
        peaks, _ = find_peaks(local_grad, height=0.15, distance=10)
        new_left = (peaks[np.argmax(local_grad[peaks])] + sl) if len(peaks) > 0 else prev_left
        
        sl = max(0, prev_right - SEARCH_RADIUS)
        sr = min(w, prev_right + SEARCH_RADIUS)
        local_grad = -grad[sl:sr]
        peaks, _ = find_peaks(local_grad, height=0.15, distance=10)
        new_right = (peaks[np.argmax(local_grad[peaks])] + sl) if len(peaks) > 0 else prev_right
        
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
    
    lc, _ = robust_fit(y_arr, left_arr)
    rc, _ = robust_fit(y_arr, right_arr)
    return lc, rc


def board_x(board_num, y, lc, rc):
    """Get x position of a board number at a given y level.
    Board 0 = right edge (larger x), Board 40 = left edge (smaller x).
    """
    x_right = np.polyval(rc, y)  # board 0
    x_left = np.polyval(lc, y)   # board 40
    return x_right + (x_left - x_right) * board_num / 40.0


def draw_boards(frame, lc, rc, foul_y):
    """Draw board lines and dots on the approach area."""
    h, w = frame.shape[:2]
    out = frame.copy()
    
    # --- Lane outline ---
    slope_diff = rc[0] - lc[0]
    intercept_diff = rc[1] - lc[1]
    if slope_diff > 0:
        pin_y = max(50, int((30 - intercept_diff) / slope_diff))
    else:
        pin_y = 50
    
    lt_x = int(np.polyval(lc, pin_y))
    rt_x = int(np.polyval(rc, pin_y))
    lb_x = int(np.polyval(lc, foul_y))
    rb_x = int(np.polyval(rc, foul_y))
    
    lane_pts = np.array([[lt_x, pin_y], [rt_x, pin_y], [rb_x, foul_y], [lb_x, foul_y]], np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [lane_pts], (0, 180, 0))
    cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
    cv2.polylines(out, [lane_pts], True, (0, 255, 0), 3)
    
    # --- Foul line ---
    cv2.line(out, (lb_x, foul_y), (rb_x, foul_y), (0, 0, 255), 4)
    cv2.putText(out, 'FOUL LINE', (rb_x + 20, foul_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # --- Approach area ---
    al_x = np.polyval(lc, h - 1)
    ar_x = np.polyval(rc, h - 1)
    
    approach_pts = np.array([
        [lb_x, foul_y], [rb_x, foul_y],
        [int(ar_x), h - 1], [int(al_x), h - 1]
    ], np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [approach_pts], (200, 150, 0))
    cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)
    
    # --- Board lines on approach (boards 0-40) ---
    # Draw every board as a thin line, every 5th board thicker
    for board in range(0, 41):
        x_top = board_x(board, foul_y, lc, rc)
        x_bot = board_x(board, h - 1, lc, rc)
        
        if board % 10 == 0:
            # Every 10th board: thick white line
            color = (255, 255, 255)
            thickness = 2
        elif board % 5 == 0:
            # Every 5th board: medium line
            color = (200, 200, 200)
            thickness = 2
        else:
            # Every board: thin subtle line
            color = (120, 120, 120)
            thickness = 1
        
        cv2.line(out, (int(x_top), foul_y), (int(x_bot), h - 1), color, thickness)
    
    # --- Board number labels at bottom of approach ---
    label_y = h - 60
    for board in range(0, 41, 5):
        x = board_x(board, label_y, lc, rc)
        label = str(board)
        font_scale = 0.8 if board % 10 == 0 else 0.6
        thickness = 2 if board % 10 == 0 else 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.putText(out, label, (int(x - text_size[0] / 2), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # --- Board number labels near foul line ---
    label_y_top = foul_y + 50
    for board in range(0, 41, 5):
        x = board_x(board, label_y_top, lc, rc)
        label = str(board)
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.putText(out, label, (int(x - text_size[0] / 2), label_y_top),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)
    
    # --- Approach dots ---
    # Real bowling approach has rows of dots at ~12ft and ~15ft from foul line
    # In our frame, the approach goes from foul_y to h-1
    # Place dot rows at roughly 1/3 and 2/3 of the approach
    approach_height = h - 1 - foul_y
    dot_rows = [
        foul_y + int(approach_height * 0.20),  # first row of dots (~12ft)
        foul_y + int(approach_height * 0.45),  # second row of dots (~15ft)
    ]
    
    dot_boards = [10, 15, 20, 25, 30]
    
    for dot_y in dot_rows:
        for board in dot_boards:
            x = board_x(board, dot_y, lc, rc)
            if board == 20:
                # Big dot for board 20
                cv2.circle(out, (int(x), dot_y), 16, (0, 255, 255), -1)
                cv2.circle(out, (int(x), dot_y), 16, (0, 180, 180), 2)
            else:
                # Regular dot
                cv2.circle(out, (int(x), dot_y), 10, (0, 255, 255), -1)
                cv2.circle(out, (int(x), dot_y), 10, (0, 180, 180), 2)
    
    # --- Label the dot rows ---
    for dot_y in dot_rows:
        x_label = board_x(32, dot_y, lc, rc)
        cv2.putText(out, 'DOTS', (int(x_label) - 60, dot_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    return out


# === Main ===
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
        print("  Failed!")
        continue
    
    lc, rc = result
    
    print(f"  Left edge:  x = {lc[0]:.4f}*y + {lc[1]:.1f}")
    print(f"  Right edge: x = {rc[0]:.4f}*y + {rc[1]:.1f}")
    
    # Print board positions at foul line
    print(f"\n  Board positions at foul line (y={foul_y}):")
    for b in range(0, 41, 5):
        x = board_x(b, foul_y, lc, rc)
        print(f"    Board {b:2d}: x={x:.0f}")
    
    lane_w = np.polyval(rc, foul_y) - np.polyval(lc, foul_y)
    board_px = lane_w / 40
    print(f"\n  Lane width at foul: {lane_w:.0f}px")
    print(f"  Pixels per board: {board_px:.1f}")
    
    # Print board positions at bottom of approach
    print(f"\n  Board positions at bottom (y={h-1}):")
    for b in range(0, 41, 5):
        x = board_x(b, h - 1, lc, rc)
        print(f"    Board {b:2d}: x={x:.0f}")
    
    lane_w_bot = np.polyval(rc, h-1) - np.polyval(lc, h-1)
    board_px_bot = lane_w_bot / 40
    print(f"  Lane width at bottom: {lane_w_bot:.0f}px")
    print(f"  Pixels per board: {board_px_bot:.1f}")
    
    # Generate overlay
    out = draw_boards(frame, lc, rc, foul_y)
    
    tag = vname.replace('.MP4', '')
    out_path = f'board_overlay_{tag}.png'
    cv2.imwrite(out_path, out)
    print(f"\n  Saved {out_path}")

print("\nDone!")
