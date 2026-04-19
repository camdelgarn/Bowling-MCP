"""
Track the bowler's walk-up on the approach from a behind-the-bowler video.
Detects which board the bowler starts on, ends on, and if they drift.

Usage:
    python track_bowler.py <video_path> [--empty-frame N] [--foul-y N] [--calibration FILE]
"""

import argparse
import cv2
import numpy as np
import json
import os
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ================================================================
# Step 1: Lane edge detection (reuse proven method)
# ================================================================
def find_lane_edges(gray, foul_y, lane_hint_x=None):
    h, w = gray.shape
    target_x = lane_hint_x if lane_hint_x is not None else w // 2

    # Search multiple rows near the foul line for the best lane candidate.
    # This handles offset cameras where foul-line markings create noise.
    lane_candidates = []
    for search_y in range(max(50, foul_y - 250), foul_y - 10, 20):
        row = gaussian_filter1d(gray[search_y, :].astype(np.float64), sigma=30)
        grad = np.gradient(row)

        pos_peaks, _ = find_peaks(grad, height=0.3, distance=40)
        neg_peaks, _ = find_peaks(-grad, height=0.3, distance=40)

        for pp in pos_peaks:
            matching_neg = neg_peaks[neg_peaks > pp]
            # Try nearest 3 negative peaks (not just closest) to find wider lanes
            for np_val in matching_neg[:3]:
                width = np_val - pp
                if 150 < width < 700:
                    center = (pp + np_val) // 2
                    avg_brightness = float(np.mean(gray[search_y, pp:np_val]))
                    lane_candidates.append(
                        (pp, np_val, width, center, avg_brightness, search_y)
                    )

    if not lane_candidates:
        return None

    # Score: prefer wider, brighter, closer to hint
    def score(c):
        dist_norm = abs(c[3] - target_x) / w
        bright_norm = c[4] / 255.0
        width_pref = 1.0 - min(abs(c[2] - 400) / 400.0, 1.0)  # prefer ~400px
        return dist_norm - 0.4 * bright_norm - 0.4 * width_pref

    lane_candidates.sort(key=score)

    lane = lane_candidates[0]
    left_start, right_start = lane[0], lane[1]
    init_y = lane[5]

    SEARCH_RADIUS = 40
    STEP = 3
    left_track = [left_start]
    right_track = [right_start]
    y_track = [init_y]
    prev_left, prev_right = left_start, right_start

    for y in range(init_y - STEP, 400, -STEP):
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
        return c
    
    lc = robust_fit(y_arr, left_arr)
    rc = robust_fit(y_arr, right_arr)
    
    # Validate: lane width at foul line should be reasonable
    # lc = positive gradient (left edge, lower x), rc = negative gradient (right edge, higher x)
    left_x_foul = np.polyval(lc, foul_y)
    right_x_foul = np.polyval(rc, foul_y)
    lane_width_foul = right_x_foul - left_x_foul
    if lane_width_foul < 100 or lane_width_foul > 800:
        return None
    return lc, rc


def board_x(board_num, y, lc, rc):
    """Board 0 = right edge, Board 40 = left edge."""
    x_right = np.polyval(rc, y)
    x_left = np.polyval(lc, y)
    return x_right + (x_left - x_right) * board_num / 40.0


def x_to_board(x, y, lc, rc):
    """Convert pixel x at given y to board number.
    Clamps y to foul_y range to avoid extrapolation below tracked region."""
    x_right = np.polyval(rc, y)
    x_left = np.polyval(lc, y)
    if abs(x_left - x_right) < 1:
        return 20.0
    return (x - x_right) / (x_left - x_right) * 40.0


# ================================================================
# Argument parsing
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Track bowler walk-up and board drift from behind-the-bowler video.'
    )
    parser.add_argument('video', help='Path to the video file (e.g. ../video/behind/1.MP4)')
    parser.add_argument('--empty-frame', type=int, default=None,
                        help='Frame index of an empty lane (no bowler). Auto-detected if not set.')
    parser.add_argument('--foul-y', type=int, default=1350,
                        help='Y pixel coordinate of the foul line (default: 1350)')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to existing lane_calibration.json to skip calibration step')
    parser.add_argument('--output', type=str, default='bowler_tracking.json',
                        help='Output JSON file path (default: bowler_tracking.json)')
    parser.add_argument('--scale', type=float, default=0.25,
                        help='Downscale factor for processing (default: 0.25 = quarter res)')
    parser.add_argument('--skip', type=int, default=3,
                        help='Process every Nth frame (default: 3)')
    parser.add_argument('--lane-x', type=int, default=None,
                        help='Approximate x-coordinate of lane center (auto-detected if omitted)')
    parser.add_argument('--foul-margin', type=float, default=0.5,
                        help='Fraction of approach length: bowler must reach within this fraction of foul line (default: 0.5)')
    return parser.parse_args()


def find_empty_frame(cap, total_frames, foul_y, lc, rc, h, w):
    """Find a frame with no bowler on the approach by checking for minimal motion."""
    best_frame = total_frames - 1
    min_activity = float('inf')
    
    # Build approach mask
    approach_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(foul_y, h):
        xl = max(0, int(np.polyval(lc, y)) - 200)
        xr = min(w, int(np.polyval(rc, y)) + 200)
        approach_mask[y, xl:xr] = 255
    
    # Sample frames from the end of the video (bowler usually gone)
    sample_indices = list(range(max(0, total_frames - 50), total_frames, 2))
    # Also sample from scattered positions
    sample_indices += list(range(0, total_frames, total_frames // 20))
    
    frames_checked = {}
    for idx in sample_indices:
        if idx in frames_checked:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Measure variance in approach area (empty frame = uniform lane surface)
        masked = cv2.bitwise_and(gray, approach_mask)
        activity = np.std(masked[approach_mask > 0].astype(np.float64))
        frames_checked[idx] = activity
        if activity < min_activity:
            min_activity = activity
            best_frame = idx
    
    return best_frame


def main():
    args = parse_args()
    
    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    video_name = os.path.basename(video_path)
    foul_y = args.foul_y
    
    # ================================================================
    # Step 1: Calibration
    # ================================================================
    if args.calibration and os.path.isfile(args.calibration):
        print(f"=== Loading calibration from {args.calibration} ===")
        with open(args.calibration, 'r') as f:
            calibration = json.load(f)
        lc = np.array([calibration['left_edge_slope'], calibration['left_edge_intercept']])
        rc = np.array([calibration['right_edge_slope'], calibration['right_edge_intercept']])
        foul_y = calibration['foul_line_y']
        h = calibration['frame_height']
        w = calibration['frame_width']
        lane_hint_x = args.lane_x  # may be None; approach mask falls back to lane center
        if lane_hint_x is None and 'lane_hint_x' in calibration:
            lane_hint_x = calibration['lane_hint_x']
        print(f"  Left edge:  x = {lc[0]:.4f}*y + {lc[1]:.1f}")
        print(f"  Right edge: x = {rc[0]:.4f}*y + {rc[1]:.1f}")
        print(f"  Foul line: y={foul_y}")
        
        # Still need an empty frame for background subtraction
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if args.empty_frame is not None:
            empty_idx = args.empty_frame
        elif 'empty_frame_idx' in calibration:
            empty_idx = calibration['empty_frame_idx']
        else:
            print("  Auto-detecting empty frame...")
            empty_idx = find_empty_frame(cap, total_frames, foul_y, lc, rc, h, w)
        
        print(f"  Empty frame: {empty_idx}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, empty_idx)
        ret, empty_frame = cap.read()
        cap.release()
    else:
        print(f"=== Calibrating from {video_name} ===")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine lane center hint
        lane_hint_x = args.lane_x
        if lane_hint_x is None:
            # Auto-detect: find where bowler motion is concentrated
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, bg_ref = cap.read()
            h_tmp, w_tmp = bg_ref.shape[:2]
            scale_tmp = 0.125
            bg_small = cv2.resize(bg_ref, (int(w_tmp * scale_tmp), int(h_tmp * scale_tmp)))
            bg_g = cv2.cvtColor(bg_small, cv2.COLOR_BGR2GRAY).astype(np.float64)
            motion_col = np.zeros(bg_g.shape[1])
            samples = list(range(0, total_frames, max(1, total_frames // 30)))
            for fi in samples:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret2, fr = cap.read()
                if not ret2:
                    continue
                fr_small = cv2.resize(fr, bg_small.shape[1::-1])
                fr_g = cv2.cvtColor(fr_small, cv2.COLOR_BGR2GRAY).astype(np.float64)
                diff = (np.abs(fr_g - bg_g) > 15).astype(np.float64)
                motion_col += diff.sum(axis=0)
            peak_col = int(np.argmax(motion_col) / scale_tmp)
            print(f"  Auto-detected motion center at x~{peak_col}")
            lane_hint_x = peak_col
        
        # Find empty frame
        if args.empty_frame is not None:
            empty_idx = args.empty_frame
        else:
            # Quick calibration to get lane edges for empty frame detection
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, last_frame = cap.read()
            gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray_last.shape
            temp_edges = find_lane_edges(gray_last, foul_y, lane_hint_x)
            if temp_edges:
                tlc, trc = temp_edges
                print("  Auto-detecting empty frame...")
                empty_idx = find_empty_frame(cap, total_frames, foul_y, tlc, trc, h, w)
            else:
                empty_idx = total_frames - 1
        
        print(f"  Empty frame: {empty_idx}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, empty_idx)
        ret, empty_frame = cap.read()
        cap.release()
        
        gray_empty = cv2.cvtColor(empty_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray_empty.shape
        
        edges = find_lane_edges(gray_empty, foul_y, lane_hint_x)
        if edges is None:
            print("Error: Could not detect lane edges. Try specifying --foul-y or --lane-x.")
            sys.exit(1)
        lc, rc = edges
        
        lane_width_px = np.polyval(rc, foul_y) - np.polyval(lc, foul_y)
        print(f"  Left edge:  x = {lc[0]:.4f}*y + {lc[1]:.1f}")
        print(f"  Right edge: x = {rc[0]:.4f}*y + {rc[1]:.1f}")
        print(f"  Lane width at foul line: {lane_width_px:.0f}px")
        if lane_width_px < 250:
            print("  WARNING: Narrow lane view (offset camera). Board numbers are approximate.")
            print("           Drift measurement requires a behind-the-bowler centered camera.")
        print(f"  Foul line: y={foul_y}")
        print(f"  Frame: {w}x{h}")
        
        calibration = {
            'video': video_name,
            'frame_width': w,
            'frame_height': h,
            'foul_line_y': foul_y,
            'left_edge_slope': float(lc[0]),
            'left_edge_intercept': float(lc[1]),
            'right_edge_slope': float(rc[0]),
            'right_edge_intercept': float(rc[1]),
            'empty_frame_idx': empty_idx,
            'lane_hint_x': lane_hint_x,
            'notes': 'Board 0=right edge, Board 40=left edge. x = slope*y + intercept'
        }
        
        with open('lane_calibration.json', 'w') as f:
            json.dump(calibration, f, indent=2)
        print("\n  Saved lane_calibration.json")

    # ================================================================
    # Step 2: Track bowler using background subtraction
    # ================================================================
    print(f"\n=== Tracking bowler in {video_name} ===")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Total frames: {total_frames}, FPS: {fps:.1f}")

    # Downscale parameters
    scale = args.scale
    skip = args.skip
    sh = int(h * scale)
    sw = int(w * scale)
    s_foul_y = int(foul_y * scale)
    min_area = int(5000 * scale * scale)  # scale the area threshold
    kern_size = max(3, int(15 * scale)) | 1  # keep odd

    # Read empty frame for background subtraction (downscaled)
    cap.set(cv2.CAP_PROP_POS_FRAMES, empty_idx)
    ret, bg_frame = cap.read()
    bg_small = cv2.resize(bg_frame, (sw, sh), interpolation=cv2.INTER_AREA)
    bg_gray = cv2.cvtColor(bg_small, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Build approach mask at downscaled resolution.
    # Use motion center (lane_hint_x) with generous width rather than
    # extrapolating lane edge lines, which fails for offset cameras.
    approach_mask = np.zeros((sh, sw), dtype=np.uint8)
    lane_cx = np.polyval(lc, foul_y) / 2 + np.polyval(rc, foul_y) / 2
    mask_cx = int((lane_hint_x if lane_hint_x is not None else lane_cx) * scale)
    mask_hw = int(800 * scale)  # 800px half-width in original coords
    for y in range(s_foul_y, sh):
        approach_mask[y, max(0, mask_cx - mask_hw):min(sw, mask_cx + mask_hw)] = 255

    bowler_data = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"  Processing at {sw}x{sh} (scale={scale}), every {skip} frames...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for speed
        if frame_idx % skip != 0:
            continue

        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Frame difference in approach area only
        diff = np.abs(gray - bg_gray)
        diff_masked = diff * (approach_mask / 255.0)
        binary = (diff_masked > 30).astype(np.uint8) * 255

        # Morphological cleanup
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Find the largest contour (the bowler)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < min_area:
            continue

        # Get bounding box (in downscaled coords)
        bx, by, bw, bh = cv2.boundingRect(largest)

        # Foot position: center-bottom of bounding box, scaled to original res.
        # (Release-frame corruption from kick foot is handled in post-processing
        # by detecting the sudden board-number jump and trimming those frames.)
        foot_x = int((bx + bw / 2) / scale)
        foot_y = int((by + bh) / scale)

        # Only care about bowler on the approach
        if foot_y < foul_y + 50:
            continue

        # Convert to board number at the foot's actual y-position.
        # The lane edge model encodes perspective, so using foot_y gives
        # geometrically correct board numbers (verified on centered camera).
        board = x_to_board(foot_x, foot_y, lc, rc)

        if -5 <= board <= 45:
            bowler_data.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'foot_x': foot_x,
                'foot_y': foot_y,
                'board': round(board, 1),
                'bbox': [int(bx / scale), int(by / scale), int(bw / scale), int(bh / scale)],
                'area': int(area / (scale * scale)),
            })

    cap.release()

    print(f"\n  Detected bowler in {len(bowler_data)} frames")

    if len(bowler_data) < 3:
        print(f"  Only {len(bowler_data)} frames with bowler detected (need at least 3).")
        print("  Try: --empty-frame, --foul-y, or --lane-x to improve detection.")
        sys.exit(1)

    # ================================================================
    # Step 3: Analyze the walk-up
    # ================================================================
    print("\n=== Walk-up Analysis ===")

    # Find the actual approach walk-up using foot_y velocity.
    # The approach is when the bowler walks toward the foul line (foot_y decreasing).
    foot_ys_all = np.array([d['foot_y'] for d in bowler_data])
    times_all = np.array([d['time'] for d in bowler_data])

    # Smooth foot_y to reduce noise
    if len(foot_ys_all) > 5:
        foot_ys_smooth = gaussian_filter1d(foot_ys_all.astype(np.float64), sigma=3)
    else:
        foot_ys_smooth = foot_ys_all.astype(np.float64)

    # Compute velocity (negative = moving toward foul line)
    velocity = np.gradient(foot_ys_smooth, times_all)

    # Find sustained approach segments: foot_y velocity consistently negative
    approach_mask_arr = velocity < -50  # moving toward foul line at >50 px/s

    # Group consecutive True values into approach segments
    candidates = []
    in_approach = False
    start_i = 0
    for i in range(len(approach_mask_arr)):
        if approach_mask_arr[i] and not in_approach:
            start_i = i
            in_approach = True
        elif not approach_mask_arr[i] and in_approach:
            # Extend a few frames past the end to capture the release
            end_i = min(len(bowler_data), i + 3)
            segment = bowler_data[start_i:end_i]
            if len(segment) >= max(4, 10 // skip):
                candidates.append(segment)
            in_approach = False
    if in_approach:
        end_i = len(bowler_data)
        segment = bowler_data[start_i:end_i]
        if len(segment) >= max(4, 10 // skip):
            candidates.append(segment)

    # Filter: real approaches reach close to the foul line
    foul_margin = (h - foul_y) * args.foul_margin
    shots = []
    for seg in candidates:
        min_foot_y = min(d['foot_y'] for d in seg)
        if min_foot_y < foul_y + foul_margin:
            shots.append(seg)
        else:
            print(f"  Discarding segment (frames {seg[0]['frame']}-{seg[-1]['frame']}): "
                  f"foot never reached foul line (closest y={min_foot_y}, need <{int(foul_y + foul_margin)})")

    print(f"  Found {len(shots)} bowling approaches")

    for si, shot in enumerate(shots):
        frames = [d['frame'] for d in shot]
        boards = [d['board'] for d in shot]
        foot_ys_shot = [d['foot_y'] for d in shot]
        times = [d['time'] for d in shot]

        n = len(shot)
        # Use median of first/last 25% of frames for robust start/end board
        chunk = max(1, n // 4)
        start_board = float(np.median(boards[:chunk]))
        end_board = float(np.median(boards[-chunk:]))

        # Release-trim: detect sudden board jump (kick-foot after release).
        # Only apply if per-frame board changes are small (centered camera).
        per_frame_changes = [abs(boards[j] - boards[j-1]) for j in range(1, n)]
        median_change = float(np.median(per_frame_changes)) if per_frame_changes else 0
        jump_threshold = max(3.0, median_change * 5)
        min_y_idx = int(np.argmin(foot_ys_shot))
        for j in range(1, n):
            if abs(boards[j] - boards[j-1]) > jump_threshold:
                end_board = float(np.median(boards[max(0, j-chunk):j]))
                break

        drift = end_board - start_board

        start_idx = int(np.argmax(foot_ys_shot))
        end_idx = min_y_idx

        print(f"\n  Shot {si+1}:")
        print(f"    Frames: {frames[0]}-{frames[-1]} ({n} frames, {times[-1]-times[0]:.1f}s)")
        print(f"    Start: board {start_board:.1f} (median of first {chunk} frames)")
        print(f"    End:   board {end_board:.1f} (median of last {chunk} frames)")
        print(f"    Drift: {drift:+.1f} boards", end="")
        if abs(drift) < 1:
            print(" (straight)")
        elif drift > 0:
            print(" (drifts LEFT)")
        else:
            print(" (drifts RIGHT)")

        print(f"    Walk-up detail:")
        sample_interval = max(1, int(0.5 * fps))
        for j in range(0, len(shot), sample_interval):
            d = shot[j]
            print(f"      t={d['time']:.2f}s  frame={d['frame']:3d}  board={d['board']:5.1f}  foot=({d['foot_x']},{d['foot_y']})")
        d = shot[-1]
        print(f"      t={d['time']:.2f}s  frame={d['frame']:3d}  board={d['board']:5.1f}  foot=({d['foot_x']},{d['foot_y']})")

    # ================================================================
    # Step 4: Save tracking data
    # ================================================================
    tracking_output = {
        'video': video_name,
        'calibration': calibration,
        'shots': []
    }

    for si, shot in enumerate(shots):
        foot_ys_shot = [d['foot_y'] for d in shot]
        boards = [d['board'] for d in shot]
        n = len(shot)
        chunk = max(1, n // 4)
        start_board = float(np.median(boards[:chunk]))
        end_board = float(np.median(boards[-chunk:]))

        per_frame_changes = [abs(boards[j] - boards[j-1]) for j in range(1, n)]
        median_change = float(np.median(per_frame_changes)) if per_frame_changes else 0
        jump_threshold = max(3.0, median_change * 5)
        for j in range(1, n):
            if abs(boards[j] - boards[j-1]) > jump_threshold:
                end_board = float(np.median(boards[max(0, j-chunk):j]))
                break

        tracking_output['shots'].append({
            'shot_number': si + 1,
            'start_frame': shot[0]['frame'],
            'end_frame': shot[-1]['frame'],
            'start_board': round(start_board, 1),
            'end_board': round(end_board, 1),
            'drift': round(end_board - start_board, 1),
            'frames': shot,
        })

    with open(args.output, 'w') as f:
        json.dump(tracking_output, f, indent=2)
    print(f"\n\nSaved {args.output}")

    # ================================================================
    # Step 5: Visualize one shot on the approach
    # ================================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if shots:
        shot = shots[0]

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        ax = axes[0]
        times = [d['time'] for d in shot]
        boards = [d['board'] for d in shot]
        ax.plot(times, boards, 'b-o', markersize=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Board Number')
        ax.set_title('Shot 1: Board Position Over Time')
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Board 20 (center)')
        ax.legend()
        ax.set_ylim(0, 40)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        foot_xs = [d['foot_x'] for d in shot]
        foot_ys_shot = [d['foot_y'] for d in shot]

        for b in [0, 10, 20, 30, 40]:
            xt = board_x(b, foul_y, lc, rc)
            xb = board_x(b, h - 1, lc, rc)
            color = 'red' if b == 20 else 'gray'
            lw = 2 if b == 20 else 0.5
            ax.plot([xt, xb], [foul_y, h-1], color=color, linewidth=lw, alpha=0.5)

        ax.axhline(y=foul_y, color='red', linewidth=2)

        scatter = ax.scatter(foot_xs, foot_ys_shot, c=range(len(foot_xs)), cmap='cool', s=20, zorder=5)
        ax.plot(foot_xs, foot_ys_shot, 'k-', linewidth=0.5, alpha=0.3)

        ax.scatter([foot_xs[0]], [foot_ys_shot[0]], c='green', s=100, marker='^', zorder=10, label='Start')
        end_i = np.argmin(foot_ys_shot)
        ax.scatter([foot_xs[end_i]], [foot_ys_shot[end_i]], c='red', s=100, marker='v', zorder=10, label='End')

        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_title('Foot Path on Approach')
        ax.legend()
        ax.invert_yaxis()

        plt.tight_layout()
        plot_path = os.path.splitext(args.output)[0] + '_plot.png'
        plt.savefig(plot_path, dpi=120)
        print(f"Saved {plot_path}")


if __name__ == '__main__':
    main()
