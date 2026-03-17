"""
Save lane calibration data and track the bowler's walk-up on the approach.
Detects which board the bowler starts on, ends on, and if they drift.
"""

import cv2
import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

VIDEO_DIR = '../video/behind/'

# ================================================================
# Step 1: Lane edge detection (reuse proven method)
# ================================================================
def find_lane_edges(gray, foul_y):
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
        return c
    
    lc = robust_fit(y_arr, left_arr)
    rc = robust_fit(y_arr, right_arr)
    return lc, rc


def board_x(board_num, y, lc, rc):
    """Board 0 = right edge, Board 40 = left edge."""
    x_right = np.polyval(rc, y)
    x_left = np.polyval(lc, y)
    return x_right + (x_left - x_right) * board_num / 40.0


def x_to_board(x, y, lc, rc):
    """Convert pixel x at given y to board number."""
    x_right = np.polyval(rc, y)
    x_left = np.polyval(lc, y)
    if abs(x_left - x_right) < 1:
        return 20.0
    return (x - x_right) / (x_left - x_right) * 40.0


# ================================================================
# Step 2: Calibrate on video 1
# ================================================================
VNAME = '1.MP4'
FOUL_Y = 1350

print("=== Calibrating from 1.MP4 ===")
cap = cv2.VideoCapture(VIDEO_DIR + VNAME)
cap.set(cv2.CAP_PROP_POS_FRAMES, 770)
ret, empty_frame = cap.read()
cap.release()

gray_empty = cv2.cvtColor(empty_frame, cv2.COLOR_BGR2GRAY)
h, w = gray_empty.shape

lc, rc = find_lane_edges(gray_empty, FOUL_Y)
print(f"  Left edge:  x = {lc[0]:.4f}*y + {lc[1]:.1f}")
print(f"  Right edge: x = {rc[0]:.4f}*y + {rc[1]:.1f}")
print(f"  Foul line: y={FOUL_Y}")
print(f"  Frame: {w}x{h}")

# Save calibration
calibration = {
    'video': VNAME,
    'frame_width': w,
    'frame_height': h,
    'foul_line_y': FOUL_Y,
    'left_edge_slope': float(lc[0]),
    'left_edge_intercept': float(lc[1]),
    'right_edge_slope': float(rc[0]),
    'right_edge_intercept': float(rc[1]),
    'empty_frame_idx': 770,
    'notes': 'Board 0=right edge, Board 40=left edge. x = slope*y + intercept'
}

with open('lane_calibration.json', 'w') as f:
    json.dump(calibration, f, indent=2)
print("\nSaved lane_calibration.json")

# ================================================================
# Step 3: Track bowler using background subtraction
# ================================================================
print("\n=== Tracking bowler in 1.MP4 ===")

cap = cv2.VideoCapture(VIDEO_DIR + VNAME)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"  Total frames: {total_frames}, FPS: {fps:.1f}")

# Read empty frame for background subtraction
cap.set(cv2.CAP_PROP_POS_FRAMES, 770)
ret, bg_frame = cap.read()
bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Process all frames to find the bowler
# The bowler is on the approach (y > foul_y) 
# Detect them via frame differencing from the empty frame

bowler_data = []  # list of (frame_idx, foot_x, foot_y, board_num, bbox)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    # Frame difference in approach area only
    diff = np.abs(gray - bg_gray)
    
    # Only look at approach area (below foul line, within lane width + some margin)
    approach_mask = np.zeros_like(diff, dtype=np.uint8)
    for y in range(FOUL_Y, h):
        x_left = int(np.polyval(lc, y)) - 200  # extra margin
        x_right = int(np.polyval(rc, y)) + 200
        x_left = max(0, x_left)
        x_right = min(w, x_right)
        approach_mask[y, x_left:x_right] = 255
    
    # Threshold the difference
    diff_masked = diff * (approach_mask / 255.0)
    binary = (diff_masked > 30).astype(np.uint8) * 255
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        continue
    
    # Find the largest contour (the bowler)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < 5000:  # too small, probably noise
        continue
    
    # Get bounding box
    bx, by, bw, bh = cv2.boundingRect(largest)
    
    # Foot position: bottom-center of bounding box
    foot_x = bx + bw // 2
    foot_y = by + bh  # bottom of bbox
    
    # Only care about bowler on the approach
    if foot_y < FOUL_Y + 50:
        continue
    
    # Convert to board number
    board = x_to_board(foot_x, foot_y, lc, rc)
    
    if 0 <= board <= 40:
        bowler_data.append({
            'frame': frame_idx,
            'time': frame_idx / fps,
            'foot_x': int(foot_x),
            'foot_y': int(foot_y),
            'board': round(board, 1),
            'bbox': [int(bx), int(by), int(bw), int(bh)],
            'area': int(area),
        })

cap.release()

print(f"\n  Detected bowler in {len(bowler_data)} frames")

if len(bowler_data) == 0:
    print("  No bowler detected! May need different approach.")
    exit()

# ================================================================
# Step 4: Analyze the walk-up
# ================================================================
print("\n=== Walk-up Analysis ===")

# Group into bowling shots (continuous sequences of detection)
shots = []
current_shot = [bowler_data[0]]

for i in range(1, len(bowler_data)):
    # If gap > 10 frames, it's a new shot
    if bowler_data[i]['frame'] - bowler_data[i-1]['frame'] > 10:
        if len(current_shot) >= 10:  # minimum frames for a real approach
            shots.append(current_shot)
        current_shot = [bowler_data[i]]
    else:
        current_shot.append(bowler_data[i])

if len(current_shot) >= 10:
    shots.append(current_shot)

print(f"  Found {len(shots)} bowling approaches")

for si, shot in enumerate(shots):
    frames = [d['frame'] for d in shot]
    boards = [d['board'] for d in shot]
    foot_ys = [d['foot_y'] for d in shot]
    times = [d['time'] for d in shot]
    
    # The bowler starts far from foul line (high y) and walks toward it (lower y)
    # Find start: frame with highest foot_y (furthest from foul line)
    # Find end: frame with lowest foot_y (closest to foul line) 
    start_idx = np.argmax(foot_ys)
    end_idx = np.argmin(foot_ys)
    
    # If start_idx > end_idx, bowler is walking toward camera (away from foul) - wrong
    # The bowler walks toward the foul line, so foot_y should decrease over time
    # Actually, in our frame: higher y = closer to camera = further from foul line
    # So the bowler starts at high y and ends at low y (near foul line)
    
    start_board = boards[start_idx]
    end_board = boards[end_idx]
    drift = end_board - start_board
    
    print(f"\n  Shot {si+1}:")
    print(f"    Frames: {frames[0]}-{frames[-1]} ({len(shot)} frames, {times[-1]-times[0]:.1f}s)")
    print(f"    Start: board {start_board:.1f} at y={foot_ys[start_idx]} (t={times[start_idx]:.2f}s)")
    print(f"    End:   board {end_board:.1f} at y={foot_ys[end_idx]} (t={times[end_idx]:.2f}s)")
    print(f"    Drift: {drift:+.1f} boards", end="")
    if abs(drift) < 1:
        print(" (straight)")
    elif drift > 0:
        print(" (drifts LEFT)")  # higher board number = left
    else:
        print(" (drifts RIGHT)")  # lower board number = right
    
    # Board-by-board detail (sampled every ~0.5s)
    print(f"    Walk-up detail:")
    sample_interval = max(1, int(0.5 * fps))
    for j in range(0, len(shot), sample_interval):
        d = shot[j]
        print(f"      t={d['time']:.2f}s  frame={d['frame']:3d}  board={d['board']:5.1f}  foot=({d['foot_x']},{d['foot_y']})")
    # Always show last
    d = shot[-1]
    print(f"      t={d['time']:.2f}s  frame={d['frame']:3d}  board={d['board']:5.1f}  foot=({d['foot_x']},{d['foot_y']})")

# ================================================================
# Step 5: Save tracking data
# ================================================================
tracking_output = {
    'video': VNAME,
    'calibration': calibration,
    'shots': []
}

for si, shot in enumerate(shots):
    foot_ys = [d['foot_y'] for d in shot]
    boards = [d['board'] for d in shot]
    start_idx = np.argmax(foot_ys)
    end_idx = np.argmin(foot_ys)
    
    tracking_output['shots'].append({
        'shot_number': si + 1,
        'start_frame': shot[0]['frame'],
        'end_frame': shot[-1]['frame'],
        'start_board': boards[start_idx],
        'end_board': boards[end_idx],
        'drift': round(boards[end_idx] - boards[start_idx], 1),
        'frames': shot,
    })

with open('bowler_tracking.json', 'w') as f:
    json.dump(tracking_output, f, indent=2)
print("\n\nSaved bowler_tracking.json")

# ================================================================
# Step 6: Visualize one shot on the approach
# ================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if shots:
    shot = shots[0]  # visualize first shot
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Board vs time
    ax = axes[0]
    times = [d['time'] for d in shot]
    boards = [d['board'] for d in shot]
    ax.plot(times, boards, 'b-o', markersize=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Board Number')
    ax.set_title(f'Shot 1: Board Position Over Time')
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Board 20 (center)')
    ax.legend()
    ax.set_ylim(0, 40)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Foot path on approach (bird's eye)
    ax = axes[1]
    foot_xs = [d['foot_x'] for d in shot]
    foot_ys = [d['foot_y'] for d in shot]
    
    # Draw approach boundaries
    for b in [0, 10, 20, 30, 40]:
        xt = board_x(b, FOUL_Y, lc, rc)
        xb = board_x(b, h - 1, lc, rc)
        color = 'red' if b == 20 else 'gray'
        lw = 2 if b == 20 else 0.5
        ax.plot([xt, xb], [FOUL_Y, h-1], color=color, linewidth=lw, alpha=0.5)
    
    # Foul line
    ax.axhline(y=FOUL_Y, color='red', linewidth=2)
    
    # Foot path
    scatter = ax.scatter(foot_xs, foot_ys, c=range(len(foot_xs)), cmap='cool', s=20, zorder=5)
    ax.plot(foot_xs, foot_ys, 'k-', linewidth=0.5, alpha=0.3)
    
    # Mark start and end
    ax.scatter([foot_xs[0]], [foot_ys[0]], c='green', s=100, marker='^', zorder=10, label='Start')
    end_i = np.argmin(foot_ys)
    ax.scatter([foot_xs[end_i]], [foot_ys[end_i]], c='red', s=100, marker='v', zorder=10, label='End')
    
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_title('Foot Path on Approach')
    ax.legend()
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('debug_bowler_track.png', dpi=120)
    print("Saved debug_bowler_track.png")
