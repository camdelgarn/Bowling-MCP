"""
Analyze the bowler's approach walk-up more carefully.
Isolate the actual approach (walking toward foul line) from standing/walking back.
"""

import cv2
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

VIDEO_DIR = '../video/behind/'

# Load calibration
with open('lane_calibration.json', 'r') as f:
    cal = json.load(f)

lc = np.array([cal['left_edge_slope'], cal['left_edge_intercept']])
rc = np.array([cal['right_edge_slope'], cal['right_edge_intercept']])
FOUL_Y = cal['foul_line_y']
w_frame = cal['frame_width']
h_frame = cal['frame_height']

def x_to_board(x, y):
    x_right = np.polyval(rc, y)
    x_left = np.polyval(lc, y)
    if abs(x_left - x_right) < 1:
        return 20.0
    return (x - x_right) / (x_left - x_right) * 40.0

def board_x(board, y):
    x_right = np.polyval(rc, y)
    x_left = np.polyval(lc, y)
    return x_right + (x_left - x_right) * board / 40.0

# ================================================================
# Track bowler with better foot detection
# ================================================================
VNAME = '1.MP4'
cap = cv2.VideoCapture(VIDEO_DIR + VNAME)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Get empty frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 770)
ret, bg_frame = cap.read()
bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Build approach mask once
approach_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
for y in range(FOUL_Y, h_frame):
    x_left = max(0, int(np.polyval(lc, y)) - 300)
    x_right = min(w_frame, int(np.polyval(rc, y)) + 300)
    approach_mask[y, x_left:x_right] = 255

print("=== Processing all frames ===")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

all_detections = []
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    diff = np.abs(gray - bg_gray)
    diff_masked = diff * (approach_mask / 255.0)
    binary = (diff_masked > 30).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        all_detections.append(None)
        continue
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < 5000:
        all_detections.append(None)
        continue
    
    bx, by, bw, bh = cv2.boundingRect(largest)
    
    # Foot position: bottom-center of the detected region
    foot_x = bx + bw // 2
    foot_y = by + bh
    
    # Only on approach
    if foot_y < FOUL_Y + 20:
        all_detections.append(None)
        continue
    
    board = x_to_board(foot_x, foot_y)
    
    all_detections.append({
        'frame': frame_idx,
        'time': frame_idx / fps,
        'foot_x': int(foot_x),
        'foot_y': int(foot_y),
        'board': round(board, 1),
        'area': int(area),
        'bbox_y_top': int(by),
    })

cap.release()

# Filter to valid detections
valid = [d for d in all_detections if d is not None]
print(f"  Valid detections: {len(valid)} / {total_frames}")

# ================================================================
# Identify the actual approach
# ================================================================
# The approach is characterized by:
# 1. Bowler's top (bbox_y_top) moves upward (toward foul line, smaller y)
# 2. foot_y stays relatively stable or decreases
# 3. It's a continuous motion lasting 2-5 seconds

# Look at bbox_y_top (the bowler's head position) to find 
# when they're walking toward the foul line (y_top decreasing)
print("\n=== Identifying approach sequence ===")

# Plot foot_y over time to understand the motion
frames = [d['frame'] for d in valid]
foot_ys = [d['foot_y'] for d in valid]
boards = [d['board'] for d in valid]
bbox_tops = [d['bbox_y_top'] for d in valid]
times = [d['time'] for d in valid]

# The approach: bowler walks from high y (near camera) to low y (near foul line)
# Their bbox_y_top should decrease steadily during the approach

# Smooth the bbox_y_top to find the approach phase
if len(bbox_tops) > 10:
    smooth_tops = gaussian_filter1d(np.array(bbox_tops, dtype=np.float64), sigma=5)
    
    # Find where bbox_top is decreasing (approaching foul line)
    velocity = np.gradient(smooth_tops)  # negative = moving toward foul line
    
    # Find the longest continuous stretch where velocity < -5 (moving toward foul)
    approaching = velocity < -3
    
    # Find longest continuous True segment
    best_start = 0
    best_len = 0
    curr_start = 0
    curr_len = 0
    
    for i in range(len(approaching)):
        if approaching[i]:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
        else:
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
            curr_len = 0
    if curr_len > best_len:
        best_len = curr_len
        best_start = curr_start
    
    if best_len > 10:
        approach_start = best_start
        approach_end = best_start + best_len
        
        print(f"  Approach detected: indices {approach_start}-{approach_end}")
        print(f"  Frames: {valid[approach_start]['frame']}-{valid[approach_end-1]['frame']}")
        print(f"  Time: {valid[approach_start]['time']:.2f}s - {valid[approach_end-1]['time']:.2f}s")
        print(f"  Duration: {valid[approach_end-1]['time'] - valid[approach_start]['time']:.2f}s")
        
        approach_data = valid[approach_start:approach_end]
        
        # Start and end board
        start_board = approach_data[0]['board']
        end_board = approach_data[-1]['board']
        
        # Use median of first 5 and last 5 for stability
        start_board_med = np.median([d['board'] for d in approach_data[:min(5, len(approach_data))]])
        end_board_med = np.median([d['board'] for d in approach_data[-min(5, len(approach_data)):]])
        
        drift = end_board_med - start_board_med
        
        print(f"\n  === WALK-UP RESULTS ===")
        print(f"  Starting board: {start_board_med:.1f}")
        print(f"  Ending board:   {end_board_med:.1f}")
        print(f"  Drift: {drift:+.1f} boards", end="")
        if abs(drift) < 1:
            print(" (STRAIGHT)")
        elif drift > 0:
            print(" (drifts LEFT)")
        else:
            print(" (drifts RIGHT)")
        
        # Detailed timeline
        print(f"\n  Walk-up timeline:")
        sample = max(1, len(approach_data) // 15)
        for j in range(0, len(approach_data), sample):
            d = approach_data[j]
            print(f"    t={d['time']:5.2f}s  frame={d['frame']:3d}  board={d['board']:5.1f}  foot=({d['foot_x']:4d},{d['foot_y']:4d})  top_y={d['bbox_y_top']:4d}")
        d = approach_data[-1]
        print(f"    t={d['time']:5.2f}s  frame={d['frame']:3d}  board={d['board']:5.1f}  foot=({d['foot_x']:4d},{d['foot_y']:4d})  top_y={d['bbox_y_top']:4d}")
        
        # ================================================================
        # Visualization
        # ================================================================
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        
        # Plot 1: Full timeline - foot_y and bbox_top
        ax = axes[0, 0]
        ax.plot(times, foot_ys, 'b-', alpha=0.3, label='foot_y')
        ax.plot(times, bbox_tops, 'r-', alpha=0.3, label='head_y')
        # Highlight approach
        app_times = [d['time'] for d in approach_data]
        app_foot_ys = [d['foot_y'] for d in approach_data]
        app_bbox_tops = [d['bbox_y_top'] for d in approach_data]
        ax.plot(app_times, app_foot_ys, 'b-', linewidth=2, label='Approach foot_y')
        ax.plot(app_times, app_bbox_tops, 'r-', linewidth=2, label='Approach head_y')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Y position (px)')
        ax.set_title('Y Position Over Time')
        ax.legend()
        ax.invert_yaxis()
        
        # Plot 2: Board position during approach
        ax = axes[0, 1]
        app_boards = [d['board'] for d in approach_data]
        ax.plot(app_times, app_boards, 'b-o', markersize=3)
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Board 20')
        ax.axhline(y=start_board_med, color='green', linestyle=':', label=f'Start: {start_board_med:.1f}')
        ax.axhline(y=end_board_med, color='orange', linestyle=':', label=f'End: {end_board_med:.1f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Board Number')
        ax.set_title(f'Board Position During Approach (drift {drift:+.1f})')
        ax.legend()
        ax.set_ylim(0, 40)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Foot path on approach
        ax = axes[1, 0]
        # Draw board lines
        for b in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
            xt = board_x(b, FOUL_Y)
            xb = board_x(b, h_frame - 1)
            color = 'red' if b == 20 else ('gray' if b % 10 != 0 else 'darkgray')
            lw = 2 if b == 20 else (1 if b % 10 == 0 else 0.5)
            ax.plot([xt, xb], [FOUL_Y, h_frame-1], color=color, linewidth=lw, alpha=0.4)
            if b % 10 == 0:
                ax.text(xb, h_frame - 20, str(b), ha='center', fontsize=8, color='gray')
        
        ax.axhline(y=FOUL_Y, color='red', linewidth=2, label='Foul line')
        
        app_foot_xs = [d['foot_x'] for d in approach_data]
        colors = np.linspace(0, 1, len(approach_data))
        scatter = ax.scatter(app_foot_xs, app_foot_ys, c=colors, cmap='cool', s=20, zorder=5)
        ax.plot(app_foot_xs, app_foot_ys, 'k-', linewidth=0.5, alpha=0.3)
        
        ax.scatter([app_foot_xs[0]], [app_foot_ys[0]], c='lime', s=150, marker='^', zorder=10, 
                   edgecolors='black', label=f'Start (board {start_board_med:.1f})')
        ax.scatter([app_foot_xs[-1]], [app_foot_ys[-1]], c='red', s=150, marker='v', zorder=10,
                   edgecolors='black', label=f'End (board {end_board_med:.1f})')
        
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_title('Foot Path on Approach')
        ax.legend(loc='lower left')
        ax.invert_yaxis()
        plt.colorbar(scatter, ax=ax, label='Time progression')
        
        # Plot 4: Key frames
        ax = axes[1, 1]
        cap = cv2.VideoCapture(VIDEO_DIR + VNAME)
        
        # Show 3 key moments: start, middle, end
        key_frames = [approach_data[0], approach_data[len(approach_data)//2], approach_data[-1]]
        key_labels = ['START', 'MIDDLE', 'END']
        
        combined = None
        for ki, (kd, klabel) in enumerate(zip(key_frames, key_labels)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, kd['frame'])
            ret, kframe = cap.read()
            if ret:
                # Crop to approach area
                crop_left = max(0, kd['foot_x'] - 400)
                crop_right = min(w_frame, kd['foot_x'] + 400)
                crop_top = max(FOUL_Y, kd['bbox_y_top'] - 100)
                crop_bot = min(h_frame, kd['foot_y'] + 100)
                
                crop = kframe[crop_top:crop_bot, crop_left:crop_right]
                
                # Draw foot marker
                fx = kd['foot_x'] - crop_left
                fy = kd['foot_y'] - crop_top
                cv2.circle(crop, (fx, fy), 15, (0, 255, 255), 3)
                cv2.putText(crop, f"{klabel} B{kd['board']:.0f}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                
                # Resize all crops to same size
                crop = cv2.resize(crop, (400, 500))
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                if combined is None:
                    combined = crop_rgb
                else:
                    combined = np.hstack([combined, crop_rgb])
        
        cap.release()
        
        if combined is not None:
            ax.imshow(combined)
            ax.set_title('Key Frames: Start → Middle → End')
            ax.axis('off')
        
        plt.suptitle(f'Bowler Walk-Up Analysis - {VNAME}\n'
                     f'Start: Board {start_board_med:.1f} → End: Board {end_board_med:.1f} | '
                     f'Drift: {drift:+.1f} boards {"(LEFT)" if drift > 0 else "(RIGHT)" if drift < 0 else "(STRAIGHT)"}',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('bowler_analysis.png', dpi=120)
        print("\nSaved bowler_analysis.png")
    else:
        print("  Could not identify a clear approach sequence")
else:
    print("  Not enough data points")
