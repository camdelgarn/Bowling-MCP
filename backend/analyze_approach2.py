"""
Bowler walk-up analysis using foot_y velocity to detect the actual approach.
The approach is when foot_y drops rapidly (bowler walking toward foul line).
"""

import cv2
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

VIDEO_DIR = '../video/behind/'

# Load calibration
with open('lane_calibration.json', 'r') as f:
    cal = json.load(f)

lc = np.array([cal['left_edge_slope'], cal['left_edge_intercept']])
rc = np.array([cal['right_edge_slope'], cal['right_edge_intercept']])
FOUL_Y = cal['foul_line_y']
h_frame = cal['frame_height']
w_frame = cal['frame_width']

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

# Load pre-tracked data from bowler_tracking.json
with open('bowler_tracking.json', 'r') as f:
    tracking = json.load(f)

shot_data = tracking['shots'][0]['frames']
fps = 30.0

print(f"Loaded {len(shot_data)} detections from bowler_tracking.json")

# Extract arrays
frames_arr = np.array([d['frame'] for d in shot_data])
times_arr = np.array([d['time'] for d in shot_data])
foot_xs = np.array([d['foot_x'] for d in shot_data])
foot_ys = np.array([d['foot_y'] for d in shot_data])
boards_arr = np.array([d['board'] for d in shot_data])

# Print full timeline
print("\n=== Full timeline (every 1s) ===")
for i in range(0, len(shot_data), int(fps)):
    d = shot_data[i]
    print(f"  t={d['time']:5.2f}s  f={d['frame']:3d}  board={d['board']:5.1f}  "
          f"foot=({d['foot_x']:4d},{d['foot_y']:4d})")

# === APPROACH DETECTION USING FOOT_Y ===
# The approach: bowler stands still (foot_y ~2762), then walks toward foul line
# (foot_y drops from ~2762 to ~1488 over ~2s), then stands at foul line.
# We find the rapid descent after a stable period.

smooth_fy = gaussian_filter1d(foot_ys.astype(np.float64), sigma=3)
fy_velocity = np.gradient(smooth_fy)

# Find where the bowler is standing still on the approach (foot_y stable, not at bottom of frame)
# Foot_y around 2700-2800 and stable = on the approach, waiting to bowl
# Then foot_y drops rapidly = the actual walk-up

# Find first sustained period where foot_y < 2900 (not clipped at frame bottom)
# AND foot_y > 2500 (still at back of approach) AND velocity is near zero
standing_frames = []
for i in range(len(smooth_fy)):
    if 2500 < smooth_fy[i] < 2900 and abs(fy_velocity[i]) < 10:
        standing_frames.append(i)

if standing_frames:
    # The approach starts when foot_y starts dropping from this standing position
    # Look for the last standing frame before a sustained drop
    last_stand = standing_frames[-1]
    
    # Scan forward from last standing position to find when velocity becomes
    # strongly negative (walking forward)
    approach_start = last_stand
    for i in range(last_stand, len(smooth_fy)):
        if fy_velocity[i] < -15:  # significant forward motion
            approach_start = i
            break
    
    # Scan backward to include the very start of motion
    while approach_start > 0 and smooth_fy[approach_start - 1] > smooth_fy[approach_start]:
        approach_start -= 1
    
    # Find end: where foot_y stabilizes near the foul line (< 1550)
    approach_end = len(smooth_fy) - 1
    for i in range(approach_start, len(smooth_fy)):
        if smooth_fy[i] < 1550 and abs(fy_velocity[i]) < 10:
            approach_end = i
            break
else:
    # Fallback: find max negative velocity stretch
    approach_start = np.argmin(fy_velocity)
    approach_end = approach_start + 30

print(f"\n  Approach detected:")
print(f"  Indices: {approach_start} to {approach_end}")
print(f"  Frames: {frames_arr[approach_start]} to {frames_arr[approach_end]}")
print(f"  Time: {times_arr[approach_start]:.2f}s to {times_arr[approach_end]:.2f}s")
print(f"  Duration: {times_arr[approach_end] - times_arr[approach_start]:.2f}s")
print(f"  Foot Y: {foot_ys[approach_start]} -> {foot_ys[approach_end]}")

app = shot_data[approach_start:approach_end+1]

app_times = [d['time'] for d in app]
app_boards = [d['board'] for d in app]
app_foot_xs = [d['foot_x'] for d in app]
app_foot_ys = [d['foot_y'] for d in app]

# Start and end boards (median of first/last several frames for stability)
n_avg = min(5, max(1, len(app) // 5))
start_board = np.median(app_boards[:n_avg])
end_board = np.median(app_boards[-n_avg:])
drift = end_board - start_board

print(f"\n{'='*50}")
print(f"  BOWLER WALK-UP ANALYSIS")
print(f"{'='*50}")
print(f"  Starting board: {start_board:.1f}")
print(f"  Ending board:   {end_board:.1f}")
print(f"  Drift: {drift:+.1f} boards", end="")
if abs(drift) < 1:
    print(" (STRAIGHT)")
elif drift > 0:
    print(" (drifts LEFT)")
else:
    print(" (drifts RIGHT)")

print(f"\n  Step-by-step:")
for d in app:
    print(f"    t={d['time']:5.2f}s  f={d['frame']:3d}  board={d['board']:5.1f}  "
          f"foot=({d['foot_x']:4d},{d['foot_y']:4d})")

# ================================================================
# Visualization 
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Plot 1: Full foot_y timeline with approach highlighted
ax = axes[0, 0]
ax.plot(times_arr, smooth_fy, 'b-', alpha=0.5, label='Foot Y (full)')
ax.plot([times_arr[i] for i in range(approach_start, approach_end+1)],
        [smooth_fy[i] for i in range(approach_start, approach_end+1)],
        'r-', linewidth=3, label='Approach')
ax.axhline(y=FOUL_Y, color='green', linestyle='--', label='Foul line')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Foot Y position')
ax.set_title('Foot Position Timeline')
ax.legend()
ax.invert_yaxis()

# Plot 2: Board position during approach
ax = axes[0, 1]
ax.plot(app_times, app_boards, 'b-o', markersize=4)
ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Board 20')
ax.axhline(y=start_board, color='green', linestyle=':', linewidth=2, 
           label=f'Start: {start_board:.1f}')
ax.axhline(y=end_board, color='orange', linestyle=':', linewidth=2, 
           label=f'End: {end_board:.1f}')
ax.fill_between(app_times, start_board, end_board, alpha=0.1, color='blue')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Board Number')
ax.set_title(f'Board Position (drift {drift:+.1f})')
ax.legend()
ax.set_ylim(max(0, min(app_boards) - 5), min(40, max(app_boards) + 5))
ax.grid(True, alpha=0.3)

# Plot 3: Foot path on approach
ax = axes[1, 0]
for b in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
    xt = board_x(b, FOUL_Y)
    xb = board_x(b, h_frame - 1)
    color = 'red' if b == 20 else ('gray' if b % 10 != 0 else 'darkgray')
    lw = 2 if b == 20 else (1 if b % 10 == 0 else 0.5)
    ax.plot([xt, xb], [FOUL_Y, h_frame-1], color=color, linewidth=lw, alpha=0.4)
    if b % 10 == 0:
        ax.text(xb, h_frame - 20, str(b), ha='center', fontsize=8, color='gray')

ax.axhline(y=FOUL_Y, color='red', linewidth=2, label='Foul line')

colors = np.linspace(0, 1, len(app))
scatter = ax.scatter(app_foot_xs, app_foot_ys, c=colors, cmap='cool', s=30, zorder=5)
ax.plot(app_foot_xs, app_foot_ys, 'k-', linewidth=0.5, alpha=0.3)
ax.scatter([app_foot_xs[0]], [app_foot_ys[0]], c='lime', s=200, marker='^', 
           zorder=10, edgecolors='black', label=f'Start B{start_board:.0f}')
ax.scatter([app_foot_xs[-1]], [app_foot_ys[-1]], c='red', s=200, marker='v', 
           zorder=10, edgecolors='black', label=f'End B{end_board:.0f}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Body Center Path')
ax.legend(loc='lower left')
ax.invert_yaxis()

# Plot 4: Key frames from the approach
ax = axes[1, 1]
cap = cv2.VideoCapture(VIDEO_DIR + '1.MP4')

key_indices = [0, len(app)//4, len(app)//2, 3*len(app)//4, len(app)-1]
crops = []
for ki in key_indices:
    d = app[ki]
    cap.set(cv2.CAP_PROP_POS_FRAMES, d['frame'])
    ret, kframe = cap.read()
    if ret:
        cx = d['foot_x']
        cy = d['foot_y'] - 400
        crop_half_w = 300
        crop_half_h = 500
        y1 = max(0, cy - crop_half_h)
        y2 = min(h_frame, cy + crop_half_h)
        x1 = max(0, cx - crop_half_w)
        x2 = min(w_frame, cx + crop_half_w)
        crop = kframe[y1:y2, x1:x2]
        crop = cv2.resize(crop, (200, 350))
        cv2.putText(crop, f"B{d['board']:.0f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(crop, f"t={d['time']:.1f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
cap.release()

if crops:
    combined = np.hstack(crops)
    ax.imshow(combined)
    ax.set_title('Key Frames During Approach')
    ax.axis('off')

direction = "LEFT" if drift > 0 else "RIGHT" if drift < 0 else "STRAIGHT"
plt.suptitle(f'Bowler Walk-Up Analysis - 1.MP4\n'
             f'Start: Board {start_board:.1f} -> End: Board {end_board:.1f} | '
             f'Drift: {drift:+.1f} boards ({direction})',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('bowler_analysis.png', dpi=120)
print("\nSaved bowler_analysis.png")
