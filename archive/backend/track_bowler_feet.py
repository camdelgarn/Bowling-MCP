"""Track bowler feet position through GX010014 using lane calibration."""
import cv2
import sys
import json
import os
import numpy as np

sys.path.insert(0, r"C:\Development\Bowling-MCP\backend")

VIDEO = r"C:\video\behind\GX010014.MP4"
OUT = r"C:\Development\Bowling-MCP\backend"

# Load lane config (see LANE_CONFIG.md)
with open(os.path.join(OUT, "lane_config.json")) as f:
    lane_cfg = json.load(f)
BOARDS_PER_LANE = lane_cfg["boards_per_lane"]
BOWLER_HANDEDNESS = lane_cfg.get("bowler_handedness", "right")

# Load calibration
with open(f"{OUT}/bowler_calibration.json") as f:
    calib = json.load(f)

left_coeffs = np.array(calib["left_coeffs"])
right_coeffs = np.array(calib["right_coeffs"])
row_summaries = calib["row_data_summary"]

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fc} frames, {fps:.1f} fps, {fc/fps:.1f}s")

# Get reference frame (frame 30 = no bowler)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, ref_4k = cap.read()
ref = cv2.resize(ref_4k, (1920, 1080))
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)


def px_per_board_at(y):
    ys = [rs["mean_y"] for rs in row_summaries]
    ppbs = [rs["px_per_board"] for rs in row_summaries]
    return float(np.interp(y, ys, ppbs))


def find_bowler_feet(frame_gray, ref_gray):
    diff = cv2.absdiff(frame_gray, ref_gray)
    # y_top should be above the foul line (first dot row ~676, foul line above that)
    y_top, y_bot = 550, 1000
    roi = diff[y_top:y_bot, :]
    _, mask = cv2.threshold(roi.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 2000:
        return None
    bx, by, bw, bh = cv2.boundingRect(largest)
    by_abs = by + y_top
    foot_y = by_abs + bh
    # Feet: bottom 20% of bounding box
    feet_top = max(0, by + int(bh * 0.8))
    feet_mask = mask[feet_top:by + bh, bx:bx + bw]
    if feet_mask.sum() > 0:
        _, xs = np.where(feet_mask > 0)
        foot_x = int(np.mean(xs)) + bx
    else:
        foot_x = bx + bw // 2
    left_x = np.polyval(left_coeffs, foot_y)
    right_x = np.polyval(right_coeffs, foot_y)
    ppb = px_per_board_at(foot_y)
    if BOWLER_HANDEDNESS == "right":
        # Board 1 = right gutter edge, board 39 = left gutter edge
        board = (right_x - foot_x) / ppb
    else:
        # Board 1 = left gutter edge, board 39 = right gutter edge
        board = (foot_x - left_x) / ppb
    return {
        "foot_x": int(foot_x), "foot_y": int(foot_y),
        "board": round(float(board), 1), "area": int(area),
        "bbox": (int(bx), int(by_abs), int(bw), int(bh)),
    }


# Track every 2nd frame
tracking = []
for fi in range(0, fc, 2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, f4k = cap.read()
    if not ret:
        break
    frame = cv2.resize(f4k, (1920, 1080))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = find_bowler_feet(gray, ref_gray)
    t = fi / fps
    if r:
        tracking.append({"frame": fi, "time": round(t, 2), **r})

print(f"\nTracked {len(tracking)} frames with bowler detected")

if len(tracking) < 5:
    print("Not enough tracking data!")
    cap.release()
    sys.exit(1)

# --- Identify phases ---
# The bowler: walks on, picks up ball, stands at starting position,
# then walks down the approach toward the foul line.
#
# The APPROACH is identified by:
#   1. A "stance" period where foot_y and board are stable (standing still)
#   2. Followed by foot_y DECREASING (moving toward foul line / away from camera)
#
# foot_y decreasing = moving toward pins (perspective: foul line is further up)

times = np.array([d["time"] for d in tracking])
foot_ys = np.array([d["foot_y"] for d in tracking])
boards = np.array([d["board"] for d in tracking])

# Smooth foot_y to reduce noise (rolling mean, window=5)
win = min(5, len(foot_ys))
smooth_fy = np.convolve(foot_ys, np.ones(win) / win, mode="same")

# Find "stance" periods: foot_y stable (std < 10 over 1-second windows)
# and board stable (std < 2 over same windows).
window_frames = max(3, int(fps / 2 / 2))  # ~0.5s of every-2nd-frame data
stance_candidates = []
for i in range(len(tracking) - window_frames):
    fy_std = np.std(foot_ys[i:i + window_frames])
    bd_std = np.std(boards[i:i + window_frames])
    if fy_std < 10 and bd_std < 2:
        stance_candidates.append(i)

if not stance_candidates:
    print("No stance period detected!")
    cap.release()
    sys.exit(1)

# Group contiguous stance frames into stance periods
stance_periods = []
current_start = stance_candidates[0]
for k in range(1, len(stance_candidates)):
    if stance_candidates[k] - stance_candidates[k - 1] > 3:
        stance_periods.append((current_start, stance_candidates[k - 1]))
        current_start = stance_candidates[k]
stance_periods.append((current_start, stance_candidates[-1]))

print(f"\nStance periods found: {len(stance_periods)}")
for si, (s, e) in enumerate(stance_periods):
    dur = times[min(e + window_frames - 1, len(times) - 1)] - times[s]
    print(f"  Stance {si + 1}: t={times[s]:.1f}-{times[min(e + window_frames - 1, len(times) - 1)]:.1f}s "
          f"({dur:.1f}s), board~{np.mean(boards[s:e + 1]):.1f}, foot_y~{np.mean(foot_ys[s:e + 1]):.0f}")

# The real stance is the LONGEST stable period where the bowler is ON the lane
# (board 1-39) before the approach.
on_lane_stances = [(s, e) for s, e in stance_periods
                   if 1 <= np.mean(boards[s:e + 1]) <= 39]
if not on_lane_stances:
    on_lane_stances = stance_periods  # fallback
best_stance = max(on_lane_stances, key=lambda p: p[1] - p[0])
stance_end_idx = best_stance[1] + window_frames - 1
stance_end_idx = min(stance_end_idx, len(tracking) - 1)

stance_board = float(np.mean(boards[best_stance[0]:stance_end_idx + 1]))
stance_foot_y = float(np.mean(foot_ys[best_stance[0]:stance_end_idx + 1]))
print(f"\nSelected stance: t={times[best_stance[0]]:.1f}-{times[stance_end_idx]:.1f}s, "
      f"board={stance_board:.1f}, foot_y={stance_foot_y:.0f}")

# The approach starts RIGHT AFTER the stance: foot_y begins decreasing
# (bowler walks toward foul line). Find the first frame after stance
# where foot_y drops significantly.
approach_start_idx = None
for i in range(stance_end_idx, min(stance_end_idx + 30, len(tracking))):
    if foot_ys[i] < stance_foot_y - 5:
        approach_start_idx = i
        break

if approach_start_idx is None:
    # foot_y didn't decrease — maybe bowler walks forward (foot_y increases
    # in this camera). Look for board number changing instead.
    for i in range(stance_end_idx, min(stance_end_idx + 30, len(tracking))):
        if abs(boards[i] - stance_board) > 3:
            approach_start_idx = i
            break

if approach_start_idx is None:
    print("No approach detected after stance!")
    cap.release()
    sys.exit(1)

# The approach ends when there's a gap in tracking (bowler at foul line,
# tracker loses them) or foot_y stabilizes at a new low.
approach_end_idx = approach_start_idx
for i in range(approach_start_idx + 1, len(tracking)):
    # Gap detection: time jump > 1 second means bowler left the detection zone
    if times[i] - times[i - 1] > 1.0:
        approach_end_idx = i - 1
        break
    approach_end_idx = i
    # If foot_y rises back significantly, the approach is over
    if foot_ys[i] > foot_ys[approach_start_idx] + 30:
        approach_end_idx = i - 1
        break

# The approach "start" is actually the last stance frame (where the bowler
# started from), and the "end" is where they reach the foul line
start_data = tracking[stance_end_idx]
end_data = tracking[approach_end_idx]

# Find the minimum foot_y during the approach (closest to foul line)
approach_slice = tracking[approach_start_idx:approach_end_idx + 1]
if approach_slice:
    foul_line_frame = min(approach_slice, key=lambda d: d["foot_y"])
else:
    foul_line_frame = end_data

print(f"\n{'='*55}")
print(f" BOWLER APPROACH TRACKING - GX010014.MP4")
print(f"{'='*55}")
print(f"\n STANCE / STARTING POSITION:")
print(f"   Time: {start_data['time']:.2f}s (frame {start_data['frame']})")
print(f"   Foot: x={start_data['foot_x']}, y={start_data['foot_y']}")
print(f"   Board: {start_data['board']:.1f}")
print(f"\n APPROACH START (first step):")
app_start = tracking[approach_start_idx]
print(f"   Time: {app_start['time']:.2f}s (frame {app_start['frame']})")
print(f"   Foot: x={app_start['foot_x']}, y={app_start['foot_y']}")
print(f"   Board: {app_start['board']:.1f}")
print(f"\n FOUL LINE / SLIDE:")
print(f"   Time: {foul_line_frame['time']:.2f}s (frame {foul_line_frame['frame']})")
print(f"   Foot: x={foul_line_frame['foot_x']}, y={foul_line_frame['foot_y']}")
print(f"   Board: {foul_line_frame['board']:.1f}")
print(f"\n Duration: {foul_line_frame['time'] - start_data['time']:.2f}s")
print(f" Board drift: {start_data['board']:.1f} -> {foul_line_frame['board']:.1f} "
      f"({foul_line_frame['board'] - start_data['board']:+.1f} boards)")

# Save annotated frames
for label, data in [("stance", start_data), ("foul_line", foul_line_frame)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, data["frame"])
    ret, f4k = cap.read()
    if ret:
        frame = cv2.resize(f4k, (1920, 1080))
        fy = data["foot_y"]
        fx = data["foot_x"]
        lx = int(np.polyval(left_coeffs, fy))
        rx = int(np.polyval(right_coeffs, fy))
        cv2.line(frame, (lx, fy - 15), (lx, fy + 15), (0, 255, 0), 2)
        cv2.line(frame, (rx, fy - 15), (rx, fy + 15), (0, 255, 0), 2)
        cv2.circle(frame, (fx, fy), 12, (0, 0, 255), 3)
        cv2.putText(frame, f"Board {data['board']:.1f}",
                   (fx + 15, fy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"t={data['time']:.2f}s ({label})",
                   (fx + 15, fy + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(f"{OUT}/bowler_{label}.png", frame)
        print(f" Saved bowler_{label}.png")

# Save tracking JSON
save_data = {
    "stance": {"time": start_data["time"], "frame": start_data["frame"],
               "foot_x": start_data["foot_x"], "foot_y": start_data["foot_y"],
               "board": start_data["board"]},
    "foul_line": {"time": foul_line_frame["time"], "frame": foul_line_frame["frame"],
                  "foot_x": foul_line_frame["foot_x"], "foot_y": foul_line_frame["foot_y"],
                  "board": foul_line_frame["board"]},
    "duration_s": round(foul_line_frame["time"] - start_data["time"], 2),
    "board_drift": round(foul_line_frame["board"] - start_data["board"], 1),
    "stance_board": round(stance_board, 1),
    "all_frames": [{"time": d["time"], "frame": d["frame"],
                   "foot_x": d["foot_x"], "foot_y": d["foot_y"],
                   "board": d["board"]} for d in tracking],
}
with open(f"{OUT}/bowler_tracking.json", "w") as f:
    json.dump(save_data, f, indent=2)
print(f" Saved bowler_tracking.json")

cap.release()
