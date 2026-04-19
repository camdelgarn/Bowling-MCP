"""Check bowler video for clear frames and run lane detection."""
import cv2
import sys
import numpy as np

sys.path.insert(0, r"C:\Development\Bowling-MCP\backend")
from detect_lane_livestream import detect_lane, find_all_dot_rows
from draw_approach_boards import draw_approach_boards

VIDEO = r"C:\video\behind\GX010014.MP4"
OUT_DIR = r"C:\Development\Bowling-MCP\backend"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fc} frames, {fps:.1f} fps, {fc/fps:.1f}s")

# Try early frames for best calibration (bowler not yet in frame)
best_frame = None
best_result = None
best_rows = 0

for fi in [0, 15, 30, 45, 60]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame_4k = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame_4k, (1920, 1080))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = find_all_dot_rows(gray)
    five_dot = [r for r in rows if len(r) >= 5]
    print(f"\nFrame {fi} (t={fi/fps:.1f}s): {len(rows)} runs, "
          f"{len(five_dot)} with 5+ dots")
    if len(five_dot) > best_rows:
        best_rows = len(five_dot)
        best_frame = frame.copy()
        print(f"  -> New best: {best_rows} five-dot rows")

if best_frame is None:
    print("No frames read!")
    cap.release()
    sys.exit(1)

print(f"\n=== Running lane detection on best frame ({best_rows} five-dot rows) ===")
result = detect_lane(best_frame)
if result:
    rds = result.get("row_data", [])
    print(f"\nCalibration: {len(rds)} rows")
    for i, rd in enumerate(rds):
        print(f"  Row {i+1}: y~{rd['mean_y']:.0f}, {rd['dot_count']} dots, "
              f"boards={rd['board_numbers']}")

    out = draw_approach_boards(best_frame, result)
    cv2.imwrite(f"{OUT_DIR}/bowler_boards.png", out)
    cv2.imwrite(f"{OUT_DIR}/bowler_frame.jpg", best_frame)
    print(f"\nSaved bowler_frame.jpg and bowler_boards.png")

    # Store result for use by bowler tracking
    import json
    calib = {
        "left_coeffs": result["left_coeffs"].tolist(),
        "right_coeffs": result["right_coeffs"].tolist(),
        "y_range": list(result["y_range"]),
        "row_data_summary": [
            {"mean_y": rd["mean_y"], "px_per_board": rd["px_per_board"],
             "dot_count": rd["dot_count"], "board_numbers": rd["board_numbers"]}
            for rd in rds
        ],
    }
    with open(f"{OUT_DIR}/bowler_calibration.json", "w") as f:
        json.dump(calib, f, indent=2)
    print("Saved bowler_calibration.json")
else:
    print("Lane detection failed!")

cap.release()
