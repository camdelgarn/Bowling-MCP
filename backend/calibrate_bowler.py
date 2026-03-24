"""Calibrate lane from clear frame, then track bowler feet."""
import cv2
import sys
import json
import numpy as np

sys.path.insert(0, r"C:\Development\Bowling-MCP\backend")
from detect_lane_livestream import detect_lane
from draw_approach_boards import draw_approach_boards

VIDEO = r"C:\video\behind\GX010014.MP4"
OUT = r"C:\Development\Bowling-MCP\backend"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

# Use frame 30 (t=1.0s) — bowler hasn't started, both dot rows visible
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame_4k = cap.read()
cap.release()

frame = cv2.resize(frame_4k, (1920, 1080))
cv2.imwrite(f"{OUT}/bowler_frame.jpg", frame)

result = detect_lane(frame)
if result:
    rds = result.get("row_data", [])
    print(f"\nCalibration: {len(rds)} rows")
    for i, rd in enumerate(rds):
        print(f"  Row {i+1}: y~{rd['mean_y']:.0f}, {rd['dot_count']} dots, "
              f"boards={rd['board_numbers']}, px/board={rd['px_per_board']:.2f}")

    out = draw_approach_boards(frame, result)
    cv2.imwrite(f"{OUT}/bowler_boards.png", out)
    print(f"\nSaved bowler_frame.jpg and bowler_boards.png")

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
    with open(f"{OUT}/bowler_calibration.json", "w") as f:
        json.dump(calib, f, indent=2)
    print("Saved bowler_calibration.json")
else:
    print("Lane detection FAILED")
