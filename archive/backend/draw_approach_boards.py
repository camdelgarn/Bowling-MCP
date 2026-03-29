#!/usr/bin/env python3
"""Generate a PNG showing the bowling approach with board lines overlaid.

Uses the lane detection calibration to draw perspective-correct board lines
from the foul line area back through the approach, with board numbers labeled.
"""

import cv2
import numpy as np
import os
import sys

# Import detection from the main module
from detect_lane_livestream import detect_lane, BOARDS_PER_LANE

def draw_approach_boards(frame: np.ndarray, detection: dict) -> np.ndarray:
    """Draw board lines across the full approach area.

    Each board line is a single straight line computed from its position at two
    calibrated dot rows (or VP + one row).  No per-row interpolation kinks.
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    lc = detection["left_coeffs"]
    rc = detection["right_coeffs"]
    row_data = detection["row_data"]
    y_top, y_bot = detection["y_range"]

    # Semi-transparent lane fill
    overlay = out.copy()
    left_pts, right_pts = [], []
    for y in range(y_top, y_bot + 1, 3):
        lx = int(np.polyval(lc, y))
        rx = int(np.polyval(rc, y))
        left_pts.append([lx, y])
        right_pts.append([rx, y])
    polygon = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    cv2.fillPoly(overlay, [polygon], (40, 40, 40))
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

    # Build a straight-line equation (x = slope*y + intercept) for every board.
    # Use the two dot rows to get each board's x at two y values, then fit line.
    # For boards not directly measured by dots we extrapolate from the known
    # per-board spacing in each row (bx_fit from analyze_dot_row).
    def board_line(board_num):
        """Return (slope, intercept) for x = slope*y + intercept for this board."""
        if len(row_data) >= 2:
            pts_y, pts_x = [], []
            for rd in row_data:
                bx = np.polyval(rd["bx_fit"], board_num)
                by = np.polyval(rd["by_fit"], board_num) if "by_fit" in rd else rd["mean_y"]
                pts_y.append(by)
                pts_x.append(bx)
            coeffs = np.polyfit(pts_y, pts_x, 1)
            return coeffs[0], coeffs[1]
        else:
            # Single row: use edge slope from detection
            rd = row_data[0]
            bx = np.polyval(rd["bx_fit"], board_num)
            by = rd["mean_y"]
            # Slope from left edge as proxy
            slope = lc[0]
            intercept = bx - slope * by
            return slope, intercept

    # Precompute line for each board
    board_lines = {}
    for board in range(BOARDS_PER_LANE + 1):
        board_lines[board] = board_line(board)

    # Draw every board line as a single straight line
    for board in range(BOARDS_PER_LANE + 1):
        slope, intercept = board_lines[board]
        x_top = int(slope * y_top + intercept)
        x_bot = int(slope * y_bot + intercept)

        # Skip if entirely off-screen
        if max(x_top, x_bot) < 0 or min(x_top, x_bot) >= w:
            continue

        # Color coding
        if board == 0 or board == BOARDS_PER_LANE:
            color = (0, 255, 0)
            thickness = 2
        elif board % 10 == 0:
            color = (0, 220, 255)
            thickness = 2
        elif board % 5 == 0:
            color = (255, 200, 0)
            thickness = 1
        else:
            color = (100, 100, 100)
            thickness = 1

        # Clip to image bounds
        pt1 = (np.clip(x_top, 0, w - 1), y_top)
        pt2 = (np.clip(x_bot, 0, w - 1), y_bot)
        cv2.line(out, pt1, pt2, color, thickness)

    # Label board numbers at multiple y positions
    label_ys = [int(rd["mean_y"]) for rd in row_data]
    label_ys.append(min(y_bot - 30, int(max(rd["mean_y"] for rd in row_data) + 150)))

    for y_label in label_ys:
        for board in range(0, BOARDS_PER_LANE + 1, 5):
            slope, intercept = board_lines[board]
            bx = int(slope * y_label + intercept)
            if 10 <= bx < w - 30:
                tw, _ = cv2.getTextSize(str(board), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                tx = bx - tw[0] // 2
                ty = y_label - 15
                cv2.rectangle(out, (tx - 2, ty - 12), (tx + tw[0] + 2, ty + 4),
                              (0, 0, 0), -1)
                cv2.putText(out, str(board), (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw detected dots with board numbers
    dot_colors = [(0, 100, 255), (255, 0, 200), (255, 160, 0)]
    for ri, rd in enumerate(row_data):
        color = dot_colors[ri % len(dot_colors)]
        boards = rd["board_numbers"]
        for di, d in enumerate(rd["dots"]):
            cv2.circle(out, (d["x"], d["y"]), 8, color, 2)
            label = f"B{boards[di]}"
            cv2.putText(out, label, (d["x"] + 12, d["y"] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Edge lines (bold dots)
    for y in range(y_top, y_bot + 1, 2):
        lx = int(np.polyval(lc, y))
        rx = int(np.polyval(rc, y))
        cv2.circle(out, (lx, y), 1, (0, 255, 0), -1)
        cv2.circle(out, (rx, y), 1, (0, 255, 0), -1)

    # Title and legend
    cv2.putText(out, "APPROACH - BOARD MAP", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    legend = [
        ("Green = edges (0, 39)", (0, 255, 0)),
        ("Yellow = every 10 boards", (0, 220, 255)),
        ("Cyan = every 5 boards (dots)", (255, 200, 0)),
        ("Gray = individual boards", (140, 140, 140)),
    ]
    for i, (text, color) in enumerate(legend):
        cv2.putText(out, text, (20, 65 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Row info
    for ri, rd in enumerate(row_data):
        text = (f"Row {ri+1}: {rd['dot_count']} dots, boards {rd['board_numbers']}, "
                f"{rd['px_per_board']:.1f} px/board")
        cv2.putText(out, text, (20, h - 60 + ri * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out


def main():
    frame_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "livestream_frame.jpg")
    if not os.path.exists(frame_path):
        print(f"No frame found at {frame_path}")
        sys.exit(1)

    frame = cv2.imread(frame_path)
    print(f"Loaded frame: {frame.shape[1]}x{frame.shape[0]}")

    detection = detect_lane(frame)
    if detection is None:
        print("Lane detection failed")
        sys.exit(1)

    result = draw_approach_boards(frame, detection)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approach_boards.png")
    cv2.imwrite(out_path, result)
    print(f"\nSaved approach board map to: {out_path}")


if __name__ == "__main__":
    main()
