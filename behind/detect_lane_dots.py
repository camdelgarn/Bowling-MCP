#!/usr/bin/env python3
"""
Find frame without bowler and detect lane dots for accurate board positioning
"""

import cv2
import numpy as np
import os

def find_frame_without_bowler(video_path):
    """
    Find a frame where the bowler is not visible (lane dots should be clear)
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None

    print(f"Scanning video for frame without bowler: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // 50)  # Check ~50 frames

    best_frame = None
    min_contours = float('inf')

    for frame_idx in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert to grayscale and detect contours (lane markings/dots)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold to find dark spots on light background (lane dots)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours that could be lane dots (roughly circular, appropriate size)
        dot_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:  # Size range for dots
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                if circularity > 0.7:  # Roughly circular
                    dot_contours.append(contour)

        # Look for frames with many dots (lane markings visible)
        if len(dot_contours) > min_contours:
            min_contours = len(dot_contours)
            best_frame = frame.copy()
            best_frame_idx = frame_idx

    cap.release()

    if best_frame is not None:
        print(f"Found frame {best_frame_idx} with {min_contours} potential dots")
        return best_frame
    else:
        print("No suitable frame found")
        return None

def detect_lane_dots(frame):
    """
    Detect the lane dots and determine board positions
    """
    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold to find dark dots on light lane
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for dot-like contours
    dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 500:  # Size range for dots
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity > 0.7:  # Roughly circular
                # Get center point
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy, area))

    print(f"Found {len(dots)} potential lane dots")

    # Group dots by y-position (should be roughly horizontal lines)
    y_positions = {}
    for x, y, area in dots:
        y_key = y // 10  # Group within 10 pixels vertically
        if y_key not in y_positions:
            y_positions[y_key] = []
        y_positions[y_key].append((x, y, area))

    # Find the main line of dots (approach area)
    main_line = None
    max_dots = 0
    for y_key, line_dots in y_positions.items():
        if len(line_dots) > max_dots:
            max_dots = len(line_dots)
            main_line = line_dots

    if main_line and len(main_line) >= 3:
        # Sort by x position
        main_line.sort(key=lambda x: x[0])

        print(f"Main dot line has {len(main_line)} dots")

        # The center dot should be board 20
        center_x = width // 2
        center_dot = min(main_line, key=lambda x: abs(x[0] - center_x))

        print(f"Center dot (board 20) at x={center_dot[0]}")

        # Calculate spacing between dots
        x_positions = [x for x, y, area in main_line]
        spacing = np.diff(x_positions)
        avg_spacing = np.mean(spacing) if len(spacing) > 0 else 0

        print(f"Average dot spacing: {avg_spacing:.1f} pixels")

        # Estimate board positions
        board_20_x = center_dot[0]
        boards_per_dot = 5  # Small dots every 5 boards
        pixels_per_board = avg_spacing / boards_per_dot

        print(f"Estimated pixels per board: {pixels_per_board:.1f}")

        # Calculate board positions
        board_positions = {}
        for board in range(0, 41, 5):  # Every 5 boards
            if board == 20:
                board_positions[board] = board_20_x
            else:
                offset = (board - 20) * pixels_per_board
                board_positions[board] = int(board_20_x + offset)

        return board_positions, main_line

    return None, None

def create_dots_visualization(video_path, output_png="lane_dots_detected.png"):
    """
    Create visualization with detected lane dots and board positions
    """
    # Find frame without bowler
    frame = find_frame_without_bowler(video_path)
    if frame is None:
        print("Could not find suitable frame")
        return False

    # Detect dots
    board_positions, dots = detect_lane_dots(frame)

    if board_positions:
        # Draw detected dots
        for x, y, area in dots:
            cv2.circle(frame, (x, y), 5, (0, 255, 255), 2)  # Yellow circles

        # Draw board lines
        height, width = frame.shape[:2]
        for board, x_pos in board_positions.items():
            if 0 <= x_pos < width:
                cv2.line(frame, (x_pos, height), (x_pos, int(height * 0.7)), (255, 0, 255), 2)
                cv2.putText(frame, f"B{board}", (x_pos + 5, int(height * 0.75)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Mark center (board 20)
        if 20 in board_positions:
            center_x = board_positions[20]
            cv2.line(frame, (center_x, height), (center_x, int(height * 0.5)), (0, 255, 0), 3)
            cv2.putText(frame, "BOARD 20 (CENTER)", (center_x + 10, int(height * 0.55)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(output_png, frame)
    print(f"Saved dots detection visualization to: {output_png}")
    return True

def main():
    video_path = "../video/behind/1.MP4"
    create_dots_visualization(video_path)

if __name__ == "__main__":
    main()