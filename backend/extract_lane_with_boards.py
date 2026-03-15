#!/usr/bin/env python3
"""
Extract frame and draw lane boundary lines with board markers
"""

import cv2
import numpy as np
import os

def extract_frame_with_lane_lines(video_path, output_png="lane_boundaries_with_boards.png"):
    """
    Extract a frame and draw lane boundary lines with board markers
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False

    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return False

    # Get a frame from the middle of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        cap.release()
        return False

    print(f"Extracted frame {middle_frame} of {total_frames}")

    # Analyze the frame for lane boundaries
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)

    lane_boundaries = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

            # Look for near-vertical lines (lane boundaries)
            if abs(angle) > 70:  # Nearly vertical
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 300:  # Long enough to be lane boundary
                    lane_boundaries.append((x1, y1, x2, y2, length))

    print(f"Found {len(lane_boundaries)} potential lane boundaries")

    # Sort by x-position to find left and right boundaries
    if len(lane_boundaries) >= 2:
        # Sort by average x position
        lane_boundaries.sort(key=lambda x: (min(x[0], x[2]) + max(x[0], x[2])) / 2)

        # Take the leftmost and rightmost as lane boundaries
        left_boundary = lane_boundaries[0]
        right_boundary = lane_boundaries[-1]

        print(f"Left boundary: x={left_boundary[0]}-{left_boundary[2]}")
        print(f"Right boundary: x={right_boundary[0]}-{right_boundary[2]}")

        # Draw the boundary lines extended to the back
        extend_y1 = height
        extend_y2 = int(height * 0.3)

        # For left boundary
        lx1, ly1, lx2, ly2 = left_boundary[:4]
        if lx2 != lx1:
            slope_left = (ly2 - ly1) / (lx2 - lx1)
            extend_x1_left = lx1 + (extend_y1 - ly1) / slope_left
            extend_x2_left = lx1 + (extend_y2 - ly1) / slope_left
            cv2.line(frame, (int(extend_x1_left), extend_y1), (int(extend_x2_left), extend_y2), (0, 255, 0), 5)

        # For right boundary
        rx1, ry1, rx2, ry2 = right_boundary[:4]
        if rx2 != rx1:
            slope_right = (ry2 - ry1) / (rx2 - rx1)
            extend_x1_right = rx1 + (extend_y1 - ry1) / slope_right
            extend_x2_right = rx1 + (extend_y2 - ry1) / slope_right
            cv2.line(frame, (int(extend_x1_right), extend_y1), (int(extend_x2_right), extend_y2), (0, 255, 0), 5)

        # Draw original detected lines in red
        cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
        cv2.line(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

        # Add board number markers (corrected based on user's info)
        # Board 20 is center, board 18 is where left foot is positioned
        # Counting from right (ball side): 0 to 40

        # From user's info: left foot at board 18, center is board 20
        # So board 18 is 2 boards left of center
        lane_width_pixels = extend_x1_right - extend_x1_left
        boards_total = 40

        # Calculate pixels per board
        pixels_per_board = lane_width_pixels / boards_total

        # Board 20 (center) should be at the center of the lane
        center_x = (extend_x1_left + extend_x1_right) / 2
        board_20_x = center_x

        # Board 18 (user's left foot) - 2 boards left of center
        board_18_x = board_20_x - (2 * pixels_per_board)

        # Now calculate all board positions relative to board 20
        key_boards = [0, 5, 10, 15, 18, 20, 25, 30, 35, 40]

        for board in key_boards:
            if board == 20:
                board_x = int(board_20_x)
            else:
                offset = (board - 20) * pixels_per_board
                board_x = int(board_20_x + offset)

            # Make sure it's within frame bounds
            if 0 <= board_x < width:
                cv2.line(frame, (board_x, height), (board_x, int(height * 0.8)), (255, 255, 0), 2)
                cv2.putText(frame, f"B{board}", (board_x + 5, int(height * 0.85)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Special marking for board 18 (user's position)
                if board == 18:
                    cv2.line(frame, (board_x, height), (board_x, int(height * 0.6)), (0, 0, 255), 3)
                    cv2.putText(frame, "YOUR LEFT FOOT (B18)", (board_x + 10, int(height * 0.65)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Add labels
        cv2.putText(frame, "LEFT LANE EDGE", (int(extend_x2_left) + 10, extend_y2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "RIGHT LANE EDGE", (int(extend_x2_right) - 200, extend_y2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the frame
    cv2.imwrite(output_png, frame)
    print(f"Saved frame with lane lines and board markers to: {output_png}")

    cap.release()
    return True

def main():
    video_path = "../video/behind/1.MP4"
    extract_frame_with_lane_lines(video_path)

if __name__ == "__main__":
    main()