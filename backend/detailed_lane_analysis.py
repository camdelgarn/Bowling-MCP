#!/usr/bin/env python3
"""
Detailed analysis of bowling lane from behind camera
"""

import cv2
import numpy as np
import os

def analyze_bowling_lane_frame(frame_path):
    """
    Analyze a specific frame for bowling lane features
    """
    if not os.path.exists(frame_path):
        print(f"Frame not found: {frame_path}")
        return

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Failed to load frame: {frame_path}")
        return

    height, width = frame.shape[:2]
    print(f"\nAnalyzing {frame_path} ({width}x{height})")

    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Look for lane markings (typically dark lines on light surface)
    # Bowling lanes have specific patterns:
    # - Foul line (red/black stripe)
    # - Approach boards (numbered 1-39/40 from center)
    # - Lane boundaries
    # - Pin deck area

    # Find edges
    edges = cv2.Canny(gray, 50, 150)

    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=20)

    if lines is not None:
        print(f"Found {len(lines)} line segments")

        # Analyze line orientations and positions
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

            if abs(angle) < 30:  # Horizontal
                horizontal_lines.append((x1, y1, x2, y2, length))
            elif abs(angle) > 60:  # Vertical
                vertical_lines.append((x1, y1, x2, y2, length))

        print(f"Horizontal lines: {len(horizontal_lines)}")
        print(f"Vertical lines: {len(vertical_lines)}")

        # Look for bowling-specific patterns
        # The foul line is typically a thick black/red stripe
        # Approach boards are numbered and have specific spacing

        # Check for potential foul line (thick horizontal line near bottom)
        bottom_third_y = int(height * 0.7)
        foul_line_candidates = [line for line in horizontal_lines if line[1] > bottom_third_y and line[3] > bottom_third_y]

        if foul_line_candidates:
            print(f"Potential foul line candidates: {len(foul_line_candidates)}")
            # Find the thickest/longest one
            thickest = max(foul_line_candidates, key=lambda x: x[4])
            print(f"Thickest line: length={thickest[4]:.1f}, y-pos={thickest[1]}")

        # Look for lane boundaries (long vertical lines)
        if vertical_lines:
            long_verticals = [line for line in vertical_lines if line[4] > 200]
            print(f"Long vertical lines (potential lane boundaries): {len(long_verticals)}")

            if long_verticals:
                # Lane width is about 42 inches, but in pixels depends on distance
                leftmost = min(long_verticals, key=lambda x: min(x[0], x[2]))
                rightmost = max(long_verticals, key=lambda x: max(x[0], x[2]))
                lane_width_pixels = max(leftmost[0], leftmost[2], rightmost[0], rightmost[2]) - min(leftmost[0], leftmost[2], rightmost[0], rightmost[2])
                print(f"Estimated lane width in pixels: {lane_width_pixels}")

    # Look for text/boards (lane board numbers)
    # Convert to binary for OCR-like analysis
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours that might be text
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        # Text-like shapes (tall, narrow rectangles)
        if 0.2 < aspect_ratio < 5 and 50 < area < 5000:
            text_candidates.append((x, y, w, h, area))

    print(f"Text-like contours found: {len(text_candidates)}")

    if text_candidates:
        # Group by y-position (same line)
        y_positions = {}
        for x, y, w, h, area in text_candidates:
            y_key = y // 20  # Group within 20 pixels vertically
            if y_key not in y_positions:
                y_positions[y_key] = []
            y_positions[y_key].append((x, y, w, h))

        print(f"Grouped into {len(y_positions)} horizontal lines of text")

        # Look for evenly spaced text (board numbers)
        for y_key, items in y_positions.items():
            if len(items) > 3:  # At least 4 items
                x_positions = sorted([item[0] for item in items])
                spacing = np.diff(x_positions)
                avg_spacing = np.mean(spacing)
                std_spacing = np.std(spacing)

                if std_spacing < avg_spacing * 0.3:  # Fairly even spacing
                    print(f"Evenly spaced text line at y~{y_key*20}: {len(items)} items, spacing~{avg_spacing:.1f}px")
                    print(f"  X positions: {x_positions}")

                    # Estimate board position
                    # Lane center is at board 20 (39 boards total, 19 on each side)
                    # Each board is about 1 inch wide
                    center_x = width / 2
                    pixels_per_board = avg_spacing  # Approximate

                    # Find which board is at center
                    center_item_idx = np.argmin([abs(x - center_x) for x in x_positions])
                    center_board_number = center_item_idx + 1  # Assuming leftmost is board 1

                    print(f"  Estimated center at board ~{center_board_number} (leftmost=1, center=20)")

def main():
    # Analyze all extracted frames
    for i in range(11):
        frame_path = f"frames/frame_{i:03d}.jpg"
        analyze_bowling_lane_frame(frame_path)

if __name__ == "__main__":
    main()