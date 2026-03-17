#!/usr/bin/env python3
"""
Detect lane dots in a specific frame
"""

import cv2
import numpy as np
import os

def detect_dots_in_frame(frame_path):
    """
    Detect lane dots in a specific frame
    """
    if not os.path.exists(frame_path):
        print(f"Frame not found: {frame_path}")
        return

    print(f"Analyzing frame: {frame_path}")

    frame = cv2.imread(frame_path)
    if frame is None:
        print("Failed to load frame")
        return

    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try different thresholds to find dots
    for thresh_val in [80, 100, 120, 140]:
        print(f"\nTrying threshold: {thresh_val}")

        # Threshold to find dark dots on light lane
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Found {len(contours)} contours")

        # Filter for dot-like contours
        dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 1000:  # Wider range for dots
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Get center point
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy, area, circularity))

        print(f"Found {len(dots)} potential dots")

        # Group dots by y-position
        y_positions = {}
        for x, y, area, circ in dots:
            y_key = y // 15  # Group within 15 pixels vertically
            if y_key not in y_positions:
                y_positions[y_key] = []
            y_positions[y_key].append((x, y, area, circ))

        print(f"Dots grouped into {len(y_positions)} horizontal lines")

        # Show the groups
        for y_key, line_dots in y_positions.items():
            if len(line_dots) >= 3:  # At least 3 dots
                avg_y = sum(y for x, y, a, c in line_dots) / len(line_dots)
                print(f"  Line at y~{avg_y:.0f}: {len(line_dots)} dots")

                # Sort by x
                line_dots.sort(key=lambda x: x[0])
                x_positions = [x for x, y, a, c in line_dots]
                print(f"    X positions: {x_positions}")

                # Check spacing
                if len(x_positions) >= 2:
                    spacing = np.diff(x_positions)
                    avg_spacing = np.mean(spacing)
                    print(f"    Average spacing: {avg_spacing:.1f} pixels")

def main():
    # Try a few frames
    frames_to_check = [
        "inspection_frames/frame_05_0195.jpg",  # Middle of video
        "inspection_frames/frame_10_0390.jpg",  # Later in video
        "inspection_frames/frame_15_0585.jpg",  # Even later
    ]

    for frame_path in frames_to_check:
        detect_dots_in_frame(frame_path)

if __name__ == "__main__":
    main()