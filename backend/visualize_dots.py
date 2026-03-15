#!/usr/bin/env python3
"""
Create visualization of detected dots on a frame
"""

import cv2
import numpy as np
import os

def visualize_dots_on_frame(frame_path, output_path="dots_visualization.png"):
    """
    Load a frame and visualize detected dots
    """
    if not os.path.exists(frame_path):
        print(f"Frame not found: {frame_path}")
        return

    frame = cv2.imread(frame_path)
    if frame is None:
        print("Failed to load frame")
        return

    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}")

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use threshold that seemed to work
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for dots
    dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 500:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity > 0.5:  # Less strict circularity
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy, area))

    print(f"Found {len(dots)} dots")

    # Draw all dots
    for x, y, area in dots:
        color = (0, 255, 255)  # Yellow
        if 100 < area < 300:  # Medium dots
            color = (0, 255, 0)  # Green
        elif area >= 300:  # Large dots
            color = (255, 0, 0)  # Red for large dots
        cv2.circle(frame, (x, y), 8, color, 2)
        cv2.putText(frame, f"{area}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Look for center dot (board 20)
    center_x = width // 2
    center_dots = [(x, y, area) for x, y, area in dots if abs(x - center_x) < 100]  # Within 100px of center

    if center_dots:
        # Find the largest dot near center
        center_dot = max(center_dots, key=lambda x: x[2])
        cx, cy, carea = center_dot

        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), 3)  # Magenta for center
        cv2.putText(frame, "CENTER DOT (BOARD 20)", (cx+20, cy-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        print(f"Center dot at ({cx}, {cy}) with area {carea}")

        # Look for dots at regular intervals
        # Assume 5 boards spacing
        estimated_spacing = 150  # pixels between board markers (guess)

        for offset in [-4, -3, -2, -1, 1, 2, 3, 4]:
            expected_x = cx + offset * estimated_spacing
            # Find closest dot to this position
            nearby_dots = [(x, y, area) for x, y, area in dots if abs(x - expected_x) < 50]
            if nearby_dots:
                closest = min(nearby_dots, key=lambda d: abs(d[0] - expected_x))
                dx, dy, darea = closest
                board_num = 20 + offset * 5
                cv2.circle(frame, (dx, dy), 12, (0, 165, 255), 2)  # Orange
                cv2.putText(frame, f"B{board_num}", (dx+15, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                print(f"Board {board_num} dot at ({dx}, {dy})")

    cv2.imwrite(output_path, frame)
    print(f"Saved visualization to: {output_path}")

def main():
    # Use frame from middle of video
    frame_path = "inspection_frames/frame_10_0390.jpg"
    visualize_dots_on_frame(frame_path)

if __name__ == "__main__":
    main()