#!/usr/bin/env python3
"""
Find frame without bowler and outline approach and lane regions
"""

import cv2
import numpy as np
import os

def find_empty_lane_frame(video_path):
    """
    Find a frame where the bowler is not visible (just the lane)
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None

    print(f"Scanning for empty lane frame: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check frames at different points - focus on the best range
    frame_positions = [730, 725, 735, 720, 740, 0, 10, 20]

    best_frame = None
    min_person_score = float('inf')

    for frame_idx in frame_positions:
        if frame_idx >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert to grayscale and check for person-like dark regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Focus on the lane area where bowler would be
        lane_region = gray[height//2:, :]  # Bottom half
        _, thresh = cv2.threshold(lane_region, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours and calculate person score
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Person score: sum of areas of large connected components (potential person parts)
        person_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Large enough to be part of a person
                person_score += area
        
        print(f"Frame {frame_idx}: person score = {person_score}")

        if person_score < min_person_score:
            min_person_score = person_score
            best_frame = frame.copy()

    cap.release()

    if best_frame is not None:
        print(f"Using frame with person score {min_person_score}")
        return best_frame
    else:
        print("No frames found")
        return None

def detect_approach_and_lane_boundaries(frame):
    """
    Detect the approach and lane boundaries using edge detection and line finding
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
    
    if lines is None:
        print("No lines detected")
        return None, None
    
    # Categorize lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        if angle < 0:
            angle += 180
        
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if abs(angle - 90) < 30:  # Vertical lines
            vertical_lines.append((x1, y1, x2, y2, length))
        elif abs(angle) < 30 or abs(angle - 180) < 30:  # Horizontal lines
            horizontal_lines.append((x1, y1, x2, y2, length))
    
    print(f"Found {len(horizontal_lines)} horizontal lines, {len(vertical_lines)} vertical lines")
    
    # Debug: print some horizontal lines
    print("Sample horizontal lines (bottom 10):")
    sorted_horiz = sorted(horizontal_lines, key=lambda x: min(x[1], x[3]), reverse=True)
    for i, (x1, y1, x2, y2, length) in enumerate(sorted_horiz[:10]):
        print(f"  Line {i}: y={min(y1,y2)}, length={length:.1f}")
    
    # Find foul line - use the lowest horizontal line as approximation
    if horizontal_lines:
        # Sort by y position (lowest first)
        sorted_horiz = sorted(horizontal_lines, key=lambda x: min(x[1], x[3]), reverse=True)
        foul_line = sorted_horiz[0]  # Lowest y position
        print(f"Using lowest horizontal line as foul line at y ≈ {min(foul_line[1], foul_line[3])}")
    else:
        print("No horizontal lines found")
        return None, None
    
    # Find lane boundaries (long vertical lines)
    lane_left = None
    lane_right = None
    
    if vertical_lines:
        print("Sample vertical lines:")
        sorted_vert = sorted(vertical_lines, key=lambda x: x[4], reverse=True)  # Sort by length
        for i, (x1, y1, x2, y2, length) in enumerate(sorted_vert[:10]):
            print(f"  Vertical {i}: x={min(x1,x2)}, length={length:.1f}")
        
        # Find the leftmost and rightmost long vertical lines
        long_verticals = [line for line in vertical_lines if line[4] > height * 0.2]  # 20% of height
        print(f"Long vertical lines (> {height * 0.2:.0f} pixels): {len(long_verticals)}")
        
        if long_verticals:
            # Sort by x position
            long_verticals.sort(key=lambda x: min(x[0], x[2]))
            lane_left = long_verticals[0]
            lane_right = long_verticals[-1]
            print(f"Lane boundaries: left at x ≈ {min(lane_left[0], lane_left[2])}, right at x ≈ {max(lane_right[0], lane_right[2])}")
        else:
            print("No long vertical lines found, trying lower threshold")
            long_verticals = [line for line in vertical_lines if line[4] > height * 0.1]  # 10% of height
            if long_verticals:
                long_verticals.sort(key=lambda x: min(x[0], x[2]))
                lane_left = long_verticals[0]
                lane_right = long_verticals[-1]
                print(f"Lane boundaries (lower threshold): left at x ≈ {min(lane_left[0], lane_left[2])}, right at x ≈ {max(lane_right[0], lane_right[2])}")
    else:
        print("No vertical lines found")
    
    return foul_line, (lane_left, lane_right)

def create_approach_lane_outline(video_path, output_png="approach_lane_outline.png"):
    """
    Create visualization with approach and lane outlines
    """
    # Find empty frame
    frame = find_empty_lane_frame(video_path)
    if frame is None:
        print("Could not find suitable frame")
        return False

    height, width = frame.shape[:2]

    # Detect boundaries
    foul_line, (lane_left, lane_right) = detect_approach_and_lane_boundaries(frame)
    
    if foul_line is None:
        print("Could not detect foul line")
        return False
    
    if lane_left is None or lane_right is None:
        print("Could not detect lane boundaries")
        return False

    # Draw outlines
    # Approach: from bottom to foul line, full width
    foul_y = min(foul_line[1], foul_line[3])
    cv2.rectangle(frame, (0, height), (width, foul_y), (255, 0, 0), 3)  # Blue rectangle for approach
    cv2.putText(frame, "APPROACH", (width//2 - 100, height - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    
    # Lane: from foul line to top, between lane boundaries
    left_x = min(lane_left[0], lane_left[2])
    right_x = max(lane_right[0], lane_right[2])
    cv2.rectangle(frame, (left_x, foul_y), (right_x, 0), (0, 255, 0), 3)  # Green rectangle for lane
    cv2.putText(frame, "LANE", ((left_x + right_x)//2 - 50, foul_y - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # Draw the detected lines for reference
    if foul_line:
        cv2.line(frame, (foul_line[0], foul_line[1]), (foul_line[2], foul_line[3]), (0, 0, 255), 2)
    if lane_left:
        cv2.line(frame, (lane_left[0], lane_left[1]), (lane_left[2], lane_left[3]), (255, 255, 0), 2)
    if lane_right:
        cv2.line(frame, (lane_right[0], lane_right[1]), (lane_right[2], lane_right[3]), (255, 255, 0), 2)

    cv2.imwrite(output_png, frame)
    print(f"Saved approach and lane outline to: {output_png}")
    return True

def main():
    video_path = "../video/behind/1.MP4"
    create_approach_lane_outline(video_path)

if __name__ == "__main__":
    main()