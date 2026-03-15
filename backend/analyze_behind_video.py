#!/usr/bin/env python3
"""
Analyze bowling videos from behind camera
Extract frames and analyze board positions
"""

import cv2
import os
import sys

def analyze_video(video_path, output_dir="frames"):
    """
    Analyze a video file and extract frames for analysis
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")

    # Extract frames at regular intervals
    frame_interval = max(1, total_frames // 10)  # Extract ~10 frames
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Extract frame at intervals
        if frame_count % frame_interval == 0 or frame_count == 1:
            frame_filename = f"{output_dir}/frame_{extracted_count:03d}.jpg"
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            print(f"Extracted frame {extracted_count}: {frame_filename}")

            # Analyze the frame for bowling lane features
            analyze_frame(frame, frame_count)

    cap.release()
    print(f"Extracted {extracted_count} frames")
    return True

def analyze_frame(frame, frame_number):
    """
    Analyze a single frame for bowling lane features
    """
    height, width = frame.shape[:2]

    # Look for lane markings, boards, etc.
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Look for horizontal lines (lane boards are typically horizontal)
    edges = cv2.Canny(gray, 50, 150)

    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, 3.14159/180, 100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        print(f"  Frame {frame_number}: Found {len(lines)} line segments")

        # Count horizontal vs vertical lines
        horizontal_lines = 0
        vertical_lines = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < abs(x2 - x1):  # More horizontal
                horizontal_lines += 1
            else:
                vertical_lines += 1

        print(f"    Horizontal lines: {horizontal_lines}, Vertical lines: {vertical_lines}")

        # Look for lane-specific patterns
        # Bowling lanes are 60 feet long, bowler stands around 15-20 feet from pins
        # From behind, we should see the approach and possibly lane markings

    else:
        print(f"  Frame {frame_number}: No significant lines detected")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_video.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    analyze_video(video_file)

if __name__ == "__main__":
    main()