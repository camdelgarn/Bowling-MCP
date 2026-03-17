#!/usr/bin/env python3
"""
Extract frames from video to manually check for dots
"""

import cv2
import os

def extract_frames_for_inspection(video_path, num_frames=10):
    """
    Extract several frames to manually inspect for lane dots
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    print(f"Extracting frames from: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // num_frames

    os.makedirs("inspection_frames", exist_ok=True)

    for i in range(num_frames):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if ret:
            filename = f"inspection_frames/frame_{i:02d}_{frame_idx:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Extracted {filename}")

    cap.release()
    print(f"Extracted {num_frames} frames for inspection")

def main():
    video_path = "../video/behind/1.MP4"
    extract_frames_for_inspection(video_path, 20)  # Extract 20 frames

if __name__ == "__main__":
    main()