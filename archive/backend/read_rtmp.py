#!/usr/bin/env python3

import cv2
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_rtmp.py <url>")
        return

    url = sys.argv[1]
    print(f"Connecting to {url}")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Failed to open")
        return

    print("Opened successfully")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    frame_count = 0
    while frame_count < 10:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            print(f"Read frame {frame_count}")
        else:
            print("Failed to read frame")
            break

    cap.release()
    print("Done")

if __name__ == "__main__":
    main()