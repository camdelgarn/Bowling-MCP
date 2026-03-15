#!/usr/bin/env python3
"""
RTMP Stream Reader
Reads frames from an RTMP stream and processes them
"""

import cv2
import time
import sys

def read_rtmp_stream(rtmp_url, display=False, save_frames=False, save_video=False, max_frames=10):
    """
    Read frames from RTMP stream

    Args:
        rtmp_url (str): RTMP stream URL (e.g., 'rtmp://10.0.0.57/live/stream')
        display (bool): Whether to display frames in a window
        save_frames (bool): Whether to save frames as images
        save_video (bool): Whether to save as video file
        max_frames (int): Maximum number of frames to read
    """
    print("Connecting to RTMP stream: {}".format(rtmp_url))

    # Open the RTMP stream
    cap = cv2.VideoCapture(rtmp_url)

    if not cap.isOpened():
        print("Failed to open RTMP stream")
        print("Possible issues:")
        print("- RTMP server not running")
        print("- Incorrect URL")
        print("- Firewall blocking RTMP (port 1935)")
        print("- Stream not active")
        return False

    print("Successfully connected to RTMP stream")

    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Stream properties:")
    print("  Resolution: {}x{}".format(width, height))
    print("  FPS: {}".format(fps))

    # Initialize video writer if saving video
    video_writer = None
    if save_video:
        output_file = "rtmp_capture.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        if video_writer.isOpened():
            print("Saving video to: {}".format(output_file))
        else:
            print("Failed to create video writer")
            return False

    frame_count = 0
    start_time = time.time()

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame from stream")
                break

            frame_count += 1

            if save_video and video_writer:
                video_writer.write(frame)

            if save_frames:
                filename = "frame_{:06d}.jpg".format(frame_count)
                cv2.imwrite(filename, frame)
                print("Saved {}".format(filename))

            # Print progress
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print("Read {} frames, actual FPS: {:.2f}".format(frame_count, fps_actual))

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()

    elapsed = time.time() - start_time
    if elapsed > 0:
        avg_fps = frame_count / elapsed
        print("Session summary: {} frames in {:.2f}s ({:.2f} FPS)".format(frame_count, elapsed, avg_fps))

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rtmp_reader.py <rtmp_url> [--save-video]")
        sys.exit(1)

    url = sys.argv[1]
    save_video = "--save-video" in sys.argv

    # For 10 seconds at 30 FPS
    max_frames = 300 if save_video else 10

    success = read_rtmp_stream(url, display=False, save_frames=False, save_video=save_video, max_frames=max_frames)
    if success:
        sys.exit(0)
    else:
        sys.exit(1)