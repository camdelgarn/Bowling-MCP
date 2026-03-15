#!/usr/bin/env python3
"""
USB Camera Detection Script
Helps find the device index for USB cameras like Logitech BRIO 4K

Note: On Windows, OpenCV windows may sometimes not respond to key presses.
If the visual test doesn't work, use the simple console test instead.
Make sure to click on the OpenCV window to give it focus before pressing keys.
"""

import cv2
import sys

def detect_usb_cameras():
    """Detect available USB cameras and their indices"""
    print("Detecting USB cameras...")
    print("Note: Close other camera applications first")
    print("-" * 40)

    found_cameras = []

    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows

        if cap.isOpened():
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"✓ Camera found at index {i}:")
            print(f"  Resolution: {int(width)}x{int(height)}")
            print(f"  FPS: {fps}")
            print()

            found_cameras.append(i)
            cap.release()
        else:
            print(f"✗ No camera at index {i}")

    return found_cameras

def test_camera(index):
    """Test a specific camera index with visual feedback"""
    print(f"\nTesting camera at index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Failed to open camera")
        return

    print("Camera opened successfully!")
    print("Press 'q' in the video window to quit test")
    print("Press 's' in the video window to save a test frame")
    print("Make sure the OpenCV window has focus (click on it)")

    window_name = f'Camera {index} Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # 27 is ESC key
                print("Quitting test...")
                break
            elif key == ord('s'):
                filename = f'camera_{index}_test.jpg'
                cv2.imwrite(filename, frame)
                print(f"Test frame saved as {filename}")
        else:
            print("Failed to read frame")
            break

    cap.release()
    cv2.destroyWindow(window_name)
    print("Test completed.")

def test_camera_simple(index):
    """Simple test without OpenCV window (console-based)"""
    print(f"\nTesting camera at index {index} (simple mode)...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Failed to open camera")
        return False

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Camera opened successfully!")
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    # Capture a few frames
    frame_count = 0
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            frame_count += 1
        else:
            break

    cap.release()

    if frame_count > 0:
        print(f"Successfully captured {frame_count}/10 frames")
        return True
    else:
        print("Failed to capture any frames")
        return False

if __name__ == "__main__":
    print("USB Camera Detection Tool")
    print("=" * 40)

    cameras = detect_usb_cameras()

    if not cameras:
        print("\nNo cameras detected!")
        print("Troubleshooting:")
        print("- Ensure camera is connected and powered on")
        print("- Close other applications using the camera")
        print("- Try a different USB port")
        print("- Install camera drivers if needed")
        sys.exit(1)

    print(f"\nFound {len(cameras)} camera(s)")

    if len(cameras) == 1:
        print("\nChoose test mode:")
        print("1. Visual test (with video window)")
        print("2. Simple test (console only)")
        try:
            mode = int(input("Enter mode (1 or 2): "))
            if mode == 1:
                test_camera(cameras[0])
            elif mode == 2:
                test_camera_simple(cameras[0])
            else:
                print("Invalid mode")
        except ValueError:
            print("Invalid input")
    else:
        print("Available cameras:")
        for i, cam_idx in enumerate(cameras):
            print(f"{i+1}. Index {cam_idx}")

        try:
            choice = int(input("Enter camera number to test (0 to skip): "))
            if choice > 0 and choice <= len(cameras):
                print("\nChoose test mode:")
                print("1. Visual test (with video window)")
                print("2. Simple test (console only)")
                try:
                    mode = int(input("Enter mode (1 or 2): "))
                    if mode == 1:
                        test_camera(cameras[choice-1])
                    elif mode == 2:
                        test_camera_simple(cameras[choice-1])
                    else:
                        print("Invalid mode")
                except ValueError:
                    print("Invalid input")
        except ValueError:
            print("Invalid choice")

    print("\nUpdate your CAMERA_CONFIGS in multi_camera_capture.py with the correct index!")