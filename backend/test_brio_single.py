#!/usr/bin/env python3
"""
Single Camera Test for BRIO 4K
Tests the Logitech BRIO 4K camera with the bowling application setup
"""

import cv2
import time
import json
import os

def load_lane_calibration():
    """Load lane camera calibration for perspective correction"""
    calib_file = 'lane_calibration.json'
    if os.path.exists(calib_file):
        with open(calib_file, 'r') as f:
            return json.load(f)
    return None

def apply_perspective_correction(frame, calibration):
    """Apply perspective correction to lane camera frames"""
    if not calibration:
        return frame

    # Define target rectangle (corrected view)
    height, width = frame.shape[:2]
    dst_points = np.array([
        [100, height-100],  # Bottom left
        [width-100, height-100],  # Bottom right
        [100, 100],  # Top left
        [width-100, 100]  # Top right
    ], dtype=np.float32)

    # Source points from calibration
    src_points = np.array(calibration['camera_points'], dtype=np.float32)

    # Calculate perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transform
    corrected = cv2.warpPerspective(frame, matrix, (width, height))

    return corrected

def test_single_camera():
    """Test the BRIO 4K camera with application processing"""
    print("Testing BRIO 4K Camera for Bowling Application")
    print("=" * 50)

    # Camera configuration - adjust index if needed
    camera_index = 1  # Based on your current config
    print(f"Testing camera at index {camera_index}...")

    # Load calibration if available
    calibration = load_lane_calibration()
    if calibration:
        print("✓ Found lane calibration file")
    else:
        print("⚠ No calibration file found (this is OK for basic testing)")

    # Open camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Failed to open camera!")
        print("\nTroubleshooting:")
        print("- Check USB connection")
        print("- Try different camera index (0, 1, 2, etc.)")
        print("- Close other camera applications")
        print("- Run 'python detect_usb_cameras.py' to find correct index")
        return False

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("✓ Camera opened successfully!")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print("\nTesting frame capture and processing...")

    # Test frame capture and processing
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    window_name = 'BRIO 4K Test - Press Q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Controls:")
    print("- Press 'Q' to quit test")
    print("- Press 'S' to save current frame")
    print("- Press 'C' to toggle perspective correction")
    print("\nMake sure to click on the video window to give it focus!")

    correction_enabled = bool(calibration)

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                frame_count += 1

                # Apply perspective correction if enabled and calibrated
                if correction_enabled and calibration:
                    display_frame = apply_perspective_correction(frame, calibration)
                    processed_count += 1
                else:
                    display_frame = frame

                # Add status text
                status_text = f"Frame: {frame_count}"
                if correction_enabled and calibration:
                    status_text += " | Perspective: ON"
                else:
                    status_text += " | Perspective: OFF"

                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(window_name, display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):
                    filename = f'brio_test_frame_{frame_count}.jpg'
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame as {filename}")
                elif key == ord('c') and calibration:
                    correction_enabled = not correction_enabled
                    status = "ON" if correction_enabled else "OFF"
                    print(f"Perspective correction: {status}")

            else:
                print("Failed to read frame from camera")
                break

            # Limit test to 30 seconds or 300 frames
            if time.time() - start_time > 30 or frame_count >= 300:
                break

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    # Test results
    elapsed = time.time() - start_time
    print("\nTest Results:")
    print(f"✓ Total frames captured: {frame_count}")
    print(f"✓ Frames processed: {processed_count}")
    print(f"✓ Test duration: {elapsed:.1f} seconds")
    print(f"✓ Average FPS: {frame_count/elapsed:.1f}")

    if frame_count > 0:
        print("✓ Camera test PASSED!")
        print("\nNext steps:")
        print("1. Run 'python lane_calibration.py' to calibrate perspective")
        print("2. Update camera index in multi_camera_capture.py if needed")
        print("3. Test with full multi-camera setup when other cameras arrive")
        return True
    else:
        print("❌ Camera test FAILED!")
        return False

if __name__ == "__main__":
    import numpy as np  # Import here to avoid import issues
    success = test_single_camera()
    exit(0 if success else 1)