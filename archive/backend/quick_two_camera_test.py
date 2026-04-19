#!/usr/bin/env python3
"""
Quick Two-Camera Test for BRIO + GoPro
Tests the two configurations you requested
"""

import cv2
import threading
import time
import json
import os

def load_lane_calibration():
    """Load lane camera calibration"""
    calib_file = 'lane_calibration.json'
    if os.path.exists(calib_file):
        with open(calib_file, 'r') as f:
            return json.load(f)
    return None

def apply_perspective_correction(frame, calibration):
    """Apply perspective correction"""
    if not calibration:
        return frame

    height, width = frame.shape[:2]
    dst_points = np.array([
        [100, height-100], [width-100, height-100],
        [100, 100], [width-100, 100]
    ], dtype=np.float32)
    src_points = np.array(calibration['camera_points'], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(frame, matrix, (width, height))

def test_camera_connection(name, camera_type, source):
    """Test if a camera connection works"""
    if camera_type == 'usb':
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"✓ {name}: {width}x{height}")
        return True
    else:
        print(f"❌ {name}: Failed to connect")
        return False

def run_two_camera_test(config_name, camera_configs):
    """Run a two-camera test"""
    print(f"\n{config_name}")
    print("=" * 50)

    # Test connections first
    print("Testing camera connections...")
    connections_ok = True
    for cam_name, config in camera_configs.items():
        if not test_camera_connection(cam_name.upper(), config['type'], config['source']):
            connections_ok = False

    if not connections_ok:
        print("❌ Some cameras failed to connect. Check connections and try again.")
        return False

    # Load calibration
    calibration = load_lane_calibration()
    if calibration:
        print("✓ Lane calibration loaded")
    else:
        print("⚠ No lane calibration (run 'python lane_calibration.py' later)")

    # Start capture threads
    threads = []
    stop_flag = [False]  # Use list to modify from threads

    def capture_thread(camera_name, config):
        camera_type = config['type']
        source = config['source']

        if camera_type == 'usb':
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        window_name = f'{camera_name.upper()} - {config_name}'
        frame_count = 0
        start_time = time.time()

        while not stop_flag[0]:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                # Apply correction to lane camera
                if camera_name == 'lane' and calibration:
                    frame = apply_perspective_correction(frame, calibration)

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_flag[0] = True
                    break
            else:
                break

        cap.release()
        fps = frame_count / (time.time() - start_time)
        print(f"{camera_name.upper()}: {frame_count} frames at {fps:.1f} FPS")

    # Start threads
    for camera_name, config in camera_configs.items():
        thread = threading.Thread(target=capture_thread, args=(camera_name, config))
        threads.append(thread)
        thread.start()

    print("\nCapturing... Press 'q' in any window to stop")

    # Wait for threads
    try:
        while not stop_flag[0] and any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_flag[0] = True

    # Cleanup
    cv2.destroyAllWindows()

    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=2)

    print("✓ Test completed successfully!")
    return True

def main():
    print("BRIO + GoPro Two-Camera Test")
    print("Testing your requested configurations")
    print("=" * 50)

    # BRIO is at index 1 based on current config
    brio_index = 1

    # Determine GoPro connection method
    print("How is your GoPro connected?")
    print("1. USB cable (webcam mode)")
    print("2. WiFi/RTSP stream")
    print("3. Test both methods")

    choice = input("Enter choice (1-3): ").strip()

    configurations = []

    if choice in ['1', '3']:
        # Find GoPro USB index (different from BRIO)
        print("\nFinding GoPro USB camera...")
        gopro_usb = None
        for i in range(5):
            if i != brio_index:  # Skip BRIO index
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    if width < 3000:  # GoPro likely lower resolution than BRIO 4K
                        gopro_usb = i
                        cap.release()
                        break
                cap.release()

        if gopro_usb is None:
            print("Could not auto-detect GoPro. Trying index 0 or 2...")
            gopro_usb = 0 if brio_index != 0 else 2

        print(f"Using GoPro at USB index {gopro_usb}")

        configurations.extend([
            ("Configuration 1: BRIO Lane + GoPro Behind (USB)", {
                'lane': {'type': 'usb', 'source': brio_index},
                'behind': {'type': 'usb', 'source': gopro_usb}
            }),
            ("Configuration 2: BRIO Lane + GoPro Side (USB)", {
                'lane': {'type': 'usb', 'source': brio_index},
                'side': {'type': 'usb', 'source': gopro_usb}
            })
        ])

    if choice in ['2', '3']:
        rtsp_url = input("Enter GoPro RTSP URL (default: rtsp://192.168.1.104:554/stream): ").strip()
        if not rtsp_url:
            rtsp_url = "rtsp://192.168.1.104:554/stream"

        configurations.extend([
            ("Configuration 1: BRIO Lane + GoPro Behind (RTSP)", {
                'lane': {'type': 'usb', 'source': brio_index},
                'behind': {'type': 'rtsp', 'source': rtsp_url}
            }),
            ("Configuration 2: BRIO Lane + GoPro Side (RTSP)", {
                'lane': {'type': 'usb', 'source': brio_index},
                'side': {'type': 'rtsp', 'source': rtsp_url}
            })
        ])

    # Run tests
    results = []
    for config_name, camera_configs in configurations:
        success = run_two_camera_test(config_name, camera_configs)
        results.append((config_name, success))

        if len(configurations) > 1:
            cont = input("\nTest next configuration? (y/n): ").lower()
            if cont != 'y':
                break

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for config_name, success in results:
        status = "✓ WORKING" if success else "❌ FAILED"
        print(f"{status}: {config_name}")

    print("\nTo use the working configuration:")
    print("1. Update CAMERA_CONFIGS in multi_camera_capture.py")
    print("2. Run: python multi_camera_capture.py")

if __name__ == "__main__":
    import numpy as np
    main()