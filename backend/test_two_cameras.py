#!/usr/bin/env python3
"""
Two-Camera Test for Bowling Setup
Tests BRIO 4K + GoPro in different configurations
"""

import cv2
import threading
import time
import numpy as np
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

class TwoCameraProcessor:
    def __init__(self, config_name, camera_configs):
        self.config_name = config_name
        self.camera_configs = camera_configs
        self.processed_frames = {cam: [] for cam in camera_configs.keys()}
        self.is_running = True
        self.lane_calibration = load_lane_calibration()
        self.threads = []

    def process_camera_stream(self, camera_name, config):
        """Process individual camera stream"""
        camera_type = config['type']
        source = config['source']

        if camera_type == 'usb':
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        elif camera_type == 'rtsp':
            cap = cv2.VideoCapture(source)
        else:
            print(f"Unknown camera type for {camera_name}: {camera_type}")
            return

        if not cap.isOpened():
            print(f"❌ Failed to open {camera_name} camera: {source}")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        start_time = time.time()
        window_name = f'{camera_name.upper()} Camera - {self.config_name}'

        print(f"✓ {camera_name.upper()} camera opened successfully")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {camera_name}")
                break

            timestamp = time.time()
            frame_count += 1

            # Apply perspective correction for lane camera
            if camera_name == 'lane' and self.lane_calibration:
                frame = apply_perspective_correction(frame, self.lane_calibration)

            self.processed_frames[camera_name].append({
                'timestamp': timestamp,
                'frame': frame,
                'camera': camera_name,
                'detections': []
            })

            # Display frame
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break

        cap.release()
        fps = frame_count / (time.time() - start_time)
        print(f"{camera_name.upper()} stream ended. {frame_count} frames at {fps:.1f} FPS")

    def start_capture(self):
        """Start two-camera capture"""
        print(f"\nStarting {self.config_name} configuration...")
        print("Press 'q' in any camera window to stop all cameras")
        print("-" * 50)

        for camera_name, config in self.camera_configs.items():
            thread = threading.Thread(
                target=self.process_camera_stream,
                args=(camera_name, config),
                daemon=True
            )
            self.threads.append(thread)
            thread.start()

        # Wait for threads
        try:
            while self.is_running and any(t.is_alive() for t in self.threads):
                time.sleep(1)
        except KeyboardInterrupt:
            self.is_running = False

        # Cleanup
        cv2.destroyAllWindows()

        # Show results
        self.show_results()

    def show_results(self):
        """Show test results"""
        print(f"\n{self.config_name} Test Results:")
        print("=" * 40)

        total_frames = 0
        for camera_name, frames in self.processed_frames.items():
            frame_count = len(frames)
            total_frames += frame_count
            print(f"✓ {camera_name.upper()}: {frame_count} frames captured")

        if total_frames > 0:
            print(f"\n✓ Two-camera test PASSED! Total: {total_frames} frames")
            print("\nConfiguration working correctly!")
        else:
            print("\n❌ Test FAILED - no frames captured")

def test_configuration(config_name, camera_configs):
    """Test a specific two-camera configuration"""
    print(f"\nTesting Configuration: {config_name}")
    print("=" * 60)

    processor = TwoCameraProcessor(config_name, camera_configs)
    processor.start_capture()

    return len(processor.processed_frames['lane']) > 0  # Return success if lane camera worked

def main():
    print("Two-Camera Bowling Setup Test")
    print("Testing BRIO 4K + GoPro configurations")
    print("=" * 50)

    # First determine GoPro connection method
    print("How is your GoPro connected?")
    print("1. USB connection (appears as webcam)")
    print("2. RTSP stream (network/WiFi)")
    print("3. Test both methods")

    while True:
        try:
            choice = int(input("Enter choice (1-3): "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Enter a number.")

    # Test configurations
    configurations = []

    if choice == 1 or choice == 3:
        # USB GoPro configurations
        print("\nTesting with GoPro as USB camera...")
        print("Make sure GoPro is connected via USB and set to webcam mode")

        # Find GoPro USB index
        gopro_usb_index = None
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Check if this might be the GoPro (different resolution than BRIO)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                if width != 3840:  # BRIO is 4K, GoPro might be different
                    gopro_usb_index = i
                cap.release()

        if gopro_usb_index is None:
            print("Could not find GoPro USB camera. Trying index 2...")
            gopro_usb_index = 2

        print(f"Using GoPro at USB index {gopro_usb_index}")

        configurations.extend([
            ("BRIO Lane + GoPro Behind (USB)", {
                'lane': {'type': 'usb', 'source': 1},  # BRIO
                'behind': {'type': 'usb', 'source': gopro_usb_index}  # GoPro
            }),
            ("BRIO Lane + GoPro Side (USB)", {
                'lane': {'type': 'usb', 'source': 1},  # BRIO
                'side': {'type': 'usb', 'source': gopro_usb_index}  # GoPro
            })
        ])

    if choice == 2 or choice == 3:
        # RTSP GoPro configurations
        print("\nTesting with GoPro as RTSP stream...")
        print("Make sure GoPro is connected to WiFi and RTSP is enabled")

        rtsp_url = input("Enter GoPro RTSP URL (or press Enter for default): ").strip()
        if not rtsp_url:
            rtsp_url = "rtsp://192.168.1.104:554/stream"  # Default

        configurations.extend([
            ("BRIO Lane + GoPro Behind (RTSP)", {
                'lane': {'type': 'usb', 'source': 1},  # BRIO
                'behind': {'type': 'rtsp', 'source': rtsp_url}  # GoPro
            }),
            ("BRIO Lane + GoPro Side (RTSP)", {
                'lane': {'type': 'usb', 'source': 1},  # BRIO
                'side': {'type': 'rtsp', 'source': rtsp_url}  # GoPro
            })
        ])

    # Run tests
    results = []
    for config_name, camera_configs in configurations:
        success = test_configuration(config_name, camera_configs)
        results.append((config_name, success))

        # Ask if user wants to continue
        if len(configurations) > 1:
            cont = input("\nTest next configuration? (y/n): ").lower()
            if cont != 'y':
                break

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for config_name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"{status}: {config_name}")

    print("\nNext steps:")
    print("1. Choose your preferred working configuration")
    print("2. Update multi_camera_capture.py with the working camera settings")
    print("3. Run lane calibration: python lane_calibration.py")
    print("4. Test with AI inference when ready")

if __name__ == "__main__":
    main()