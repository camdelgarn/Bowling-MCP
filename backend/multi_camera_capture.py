import cv2
import threading
import time
import numpy as np
import json
import os

# Camera configuration - adjust these URLs for your setup
CAMERA_CONFIGS = {
    'lane': {'type': 'usb', 'source': 1},  # Logitech BRIO 4K via USB (camera index 0)
    'side': {'type': 'rtsp', 'source': 'rtsp://192.168.1.102:554/stream'},  # RTSP camera
    'behind': {'type': 'rtsp', 'source': 'rtsp://192.168.1.103:554/stream'}  # RTSP camera
}

class MultiCameraProcessor:
    def __init__(self):
        self.cameras = {}
        self.processed_frames = {cam: [] for cam in CAMERA_CONFIGS.keys()}
        self.is_running = True
        self.lane_calibration = self.load_lane_calibration()

    def load_lane_calibration(self):
        """Load lane camera calibration for perspective correction"""
        calib_file = 'lane_calibration.json'
        if os.path.exists(calib_file):
            with open(calib_file, 'r') as f:
                return json.load(f)
        return None

    def process_camera_stream(self, camera_name, config):
        """Process individual camera stream (USB or RTSP)"""
        camera_type = config['type']
        source = config['source']

        if camera_type == 'usb':
            cap = cv2.VideoCapture(source)  # USB camera index
        elif camera_type == 'rtsp':
            cap = cv2.VideoCapture(source)  # RTSP URL
        elif camera_type == 'rtmp':
            cap = cv2.VideoCapture(source)  # RTMP stream URL
        else:
            print(f"Unknown camera type: {camera_type}")
            return

        if not cap.isOpened():
            print(f"Failed to open {camera_name} camera: {source}")
            return

        # Set camera properties for consistent capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        start_time = time.time()

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {camera_name}")
                break

            timestamp = time.time()
            frame_count += 1

            # Process frame with AI (you can add your detection logic here)
            processed_frame = self.process_frame_with_ai(frame, camera_name, timestamp)

            self.processed_frames[camera_name].append(processed_frame)

            # Optional: Display frame (comment out for headless operation)
            cv2.imshow(f'{camera_name} Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        fps = frame_count / (time.time() - start_time)
        print(f"{camera_name} stream ended. Processed {frame_count} frames at {fps:.1f} FPS")

    def apply_perspective_correction(self, frame):
        """Apply perspective correction to lane camera frames"""
        if not self.lane_calibration:
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
        src_points = np.array(self.lane_calibration['camera_points'], dtype=np.float32)

        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply transform
        corrected = cv2.warpPerspective(frame, matrix, (width, height))

        return corrected

    def process_frame_with_ai(self, frame, camera_name, timestamp):
        """Process frame with AI inference"""
        # Apply perspective correction for lane camera if calibrated
        if camera_name == 'lane' and self.lane_calibration:
            frame = self.apply_perspective_correction(frame)

        # For now, just return the frame with timestamp
        # You can integrate your Roboflow workflow here
        return {
            'timestamp': timestamp,
            'frame': frame,
            'camera': camera_name,
            'detections': []  # Add your detection results
        }

    def start_capture(self):
        """Start multi-camera capture"""
        threads = []
        for camera_name, config in CAMERA_CONFIGS.items():
            thread = threading.Thread(
                target=self.process_camera_stream,
                args=(camera_name, config),
                daemon=True
            )
            threads.append(thread)
            thread.start()

        # Wait for threads or handle main loop
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.is_running = False

        # Cleanup
        cv2.destroyAllWindows()

        # Save processed data
        self.save_results()

    def save_results(self):
        """Save processed frames or results"""
        for camera_name, frames in self.processed_frames.items():
            print(f"{camera_name}: {len(frames)} frames processed")

        # Add your saving logic here (e.g., save to video files, JSON, etc.)

if __name__ == "__main__":
    processor = MultiCameraProcessor()
    print("Starting multi-camera bowling capture...")
    print("Press Ctrl+C to stop")
    processor.start_capture()