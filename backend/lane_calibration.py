import cv2
import numpy as np
import json
import os

class LaneCameraCalibrator:
    """
    Calibrate camera for perspective correction on bowling lane
    Use this for tripod-mounted cameras with angled views
    """

    def __init__(self, camera_url='rtsp://192.168.1.101:554/stream'):
        self.camera_url = camera_url
        self.calibration_file = 'lane_calibration.json'

    def capture_calibration_image(self):
        """Capture a frame for calibration"""
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print("Failed to open camera")
            return None

        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite('calibration_frame.jpg', frame)
            print("Calibration frame saved as 'calibration_frame.jpg'")
            return frame
        return None

    def calibrate_perspective(self, frame):
        """
        Interactive calibration - click on lane corners and reference points
        Order: foul line left, foul line right, pin deck left, pin deck right
        """
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                print(f"Point {len(points)}: ({x}, {y})")
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Calibration', frame)

                if len(points) == 4:
                    cv2.destroyAllWindows()
                    self.save_calibration(points)

        print("Click on 4 points in order:")
        print("1. Left edge of foul line")
        print("2. Right edge of foul line")
        print("3. Left edge of pin deck")
        print("4. Right edge of pin deck")

        cv2.imshow('Calibration', frame)
        cv2.setMouseCallback('Calibration', click_event)
        cv2.waitKey(0)

        return points

    def save_calibration(self, points):
        """Save calibration points to file"""
        calibration_data = {
            'camera_points': points,
            'lane_width_pixels': abs(points[1][0] - points[0][0]),  # Distance between foul line points
            'lane_length_pixels': abs(points[2][1] - points[0][1]),  # Distance from foul line to pin deck
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }

        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"Calibration saved to {self.calibration_file}")

    def load_calibration(self):
        """Load existing calibration"""
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r') as f:
                return json.load(f)
        return None

    def apply_perspective_correction(self, frame):
        """Apply perspective correction to a frame"""
        calib = self.load_calibration()
        if not calib:
            print("No calibration found. Run calibration first.")
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
        src_points = np.array(calib['camera_points'], dtype=np.float32)

        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply transform
        corrected = cv2.warpPerspective(frame, matrix, (width, height))

        return corrected

def main():
    calibrator = LaneCameraCalibrator()

    print("Bowling Lane Camera Calibration Tool")
    print("1. Capture calibration frame")
    print("2. Calibrate perspective")
    print("3. Test perspective correction")
    print("4. Exit")

    while True:
        choice = input("Choose option (1-4): ")

        if choice == '1':
            frame = calibrator.capture_calibration_image()
            if frame is not None:
                print("Frame captured. Now run calibration (option 2).")

        elif choice == '2':
            if os.path.exists('calibration_frame.jpg'):
                frame = cv2.imread('calibration_frame.jpg')
                calibrator.calibrate_perspective(frame.copy())
            else:
                print("Capture calibration frame first (option 1).")

        elif choice == '3':
            calib = calibrator.load_calibration()
            if calib:
                # Test with live feed
                cap = cv2.VideoCapture(calibrator.camera_url)
                while True:
                    ret, frame = cap.read()
                    if ret:
                        corrected = calibrator.apply_perspective_correction(frame)
                        cv2.imshow('Original', frame)
                        cv2.imshow('Corrected', corrected)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                cap.release()
                cv2.destroyAllWindows()
            else:
                print("Run calibration first (option 2).")

        elif choice == '4':
            break

if __name__ == "__main__":
    main()