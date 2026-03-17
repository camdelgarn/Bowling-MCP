import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
import processing

def analyze_single_image(image_path, output_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not load image")
        return

    # Detect objects
    ball_detections = processing.detect_ball_combined(frame)
    lane_detections = processing.detect_lane(frame)
    rack_detections = processing.detect_rack(frame)
    approach_detections = processing.detect_approach(frame)

    # Draw detections
    for detection in ball_detections:
        cv2.circle(frame, (int(detection['center_x']), int(detection['center_y'])), int(detection['radius']), (0, 255, 0), 2)
        cv2.putText(frame, 'Ball', (int(detection['center_x'] - detection['radius']), int(detection['center_y'] - detection['radius'] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for line in lane_detections:
        cv2.line(frame, (line['x1'], line['y1']), (line['x2'], line['y2']), (255, 0, 0), 2)

    for rect in rack_detections:
        cv2.rectangle(frame, (rect['x'], rect['y']), (rect['x'] + rect['w'], rect['y'] + rect['h']), (0, 0, 255), 2)
        cv2.putText(frame, 'Rack', (rect['x'], rect['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for rect in approach_detections:
        cv2.rectangle(frame, (rect['x'], rect['y']), (rect['x'] + rect['w'], rect['y'] + rect['h']), (255, 255, 0), 2)
        cv2.putText(frame, 'Approach', (rect['x'], rect['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imwrite(output_path, frame)
    print(f"Saved analyzed image to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python analyze_single_image.py input.jpg output.jpg")
        sys.exit(1)
    analyze_single_image(sys.argv[1], sys.argv[2])