# detect_approach_boards.py
"""
Detects approach boards and dot positions in bowling lane videos.
Uses visual differences to identify boards and dots, leveraging tape measure videos for perspective calibration.

Assumptions:
- Boards are visually distinct and can be detected via image processing (e.g., edge detection, intensity changes).
- Dots are visible and can be detected as circular features.
- Tape measure videos are used to help with perspective correction, not for direct pixel-to-distance mapping.

Steps:
1. Load video and extract frames.
2. Apply perspective correction using tape measure reference frames.
3. Detect boards using edge or line detection.
4. Detect dots using circle detection (e.g., HoughCircles).
5. Identify center board (board 20) using the largest dot in each set.
6. Map detected dots to board numbers.

This script is a scaffold for further development.
"""

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import time

# Placeholder for loading and processing video

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

# Placeholder for perspective correction using tape measure frames
def correct_perspective(frame, reference_points):
    # TODO: Implement perspective correction
    return frame

# Placeholder for board detection
def detect_boards(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    board_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Only keep nearly vertical lines (boards)
            if abs(x2 - x1) < 20 and abs(y2 - y1) > 50:
                board_lines.append((x1, y1, x2, y2))
    return board_lines

# Placeholder for dot detection
def detect_dots(frame, tape_left=None, tape_right=None, roi_top=1790, roi_bottom=1860):
    h, w = frame.shape[:2]
    # If tape measure bounds are given, use them for horizontal ROI
    roi_left = tape_left if tape_left is not None else 0
    roi_right = tape_right if tape_right is not None else w
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    print(f"Dot detection ROI: left={roi_left}, right={roi_right}, top={roi_top}, bottom={roi_bottom}")
    cv2.imwrite('dot_detection_roi.png', roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=50, param2=30, minRadius=10, maxRadius=40)
    dots = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Adjust x and y coordinates back to original frame
            x, y, r = i[0] + roi_left, i[1] + roi_top, i[2]
            dots.append((x, y, r))
    # Sort by x (horizontal position) to get left-to-right order
    dots = sorted(dots, key=lambda d: d[0])
    print(f"Detected dot coordinates (x, y, r): {dots}")
    return dots[:5], (roi_top, roi_bottom)

# Function to extract tape measure region and perform OCR

def ocr_tape_measure(frame, boards=None):
    h, w = frame.shape[:2]
    # Default ROI values
    strip_width = int(w * 0.08)  # 8% of frame width
    strip_left = int(w * 0.05)   # 5% from left (adjust this to move ROI horizontally)
    strip_right = strip_left + strip_width
    # If boards are detected, use their min/max x as the ROI
    if boards and len(boards) > 0:
        x_coords = [x1 for (x1, _, x2, _) in boards] + [x2 for (x1, _, x2, _) in boards]
        min_x = max(0, min(x_coords) - 10)
        max_x = min(w, max(x_coords) + 10)
        strip_left = min_x
        strip_right = max_x
    roi_top = h - int(h * 0.18) # 18% from bottom (adjust to move ROI vertically)
    roi_bottom = roi_top + int(h * 0.08) # 8% of frame height
    tape_roi = frame[roi_top:roi_bottom, strip_left:strip_right]
    print(f"OCR ROI: left={strip_left}, right={strip_right}, top={roi_top}, bottom={roi_bottom}")
    cv2.imwrite('tape_measure_ocr_raw.png', tape_roi)
    gray = cv2.cvtColor(tape_roi, cv2.COLOR_BGR2GRAY)
    # Try both OTSU and adaptive thresholding
    _, thresh_otsu = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    # Optionally resize for better OCR
    scale = 2
    resized_otsu = cv2.resize(thresh_otsu, (tape_roi.shape[1]*scale, tape_roi.shape[0]*scale))
    resized_adapt = cv2.resize(thresh_adapt, (tape_roi.shape[1]*scale, tape_roi.shape[0]*scale))
    cv2.imwrite('tape_measure_ocr_thresh_otsu.png', resized_otsu)
    cv2.imwrite('tape_measure_ocr_thresh_adapt.png', resized_adapt)
    # OCR both versions
    config = '--psm 6 digits'
    text_otsu = pytesseract.image_to_string(resized_otsu, config=config)
    text_adapt = pytesseract.image_to_string(resized_adapt, config=config)
    print('OCR tape measure region (OTSU):', repr(text_otsu))
    print('OCR tape measure region (ADAPT):', repr(text_adapt))
    # Prefer the one with more digits
    text = text_otsu if sum(c.isdigit() for c in text_otsu) >= sum(c.isdigit() for c in text_adapt) else text_adapt
    if any(char.isdigit() for char in text):
        print("OCR found numbers:", text)
    else:
        print("OCR did not find any numbers.")
    return text

# Function to detect the yellow tape measure by color thresholding in HSV space
def detect_tape_measure(frame, roi_top=1790, roi_bottom=1860):
    roi = frame[roi_top:roi_bottom, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Widened yellow color range in HSV
    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Add timestamp to filenames for uniqueness
    ts = int(time.time())
    mask_filename = f'tape_measure_yellow_mask_{ts}.png'
    detected_filename = f'tape_measure_yellow_detected_{ts}.png'
    cv2.imwrite(mask_filename, mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (assume it's the tape measure)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        left = x
        right = x + w
        print(f"Tape measure detected: left={left}, right={right} (in ROI coordinates)")
        tape_roi_vis = roi.copy()
        cv2.rectangle(tape_roi_vis, (left, 0), (right, roi.shape[0]-1), (0, 255, 255), 2)
        cv2.imwrite(detected_filename, tape_roi_vis)
        # Return as absolute coordinates in the frame
        return left, right, roi_top, roi_bottom
    else:
        print("Tape measure not detected by color thresholding.")
        cv2.imwrite(detected_filename, roi)
        return None

# Main processing function
def process_approach_video(video_path):
    cap = load_video(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boards = detect_boards(frame)
        dots = detect_dots(frame)
        # Draw detected boards
        for (x1, y1, x2, y2) in boards:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw detected dots (first 5)
        for (x, y, r) in dots:
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
        # Resize frame for display
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Approach Boards and Dots', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to save approach boards and dots image
def save_approach_boards_image(video_path, output_path='approach_boards_detected.png'):
    cap = load_video(video_path)
    ret, frame = cap.read()
    if not ret:
        print('Failed to read frame from video.')
        cap.release()
        return
    # Detect tape measure in the ROI first
    tape_result = detect_tape_measure(frame, roi_top=1750, roi_bottom=1900)
    tape_left, tape_right = (tape_result[0], tape_result[1]) if tape_result else (None, None)
    # Use tape measure bounds for dot detection
    dots, (roi_top, roi_bottom) = detect_dots(frame, tape_left=tape_left, tape_right=tape_right, roi_top=1750, roi_bottom=1900)
    boards = detect_boards(frame)
    for (x1, y1, x2, y2) in boards:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x, y, r) in dots:
        cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
    cv2.rectangle(frame, (tape_left if tape_left else 0, 1750), (tape_right if tape_right else frame.shape[1], 1900), (255, 255, 0), 2)
    ocr_tape_measure(frame, boards)
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imwrite(output_path, display_frame)
    print(f'Saved detected boards and dots image to {output_path}')
    cap.release()

if __name__ == "__main__":
    # Example usage
    # process_approach_video("C:/video/lane/tapemeasure/39inch_front_dots_tape.MP4")
    # To save a PNG with detected boards and dots, uncomment below:
    save_approach_boards_image("C:/video/lane/tapemeasure/39inch_front_dots_tape.MP4")
