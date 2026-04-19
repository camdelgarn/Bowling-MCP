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
import scipy.signal
import json
import os

# Placeholder for loading and processing video

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

# Placeholder for perspective correction using tape measure frames
def correct_perspective(frame, reference_points):
    # TODO: Implement perspective correction
    return frame

# Placeholder for board detection
def detect_boards(frame, roi_top=0, roi_bottom=None):
    h, w = frame.shape[:2]
    if roi_bottom is None:
        roi_bottom = h
    roi = frame[roi_top:roi_bottom, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    cv2.imwrite('canny_edges_debug.png', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=15)
    board_lines = []
    x_positions = []
    print(f"Raw Hough lines found: {len(lines) if lines is not None else 0}")
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) > 80:
                x_positions.append(x1)
                board_lines.append((x1, y1 + roi_top, x2, y2 + roi_top))
    x_positions = sorted(x_positions)
    merged_x = []
    for x in x_positions:
        if not merged_x or abs(x - merged_x[-1]) > 6:
            merged_x.append(x)
    merged_lines = [(x, roi_top, x, roi_bottom) for x in merged_x]
    print(f"Detected {len(merged_lines)} vertical board lines at x positions: {merged_x}")
    return merged_lines

def detect_boards_vertical_projection(frame, roi_top=0, roi_bottom=None):
    h, w = frame.shape[:2]
    if roi_bottom is None:
        roi_bottom = h
    roi = frame[roi_top:roi_bottom, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Vertical gradient (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.abs(sobelx)
    sobelx = (sobelx / sobelx.max() * 255).astype(np.uint8)
    cv2.imwrite('sobelx_debug.png', sobelx)
    # Vertical projection
    vertical_proj = np.sum(sobelx, axis=0)
    proj_img = np.tile(vertical_proj, (roi.shape[0], 1))
    proj_img = (proj_img / proj_img.max() * 255).astype(np.uint8)
    cv2.imwrite('vertical_projection_debug.png', proj_img)
    # Peak finding
    peaks, _ = scipy.signal.find_peaks(vertical_proj, distance=10, prominence=20)
    print(f"Vertical projection peaks (board candidates): {peaks}")
    board_lines = [(int(x), roi_top, int(x), roi_bottom) for x in peaks]
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
def interactive_select_roi(frame, max_display_height=900):
    # Resize frame for display if too large
    h, w = frame.shape[:2]
    scale = 1.0
    if h > max_display_height:
        scale = max_display_height / h
        display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        display_frame = frame.copy()
    clone = display_frame.copy()
    selected = {'left': None, 'right': None, 'top': None, 'bottom': None, 'count': 0}
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if selected['count'] == 0:
                selected['left'] = x
                cv2.line(clone, (x, 0), (x, clone.shape[0]), (0, 255, 0), 2)
                cv2.imshow('Select Approach ROI', clone)
                print(f"Left edge selected at x={x}. Now click the right edge.")
                selected['count'] += 1
            elif selected['count'] == 1:
                selected['right'] = x
                cv2.line(clone, (x, 0), (x, clone.shape[0]), (255, 0, 0), 2)
                cv2.imshow('Select Approach ROI', clone)
                print(f"Right edge selected at x={x}. Now click the TOP edge.")
                selected['count'] += 1
            elif selected['count'] == 2:
                selected['top'] = y
                cv2.line(clone, (0, y), (clone.shape[1], y), (0, 255, 255), 2)
                cv2.imshow('Select Approach ROI', clone)
                print(f"Top edge selected at y={y}. Now click the BOTTOM edge.")
                selected['count'] += 1
            elif selected['count'] == 3:
                selected['bottom'] = y
                cv2.line(clone, (0, y), (clone.shape[1], y), (255, 0, 255), 2)
                cv2.imshow('Select Approach ROI', clone)
                print(f"Bottom edge selected at y={y}. Press any key to continue.")
                selected['count'] += 1
    cv2.imshow('Select Approach ROI', clone)
    cv2.setMouseCallback('Select Approach ROI', click_event)
    print("Click LEFT edge, RIGHT edge, TOP edge, then BOTTOM edge of the approach, then press any key.")
    cv2.waitKey(0)
    cv2.destroyWindow('Select Approach ROI')
    # Map selected coordinates back to original frame size
    if (selected['left'] is not None and selected['right'] is not None and selected['top'] is not None and selected['bottom'] is not None and selected['left'] < selected['right'] and selected['top'] < selected['bottom']):
        left = int(selected['left'] / scale)
        right = int(selected['right'] / scale)
        top = int(selected['top'] / scale)
        bottom = int(selected['bottom'] / scale)
        return left, right, top, bottom
    else:
        print("ROI selection failed or invalid. Aborting.")
        return None, None, None, None

def save_approach_boards_image(video_path, output_path='approach_boards_detected.png', roi_config_path=None):
    cap = load_video(video_path)
    ret, frame = cap.read()
    if not ret:
        print('Failed to read frame from video.')
        cap.release()
        return
    # Try to load ROI config if provided
    if roi_config_path and os.path.exists(roi_config_path):
        with open(roi_config_path, 'r') as f:
            roi_cfg = json.load(f)
        left_x = roi_cfg['roi']['left']
        right_x = roi_cfg['roi']['right']
        top_y = roi_cfg['roi']['top']
        bottom_y = roi_cfg['roi']['bottom']
        print(f"Loaded ROI from config: left={left_x}, right={right_x}, top={top_y}, bottom={bottom_y}")
    else:
        # Interactive ROI selection (left, right, top, bottom)
        left_x, right_x, top_y, bottom_y = interactive_select_roi(frame)
        if None in (left_x, right_x, top_y, bottom_y):
            print("ROI selection failed. Aborting.")
            cap.release()
            return
    # Crop to selected ROI for board detection
    roi_frame = frame[top_y:bottom_y, left_x:right_x]
    board_lines = detect_boards_vertical_projection(roi_frame, roi_top=0, roi_bottom=None)
    detected_peaks = [x1 for (x1, _, _, _) in board_lines]
    # Estimate average spacing
    if len(detected_peaks) < 2:
        print("Not enough peaks for spacing estimation.")
        cap.release()
        return
    detected_peaks = np.array(detected_peaks)
    spacings = np.diff(detected_peaks)
    avg_spacing = np.median(spacings)
    print(f"Estimated average board spacing: {avg_spacing:.2f} pixels")
    # Generate board positions (in ROI coordinates)
    num_boards = 39
    w_roi = roi_frame.shape[1]
    board_positions = [int(detected_peaks[0] + i * avg_spacing) for i in range(num_boards + 1) if int(detected_peaks[0] + i * avg_spacing) < w_roi]
    # Draw on original frame (convert ROI x/y to global)
    for idx, x in enumerate(board_positions):
        abs_x = x + left_x
        cv2.line(frame, (abs_x, top_y), (abs_x, bottom_y), (0, 255, 255), 1)
        if idx % 5 == 0 or idx == 0:
            cv2.putText(frame, str(idx+1), (abs_x+2, top_y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imwrite(output_path, display_frame)
    print(f'Saved detected boards image to {output_path}')
    cap.release()

def draw_numbered_boards(frame, detected_peaks, roi_top=0, roi_bottom=None, num_boards=39):
    h, w = frame.shape[:2]
    if roi_bottom is None:
        roi_bottom = h
    # If not enough peaks, skip
    if len(detected_peaks) < 2:
        print("Not enough peaks for spacing estimation.")
        return frame
    # Estimate average spacing
    detected_peaks = np.array(detected_peaks)
    spacings = np.diff(detected_peaks)
    avg_spacing = np.median(spacings)
    print(f"Estimated average board spacing: {avg_spacing:.2f} pixels")
    # Use the leftmost peak as the starting point
    start_x = detected_peaks[0]
    # Generate board positions
    board_positions = [int(start_x + i * avg_spacing) for i in range(num_boards + 1) if int(start_x + i * avg_spacing) < w]
    # Draw and label
    for idx, x in enumerate(board_positions):
        cv2.line(frame, (x, roi_top), (x, roi_bottom), (0, 255, 255), 1)
        # Label every 5th board and the first
        if idx % 5 == 0 or idx == 0:
            cv2.putText(frame, str(idx+1), (x+2, roi_top+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame

def interpolate_board_lines_perspective(merged_lines, img_height, num_boards=39):
    # Find the two outermost lane edges (left and right)
    if len(merged_lines) < 2:
        print("Not enough lane edges detected for interpolation.")
        return []
    # Find the leftmost and rightmost lines by their x at the bottom and top
    bottom_y = img_height - 1
    top_y = 0
    left_line = min(merged_lines, key=lambda l: l[0])
    right_line = max(merged_lines, key=lambda l: l[0])
    # Get bottom and top x for each edge
    left_x_bottom = left_line[0]
    left_x_top = left_line[2]
    right_x_bottom = right_line[0]
    right_x_top = right_line[2]
    print(f"Perspective interpolation: left_x_bottom={left_x_bottom}, right_x_bottom={right_x_bottom}, left_x_top={left_x_top}, right_x_top={right_x_top}")
    board_lines = []
    for i in range(num_boards + 1):
        frac = i / num_boards
        x_bottom = int(left_x_bottom + frac * (right_x_bottom - left_x_bottom))
        x_top = int(left_x_top + frac * (right_x_top - left_x_top))
        board_lines.append((x_bottom, bottom_y, x_top, top_y))
    return board_lines

def interactive_select_start_x(frame):
    clone = frame.copy()
    selected = {'x': None}
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected['x'] = x
            cv2.line(clone, (x, 0), (x, frame.shape[0]), (0, 0, 255), 2)
            cv2.imshow('Select First Board', clone)
    cv2.imshow('Select First Board', clone)
    cv2.setMouseCallback('Select First Board', click_event)
    print("Click on the image to select the starting x position for the first board, then press any key.")
    cv2.waitKey(0)
    cv2.destroyWindow('Select First Board')
    return selected['x']

if __name__ == "__main__":
    # Use lane23 video and saved ROI config
    video_file = "C:/video/behind/nobowler/lane23.MP4"
    roi_config = os.path.join(os.path.dirname(__file__), "lane23_roi_config.json")
    print(f"Using video: {video_file}")
    print(f"Using ROI config: {roi_config}")
    save_approach_boards_image(video_file, output_path="approach_boards_detected_lane23.png", roi_config_path=roi_config)
