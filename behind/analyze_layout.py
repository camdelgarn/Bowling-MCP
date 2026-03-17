#!/usr/bin/env python3
"""
Analyze lane_layout2.png to find the white lane outline, then apply it to the video.
The PNG is a screenshot/crop from the video with white lines drawn showing the lane edges.
"""

import cv2
import numpy as np
import os


def analyze_layout_white_lines(layout_path):
    """
    Find the white line pixels in the layout PNG.
    Returns the pixel coordinates of the white lines.
    """
    layout = cv2.imread(layout_path)
    if layout is None:
        print(f"Failed to load: {layout_path}")
        return None

    h, w = layout.shape[:2]
    print(f"Layout image size: {w}x{h}")

    # Convert to grayscale
    gray = cv2.cvtColor(layout, cv2.COLOR_BGR2GRAY)

    # Threshold for bright white pixels (the drawn lines)
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Save the mask so we can see what we're detecting
    cv2.imwrite("debug_white_mask.png", white_mask)

    # Get all white pixel coordinates
    white_pixels = np.where(white_mask > 0)
    if len(white_pixels[0]) == 0:
        print("No white pixels found, trying lower threshold")
        _, white_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        white_pixels = np.where(white_mask > 0)

    if len(white_pixels[0]) == 0:
        print("Still no white pixels found")
        return None

    ys = white_pixels[0]
    xs = white_pixels[1]
    print(f"Found {len(xs)} white pixels")
    print(f"  X range: {xs.min()} to {xs.max()}")
    print(f"  Y range: {ys.min()} to {ys.max()}")

    # For each row, find the leftmost and rightmost white pixel
    # This gives us the left and right edges of the lane outline
    left_edge = {}
    right_edge = {}
    for y_val in range(ys.min(), ys.max() + 1):
        row_xs = xs[ys == y_val]
        if len(row_xs) > 0:
            left_edge[y_val] = int(row_xs.min())
            right_edge[y_val] = int(row_xs.max())

    # Print some sample rows
    sample_rows = sorted(left_edge.keys())
    step = max(1, len(sample_rows) // 20)
    print("\nSample lane edges (y -> left_x, right_x, width):")
    for i in range(0, len(sample_rows), step):
        y = sample_rows[i]
        lx = left_edge[y]
        rx = right_edge[y]
        print(f"  y={y}: left={lx}, right={rx}, width={rx - lx}")

    return {
        'layout_w': w,
        'layout_h': h,
        'left_edge': left_edge,
        'right_edge': right_edge,
        'y_min': int(ys.min()),
        'y_max': int(ys.max()),
    }


def analyze_layout_blue_lines(layout_path):
    """
    Find the blue line pixels in the layout PNG (approach outline).
    Returns the pixel coordinates of the blue lines.
    """
    layout = cv2.imread(layout_path)
    if layout is None:
        print(f"Failed to load: {layout_path}")
        return None

    h, w = layout.shape[:2]

    # Convert to HSV to isolate blue
    hsv = cv2.cvtColor(layout, cv2.COLOR_BGR2HSV)

    # Blue in HSV: hue ~90-130, decent saturation and value
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.imwrite("debug_blue_mask.png", blue_mask)

    blue_pixels = np.where(blue_mask > 0)
    if len(blue_pixels[0]) == 0:
        print("No blue pixels found")
        return None

    ys = blue_pixels[0]
    xs = blue_pixels[1]
    print(f"Found {len(xs)} blue pixels")
    print(f"  X range: {xs.min()} to {xs.max()}")
    print(f"  Y range: {ys.min()} to {ys.max()}")

    # For each row, find the leftmost and rightmost blue pixel
    left_edge = {}
    right_edge = {}
    for y_val in range(ys.min(), ys.max() + 1):
        row_xs = xs[ys == y_val]
        if len(row_xs) > 0:
            left_edge[y_val] = int(row_xs.min())
            right_edge[y_val] = int(row_xs.max())

    sample_rows = sorted(left_edge.keys())
    step = max(1, len(sample_rows) // 15)
    print("\nSample approach edges (y -> left_x, right_x, width):")
    for i in range(0, len(sample_rows), step):
        y = sample_rows[i]
        lx = left_edge[y]
        rx = right_edge[y]
        print(f"  y={y}: left={lx}, right={rx}, width={rx - lx}")

    return {
        'layout_w': w,
        'layout_h': h,
        'left_edge': left_edge,
        'right_edge': right_edge,
        'y_min': int(ys.min()),
        'y_max': int(ys.max()),
    }


def find_empty_frame(video_path):
    """
    Scan video for a frame without a person (bowler).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nVideo: {vid_w}x{vid_h}, {total} frames, {fps:.1f} fps")

    # Check a range of frames, looking for the one with the least "person" activity
    best_frame = None
    best_score = float('inf')
    best_idx = -1

    # Scan a range around frame 730 which worked before,
    # plus a few other spots
    candidates = list(range(720, 745)) + list(range(0, 30))

    for idx in candidates:
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Look at bottom half for dark blobs (person)
        bottom = gray[vid_h // 2:, :]
        _, thresh = cv2.threshold(bottom, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        score = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 5000)

        if score < best_score:
            best_score = score
            best_frame = frame.copy()
            best_idx = idx

    cap.release()
    print(f"Best empty frame: {best_idx} (person score={best_score:.0f})")
    return best_frame, best_idx


def main():
    layout_path = "lane_layout2.png"
    video_path = "../video/behind/1.MP4"

    # Step 1: Analyze the layout PNG to find white line positions (lane)
    print("=== Analyzing lane_layout2.png - WHITE lines (lane) ===")
    info = analyze_layout_white_lines(layout_path)
    if info is None:
        print("Could not find white lines in layout")
        return

    # Step 1b: Analyze blue line positions (approach)
    print("\n=== Analyzing lane_layout2.png - BLUE lines (approach) ===")
    approach_info = analyze_layout_blue_lines(layout_path)
    if approach_info is None:
        print("Could not find blue lines in layout")

    # Step 2: Find an empty frame in the video
    print("\n=== Finding empty frame in video ===")
    frame, frame_idx = find_empty_frame(video_path)
    if frame is None:
        print("Could not find empty frame")
        return

    vid_h, vid_w = frame.shape[:2]

    # Save the raw empty frame for reference
    cv2.imwrite("empty_frame.png", frame)
    print(f"Saved empty frame (#{frame_idx}) to empty_frame.png")

    # Step 3: The layout PNG is a crop/screenshot from the same video.
    # We need to figure out how the layout coordinates map to the full video.
    # The layout PNG might be the same resolution as the video, or a crop.
    # Let's try to find where the layout matches in the video using template matching.
    layout_img = cv2.imread(layout_path)
    layout_h, layout_w = layout_img.shape[:2]

    print(f"\nLayout: {layout_w}x{layout_h}, Video frame: {vid_w}x{vid_h}")

    if layout_w == vid_w and layout_h == vid_h:
        # Same size - direct mapping
        print("Layout is same size as video - direct coordinate mapping")
        scale_x, scale_y = 1.0, 1.0
        offset_x, offset_y = 0, 0
    else:
        # Layout is a crop/smaller version - use template matching to find position
        print("Layout is a different size - using template matching to locate it in the video")

        # Convert both to grayscale for matching
        layout_gray = cv2.cvtColor(layout_img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try multiple scales since the layout might be scaled
        best_match_val = -1
        best_scale = 1.0
        best_loc = (0, 0)

        for scale in np.arange(0.5, 5.0, 0.1):
            new_w = int(layout_w * scale)
            new_h = int(layout_h * scale)
            if new_w > vid_w or new_h > vid_h or new_w < 50 or new_h < 50:
                continue
            resized = cv2.resize(layout_gray, (new_w, new_h))
            result = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_match_val:
                best_match_val = max_val
                best_scale = scale
                best_loc = max_loc

        print(f"Best template match: scale={best_scale:.2f}, loc={best_loc}, score={best_match_val:.3f}")

        scale_x = best_scale
        scale_y = best_scale
        offset_x = best_loc[0]
        offset_y = best_loc[1]

    # Step 4: Build clean 4-corner trapezoids for both lane and approach.
    # Both are rectangles in 3D, appearing as trapezoids due to perspective.
    # Use the white line data to build the lane and derive the approach.
    # Camera perspective: bottom of frame = closest to camera = widest.
    # The lane is the long rectangle going away from camera (narrower at top).
    # The approach is between the camera and the foul line (bottom of frame).
    # The foul line is the boundary between lane and approach.
    overlay = frame.copy()

    # --- Extract lane corners from white line data ---
    # The white data has:
    #   - Lane side edges from ~y=141 to ~y=247 (converging upward toward pins)
    #   - A horizontal bottom line at ~y=820-827 (the foul line end of the lane)
    # We need to connect these to form the full lane trapezoid.

    rows = sorted(info['left_edge'].keys())

    # Separate the side-edge rows from the bottom horizontal line rows.
    # There's a big gap in the data between ~y=249 and ~y=820.
    # Find that gap to split the data.
    side_rows = []
    bottom_rows = []
    prev_y = None
    for y in rows:
        if prev_y is not None and y - prev_y > 50:
            # Big gap - everything after this is the bottom line
            bottom_rows.append(y)
        elif bottom_rows:
            bottom_rows.append(y)
        else:
            side_rows.append(y)
        prev_y = y

    # Filter side rows: only keep rows where the width is reasonable for lane edges
    # (not the noise/scattered pixels at the very top)
    clean_side_rows = [y for y in side_rows
                       if 30 < (info['right_edge'][y] - info['left_edge'][y]) < 500]

    if not clean_side_rows:
        print("Could not find clean lane side edges")
    else:
        # Top of lane (pins end) - average first few clean rows
        n = max(3, len(clean_side_rows) // 20)
        top_rows_avg = clean_side_rows[:n]
        lane_top_y = clean_side_rows[0]
        lane_top_left = int(np.mean([info['left_edge'][r] for r in top_rows_avg]))
        lane_top_right = int(np.mean([info['right_edge'][r] for r in top_rows_avg]))

        # Bottom of lane (foul line) - use the horizontal bottom line
        if bottom_rows:
            bot_rows_avg = bottom_rows[-min(n, len(bottom_rows)):]
            lane_bot_y = int(np.mean(bottom_rows))
            lane_bot_left = int(np.mean([info['left_edge'][r] for r in bot_rows_avg]))
            lane_bot_right = int(np.mean([info['right_edge'][r] for r in bot_rows_avg]))
        else:
            # Fallback: extrapolate from side edge data
            bot_rows_avg = clean_side_rows[-n:]
            lane_bot_y = clean_side_rows[-1]
            lane_bot_left = int(np.mean([info['left_edge'][r] for r in bot_rows_avg]))
            lane_bot_right = int(np.mean([info['right_edge'][r] for r in bot_rows_avg]))

        print(f"\nLane (layout coords):")
        print(f"  Top (pins):     y={lane_top_y}, left={lane_top_left}, right={lane_top_right}, width={lane_top_right - lane_top_left}")
        print(f"  Bot (foul line): y={lane_bot_y}, left={lane_bot_left}, right={lane_bot_right}, width={lane_bot_right - lane_bot_left}")

        # Map lane to video coordinates
        def to_vid(x, y):
            return (int(x * scale_x + offset_x), int(y * scale_y + offset_y))

        v_lane_tl = to_vid(lane_top_left, lane_top_y)
        v_lane_tr = to_vid(lane_top_right, lane_top_y)
        v_lane_bl = to_vid(lane_bot_left, lane_bot_y)
        v_lane_br = to_vid(lane_bot_right, lane_bot_y)

        # Draw lane in green
        cv2.line(overlay, v_lane_tl, v_lane_tr, (0, 255, 0), 3)
        cv2.line(overlay, v_lane_bl, v_lane_br, (0, 255, 0), 3)
        cv2.line(overlay, v_lane_tl, v_lane_bl, (0, 255, 0), 3)
        cv2.line(overlay, v_lane_tr, v_lane_br, (0, 255, 0), 3)

        mid_x = (v_lane_tl[0] + v_lane_tr[0] + v_lane_bl[0] + v_lane_br[0]) // 4
        mid_y = (v_lane_tl[1] + v_lane_bl[1]) // 2
        cv2.putText(overlay, "LANE", (mid_x - 40, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        print(f"  Video top: {v_lane_tl} - {v_lane_tr}")
        print(f"  Video bot: {v_lane_bl} - {v_lane_br}")

        # Draw foul line in red
        cv2.line(overlay, v_lane_bl, v_lane_br, (0, 0, 255), 4)
        fl_mid_x = (v_lane_bl[0] + v_lane_br[0]) // 2
        cv2.putText(overlay, "FOUL LINE", (fl_mid_x - 80, v_lane_bl[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # --- Approach: extrapolate perspective lines BELOW the foul line ---
        # The approach is between the foul line and the bottom of the video frame.
        # Extend the lane side lines downward (they get wider toward the camera).

        # Calculate the slope of each side line in layout coords
        lane_height = lane_bot_y - lane_top_y
        if lane_height > 0:
            left_slope = (lane_bot_left - lane_top_left) / lane_height   # dx/dy
            right_slope = (lane_bot_right - lane_top_right) / lane_height

            # Bottom of video frame in layout coords
            approach_bot_y_layout = (vid_h - offset_y) / scale_y

            # Extrapolate side lines to bottom of frame
            dy_approach = approach_bot_y_layout - lane_bot_y
            approach_bot_left = lane_bot_left + left_slope * dy_approach
            approach_bot_right = lane_bot_right + right_slope * dy_approach

            # Approach is a bit wider than the lane extrapolation
            # (ball returns, seating area beside approach)
            extra_width = (approach_bot_right - approach_bot_left) * 0.05
            approach_bot_left -= extra_width
            approach_bot_right += extra_width

            v_approach_tl = v_lane_bl  # top of approach = foul line left
            v_approach_tr = v_lane_br  # top of approach = foul line right
            v_approach_bl = to_vid(approach_bot_left, approach_bot_y_layout)
            v_approach_br = to_vid(approach_bot_right, approach_bot_y_layout)

            # Clamp to frame
            v_approach_bl = (max(0, v_approach_bl[0]), min(vid_h, v_approach_bl[1]))
            v_approach_br = (min(vid_w, v_approach_br[0]), min(vid_h, v_approach_br[1]))

            # Draw approach in blue
            cv2.line(overlay, v_approach_tl, v_approach_tr, (255, 100, 0), 3)
            cv2.line(overlay, v_approach_bl, v_approach_br, (255, 100, 0), 3)
            cv2.line(overlay, v_approach_tl, v_approach_bl, (255, 100, 0), 3)
            cv2.line(overlay, v_approach_tr, v_approach_br, (255, 100, 0), 3)

            a_mid_x = (v_approach_tl[0] + v_approach_tr[0] + v_approach_bl[0] + v_approach_br[0]) // 4
            a_mid_y = (v_approach_tl[1] + v_approach_bl[1]) // 2
            cv2.putText(overlay, "APPROACH", (a_mid_x - 80, a_mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 3)

            print(f"\nApproach (derived from lane perspective):")
            print(f"  Top (foul line): {v_approach_tl} - {v_approach_tr}")
            print(f"  Bot (near camera): {v_approach_bl} - {v_approach_br}")

    cv2.imwrite("lane_outline_on_video.png", overlay)
    print(f"\nSaved lane + approach outline visualization to: lane_outline_on_video.png")


if __name__ == "__main__":
    main()