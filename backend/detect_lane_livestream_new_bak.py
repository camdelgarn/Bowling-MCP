#!/usr/bin/env python3
"""
Detect bowling lane from RTMP livestream using approach dot calibration.

The bowling approach has multiple rows of alignment dots:
  Row 1 (closest to camera): big center dot + 2 smaller dots each side (5 dots)
  Row 2 (2-3 ft past Row 1): same pattern (5 dots)
  Row 3 (near foul line): big center dot + 3 smaller dots each side (7 dots)
  Camera may not see Row 3.

Dots are spaced 5 boards apart. Center dot = board 20. Lane = 39 boards wide.
With 2+ dot rows we get direct perspective calibration for both edges.

Usage:
  python detect_lane_livestream.py              # Use saved frame
  python detect_lane_livestream.py --live       # Capture fresh frame
  python detect_lane_livestream.py <rtmp_url>   # Custom stream URL
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import sys
import os

RTMP_URL = "rtmp://192.168.1.7:1935/live/stream"
BOARDS_PER_LANE = 39
DOTS_SPACING_BOARDS = 5  # 5 boards between each approach dot
CENTER_BOARD = 20        # The big center dot is at board 20


def capture_frame(url: str) -> np.ndarray:
    """Capture a single frame from the RTMP stream."""
    print(f"Connecting to {url}...")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("ERROR: Could not open stream")
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream: {w}x{h} @ {fps:.1f} fps")
    for _ in range(15):
        ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("ERROR: Failed to read frame")
        sys.exit(1)
    return frame


def find_dot_rows(gray: np.ndarray) -> list[list[dict]]:
    """Find approach dot rows using blob detection.

    Returns a list of dot rows, each row sorted by x.
    Each dot: {"x": int, "y": int, "size": float, "contrast": float}
    """
    h, w = gray.shape

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 12
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.25
    params.filterByConvexity = True
    params.minConvexity = 0.35
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)

    # Search the approach area (roughly 35-85% of frame height)
    roi_top = int(h * 0.30)
    roi_bot = int(h * 0.85)
    roi = gray[roi_top:roi_bot, :]
    keypoints = detector.detect(roi)

    # Filter by contrast
    blobs = []
    for kp in keypoints:
        x, y_local = kp.pt
        y_abs = y_local + roi_top
        xi, yi = int(x), int(y_abs)
        if yi < 5 or yi >= h - 5 or xi < 5 or xi >= w - 5:
            continue
        px_val = gray[yi, xi]
        neighborhood = gray[max(0, yi - 15):yi + 15,
                            max(0, xi - 15):xi + 15].mean()
        contrast = neighborhood - px_val
        if contrast > 25:
            blobs.append({
                "x": xi, "y": yi,
                "size": round(kp.size, 1),
                "contrast": round(contrast, 1),
            })

    print(f"  Found {len(blobs)} high-contrast blobs")
    if len(blobs) < 3:
        return []

    # Cluster blobs into rows by y-coordinate
    blobs.sort(key=lambda b: b["y"])
    rows = []
    current_row = [blobs[0]]
    for b in blobs[1:]:
        if abs(b["y"] - current_row[-1]["y"]) < 40:
            current_row.append(b)
        else:
            if len(current_row) >= 3:
                rows.append(current_row)
            current_row = [b]
    if len(current_row) >= 3:
        rows.append(current_row)

    print(f"  Clustered into {len(rows)} potential rows (3+ blobs each)")

    # For each row, find the best evenly-spaced collinear dot pattern
    good_rows = []
    for row_idx, row_blobs in enumerate(rows):
        row_blobs.sort(key=lambda b: b["x"])
        best_run = _find_collinear_run(row_blobs)
        if best_run and len(best_run) >= 3:
            mean_y = int(np.mean([d["y"] for d in best_run]))
            print(f"  Row {row_idx}: {len(best_run)} dots at y≈{mean_y}, "
                  f"x=[{best_run[0]['x']}..{best_run[-1]['x']}]")
            good_rows.append(best_run)

    return good_rows


def _find_collinear_run(blobs: list[dict]) -> list[dict] | None:
    """Find the longest run of evenly-spaced collinear dots in a set of blobs."""
    best_run = []
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            dx = blobs[j]["x"] - blobs[i]["x"]
            dy = blobs[j]["y"] - blobs[i]["y"]
            dist = (dx**2 + dy**2)**0.5
            if dist < 25 or dist > 250:
                continue

            run = [blobs[i], blobs[j]]
            last = blobs[j]
            for k in range(j + 1, len(blobs)):
                cand = blobs[k]
                cdx = cand["x"] - last["x"]
                cdy = cand["y"] - last["y"]
                cdist = (cdx**2 + cdy**2)**0.5
                if abs(cdist - dist) < dist * 0.25:
                    angle_ref = np.arctan2(dy, dx)
                    angle_cand = np.arctan2(cdy, cdx)
                    if abs(angle_ref - angle_cand) < 0.2:
                        run.append(cand)
                        last = cand

            if len(run) > len(best_run):
                best_run = run

    return best_run if len(best_run) >= 3 else None


def _row_to_board_mapping(dots: list[dict]) -> dict:
    """Map a single dot row to board positions.

    Returns dict with:
      'dots': the dot list
      'center': the center (biggest) dot
      'board_positions': list of (board_num, x, y) tuples
      'px_per_board': pixels per board at this y-level
      'left_edge_x': x position of board 0
      'right_edge_x': x position of board 39
      'mean_y': average y of this row
      'bx_coeffs': polyfit coefficients for board -> x
    """
    biggest = max(dots, key=lambda d: d["size"])
    center_idx = dots.index(biggest)

    # Assign board numbers: center = board 20
    boards = []
    for i, d in enumerate(dots):
        offset = (i - center_idx) * DOTS_SPACING_BOARDS
        boards.append(CENTER_BOARD + offset)

    xs = np.array([d["x"] for d in dots], dtype=float)
    ys = np.array([d["y"] for d in dots], dtype=float)
    bs = np.array(boards, dtype=float)

    # Linear fit: x = slope * board + intercept
    bx_coeffs = np.polyfit(bs, xs, 1)
    px_per_board = bx_coeffs[0]

    left_x = np.polyval(bx_coeffs, 0)
    right_x = np.polyval(bx_coeffs, BOARDS_PER_LANE)
    mean_y = float(np.mean(ys))

    return {
        "dots": dots,
        "center": biggest,
        "boards": boards,
        "px_per_board": px_per_board,
        "left_edge_x": left_x,
        "right_edge_x": right_x,
        "mean_y": mean_y,
        "bx_coeffs": bx_coeffs,
    }


def track_edge_gradient(gray: np.ndarray, expected_xs: list[tuple[int, int]],
                        sigma: int = 15, search_radius: int = 60,
                        edge_sign: int = 1) -> dict:
    """Track a lane edge using gradient peaks near expected positions.

    Args:
        expected_xs: list of (y, expected_x) pairs
        edge_sign: +1 for left edge (dark->light), -1 for right edge (light->dark)

    Returns dict with 'x' and 'y' arrays.
    """
    h, w = gray.shape
    tracked_x = []
    tracked_y = []

    for y, ex in expected_xs:
        if y < 5 or y >= h - 5:
            continue
        row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=sigma)
        grad = np.gradient(row) * edge_sign

        sl = max(0, ex - search_radius)
        sr = min(w, ex + search_radius)
        local = grad[sl:sr]

        peaks, props = find_peaks(local, height=0.08, distance=8)
        if len(peaks) > 0:
            abs_peaks = peaks + sl
            # Score: prefer high gradient AND close to expected position
            dists = np.abs(abs_peaks - ex).astype(float)
            scores = props["peak_heights"] / (1.0 + dists / 25.0)
            best = int(abs_peaks[np.argmax(scores)])
            if abs(best - ex) < search_radius:
                tracked_x.append(best)
                tracked_y.append(y)

    return {"x": np.array(tracked_x), "y": np.array(tracked_y)}


def robust_line_fit(y_vals, x_vals, max_iter=10, threshold=15):
    """Iteratively fit a line x = slope*y + intercept, removing outliers."""
    mask = np.ones(len(y_vals), dtype=bool)
    coeffs = None
    for _ in range(max_iter):
        yf, xf = y_vals[mask], x_vals[mask]
        if len(yf) < 3:
            break
        coeffs = np.polyfit(yf, xf, 1)
        res = np.abs(x_vals - np.polyval(coeffs, y_vals))
        new_mask = res < threshold
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
    return coeffs, mask


def detect_lane(frame: np.ndarray) -> dict | None:
    """Main lane detection pipeline using approach dot rows."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"\nFrame: {w}x{h}, mean brightness: {gray.mean():.1f}")

    # Step 1: Find all dot rows
    print("\n--- Step 1: Finding approach dot rows ---")
    dot_rows_raw = find_dot_rows(gray)

    if len(dot_rows_raw) == 0:
        print("  No dot rows found! Cannot detect lane.")
        return None

    # Map each row to board positions
    row_mappings = []
    for i, row_dots in enumerate(dot_rows_raw):
        mapping = _row_to_board_mapping(row_dots)
        row_mappings.append(mapping)
        print(f"\n  Row {i+1}: {len(row_dots)} dots at y≈{mapping['mean_y']:.0f}")
        print(f"    px/board = {mapping['px_per_board']:.2f}")
        print(f"    Left edge (board 0):  x = {mapping['left_edge_x']:.0f}")
        print(f"    Right edge (board {BOARDS_PER_LANE}): x = {mapping['right_edge_x']:.0f}")
        print(f"    Lane width = {mapping['right_edge_x'] - mapping['left_edge_x']:.0f}px")
        center = mapping["center"]
        print(f"    Center dot: x={center['x']}, y={center['y']}, size={center['size']}")

    # Step 2: Compute lane edge lines from dot rows
    print("\n--- Step 2: Computing lane edges from dot geometry ---")

    if len(row_mappings) >= 2:
        # BEST CASE: 2+ rows give us direct perspective calibration
        # Sort by y (top of frame = far, bottom = near camera)
        row_mappings.sort(key=lambda r: r["mean_y"])

        # Left edge: fit line through (y, left_x) from each row
        left_points_y = [r["mean_y"] for r in row_mappings]
        left_points_x = [r["left_edge_x"] for r in row_mappings]
        right_points_y = [r["mean_y"] for r in row_mappings]
        right_points_x = [r["right_edge_x"] for r in row_mappings]

        if len(row_mappings) == 2:
            # Exactly 2 points: direct line
            left_coeffs = np.polyfit(left_points_y, left_points_x, 1)
            right_coeffs = np.polyfit(right_points_y, right_points_x, 1)
        else:
            # 3+ points: robust fit
            left_coeffs, _ = robust_line_fit(
                np.array(left_points_y), np.array(left_points_x), threshold=20)
            right_coeffs, _ = robust_line_fit(
                np.array(right_points_y), np.array(right_points_x), threshold=20)

        print(f"  Left edge:  x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}")
        print(f"  Right edge: x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}")
        print(f"  (Derived from {len(row_mappings)} dot rows)")
        edge_source = f"{len(row_mappings)}-row dots"

    else:
        # Only 1 row: need gradient to find edge slope
        print("  Only 1 dot row — will use gradient to determine edge slope")
        rm = row_mappings[0]

        # Generate expected positions at various y-levels using an estimated slope
        # Try gradient tracking near the dot-derived positions
        test_ys = list(range(int(h * 0.25), int(h * 0.80), 4))
        expected_left = [(y, int(rm["left_edge_x"])) for y in test_ys]
        expected_right = [(y, int(rm["right_edge_x"])) for y in test_ys]

        left_track = track_edge_gradient(gray, expected_left, edge_sign=1)
        right_track = track_edge_gradient(gray, expected_right, edge_sign=-1)

        if len(left_track["y"]) >= 5:
            left_coeffs, left_mask = robust_line_fit(left_track["y"], left_track["x"])
            n_in = int(np.sum(left_mask))
            print(f"  Left edge (gradient): x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f} "
                  f"({n_in}/{len(left_track['y'])} inliers)")
        else:
            # Fallback: assume a typical slope
            left_coeffs = np.array([-0.4, rm["left_edge_x"] + 0.4 * rm["mean_y"]])
            print(f"  Left edge (fallback slope): x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}")

        if len(right_track["y"]) >= 5:
            right_coeffs, right_mask = robust_line_fit(right_track["y"], right_track["x"])
            n_in = int(np.sum(right_mask))
            print(f"  Right edge (gradient): x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f} "
                  f"({n_in}/{len(right_track['y'])} inliers)")
        else:
            right_coeffs = np.array([-0.3, rm["right_edge_x"] + 0.3 * rm["mean_y"]])
            print(f"  Right edge (fallback slope): x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}")

        edge_source = "1-row dots + gradient"

    # Step 3: Refine edges with gradient (optional, for 2+ row case too)
    print("\n--- Step 3: Gradient refinement ---")
    y_min = max(50, int(min(r["mean_y"] for r in row_mappings)) - 100)
    y_max = min(h - 50, int(max(r["mean_y"] for r in row_mappings)) + 100)

    refine_ys = list(range(y_min, y_max, 4))
    expected_left = [(y, int(np.polyval(left_coeffs, y))) for y in refine_ys]
    expected_right = [(y, int(np.polyval(right_coeffs, y))) for y in refine_ys]

    left_refined = track_edge_gradient(gray, expected_left, edge_sign=1, search_radius=40)
    right_refined = track_edge_gradient(gray, expected_right, edge_sign=-1, search_radius=40)

    print(f"  Left: {len(left_refined['y'])} gradient points found")
    print(f"  Right: {len(right_refined['y'])} gradient points found")

    # If gradient tracking found good data, blend with dot-derived lines
    if len(left_refined["y"]) >= 10 and len(row_mappings) >= 2:
        # Combine dot anchor points and gradient points for better fit
        all_y = np.concatenate([np.array([r["mean_y"] for r in row_mappings]),
                                left_refined["y"]])
        all_x = np.concatenate([np.array([r["left_edge_x"] for r in row_mappings]),
                                left_refined["x"]])
        left_coeffs_r, left_mask = robust_line_fit(all_y, all_x, threshold=12)
        if left_coeffs_r is not None:
            n_in = int(np.sum(left_mask))
            print(f"  Left refined: x = {left_coeffs_r[0]:.4f}*y + {left_coeffs_r[1]:.1f} "
                  f"({n_in}/{len(all_y)} inliers)")
            left_coeffs = left_coeffs_r

    if len(right_refined["y"]) >= 10 and len(row_mappings) >= 2:
        all_y = np.concatenate([np.array([r["mean_y"] for r in row_mappings]),
                                right_refined["y"]])
        all_x = np.concatenate([np.array([r["right_edge_x"] for r in row_mappings]),
                                right_refined["x"]])
        right_coeffs_r, right_mask = robust_line_fit(all_y, all_x, threshold=12)
        if right_coeffs_r is not None:
            n_in = int(np.sum(right_mask))
            print(f"  Right refined: x = {right_coeffs_r[0]:.4f}*y + {right_coeffs_r[1]:.1f} "
                  f"({n_in}/{len(all_y)} inliers)")
            right_coeffs = right_coeffs_r

    # Determine visible y range
    y_top = max(50, y_min)
    y_bot = min(h - 50, y_max)

    # Validation: check lane width in boards at each dot row
    print(f"\n--- Lane Dimensions (y={y_top} to y={y_bot}) ---")
    for rm in row_mappings:
        y_check = int(rm["mean_y"])
        lx = np.polyval(left_coeffs, y_check)
        rx = np.polyval(right_coeffs, y_check)
        width_px = rx - lx
        width_boards = width_px / rm["px_per_board"]
        print(f"  At dot row y={y_check}: left={lx:.0f}, right={rx:.0f}, "
              f"width={width_px:.0f}px = {width_boards:.1f} boards (expect {BOARDS_PER_LANE})")

    for label, y_check in [("top", y_top), ("middle", (y_top + y_bot) // 2), ("bottom", y_bot)]:
        lx = np.polyval(left_coeffs, y_check)
        rx = np.polyval(right_coeffs, y_check)
        print(f"  {label:6s} (y={y_check:4d}): left={lx:6.0f}, right={rx:6.0f}, "
              f"width={rx - lx:.0f}px")

    # Vanishing point
    vp_y = vp_x = None
    slope_diff = right_coeffs[0] - left_coeffs[0]
    if abs(slope_diff) > 0.001:
        vp_y = (left_coeffs[1] - right_coeffs[1]) / slope_diff
        vp_x = np.polyval(left_coeffs, vp_y)
        print(f"\n  Vanishing point: ({vp_x:.0f}, {vp_y:.0f})")

    # Collect all dots for overlay
    all_dots = []
    for rm in row_mappings:
        all_dots.extend(rm["dots"])

    return {
        "left_coeffs": left_coeffs,
        "right_coeffs": right_coeffs,
        "left_refined": left_refined,
        "right_refined": right_refined,
        "row_mappings": row_mappings,
        "all_dots": all_dots,
        "y_range": (y_top, y_bot),
        "edge_source": edge_source,
        "vanishing_point": (vp_x, vp_y) if vp_y is not None else None,
    }


def draw_lane_overlay(frame: np.ndarray, detection: dict) -> np.ndarray:
    """Draw lane detection results on the frame."""
    out = frame.copy()
    h, w = frame.shape[:2]
    lc = detection["left_coeffs"]
    rc = detection["right_coeffs"]
    y_top, y_bot = detection["y_range"]

    # Lane polygon fill
    left_pts, right_pts = [], []
    for y in range(y_top, y_bot + 1, 3):
        lx = int(np.polyval(lc, y))
        rx = int(np.polyval(rc, y))
        left_pts.append([lx, y])
        right_pts.append([rx, y])

    polygon = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [polygon], (0, 180, 0))
    cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)

    # Edge lines
    for y in range(y_top, y_bot + 1, 2):
        lx = int(np.polyval(lc, y))
        rx = int(np.polyval(rc, y))
        cv2.circle(out, (lx, y), 1, (0, 255, 0), -1)
        cv2.circle(out, (rx, y), 1, (0, 0, 255), -1)

    # Gradient-refined points
    for key, color in [("left_refined", (255, 255, 0)), ("right_refined", (0, 255, 255))]:
        track = detection.get(key, {})
        xs = track.get("x", np.array([]))
        ys = track.get("y", np.array([]))
        for i in range(len(ys)):
            cv2.circle(out, (int(xs[i]), int(ys[i])), 2, color, -1)

    # Approach dots — mark each row
    row_mappings = detection.get("row_mappings", [])
    for rm in row_mappings:
        biggest = rm["center"]
        for d in rm["dots"]:
            is_center = (d["x"] == biggest["x"] and d["y"] == biggest["y"])
            color = (0, 0, 255) if is_center else (0, 255, 255)
            radius = 15 if is_center else 10
            cv2.circle(out, (d["x"], d["y"]), radius, color, 2)

    # Board tick marks at each dot row
    for rm in row_mappings:
        dot_y = int(rm["mean_y"])
        lx_at_row = np.polyval(lc, dot_y)
        rx_at_row = np.polyval(rc, dot_y)
        lane_width = rx_at_row - lx_at_row
        for b in range(0, BOARDS_PER_LANE + 1, 5):
            bx = int(lx_at_row + b * lane_width / BOARDS_PER_LANE)
            cv2.line(out, (bx, dot_y - 8), (bx, dot_y + 8), (255, 200, 0), 1)
            if b % 10 == 0:
                cv2.putText(out, str(b), (bx - 8, dot_y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

    # Labels
    cv2.putText(out, "LANE DETECTED", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    info = [f"edges: {detection['edge_source']}"]
    for i, rm in enumerate(row_mappings):
        info.append(f"row {i+1}: {len(rm['dots'])} dots, {rm['px_per_board']:.1f} px/board @ y={int(rm['mean_y'])}")
    mid_y = (y_top + y_bot) // 2
    lw = np.polyval(rc, mid_y) - np.polyval(lc, mid_y)
    info.append(f"width@mid = {lw:.0f}px")

    for i, text in enumerate(info):
        cv2.putText(out, text, (20, 80 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    vp = detection.get("vanishing_point")
    if vp and vp[0] is not None:
        vpx, vpy = int(vp[0]), int(vp[1])
        if 0 <= vpx < w and 0 <= vpy < h:
            cv2.drawMarker(out, (vpx, vpy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

    return out


def main():
    url = RTMP_URL
    for arg in sys.argv[1:]:
        if arg.startswith("rtmp://"):
            url = arg

    frame_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "livestream_frame.jpg")
    if os.path.exists(frame_path) and "--live" not in sys.argv:
        print(f"Using saved frame: {frame_path}")
        frame = cv2.imread(frame_path)
    else:
        frame = capture_frame(url)
        cv2.imwrite(frame_path, frame)
        print(f"Saved frame to {frame_path}")

    detection = detect_lane(frame)

    if detection is None:
        print("\n*** Could not detect a bowling lane. ***")
        print("Try: --live to capture a fresh frame, or check camera angle.")
        sys.exit(1)

    annotated = draw_lane_overlay(frame, detection)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lane_detected.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated frame saved to: {out_path}")
    print("LANE DETECTION SUCCESSFUL")


if __name__ == "__main__":
    main()
