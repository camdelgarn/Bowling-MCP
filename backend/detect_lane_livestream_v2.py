#!/usr/bin/env python3
"""
Detect bowling lane from RTMP livestream using approach dot calibration.

The approach area has alignment dots: a big center dot with smaller dots
on each side, spaced 5 boards apart. We find these dots, calculate board
width in pixels, and derive lane edges (39 boards wide).

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


def find_approach_dots(gray: np.ndarray) -> list[dict] | None:
    """Find the approach alignment dots using blob detection.

    Returns list of dot dicts sorted by x, or None if not enough found.
    """
    h, w = gray.shape

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 400
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.4
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)

    roi_top = int(h * 0.40)
    roi_bot = int(h * 0.80)
    roi = gray[roi_top:roi_bot, :]
    keypoints = detector.detect(roi)

    if len(keypoints) < 3:
        print(f"  Only {len(keypoints)} blobs found, need at least 3")
        return None

    blobs = []
    for kp in keypoints:
        x, y_local = kp.pt
        y_abs = y_local + roi_top
        px_val = gray[int(y_abs), int(x)]
        neighborhood = gray[max(0, int(y_abs) - 15):int(y_abs) + 15,
                            max(0, int(x) - 15):int(x) + 15].mean()
        contrast = neighborhood - px_val
        if contrast > 30:
            blobs.append({
                "x": int(x), "y": int(y_abs),
                "size": round(kp.size, 1),
                "brightness": int(px_val),
                "contrast": round(contrast, 1),
            })

    if len(blobs) < 3:
        print(f"  Only {len(blobs)} high-contrast blobs, need at least 3")
        return None

    blobs.sort(key=lambda b: b["x"])

    # Find the longest run of evenly-spaced collinear dots
    best_run = []
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            dx = blobs[j]["x"] - blobs[i]["x"]
            dy = blobs[j]["y"] - blobs[i]["y"]
            dist = (dx**2 + dy**2)**0.5
            if dist < 30 or dist > 200:
                continue

            run = [blobs[i], blobs[j]]
            last = blobs[j]
            for k in range(j + 1, len(blobs)):
                cand = blobs[k]
                cdx = cand["x"] - last["x"]
                cdy = cand["y"] - last["y"]
                cdist = (cdx**2 + cdy**2)**0.5
                if abs(cdist - dist) < dist * 0.2:
                    angle_ref = np.arctan2(dy, dx)
                    angle_cand = np.arctan2(cdy, cdx)
                    if abs(angle_ref - angle_cand) < 0.15:
                        run.append(cand)
                        last = cand

            if len(run) > len(best_run):
                best_run = run

    if len(best_run) < 3:
        print(f"  Could not find 3+ evenly spaced collinear dots")
        return None

    return best_run


def track_left_edge(gray: np.ndarray, start_x: int, start_y: int,
                    end_y: int = 50, step: int = 3,
                    search_radius: int = 50, sigma: int = 20) -> dict:
    """Track the left lane edge bidirectionally from start_y."""
    h, w = gray.shape

    def _track(sx, sy, ey, st):
        xs, ys_out = [], []
        prev_x = sx
        misses = 0
        for y in range(sy + st, ey, st):
            if y < 0 or y >= h:
                break
            row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=sigma)
            grad = np.gradient(row)
            sl = max(0, prev_x - search_radius)
            sr = min(w, prev_x + search_radius)
            local = grad[sl:sr]
            peaks, props = find_peaks(local, height=0.10, distance=8)
            if len(peaks) > 0:
                abs_peaks = peaks + sl
                dists = np.abs(abs_peaks - prev_x).astype(float)
                scores = props["peak_heights"] / (1.0 + dists / 20.0)
                new_x = int(abs_peaks[np.argmax(scores)])
                if abs(new_x - prev_x) > 40:
                    misses += 1
                    if misses > 5:
                        break
                    continue
                xs.append(new_x)
                ys_out.append(y)
                prev_x = new_x
                misses = 0
            else:
                misses += 1
                if misses > 5:
                    break
        return xs, ys_out

    ux, uy = _track(start_x, start_y, end_y, -step)
    dx, dy = _track(start_x, start_y, h - 10, step)

    all_x = ux[::-1] + [start_x] + dx
    all_y = uy[::-1] + [start_y] + dy

    return {"x": np.array(all_x), "y": np.array(all_y)}


def robust_line_fit(y_vals, x_vals, max_iter=10, threshold=15):
    """Iteratively fit a line, removing outliers."""
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
    """Main lane detection pipeline using approach dots + gradient edge tracking."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"\nFrame: {w}x{h}, mean brightness: {gray.mean():.1f}")

    # Step 1: Find approach dots
    print("\n--- Step 1: Finding approach dots ---")
    dots = find_approach_dots(gray)
    board_px = None

    if dots is None:
        print("  WARNING: Could not find approach dots, falling back to gradient-only")
        dots = []
    else:
        print(f"  Found {len(dots)} approach dots:")
        biggest = max(dots, key=lambda d: d["size"])
        for d in dots:
            label = "CENTER" if d is biggest else "      "
            print(f"    {label} x={d['x']:4d}, y={d['y']:4d}, size={d['size']:.1f}, "
                  f"contrast={d['contrast']:.0f}")

        spacings = []
        for i in range(1, len(dots)):
            dx = dots[i]["x"] - dots[i - 1]["x"]
            dy = dots[i]["y"] - dots[i - 1]["y"]
            spacings.append((dx**2 + dy**2)**0.5)
        avg_spacing = np.mean(spacings)
        board_px = avg_spacing / DOTS_SPACING_BOARDS
        print(f"\n  Dot spacing: {avg_spacing:.1f}px = {DOTS_SPACING_BOARDS} boards")
        print(f"  1 board = {board_px:.1f}px at y~{int(np.mean([d['y'] for d in dots]))}")
        print(f"  Lane width ({BOARDS_PER_LANE} boards) = {board_px * BOARDS_PER_LANE:.0f}px")

    # Step 2: Derive lane edge positions from dots
    print("\n--- Step 2: Computing lane edges ---")

    if len(dots) >= 3 and board_px:
        # The dots give us a perspective mapping: board_number -> (x, y)
        # Standard approach dots at boards 10, 15, 20, 25, 30
        # (center dot = board 20)
        biggest = max(dots, key=lambda d: d["size"])
        center_board = 20  # center dot is at board 20

        dot_boards = []
        for i, d in enumerate(dots):
            offset = (i - dots.index(biggest)) * DOTS_SPACING_BOARDS
            dot_boards.append(center_board + offset)

        dot_xs = np.array([d["x"] for d in dots], dtype=float)
        dot_ys = np.array([d["y"] for d in dots], dtype=float)
        dot_b = np.array(dot_boards, dtype=float)

        # Linear fit: board -> x and board -> y at the dot row
        bx_coeffs = np.polyfit(dot_b, dot_xs, 1)  # x = slope*board + intercept
        by_coeffs = np.polyfit(dot_b, dot_ys, 1)  # y = slope*board + intercept

        px_per_board = bx_coeffs[0]  # ~15.55
        py_per_board = by_coeffs[0]  # ~-2.35

        print(f"  Board->pixel mapping: dx={px_per_board:.2f} px/board, dy={py_per_board:.2f} px/board")

        # Left edge (board 0) and right edge (board 39) at the dot row
        left_x_at_dots = np.polyval(bx_coeffs, 0)
        left_y_at_dots = np.polyval(by_coeffs, 0)
        right_x_at_dots = np.polyval(bx_coeffs, BOARDS_PER_LANE)
        right_y_at_dots = np.polyval(by_coeffs, BOARDS_PER_LANE)

        print(f"  Left edge (board 0) at dot row: x={left_x_at_dots:.0f}, y={left_y_at_dots:.0f}")
        print(f"  Right edge (board {BOARDS_PER_LANE}) at dot row: x={right_x_at_dots:.0f}, y={right_y_at_dots:.0f}")

        # Now track the left edge with gradient, anchored near the dot-derived position
        print(f"\n  Tracking left edge from gradient (anchored at x≈{left_x_at_dots:.0f})...")
        search_x_min = int(left_x_at_dots - 4 * px_per_board)
        search_x_max = int(left_x_at_dots + 4 * px_per_board)

        # Find best gradient anchor near expected left edge
        best_left = None
        best_grad = 0
        for y_pct in range(30, 80, 5):
            y = int(h * y_pct / 100)
            row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=20)
            grad = np.gradient(row)
            # Expected left x at this y (extrapolate from dot row proportionally)
            # Use proportion: as y changes, x changes due to perspective
            # Approximate: left edge slope ≈ center dot slope (same convergence)
            expected_x = left_x_at_dots + (y - left_y_at_dots) * (-0.4)  # rough slope
            sx_min = max(0, int(expected_x - 80))
            sx_max = min(w, int(expected_x + 80))
            pos_peaks, props = find_peaks(grad[sx_min:sx_max], height=0.15, distance=8)
            if len(pos_peaks) > 0:
                abs_peaks = pos_peaks + sx_min
                for idx in range(len(abs_peaks)):
                    if props["peak_heights"][idx] > best_grad * 0.7:
                        # Prefer closest to expected position
                        if best_left is None or abs(abs_peaks[idx] - expected_x) < abs(best_left[0] - expected_x):
                            best_grad = max(best_grad, props["peak_heights"][idx])
                            best_left = (int(abs_peaks[idx]), y)

        if best_left is not None:
            print(f"  Best left edge anchor: x={best_left[0]}, y={best_left[1]}")
            left_track = track_left_edge(gray, best_left[0], best_left[1])
            n_pts = len(left_track["y"])
            print(f"  Tracked {n_pts} points (y={left_track['y'].min()} to y={left_track['y'].max()})")

            if n_pts >= 5:
                left_coeffs, left_mask = robust_line_fit(left_track["y"], left_track["x"])
                left_inliers = int(np.sum(left_mask))
                left_slope = left_coeffs[0]
                print(f"  Left gradient fit: x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}  "
                      f"({left_inliers}/{n_pts} inliers)")
            else:
                left_slope = -0.4  # fallback slope estimate
                left_track = {"x": np.array([int(left_x_at_dots)]),
                              "y": np.array([int(left_y_at_dots)])}
        else:
            print("  No gradient anchor found, using dot-derived position only")
            left_slope = -0.4
            left_track = {"x": np.array([int(left_x_at_dots)]),
                          "y": np.array([int(left_y_at_dots)])}

        # Build left edge line: use gradient slope, anchored at dot-derived position
        left_intercept = left_x_at_dots - left_slope * left_y_at_dots
        left_coeffs = np.array([left_slope, left_intercept])
        print(f"  Final left edge: x = {left_slope:.4f}*y + {left_intercept:.1f}")

        # Right edge: same approach — use slope from perspective geometry
        # The right edge should converge to same vanishing point
        # Slope ratio: right_slope / left_slope ≈ right_x_shift / left_x_shift
        # Use dot line to estimate right edge slope
        right_intercept = right_x_at_dots - left_slope * right_y_at_dots
        right_coeffs = np.array([left_slope, right_intercept])

        # But actually the right edge has a different slope because it's on the
        # other side. Use the dot line: extrapolate how x,y changes for right edge.
        # The diagonal from left to right dot-row positions suggests a slight
        # perspective correction
        dot_line_slope = (dot_xs[-1] - dot_xs[0]) / (dot_ys[-1] - dot_ys[0])
        # Right edge should be slightly less steep than left (converging)
        right_slope = left_slope * 0.95  # slight convergence adjustment
        right_intercept = right_x_at_dots - right_slope * right_y_at_dots
        right_coeffs = np.array([right_slope, right_intercept])
        right_source = "dots"

        print(f"  Final right edge: x = {right_slope:.4f}*y + {right_intercept:.1f}")

        y_top = max(50, int(left_track["y"].min())) if len(left_track["y"]) > 1 else 100
        y_bot = min(h - 50, int(left_track["y"].max())) if len(left_track["y"]) > 1 else h - 100

    if best_left is None:
        print("  Could not find left lane edge!")
        return None

    print(f"  Best left edge anchor: x={best_left[0]}, y={best_left[1]}, gradient={best_grad:.2f}")

    left_track = track_left_edge(gray, best_left[0], best_left[1])
    n_pts = len(left_track["y"])
    print(f"  Tracked {n_pts} points (y={left_track['y'].min()} to y={left_track['y'].max()})")

    if n_pts < 5:
        print("  Not enough tracking points")
        return None

    left_coeffs, left_mask = robust_line_fit(left_track["y"], left_track["x"])
    left_inliers = int(np.sum(left_mask))
    print(f"  Left fit: x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}  "
          f"({left_inliers}/{n_pts} inliers)")

    # Step 3: Derive right edge
    print("\n--- Step 3: Determining right lane edge ---")
    right_source = "gradient"

    if len(dots) >= 3 and board_px:
        # Use dot calibration
        biggest = max(dots, key=lambda d: d["size"])
        center_x = biggest["x"]
        dot_mean_y = float(np.mean([d["y"] for d in dots]))

        left_at_dots = np.polyval(left_coeffs, dot_mean_y)
        lane_width_at_dots = BOARDS_PER_LANE * board_px
        right_at_dots = left_at_dots + lane_width_at_dots

        measured_boards = (center_x - left_at_dots) / board_px
        print(f"  Center dot at {measured_boards:.1f} boards from left edge (expect ~20)")

        # Estimate right edge slope from perspective geometry
        # Use two reference points with scaled lane width
        y1 = int(h * 0.35)
        y2 = int(h * 0.70)
        left_y1 = np.polyval(left_coeffs, y1)
        left_y2 = np.polyval(left_coeffs, y2)

        # Perspective scaling: lane width narrows as y decreases (further away)
        # Scale factor roughly proportional to (dot_y - vanishing_y) / (y - vanishing_y)
        # Approximate: use left edge slope to estimate width change
        scale_y1 = 1.0 + left_coeffs[0] * (y1 - dot_mean_y) / lane_width_at_dots * 0.5
        scale_y2 = 1.0 + left_coeffs[0] * (y2 - dot_mean_y) / lane_width_at_dots * 0.5

        right_y1 = left_y1 + lane_width_at_dots * scale_y1
        right_y2 = left_y2 + lane_width_at_dots * scale_y2

        right_coeffs = np.polyfit([y1, dot_mean_y, y2],
                                  [right_y1, right_at_dots, right_y2], 1)
        right_source = "dots"
        print(f"  Right edge (from dots): x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}")
    else:
        # Fallback: gradient-based scan with fine sigma
        right_points_y = []
        right_points_x = []
        for y_pct in range(25, 80, 5):
            y = int(h * y_pct / 100)
            left_x = int(np.polyval(left_coeffs, y))
            row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=12)
            grad = np.gradient(row)
            search_start = left_x + 200
            search_end = min(w, left_x + 1200)
            if search_start >= search_end:
                continue
            segment = -grad[search_start:search_end]
            peaks, _ = find_peaks(segment, height=0.3, distance=15)
            if len(peaks) > 0:
                right_points_y.append(y)
                right_points_x.append(int(peaks[0]) + search_start)

        if len(right_points_y) >= 3:
            right_coeffs, _ = robust_line_fit(
                np.array(right_points_y), np.array(right_points_x), threshold=30)
            print(f"  Right edge (gradient): x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}")
        else:
            print("  Could not determine right edge!")
            return None

    # Results
    y_top = int(left_track["y"].min())
    y_bot = int(left_track["y"].max())
    print(f"\n--- Lane Dimensions (y={y_top} to y={y_bot}) ---")
    for label, y_check in [("top", y_top), ("middle", (y_top + y_bot) // 2), ("bottom", y_bot)]:
        lx = np.polyval(left_coeffs, y_check)
        rx = np.polyval(right_coeffs, y_check)
        if board_px:
            width_boards = (rx - lx) / board_px
            print(f"  {label:6s} (y={y_check:4d}): left={lx:6.0f}, right={rx:6.0f}, "
                  f"width={rx - lx:.0f}px (~{width_boards:.1f} boards)")
        else:
            print(f"  {label:6s} (y={y_check:4d}): left={lx:6.0f}, right={rx:6.0f}, "
                  f"width={rx - lx:.0f}px")

    vp_y = vp_x = None
    if abs(right_coeffs[0] - left_coeffs[0]) > 0.001:
        vp_y = (left_coeffs[1] - right_coeffs[1]) / (right_coeffs[0] - left_coeffs[0])
        vp_x = np.polyval(left_coeffs, vp_y)
        print(f"\n  Vanishing point: ({vp_x:.0f}, {vp_y:.0f})")

    return {
        "left_coeffs": left_coeffs,
        "right_coeffs": right_coeffs,
        "left_track": left_track,
        "dots": dots,
        "board_px": board_px,
        "y_range": (y_top, y_bot),
        "right_source": right_source,
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

    # Left edge tracked points
    track = detection["left_track"]
    for i in range(len(track["y"])):
        cv2.circle(out, (int(track["x"][i]), int(track["y"][i])), 2, (255, 255, 0), -1)

    # Approach dots
    dots = detection.get("dots", [])
    if dots:
        biggest = max(dots, key=lambda d: d["size"])
        for d in dots:
            color = (0, 0, 255) if d is biggest else (0, 255, 255)
            radius = 15 if d is biggest else 10
            cv2.circle(out, (d["x"], d["y"]), radius, color, 2)

    # Board tick marks at dot level
    board_px = detection.get("board_px")
    if board_px and dots:
        biggest = max(dots, key=lambda d: d["size"])
        dot_y = biggest["y"]
        lx_at_dot = int(np.polyval(lc, dot_y))
        for b in range(0, BOARDS_PER_LANE + 1, 5):
            bx = int(lx_at_dot + b * board_px)
            cv2.line(out, (bx, dot_y - 8), (bx, dot_y + 8), (255, 200, 0), 1)
            if b % 10 == 0:
                cv2.putText(out, str(b), (bx - 8, dot_y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

    # Labels
    cv2.putText(out, "LANE DETECTED", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    info = []
    if board_px:
        info.append(f"1 board = {board_px:.1f}px")
    mid_y = (y_top + y_bot) // 2
    lw = np.polyval(rc, mid_y) - np.polyval(lc, mid_y)
    info.append(f"width@mid = {lw:.0f}px")
    info.append(f"right edge: {detection['right_source']}")
    for i, text in enumerate(info):
        cv2.putText(out, text, (20, 80 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
