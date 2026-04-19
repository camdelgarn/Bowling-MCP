#!/usr/bin/env python3
"""
Detect bowling lane from RTMP livestream.
Captures a frame, finds lane edges via gradient analysis, and saves annotated output.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import sys

RTMP_URL = "rtmp://192.168.1.7:1935/live/stream"


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

    # Skip a few frames to stabilize
    for _ in range(10):
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: Failed to read frame")
        sys.exit(1)
    return frame


def find_lane_edges_at_y(gray: np.ndarray, y: int, sigma: int = 30,
                         min_width: int = 100, max_width: int = 1400) -> list[dict]:
    """Find lane-width bright bands at a given y coordinate."""
    h, w = gray.shape
    if y < 0 or y >= h:
        return []

    row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=sigma)
    grad = np.gradient(row)

    pos_peaks, _ = find_peaks(grad, height=0.2, distance=30)
    neg_peaks, _ = find_peaks(-grad, height=0.2, distance=30)

    candidates = []
    for pp in pos_peaks:
        matching = neg_peaks[neg_peaks > pp]
        if len(matching) == 0:
            continue
        np_close = matching[0]
        width = np_close - pp
        if min_width < width < max_width:
            center = (pp + np_close) // 2
            brightness = float(np.mean(row[pp:np_close]))
            candidates.append({
                "left": int(pp), "right": int(np_close),
                "width": int(width), "center": int(center),
                "brightness": round(brightness, 1),
            })
    return candidates


def refine_right_edge(gray: np.ndarray, y: int, left_x: int,
                      sigma_fine: int = 12) -> int | None:
    """Find the right lane edge using finer sigma to resolve the gutter.
    Looks for the first significant bright->dark transition after left_x + min_lane."""
    h, w = gray.shape
    if y < 0 or y >= h:
        return None

    row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=sigma_fine)
    grad = np.gradient(row)

    # Look for negative gradient peaks (bright->dark) to the right of the left edge
    # Minimum lane width ~300px, max ~1200px at this resolution
    search_start = left_x + 200
    search_end = min(w, left_x + 1200)
    if search_start >= search_end:
        return None

    segment = -grad[search_start:search_end]
    peaks, props = find_peaks(segment, height=0.3, distance=15)
    if len(peaks) == 0:
        return None

    # Return the first significant peak (closest to left edge = the lane's own gutter)
    return int(peaks[0]) + search_start
    return candidates


def _track_one_direction(gray: np.ndarray, left_start: int, right_start: int,
                         start_y: int, end_y: int, step: int,
                         search_radius: int, sigma: int,
                         max_width: int, max_jump: int,
                         max_misses: int = 5) -> tuple[list, list, list]:
    """Track lane edges in one direction (step > 0 = downward, step < 0 = upward)."""
    h, w = gray.shape
    left_track, right_track, y_track = [], [], []
    prev_left, prev_right = left_start, right_start
    misses = 0

    y_range = range(start_y + step, end_y, step)
    for y in y_range:
        if y < 0 or y >= h:
            break
        row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=sigma)
        grad = np.gradient(row)

        lane_center = (prev_left + prev_right) // 2
        lane_brightness = np.mean(row[max(0, lane_center - 80):min(w, lane_center + 80)])
        if lane_brightness < 25:
            misses += 1
            if misses > max_misses:
                break
            continue

        # Left edge: positive gradient (prefer closest to prev_left)
        sl = max(0, prev_left - search_radius)
        sr = min(w, prev_left + search_radius)
        local_grad = grad[sl:sr]
        peaks, props = find_peaks(local_grad, height=0.10, distance=8)
        if len(peaks) > 0:
            abs_peaks = peaks + sl
            distances = np.abs(abs_peaks - prev_left).astype(float)
            # Score: gradient strength / (1 + distance/20) — prefer close & strong
            scores = props['peak_heights'] / (1.0 + distances / 20.0)
            new_left = abs_peaks[np.argmax(scores)]
        else:
            new_left = prev_left

        # Right edge: negative gradient (prefer closest to prev_right)
        sl = max(0, prev_right - search_radius)
        sr = min(w, prev_right + search_radius)
        local_grad = -grad[sl:sr]
        peaks, props = find_peaks(local_grad, height=0.10, distance=8)
        if len(peaks) > 0:
            abs_peaks = peaks + sl
            distances = np.abs(abs_peaks - prev_right).astype(float)
            scores = props['peak_heights'] / (1.0 + distances / 20.0)
            new_right = abs_peaks[np.argmax(scores)]
        else:
            new_right = prev_right

        new_width = new_right - new_left
        if new_width < 50 or new_width > max_width:
            misses += 1
            if misses > max_misses:
                break
            # Use linear extrapolation instead of breaking
            if len(left_track) >= 2:
                new_left = 2 * left_track[-1] - left_track[-2]
                new_right = 2 * right_track[-1] - right_track[-2]
            else:
                continue

        left_jump = abs(new_left - prev_left)
        right_jump = abs(new_right - prev_right)
        if left_jump > max_jump or right_jump > max_jump:
            # Try linear extrapolation for large jumps
            if len(left_track) >= 2:
                ext_left = 2 * left_track[-1] - left_track[-2]
                ext_right = 2 * right_track[-1] - right_track[-2]
                # Use extrapolated if it's reasonable
                if abs(ext_left - prev_left) <= max_jump:
                    new_left = ext_left
                if abs(ext_right - prev_right) <= max_jump:
                    new_right = ext_right
                # If still too jumpy, skip
                if abs(new_left - prev_left) > max_jump or abs(new_right - prev_right) > max_jump:
                    misses += 1
                    if misses > max_misses:
                        break
                    continue
            else:
                misses += 1
                if misses > max_misses:
                    break
                continue

        left_track.append(int(new_left))
        right_track.append(int(new_right))
        y_track.append(y)
        prev_left = new_left
        prev_right = new_right
        misses = 0  # reset on success

    return left_track, right_track, y_track


def track_lane_edges(gray: np.ndarray, left_start: int, right_start: int,
                     start_y: int, top_y: int = 50, step: int = 3,
                     search_radius: int = 60, sigma: int = 20) -> dict:
    """Track lane edges bidirectionally from start_y."""
    h, w = gray.shape
    max_width = int(w * 0.85)  # dynamic based on frame width
    max_jump = 45

    # Track upward
    ul, ur, uy = _track_one_direction(
        gray, left_start, right_start, start_y, top_y, -step,
        search_radius, sigma, max_width, max_jump)

    # Track downward
    dl, dr, dy = _track_one_direction(
        gray, left_start, right_start, start_y, h - 10, step,
        search_radius, sigma, max_width, max_jump)

    # Combine: upward (reversed) + anchor + downward
    left_track = ul[::-1] + [left_start] + dl
    right_track = ur[::-1] + [right_start] + dr
    y_track = uy[::-1] + [start_y] + dy

    return {
        "y": np.array(y_track),
        "left": np.array(left_track),
        "right": np.array(right_track),
    }


def robust_line_fit(y_vals, x_vals, max_iter=10, threshold=20):
    """Iteratively fit a line, removing outliers."""
    mask = np.ones(len(y_vals), dtype=bool)
    coeffs = None
    for _ in range(max_iter):
        yf, xf = y_vals[mask], x_vals[mask]
        if len(yf) < 5:
            break
        coeffs = np.polyfit(yf, xf, 1)
        res = np.abs(x_vals - np.polyval(coeffs, y_vals))
        new_mask = res < threshold
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
    return coeffs, mask


def detect_lane(frame: np.ndarray) -> dict | None:
    """Main lane detection pipeline."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"\nFrame: {w}x{h}, mean brightness: {gray.mean():.1f}")

    # Scan for lane candidates at multiple y levels
    print("\n--- Scanning for lane candidates ---")
    all_candidates = []  # (y, y_pct, lane_dict)

    for y_pct in range(90, 20, -5):
        y = int(h * y_pct / 100)
        candidates = find_lane_edges_at_y(gray, y, sigma=30)
        if candidates:
            # Pick the brightest candidate near center
            candidates.sort(key=lambda c: abs(c["center"] - w // 2))
            lane = candidates[0]
            print(f"  y={y:4d} ({y_pct}%): lane x={lane['left']}-{lane['right']}, "
                  f"w={lane['width']}, bright={lane['brightness']}")
            all_candidates.append((y, y_pct, lane))

    if not all_candidates:
        print("\nNo bowling lane detected in frame!")
        return None

    # Pick anchor: prefer middle of frame (40-70%), with good brightness,
    # and consistent with neighbors
    def anchor_score(idx):
        y, y_pct, lane = all_candidates[idx]
        # Prefer mid-frame
        pct_score = 1.0 - abs(y_pct - 55) / 50.0
        # Prefer brighter
        bright_score = lane["brightness"] / 200.0
        # Prefer consistency with neighbors
        consistency = 0
        for j in range(max(0, idx - 2), min(len(all_candidates), idx + 3)):
            if j == idx:
                continue
            other = all_candidates[j][2]
            center_diff = abs(lane["center"] - other["center"])
            if center_diff < 100:
                consistency += 1
        return pct_score + bright_score + consistency * 0.3

    best_idx = max(range(len(all_candidates)), key=anchor_score)
    best_y, best_pct, best_lane = all_candidates[best_idx]

    # Refine right edges using finer sigma to resolve gutter between lanes
    print("\n--- Refining right edges with fine sigma ---")
    refined_rights = []
    for cy, cy_pct, clane in all_candidates:
        if clane["brightness"] < 80:
            continue
        refined = refine_right_edge(gray, cy, clane["left"])
        if refined is not None:
            refined_rights.append((cy, refined))
            orig_r = clane["right"]
            print(f"  y={cy:4d}: coarse right={orig_r}, refined right={refined} (delta={refined - orig_r:+d})")

    # Use refined right edge for the anchor too
    anchor_refined = refine_right_edge(gray, best_y, best_lane["left"])
    if anchor_refined is not None:
        best_lane = {**best_lane, "right": anchor_refined,
                     "width": anchor_refined - best_lane["left"]}

    print(f"\n--- Best lane anchor at y={best_y} ({best_pct}%): "
          f"x={best_lane['left']}-{best_lane['right']}, w={best_lane['width']} ---")

    # Track edges from the best anchor point
    track = track_lane_edges(gray, best_lane["left"], best_lane["right"], best_y)
    n_points = len(track["y"])
    print(f"  Tracked {n_points} points (y={track['y'].min()} to y={track['y'].max()})")

    if n_points < 5:
        print("  Not enough tracking points for a reliable fit.")
        return None

    # Fit lines to the tracked edges
    left_coeffs, left_mask = robust_line_fit(track["y"], track["left"])
    right_coeffs, right_mask = robust_line_fit(track["y"], track["right"])

    left_inliers = int(np.sum(left_mask))
    right_inliers = int(np.sum(right_mask))
    print(f"  Left fit:  x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}  ({left_inliers}/{n_points} inliers)")
    print(f"  Right fit: x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}  ({right_inliers}/{n_points} inliers)")

    # Fallback: if right edge tracking is poor (<30% inliers), use refined scan data
    min_inlier_ratio = 0.30
    if right_inliers / n_points < min_inlier_ratio and len(refined_rights) >= 3:
        print(f"  Right edge poor ({right_inliers}/{n_points}), using {len(refined_rights)} refined scan points...")
        scan_y_f = np.array([r[0] for r in refined_rights])
        scan_right_f = np.array([r[1] for r in refined_rights])
            best_inliers = 0
            best_coeffs = right_coeffs
            # Try all pairs for RANSAC
            for i in range(len(good)):
                for j in range(i + 1, len(good)):
                    y2 = np.array([scan_y_f[i], scan_y_f[j]])
                    x2 = np.array([scan_right_f[i], scan_right_f[j]])
                    c2 = np.polyfit(y2, x2, 1)
                    res = np.abs(scan_right_f - np.polyval(c2, scan_y_f))
                    n_in = int(np.sum(res < 40))
                    if n_in > best_inliers:
                        best_inliers = n_in
                        best_coeffs = c2
            # Refit using all inliers
            res = np.abs(scan_right_f - np.polyval(best_coeffs, scan_y_f))
            inlier_mask = res < 40
            if np.sum(inlier_mask) >= 2:
                right_coeffs = np.polyfit(scan_y_f[inlier_mask], scan_right_f[inlier_mask], 1)
                right_inliers = int(np.sum(inlier_mask))
            print(f"  RANSAC right fit: x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}  "
                  f"({right_inliers}/{len(good)} scan inliers)")

    if left_inliers / n_points < min_inlier_ratio and len(all_candidates) >= 3:
        print(f"  Left edge poor ({left_inliers}/{n_points}), falling back to scan-level RANSAC...")
        good = [(c[0], c[2]["left"]) for c in all_candidates if c[2]["brightness"] > 80]
        if len(good) >= 3:
            scan_y_f = np.array([g[0] for g in good])
            scan_left_f = np.array([g[1] for g in good])
            best_inliers = 0
            best_coeffs = left_coeffs
            for i in range(len(good)):
                for j in range(i + 1, len(good)):
                    y2 = np.array([scan_y_f[i], scan_y_f[j]])
                    x2 = np.array([scan_left_f[i], scan_left_f[j]])
                    c2 = np.polyfit(y2, x2, 1)
                    res = np.abs(scan_left_f - np.polyval(c2, scan_y_f))
                    n_in = int(np.sum(res < 40))
                    if n_in > best_inliers:
                        best_inliers = n_in
                        best_coeffs = c2
            res = np.abs(scan_left_f - np.polyval(best_coeffs, scan_y_f))
            inlier_mask = res < 40
            if np.sum(inlier_mask) >= 2:
                left_coeffs = np.polyfit(scan_y_f[inlier_mask], scan_left_f[inlier_mask], 1)
                left_inliers = int(np.sum(inlier_mask))
            print(f"  RANSAC left fit: x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}  "
                  f"({left_inliers}/{len(good)} scan inliers)")

    # Lane widths at key y levels
    print("\n--- Lane dimensions ---")
    for label, y_check in [("top of track", int(track["y"].min())),
                           ("middle", int((track["y"].min() + track["y"].max()) / 2)),
                           ("bottom/anchor", int(track["y"].max()))]:
        lx = np.polyval(left_coeffs, y_check)
        rx = np.polyval(right_coeffs, y_check)
        print(f"  {label} (y={y_check}): left={lx:.0f}, right={rx:.0f}, width={rx - lx:.0f}")

    # Vanishing point
    vp_y = vp_x = None
    if abs(right_coeffs[0] - left_coeffs[0]) > 0.001:
        vp_y = (left_coeffs[1] - right_coeffs[1]) / (right_coeffs[0] - left_coeffs[0])
        vp_x = np.polyval(left_coeffs, vp_y)
        print(f"\n  Vanishing point: ({vp_x:.0f}, {vp_y:.0f})")

    return {
        "left_coeffs": left_coeffs,
        "right_coeffs": right_coeffs,
        "track": track,
        "anchor_y": best_y,
        "anchor_lane": best_lane,
        "vanishing_point": (vp_x, vp_y) if vp_y is not None else None,
    }


def draw_lane_overlay(frame: np.ndarray, detection: dict) -> np.ndarray:
    """Draw lane detection results on the frame."""
    out = frame.copy()
    h, w = frame.shape[:2]
    lc = detection["left_coeffs"]
    rc = detection["right_coeffs"]
    track = detection["track"]

    y_min = int(track["y"].min())
    y_max = int(track["y"].max())

    # Draw lane polygon
    left_pts = []
    right_pts = []
    for y in range(y_min, y_max + 1, 3):
        lx = int(np.polyval(lc, y))
        rx = int(np.polyval(rc, y))
        left_pts.append([lx, y])
        right_pts.append([rx, y])

    polygon = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [polygon], (0, 180, 0))  # green lane fill
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

    # Draw edge lines
    for y in range(y_min, y_max + 1, 2):
        lx = int(np.polyval(lc, y))
        rx = int(np.polyval(rc, y))
        cv2.circle(out, (lx, y), 1, (0, 255, 0), -1)
        cv2.circle(out, (rx, y), 1, (0, 0, 255), -1)

    # Draw tracked points
    for i in range(len(track["y"])):
        y = int(track["y"][i])
        cv2.circle(out, (int(track["left"][i]), y), 2, (255, 255, 0), -1)
        cv2.circle(out, (int(track["right"][i]), y), 2, (255, 0, 255), -1)

    # Draw anchor line
    anchor_y = detection["anchor_y"]
    lane = detection["anchor_lane"]
    cv2.line(out, (lane["left"], anchor_y), (lane["right"], anchor_y), (0, 255, 255), 2)

    # Vanishing point
    vp = detection.get("vanishing_point")
    if vp and vp[0] is not None:
        vpx, vpy = int(vp[0]), int(vp[1])
        if 0 <= vpx < w and 0 <= vpy < h:
            cv2.drawMarker(out, (vpx, vpy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

    # Labels
    cv2.putText(out, "LANE DETECTED", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    lw = lane["right"] - lane["left"]
    cv2.putText(out, f"Width at anchor: {lw}px", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else RTMP_URL

    # Use saved frame if available, otherwise capture live
    import os
    frame_path = os.path.join(os.path.dirname(__file__), "livestream_frame.jpg")
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
    out_path = os.path.join(os.path.dirname(__file__), "lane_detected.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated frame saved to: {out_path}")
    print("LANE DETECTION SUCCESSFUL")


if __name__ == "__main__":
    main()
