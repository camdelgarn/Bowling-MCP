#!/usr/bin/env python3
"""
Detect bowling lane from RTMP livestream using multi-row approach dot calibration.

Bowling approach has 2-3 rows of alignment dots:
  Row 1 (12 ft from foul line): big center dot + 3 smaller dots each side (7 dots)
  Row 2 (15 ft from foul line): big center dot + 2 smaller dots each side (5 dots)
  Row 3 (further back, may not be visible): similar pattern

All dots are spaced 5 boards apart, center dot = board 20.
With 2+ rows we get exact perspective calibration for both lane edges.

Usage:
  python detect_lane_livestream.py              # Use saved frame
  python detect_lane_livestream.py --live       # Capture fresh frame
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import sys
import os
import json

RTMP_URL = "rtmp://192.168.1.7:1935/live/stream"

# Load lane configuration from lane_config.json (see LANE_CONFIG.md for docs)
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lane_config.json")
with open(_config_path) as _f:
    LANE_CONFIG = json.load(_f)

BOARDS_PER_LANE = LANE_CONFIG["boards_per_lane"]
DOTS_SPACING_BOARDS = LANE_CONFIG["dot_spacing_boards"]
CENTER_DOT_BOARD = LANE_CONFIG["center_dot_board"]
EXPECTED_DOT_COUNT = LANE_CONFIG["dot_count"]
BOWLER_HANDEDNESS = LANE_CONFIG.get("bowler_handedness", "right")


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


def find_all_dot_rows(gray: np.ndarray) -> list[list[dict]]:
    """Find all rows of approach dots using blob detection.

    Returns a list of dot rows, each row sorted by x.
    Each dot is a dict with x, y, size, contrast.
    """
    h, w = gray.shape

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 8
    params.maxArea = 600
    params.filterByCircularity = True
    params.minCircularity = 0.20
    params.filterByConvexity = True
    params.minConvexity = 0.30
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)

    # Search bottom 60% of frame (approach area)
    roi_top = int(h * 0.35)
    roi_bot = int(h * 0.85)
    roi = gray[roi_top:roi_bot, :]
    keypoints = detector.detect(roi)

    print(f"  Raw blobs found: {len(keypoints)}")

    # Filter to high-contrast blobs (dark dot on lighter lane)
    blobs = []
    for kp in keypoints:
        x, y_local = kp.pt
        y_abs = y_local + roi_top
        ix, iy = int(x), int(y_abs)
        if iy < 0 or iy >= h or ix < 0 or ix >= w:
            continue
        # Use minimum pixel in a small patch around center — blob detector
        # center can be off by 1-2px, missing the dark core
        r = max(2, int(kp.size / 2))
        cy1 = max(0, iy - r)
        cy2 = min(h, iy + r + 1)
        cx1 = max(0, ix - r)
        cx2 = min(w, ix + r + 1)
        px_val = gray[cy1:cy2, cx1:cx2].min()
        y1 = max(0, iy - 15)
        y2 = min(h, iy + 15)
        x1 = max(0, ix - 15)
        x2 = min(w, ix + 15)
        neighborhood = gray[y1:y2, x1:x2].mean()
        contrast = neighborhood - px_val
        if contrast > 40:
            blobs.append({
                "x": ix, "y": iy,
                "size": round(kp.size, 1),
                "contrast": round(contrast, 1),
            })

    print(f"  High-contrast blobs: {len(blobs)}")
    if len(blobs) < 3:
        return []

    blobs.sort(key=lambda b: b["x"])

    # Cluster blobs into rows by y-proximity
    # Use wider tolerance (60px) because dot rows tilt due to perspective
    by_y = sorted(blobs, key=lambda b: b["y"])
    rows_clusters = []
    used = set()

    for i, b in enumerate(by_y):
        if i in used:
            continue
        cluster = [b]
        used.add(i)
        # Grow cluster: include anything within 60px of any existing member
        changed = True
        while changed:
            changed = False
            for j in range(len(by_y)):
                if j in used:
                    continue
                for member in cluster:
                    if abs(by_y[j]["y"] - member["y"]) < 60:
                        cluster.append(by_y[j])
                        used.add(j)
                        changed = True
                        break
        if len(cluster) >= 3:
            rows_clusters.append(cluster)

    print(f"  Blob row clusters (>=3 blobs): {len(rows_clusters)}")
    for ci, cluster in enumerate(rows_clusters):
        ys = [b["y"] for b in cluster]
        xs = [b["x"] for b in cluster]
        print(f"    Cluster {ci+1}: {len(cluster)} blobs, y~{np.mean(ys):.0f}, "
              f"x range {min(xs)}-{max(xs)}")

    # For each cluster, find runs of evenly-spaced collinear dots.
    # Extract multiple runs per cluster: two real dot rows can merge into one
    # cluster when bridge blobs connect them through the y-tolerance.
    dot_rows = []
    for cluster in rows_clusters:
        remaining = list(cluster)
        remaining.sort(key=lambda b: b["x"])
        for _ in range(3):  # extract up to 3 runs per cluster
            best_run = _find_best_dot_run(remaining)
            if best_run and len(best_run) >= 3:
                dot_rows.append(best_run)
                # Remove found dots from remaining
                run_set = {(d["x"], d["y"]) for d in best_run}
                remaining = [b for b in remaining if (b["x"], b["y"]) not in run_set]
                remaining.sort(key=lambda b: b["x"])
            else:
                break
        if not any(r for r in dot_rows):
            print(f"    (cluster at y~{np.mean([b['y'] for b in cluster]):.0f}: "
                  f"no dot run found)")

    # Filter out runs with excessive y-span. Real approach dots form a roughly
    # horizontal line; scattered lane artifacts produce runs spanning many rows.
    max_y_span = int(h * 0.05)  # ~54px for 1080p
    filtered = []
    for row in dot_rows:
        ys = [d["y"] for d in row]
        y_span = max(ys) - min(ys)
        if y_span <= max_y_span:
            filtered.append(row)
        else:
            print(f"    Rejecting run at y~{np.mean(ys):.0f}: "
                  f"y_span={y_span}px exceeds {max_y_span}px")
    dot_rows = filtered

    # Sort rows by y (smaller y = closer to foul line = further from camera)
    dot_rows.sort(key=lambda row: np.mean([d["y"] for d in row]))

    return dot_rows


def _find_best_dot_run(blobs: list[dict]) -> list[dict] | None:
    """Find the longest run of evenly-spaced collinear dots in a blob list.

    When runs tie in length, prefer higher average contrast (real dots are darker).
    After finding the best run, prune any outlier dots whose contrast or size
    deviates significantly from the rest.
    """
    best_run = []
    best_score = 0

    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            dx = blobs[j]["x"] - blobs[i]["x"]
            dy = blobs[j]["y"] - blobs[i]["y"]
            dist = (dx**2 + dy**2)**0.5
            if dist < 25 or dist > 200:
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

            # Score: length * average contrast (prefer more dots AND higher contrast)
            avg_contrast = np.mean([d["contrast"] for d in run])
            score = len(run) * avg_contrast

            if score > best_score:
                best_score = score
                best_run = run

    if len(best_run) < 3:
        return None

    # Prune outlier dots from the run: remove any dot whose contrast is less
    # than 50% of the median contrast of the run (catches false positives like
    # screw covers or lane markings that happen to be collinear)
    contrasts = [d["contrast"] for d in best_run]
    median_contrast = np.median(contrasts)
    # Only prune from the ends (interior dots are spacing-validated)
    pruned = list(best_run)
    while len(pruned) > 3 and pruned[-1]["contrast"] < median_contrast * 0.5:
        pruned.pop()
    while len(pruned) > 3 and pruned[0]["contrast"] < median_contrast * 0.5:
        pruned.pop(0)

    return pruned if len(pruned) >= 3 else None


def analyze_dot_row(dots: list[dict], board_numbers: list[int]) -> dict:
    """Analyze a row of dots given their board number assignments.

    Args:
        dots: List of dot dicts sorted by x.
        board_numbers: Board number for each dot (same length as dots).

    Returns dict with center, board positions, px_per_board, left/right edge positions.
    """
    center_idx = None
    for i, b in enumerate(board_numbers):
        if b == CENTER_DOT_BOARD:
            center_idx = i
            break
    if center_idx is None:
        # CENTER_DOT_BOARD not in list; pick closest to board 20
        center_idx = min(range(len(dots)),
                         key=lambda i: abs(board_numbers[i] - CENTER_DOT_BOARD))

    center_dot = dots[center_idx]

    xs = np.array([d["x"] for d in dots], dtype=float)
    ys = np.array([d["y"] for d in dots], dtype=float)
    boards = np.array(board_numbers, dtype=float)

    # Linear fit: board number -> pixel x/y
    bx_fit = np.polyfit(boards, xs, 1)
    by_fit = np.polyfit(boards, ys, 1)

    px_per_board = bx_fit[0]
    mean_y = float(np.mean(ys))

    left_x = np.polyval(bx_fit, 0)
    left_y = np.polyval(by_fit, 0)
    right_x = np.polyval(bx_fit, BOARDS_PER_LANE)
    right_y = np.polyval(by_fit, BOARDS_PER_LANE)

    return {
        "dots": dots,
        "center": center_dot,
        "center_idx": center_idx,
        "dot_count": len(dots),
        "board_numbers": board_numbers,
        "mean_y": mean_y,
        "px_per_board": px_per_board,
        "bx_fit": bx_fit,
        "by_fit": by_fit,
        "left": (left_x, left_y),
        "right": (right_x, right_y),
    }


def _assign_boards_by_center(dots: list[dict]) -> list[int]:
    """Assign board numbers assuming middle dot is the center dot (board 20).

    Standard layout: dots are symmetric around center.
      5 dots → boards [10, 15, 20, 25, 30]
      7 dots → boards [5, 10, 15, 20, 25, 30, 35]
    """
    mid_idx = len(dots) // 2
    return [CENTER_DOT_BOARD + (i - mid_idx) * DOTS_SPACING_BOARDS
            for i in range(len(dots))]


def _assign_boards_via_perspective(ref_row: dict, other_dots: list[dict]) -> list[int]:
    """Assign board numbers to a farther row using perspective lines from a reference row.

    Dots at the same board number across rows form straight lines converging to
    the vanishing point. Try all possible board assignments for other_dots, and
    pick the one where connecting lines converge most tightly.
    """
    ref_dots = ref_row["dots"]
    ref_boards = ref_row["board_numbers"]

    n = len(other_dots)
    # Generate candidate board assignments: consecutive multiples of 5
    # The dots could be at boards like [5,10,15,20], [10,15,20,25], etc.
    min_start = -5  # allow a bit below 0
    max_start = BOARDS_PER_LANE + 5  # allow a bit above 39
    candidates = []
    for start_board in range(min_start, max_start + 1, DOTS_SPACING_BOARDS):
        assignment = [start_board + i * DOTS_SPACING_BOARDS for i in range(n)]
        # At least some boards should overlap with reference row
        overlap = set(assignment) & set(ref_boards)
        if len(overlap) >= 2:
            candidates.append(assignment)

    if not candidates:
        # Fallback: no overlap possible, just use center-of-row guess
        mid_idx = n // 2
        return [CENTER_DOT_BOARD + (i - mid_idx) * DOTS_SPACING_BOARDS
                for i in range(n)]

    best_assignment = None
    best_score = float("inf")

    for assignment in candidates:
        # Find pairs of dots at the same board number
        pairs = []
        for i, board in enumerate(assignment):
            if board in ref_boards:
                ref_idx = ref_boards.index(board)
                ref_d = ref_dots[ref_idx]
                oth_d = other_dots[i]
                pairs.append((ref_d["x"], ref_d["y"], oth_d["x"], oth_d["y"]))

        if len(pairs) < 2:
            continue

        # Each pair defines a line. Compute where these lines intersect (VP).
        # For N pairs, compute pairwise VP intersections and measure spread.
        vp_xs = []
        vp_ys = []
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                x1a, y1a, x2a, y2a = pairs[a]
                x1b, y1b, x2b, y2b = pairs[b]
                # Line a: from (x1a,y1a) to (x2a,y2a)
                # Line b: from (x1b,y1b) to (x2b,y2b)
                dxa = x2a - x1a
                dya = y2a - y1a
                dxb = x2b - x1b
                dyb = y2b - y1b
                denom = dxa * dyb - dya * dxb
                if abs(denom) < 1e-6:
                    continue
                t = ((x1b - x1a) * dyb - (y1b - y1a) * dxb) / denom
                vp_x = x1a + t * dxa
                vp_y = y1a + t * dya
                vp_xs.append(vp_x)
                vp_ys.append(vp_y)

        if len(vp_xs) < 1:
            continue

        # Score = spread of VP estimates (lower is better = tighter convergence)
        # Penalize assignments with fewer pairs: fewer intersections means less
        # confidence. With only 1 intersection, std=0 which would falsely win.
        if len(vp_xs) >= 2:
            spread = np.std(vp_xs) + np.std(vp_ys)
        else:
            # Only 1 intersection — assign large penalty so multi-pair wins
            spread = 1e6
        # Tie-break: prefer more overlapping pairs
        score = spread - len(pairs) * 0.01
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        mid_idx = n // 2
        return [CENTER_DOT_BOARD + (i - mid_idx) * DOTS_SPACING_BOARDS
                for i in range(n)]

    return best_assignment


def detect_lane(frame: np.ndarray) -> dict | None:
    """Main lane detection pipeline using multi-row approach dot calibration."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"\nFrame: {w}x{h}, mean brightness: {gray.mean():.1f}")

    # Step 1: Find all dot rows
    print("\n--- Step 1: Finding approach dot rows ---")
    dot_rows_raw = find_all_dot_rows(gray)

    if len(dot_rows_raw) == 0:
        print("  No dot rows found!")
        return None

    # Analyze rows — all rows use middle dot = center (board 20).
    # Standard bowling approach: dots are symmetric around board 20.
    row_data = []
    for i, dots in enumerate(dot_rows_raw):
        boards = _assign_boards_by_center(dots)
        method = "middle=center"

        info = analyze_dot_row(dots, boards)
        mean_y = info["mean_y"]
        n = info["dot_count"]
        bpx = info["px_per_board"]
        cx = info["center"]["x"]
        lx, ly = info["left"]
        rx, ry = info["right"]
        lane_width_boards = (rx - lx) / bpx if bpx > 0 else 0

        print(f"\n  Row {i+1}: {n} dots, center at ({cx}, {mean_y:.0f})"
              f" [{method}]")
        print(f"    Board width: {bpx:.2f} px/board")
        print(f"    Board numbers: {info['board_numbers']}")
        print(f"    Left edge (board 0): x={lx:.0f}, y={ly:.0f}")
        print(f"    Right edge (board {BOARDS_PER_LANE}): x={rx:.0f}, y={ry:.0f}")
        print(f"    Lane width: {rx - lx:.0f}px = {lane_width_boards:.1f} boards")

        row_data.append(info)

    # Pre-filter: prefer rows with more dots. If we have 5-dot rows,
    # drop any with 3 dots (likely lane surface artifacts).
    if len(row_data) > 2:
        max_dots = max(rd["dot_count"] for rd in row_data)
        min_keep = max(3, max_dots - 1)  # keep within 1 of the best
        before = len(row_data)
        row_data = [rd for rd in row_data if rd["dot_count"] >= min_keep]
        if len(row_data) < before:
            print(f"\n  Pre-filter: kept {len(row_data)} rows with >= {min_keep} dots "
                  f"(dropped {before - len(row_data)} short rows)")

    # Pre-filter: when multiple rows have similar y, keep only the one with
    # highest average contrast (real dots are much darker than artifacts).
    if len(row_data) > 1:
        row_data.sort(key=lambda rd: rd["mean_y"])
        merged = [row_data[0]]
        for rd in row_data[1:]:
            if abs(rd["mean_y"] - merged[-1]["mean_y"]) < 100:
                # Keep whichever has higher average contrast
                avg_c_new = np.mean([d["contrast"] for d in rd["dots"]])
                avg_c_old = np.mean([d["contrast"] for d in merged[-1]["dots"]])
                if avg_c_new > avg_c_old:
                    print(f"  Merging rows at y~{merged[-1]['mean_y']:.0f} and "
                          f"y~{rd['mean_y']:.0f}: keeping higher contrast "
                          f"({avg_c_new:.0f} > {avg_c_old:.0f})")
                    merged[-1] = rd
                else:
                    print(f"  Merging rows at y~{merged[-1]['mean_y']:.0f} and "
                          f"y~{rd['mean_y']:.0f}: keeping higher contrast "
                          f"({avg_c_old:.0f} > {avg_c_new:.0f})")
            else:
                merged.append(rd)
        row_data = merged

    # Perspective monotonicity: px_per_board must increase with y (rows
    # closer to camera appear wider). Remove any row that violates this.
    if len(row_data) > 1:
        row_data.sort(key=lambda rd: rd["mean_y"])
        changed = True
        while changed and len(row_data) > 1:
            changed = False
            for i in range(len(row_data) - 1):
                if row_data[i]["px_per_board"] > row_data[i + 1]["px_per_board"]:
                    # Violation: farther row has larger px_per_board.
                    # Remove whichever has fewer dots; if tied, remove the farther one.
                    if row_data[i]["dot_count"] < row_data[i + 1]["dot_count"]:
                        bad = i
                    elif row_data[i]["dot_count"] > row_data[i + 1]["dot_count"]:
                        bad = i + 1
                    else:
                        bad = i  # farther row is more likely a false positive
                    rd_bad = row_data[bad]
                    print(f"  Perspective filter: removing row at y~{rd_bad['mean_y']:.0f} "
                          f"(px/board={rd_bad['px_per_board']:.1f}, {rd_bad['dot_count']} dots) "
                          f"— violates monotonicity")
                    row_data.pop(bad)
                    changed = True
                    break

    # Step 2: Compute lane edges from dot rows
    print("\n--- Step 2: Computing lane edges ---")

    if len(row_data) >= 2:
        # Two or more rows: fit edge lines, then validate each row by
        # checking the cross-row lane width is ~39 boards. Iteratively
        # remove the worst outlier until all rows pass.
        def _fit_and_verify(rows):
            left_points = [(rd["left"][1], rd["left"][0]) for rd in rows]
            right_points = [(rd["right"][1], rd["right"][0]) for rd in rows]
            lys = np.array([p[0] for p in left_points])
            lxs = np.array([p[1] for p in left_points])
            rys = np.array([p[0] for p in right_points])
            rxs = np.array([p[1] for p in right_points])
            lc = np.polyfit(lys, lxs, 1)
            rc = np.polyfit(rys, rxs, 1)
            errors = []
            for rd in rows:
                y = rd["mean_y"]
                width = np.polyval(rc, y) - np.polyval(lc, y)
                board_count = width / rd["px_per_board"]
                errors.append(abs(board_count - BOARDS_PER_LANE))
            return lc, rc, errors

        while len(row_data) >= 2:
            lc_tmp, rc_tmp, errs = _fit_and_verify(row_data)
            worst_idx = int(np.argmax(errs))
            if errs[worst_idx] > 3:
                rd_bad = row_data[worst_idx]
                y = rd_bad["mean_y"]
                width = np.polyval(rc_tmp, y) - np.polyval(lc_tmp, y)
                board_count = width / rd_bad["px_per_board"]
                print(f"  Rejecting row at y~{y:.0f}: cross-verify = "
                      f"{board_count:.1f} boards (expected ~{BOARDS_PER_LANE})")
                row_data.pop(worst_idx)
            else:
                break

    if len(row_data) >= 2:
        print(f"  Using {len(row_data)} dot rows for perspective calibration")

        left_points = []
        right_points = []
        for rd in row_data:
            lx, ly = rd["left"]
            rx, ry = rd["right"]
            left_points.append((ly, lx))
            right_points.append((ry, rx))

        left_ys = np.array([p[0] for p in left_points])
        left_xs = np.array([p[1] for p in left_points])
        right_ys = np.array([p[0] for p in right_points])
        right_xs = np.array([p[1] for p in right_points])

        # Fit edge lines: x = slope*y + intercept
        left_coeffs = np.polyfit(left_ys, left_xs, 1)
        right_coeffs = np.polyfit(right_ys, right_xs, 1)

        print(f"  Left edge:  x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}")
        print(f"  Right edge: x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}")

        for i, rd in enumerate(row_data):
            y = rd["mean_y"]
            lx = np.polyval(left_coeffs, y)
            rx = np.polyval(right_coeffs, y)
            width = rx - lx
            boards = width / rd["px_per_board"]
            print(f"  Row {i+1} verify: width={width:.0f}px = {boards:.1f} boards at y={y:.0f}")

    elif len(row_data) == 1:
        # Single row: dots for position + gradient for slope
        print("  Single dot row — using dots + gradient for edge slope")
        rd = row_data[0]
        lx, ly = rd["left"]
        rx, ry = rd["right"]

        left_slope = _find_edge_slope(gray, lx, ly, side="left")
        right_slope = _find_edge_slope(gray, rx, ry, side="right")

        left_coeffs = np.array([left_slope, lx - left_slope * ly])
        right_coeffs = np.array([right_slope, rx - right_slope * ry])

        print(f"  Left edge:  x = {left_coeffs[0]:.4f}*y + {left_coeffs[1]:.1f}")
        print(f"  Right edge: x = {right_coeffs[0]:.4f}*y + {right_coeffs[1]:.1f}")
    else:
        return None

    # Vanishing point
    vp_y = vp_x = None
    slope_diff = right_coeffs[0] - left_coeffs[0]
    if abs(slope_diff) > 0.001:
        vp_y = (left_coeffs[1] - right_coeffs[1]) / slope_diff
        vp_x = np.polyval(left_coeffs, vp_y)
        print(f"\n  Vanishing point: ({vp_x:.0f}, {vp_y:.0f})")

    # Visible y range: extend well beyond dot rows since we have good edge lines
    dot_y_min = int(min(rd["mean_y"] for rd in row_data))
    dot_y_max = int(max(rd["mean_y"] for rd in row_data))
    y_top = max(50, dot_y_min - 300)
    y_bot = min(h - 50, dot_y_max + 300)

    # Track left edge with gradient for visual verification
    rd_ref = row_data[0]
    lx_ref, ly_ref = rd_ref["left"]
    left_track = _track_edge_from_anchor(gray, int(lx_ref), int(ly_ref),
                                          left_coeffs, search_radius=40)
    print(f"  Gradient tracking: {len(left_track['y'])} points "
          f"(y={left_track['y'].min() if len(left_track['y']) > 0 else '?'} to "
          f"y={left_track['y'].max() if len(left_track['y']) > 0 else '?'})")

    # Results summary
    print(f"\n--- Lane Dimensions (y={y_top} to y={y_bot}) ---")
    for label, y_check in [("top", y_top), ("middle", (y_top + y_bot) // 2), ("bottom", y_bot)]:
        lx = np.polyval(left_coeffs, y_check)
        rx = np.polyval(right_coeffs, y_check)
        width_px = rx - lx
        if len(row_data) >= 2:
            bpx = _interpolate_board_px(row_data, y_check)
        else:
            bpx = row_data[0]["px_per_board"]
        boards = width_px / bpx
        print(f"  {label:6s} (y={y_check:4d}): left={lx:6.0f}, right={rx:6.0f}, "
              f"width={width_px:.0f}px ({boards:.1f} boards, {bpx:.2f} px/board)")

    return {
        "left_coeffs": left_coeffs,
        "right_coeffs": right_coeffs,
        "left_track": left_track,
        "row_data": row_data,
        "y_range": (y_top, y_bot),
        "vanishing_point": (vp_x, vp_y) if vp_y is not None else None,
    }


def _find_edge_slope(gray: np.ndarray, anchor_x: float, anchor_y: float,
                     side: str = "left") -> float:
    """Find the slope of a lane edge near an anchor point using gradient tracking."""
    h, w = gray.shape
    points_y = []
    points_x = []

    for dy in range(-150, 151, 10):
        y = int(anchor_y + dy)
        if y < 10 or y >= h - 10:
            continue
        expected_x = anchor_x + dy * (-0.4)
        sx = max(0, int(expected_x - 60))
        ex = min(w, int(expected_x + 60))
        if sx >= ex:
            continue

        row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=15)
        grad = np.gradient(row)

        if side == "left":
            peaks, props = find_peaks(grad[sx:ex], height=0.10, distance=8)
        else:
            peaks, props = find_peaks(-grad[sx:ex], height=0.10, distance=8)

        if len(peaks) > 0:
            abs_peaks = peaks + sx
            dists = np.abs(abs_peaks - expected_x)
            best = np.argmin(dists)
            if dists[best] < 40:
                points_y.append(y)
                points_x.append(int(abs_peaks[best]))

    if len(points_y) >= 5:
        coeffs = np.polyfit(points_y, points_x, 1)
        return coeffs[0]
    return -0.4


def _track_edge_from_anchor(gray: np.ndarray, start_x: int, start_y: int,
                             edge_coeffs: np.ndarray,
                             search_radius: int = 40, step: int = 3) -> dict:
    """Track a lane edge bidirectionally, guided by the fitted line."""
    h, w = gray.shape
    xs, ys = [], []

    for direction in [-1, 1]:
        prev_x = start_x
        misses = 0
        for i in range(1, 300):
            y = start_y + direction * step * i
            if y < 10 or y >= h - 10:
                break

            expected_x = int(np.polyval(edge_coeffs, y))
            row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=15)
            grad = np.gradient(row)

            sl = max(0, expected_x - search_radius)
            sr = min(w, expected_x + search_radius)
            peaks, props = find_peaks(grad[sl:sr], height=0.08, distance=8)

            if len(peaks) > 0:
                abs_peaks = peaks + sl
                dists = np.abs(abs_peaks - expected_x).astype(float)
                scores = props["peak_heights"] / (1.0 + dists / 15.0)
                new_x = int(abs_peaks[np.argmax(scores)])
                if abs(new_x - prev_x) > 35:
                    misses += 1
                    if misses > 5:
                        break
                    continue
                xs.append(new_x)
                ys.append(y)
                prev_x = new_x
                misses = 0
            else:
                misses += 1
                if misses > 5:
                    break

    if xs:
        order = np.argsort(ys)
        return {"x": np.array(xs)[order], "y": np.array(ys)[order]}
    return {"x": np.array([start_x]), "y": np.array([start_y])}


def _interpolate_board_px(row_data: list[dict], y: float) -> float:
    """Interpolate board pixel width at a given y using multiple dot rows."""
    ys = [rd["mean_y"] for rd in row_data]
    bpxs = [rd["px_per_board"] for rd in row_data]
    if len(ys) == 1:
        return bpxs[0]
    coeffs = np.polyfit(ys, bpxs, 1)
    return float(np.polyval(coeffs, y))


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

    # Tracked left edge points
    track = detection.get("left_track", {})
    if "y" in track:
        for i in range(len(track["y"])):
            cv2.circle(out, (int(track["x"][i]), int(track["y"][i])), 2, (255, 255, 0), -1)

    # Dot rows with different colors
    row_data = detection.get("row_data", [])
    colors = [(0, 0, 255), (255, 0, 255), (255, 128, 0)]
    for ri, rd in enumerate(row_data):
        color = colors[ri % len(colors)]
        center = rd["center"]
        for d in rd["dots"]:
            is_center = (d["x"] == center["x"] and d["y"] == center["y"])
            radius = 15 if is_center else 10
            cv2.circle(out, (d["x"], d["y"]), radius, color, 2)
            if is_center:
                cv2.putText(out, f"Row {ri+1}", (d["x"] + 20, d["y"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Board tick marks at this row
        y_row = int(rd["mean_y"])
        lx_at_row = np.polyval(lc, y_row)
        bpx = rd["px_per_board"]
        for b in range(0, BOARDS_PER_LANE + 1, 5):
            bx = int(lx_at_row + b * bpx)
            if 0 <= bx < w:
                cv2.line(out, (bx, y_row - 8), (bx, y_row + 8), (255, 200, 0), 1)
                if b % 10 == 0:
                    cv2.putText(out, str(b), (bx - 8, y_row - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

    # Info labels
    cv2.putText(out, "LANE DETECTED", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    info = [f"{len(row_data)} dot row(s) found"]
    for ri, rd in enumerate(row_data):
        info.append(f"Row {ri+1}: {rd['dot_count']} dots, {rd['px_per_board']:.1f} px/board")
    mid_y = (y_top + y_bot) // 2
    lw = np.polyval(rc, mid_y) - np.polyval(lc, mid_y)
    info.append(f"width@mid = {lw:.0f}px")
    for i, text in enumerate(info):
        cv2.putText(out, text, (20, 80 + i * 25),
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
