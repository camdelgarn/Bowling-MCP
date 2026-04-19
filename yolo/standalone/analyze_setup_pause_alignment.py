import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


# COCO keypoint indices.
L_SHOULDER = 5
R_SHOULDER = 6
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_ANKLE = 15
R_ANKLE = 16
SPORTS_BALL = 32


def center_of(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def choose_person(boxes: List[Tuple[int, int, int, int]], prev_center: Optional[Tuple[float, float]]) -> Optional[int]:
    if not boxes:
        return None
    if prev_center is None:
        return max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    px, py = prev_center

    def score(i: int) -> float:
        cx, cy = center_of(boxes[i])
        return (cx - px) ** 2 + (cy - py) ** 2

    return min(range(len(boxes)), key=score)


def kp_xy_conf(kp_tensor, idx: int) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    if kp_tensor is None:
        return None, None
    x = float(kp_tensor.xy[0, idx, 0].item())
    y = float(kp_tensor.xy[0, idx, 1].item())
    conf = float(kp_tensor.conf[0, idx].item()) if kp_tensor.conf is not None else None
    if x <= 0 and y <= 0:
        return None, conf
    return (x, y), conf


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def choose_ball(ball_boxes: List[Tuple[int, int, int, int]], wrist: Optional[Tuple[float, float]]) -> Optional[int]:
    if wrist is None or not ball_boxes:
        return None
    return min(range(len(ball_boxes)), key=lambda i: dist(center_of(ball_boxes[i]), wrist))


def foot_point(left_ankle: Optional[Tuple[float, float]], right_ankle: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if left_ankle is not None and right_ankle is not None:
        return ((left_ankle[0] + right_ankle[0]) / 2.0, (left_ankle[1] + right_ankle[1]) / 2.0)
    if left_ankle is not None:
        return left_ankle
    if right_ankle is not None:
        return right_ankle
    return None


def find_pause_window(
    rows: List[Dict],
    min_pause_frames: int,
    speed_thresh: float,
    start_frame: int,
) -> Tuple[Optional[int], Optional[int]]:
    start = None
    count = 0
    for r in rows:
        if r["frame"] < start_frame:
            start = None
            count = 0
            continue

        if r["foot_speed"] is not None and r["foot_speed"] <= speed_thresh:
            if start is None:
                start = r["frame"]
            count += 1
            if count >= min_pause_frames:
                end = r["frame"]
                return start, end
        else:
            start = None
            count = 0
    return None, None


def _mad_thresh(rows: List[Dict], start_frame: int, speed_key: str, floor: float, k: float) -> float:
    """median + k*1.4826*MAD threshold over a single ankle speed series.

    Self-calibrates for bowler speed, camera distance, and frame rate.
    MAD is robust to the extreme delivery-slide spikes that inflate std.
    """
    speeds = sorted(
        float(r[speed_key])
        for r in rows
        if r["frame"] >= start_frame and r.get(speed_key) is not None
    )
    if len(speeds) < 5:
        return floor
    mid = len(speeds) // 2
    median = speeds[mid]
    mad = sorted(abs(s - median) for s in speeds)[mid]
    return max(floor, median + k * 1.4826 * mad)


def adaptive_speed_thresh(rows: List[Dict], start_frame: int, floor: float, k: float = 3.5) -> float:
    """Adaptive threshold using the combined foot midpoint speed (kept as fallback)."""
    return _mad_thresh(rows, start_frame, "foot_speed", floor, k)


def _peak_frames(series: List[Dict], speed_key: str, thresh: float, min_gap: int) -> List[int]:
    """Find local-peak frames in one ankle speed series above thresh."""
    peaks: List[int] = []
    last = -(10 ** 9)
    for i in range(1, len(series) - 1):
        p = series[i - 1].get(speed_key)
        c = series[i].get(speed_key)
        n = series[i + 1].get(speed_key)
        if p is None or c is None or n is None:
            continue
        frame = int(series[i]["frame"])
        if float(c) >= thresh and float(c) >= float(p) and float(c) >= float(n):
            if frame - last >= min_gap:
                peaks.append(frame)
                last = frame
    return peaks


def detect_steps_alternating(
    rows: List[Dict],
    start_frame: int,
    l_thresh: float,
    r_thresh: float,
    min_gap_frames: int,
    min_alt_step_spacing_frames: int,
    first_foot: str,
    expected_steps: int,
) -> Tuple[List[int], List[str]]:
    """Detect approach steps as alternating per-ankle speed peaks.

    Right-handed bowler: L-R-L-R-L  (first_foot='left')
    Left-handed bowler:  R-L-R-L-R  (first_foot='right')

    Returns parallel lists of step frame numbers and which foot ('left'/'right').
    """
    series = [r for r in rows if r["frame"] >= start_frame]

    l_peaks = _peak_frames(series, "left_ankle_speed", l_thresh, min_gap_frames)
    r_peaks = _peak_frames(series, "right_ankle_speed", r_thresh, min_gap_frames)

    foot_order = [
        ("left" if i % 2 == 0 else "right") if first_foot == "left"
        else ("right" if i % 2 == 0 else "left")
        for i in range(expected_steps)
    ]

    l_idx, r_idx = 0, 0
    step_frames: List[int] = []
    step_feet: List[str] = []
    last_frame = start_frame - 1

    for foot in foot_order:
        min_next_frame = last_frame + max(1, min_alt_step_spacing_frames)
        if foot == "left":
            while l_idx < len(l_peaks) and l_peaks[l_idx] < min_next_frame:
                l_idx += 1
            if l_idx < len(l_peaks):
                step_frames.append(l_peaks[l_idx])
                step_feet.append("left")
                last_frame = l_peaks[l_idx]
                l_idx += 1
        else:
            while r_idx < len(r_peaks) and r_peaks[r_idx] < min_next_frame:
                r_idx += 1
            if r_idx < len(r_peaks):
                step_frames.append(r_peaks[r_idx])
                step_feet.append("right")
                last_frame = r_peaks[r_idx]
                r_idx += 1

    return step_frames, step_feet


def line_residual_ratio(xs: List[float], frames: List[int], scale: float) -> Optional[float]:
    if not xs or not frames or len(xs) != len(frames):
        return None
    n = len(xs)
    if n < 3:
        return None

    mean_f = sum(frames) / n
    mean_x = sum(xs) / n
    denom = sum((f - mean_f) ** 2 for f in frames)
    if denom <= 1e-9:
        return None

    slope = sum((frames[i] - mean_f) * (xs[i] - mean_x) for i in range(n)) / denom
    intercept = mean_x - slope * mean_f
    residuals = [xs[i] - (slope * frames[i] + intercept) for i in range(n)]
    rms = (sum(r * r for r in residuals) / n) ** 0.5
    return float(rms / max(1.0, scale))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pickup, set/pause, left ankle line, and right-hand lateral control.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--detect-model", default="yolov8n.pt")
    parser.add_argument("--ball-model", default=None,
                        help="Path to custom ball detector weights (e.g. yolo/runs/ball_near_hand_v1/weights/best.pt). "
                             "If omitted, falls back to COCO sports-ball class from --detect-model.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--pose-conf", type=float, default=0.35)
    parser.add_argument("--ball-conf", type=float, default=0.25,
                        help="Confidence threshold for ball detection (default 0.25 for custom model; use 0.05 for COCO fallback).")
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--kp-conf", type=float, default=0.2)
    parser.add_argument("--hold-dist-ratio", type=float, default=0.30)
    parser.add_argument("--pause-speed-thresh", type=float, default=12.0)
    parser.add_argument("--min-pause-frames", type=int, default=6)
    parser.add_argument("--step-speed-thresh", type=float, default=10.0,
                        help="Floor for the adaptive per-ankle step speed threshold (px/frame).")
    parser.add_argument("--step-speed-k", type=float, default=3.5,
                        help="k in median+k*1.4826*MAD used to compute the adaptive threshold "
                             "per ankle. k=3.5 is ~35 px/frame for a typical behind-lane 60fps camera.")
    parser.add_argument("--min-step-gap-frames", type=int, default=26)
    parser.add_argument(
        "--min-alt-step-spacing-frames",
        type=int,
        default=6,
        help=(
            "Minimum spacing between consecutive alternating steps (default: 6). "
            "Higher values reduce double-counts but may undercount bowlers who take very short steps."
        ),
    )
    parser.add_argument("--bowler-hand", default="right", choices=["right", "left"],
                        help="Dominant hand of the bowler. right -> L-R-L-R-L steps; "
                             "left -> R-L-R-L-R steps (default: right).")
    parser.add_argument("--expected-steps", type=int, default=5)
    parser.add_argument("--return-window-frames", type=int, default=45)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", help="Skip writing annotated output video (faster stats-only run).")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    pose_model = YOLO(args.pose_model)
    detect_model = YOLO(args.detect_model)
    if args.ball_model:
        ball_model = YOLO(args.ball_model)
        _ball_use_custom = True
    else:
        ball_model = detect_model
        _ball_use_custom = False

    rows: List[Dict] = []
    frame_idx = 0
    prev_center = None
    prev_wrist = None
    prev_foot = None
    prev_l_ankle: Optional[Tuple[float, float]] = None
    prev_r_ankle: Optional[Tuple[float, float]] = None
    pickup_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < args.start_frame:
            frame_idx += 1
            continue
        if args.end_frame >= 0 and frame_idx > args.end_frame:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        pose = pose_model.predict(
            source=frame,
            classes=[0],
            conf=args.pose_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        det = ball_model.predict(
            source=frame,
            classes=None if _ball_use_custom else [SPORTS_BALL],
            conf=args.ball_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        p_boxes: List[Tuple[int, int, int, int]] = []
        p_kps: List = []
        if pose.boxes is not None and len(pose.boxes) > 0:
            for i, b in enumerate(pose.boxes):
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                p_boxes.append((x1, y1, x2, y2))
                p_kps.append(pose.keypoints[i] if pose.keypoints is not None else None)

        b_boxes: List[Tuple[int, int, int, int]] = []
        if det.boxes is not None and len(det.boxes) > 0:
            for b in det.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                b_boxes.append((x1, y1, x2, y2))

        p_idx = choose_person(p_boxes, prev_center)
        p_box = None
        kp = None
        body_mid_x = None
        person_h = None
        person_w = None
        l_ankle = None
        r_ankle = None
        r_wrist = None
        r_wrist_conf = None

        if p_idx is not None:
            p_box = p_boxes[p_idx]
            kp = p_kps[p_idx]
            prev_center = center_of(p_box)
            person_h = float(max(1, p_box[3] - p_box[1]))
            person_w = float(max(1, p_box[2] - p_box[0]))

            l_sh, _ = kp_xy_conf(kp, L_SHOULDER)
            r_sh, _ = kp_xy_conf(kp, R_SHOULDER)
            l_hip, _ = kp_xy_conf(kp, L_HIP)
            r_hip, _ = kp_xy_conf(kp, R_HIP)
            l_ankle, _ = kp_xy_conf(kp, L_ANKLE)
            r_ankle, _ = kp_xy_conf(kp, R_ANKLE)
            r_wrist, r_wrist_conf = kp_xy_conf(kp, R_WRIST)

            mids = []
            for pt in (l_sh, r_sh, l_hip, r_hip):
                if pt is not None:
                    mids.append(pt[0])
            if mids:
                body_mid_x = float(mean(mids))

            x1, y1, x2, y2 = p_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 210, 60), 2)

        if r_wrist is not None and (r_wrist_conf is not None and r_wrist_conf < args.kp_conf):
            r_wrist = None

        b_idx = choose_ball(b_boxes, r_wrist)
        ball_center = None
        if b_idx is not None:
            ball_center = center_of(b_boxes[b_idx])

        is_holding = False
        wrist_ball_dist = None
        if r_wrist is not None and ball_center is not None and person_h is not None:
            wrist_ball_dist = dist(r_wrist, ball_center)
            is_holding = wrist_ball_dist <= (args.hold_dist_ratio * person_h)

        if pickup_frame is None and is_holding:
            pickup_frame = frame_idx

        wrist_speed = None
        if r_wrist is not None and prev_wrist is not None:
            wrist_speed = dist(r_wrist, prev_wrist)
        prev_wrist = r_wrist

        foot = foot_point(l_ankle, r_ankle)
        foot_speed = None
        if foot is not None and prev_foot is not None:
            foot_speed = dist(foot, prev_foot)
        prev_foot = foot

        l_ankle_speed = None
        if l_ankle is not None and prev_l_ankle is not None:
            l_ankle_speed = dist(l_ankle, prev_l_ankle)
        prev_l_ankle = l_ankle

        r_ankle_speed = None
        if r_ankle is not None and prev_r_ankle is not None:
            r_ankle_speed = dist(r_ankle, prev_r_ankle)
        prev_r_ankle = r_ankle

        wrist_offset = None
        wrist_offset_ratio = None
        if r_wrist is not None and body_mid_x is not None:
            wrist_offset = r_wrist[0] - body_mid_x
            if person_w is not None:
                wrist_offset_ratio = wrist_offset / person_w

        if l_ankle is not None:
            ax, ay = int(l_ankle[0]), int(l_ankle[1])
            cv2.circle(frame, (ax, ay), 6, (255, 200, 0), -1)
            cv2.putText(frame, "L ankle", (ax + 8, ay - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)
        if r_ankle is not None:
            rax, ray = int(r_ankle[0]), int(r_ankle[1])
            cv2.circle(frame, (rax, ray), 6, (255, 120, 0), -1)
            cv2.putText(frame, "R ankle", (rax + 8, ray - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 2)
        if r_wrist is not None:
            wx, wy = int(r_wrist[0]), int(r_wrist[1])
            cv2.circle(frame, (wx, wy), 6, (0, 200, 255), -1)
            cv2.putText(frame, "R wrist", (wx + 8, wy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
        if ball_center is not None:
            bx, by = int(ball_center[0]), int(ball_center[1])
            cv2.circle(frame, (bx, by), 7, (0, 120, 255), 2)
            cv2.putText(frame, "ball", (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 2)

        rows.append(
            {
                "frame": frame_idx,
                "pickup_frame": pickup_frame,
                "person_bbox": p_box,
                "person_height": person_h,
                "person_width": person_w,
                "left_ankle": l_ankle,
                "right_ankle": r_ankle,
                "foot_point": foot,
                "foot_speed": foot_speed,
                "left_ankle_speed": l_ankle_speed,
                "right_ankle_speed": r_ankle_speed,
                "right_wrist": r_wrist,
                "ball_center": ball_center,
                "body_mid_x": body_mid_x,
                "is_holding": is_holding,
                "wrist_ball_distance": wrist_ball_dist,
                "wrist_speed": wrist_speed,
                "wrist_offset_from_mid": wrist_offset,
                "wrist_offset_ratio": wrist_offset_ratio,
            }
        )

        if writer is not None:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Phase 1: start of process is ball pickup.
    search_start = pickup_frame if pickup_frame is not None else 0

    # Phase 2: pause is defined only by foot motion after pickup.
    pause_start, pause_end = find_pause_window(
        rows,
        args.min_pause_frames,
        args.pause_speed_thresh,
        start_frame=search_start,
    )

    # Phase 3: approach starts after pause; steps are per-ankle alternating peaks.
    return_start = pause_end + 1 if pause_end is not None else None
    return_end = len(rows) - 1 if return_start is not None else None

    step_frames: List[int] = []
    step_feet: List[str] = []
    effective_l_thresh = args.step_speed_thresh
    effective_r_thresh = args.step_speed_thresh
    first_foot = "left" if args.bowler_hand == "right" else "right"
    if return_start is not None:
        effective_l_thresh = _mad_thresh(
            rows, return_start, "left_ankle_speed", floor=args.step_speed_thresh, k=args.step_speed_k
        )
        effective_r_thresh = _mad_thresh(
            rows, return_start, "right_ankle_speed", floor=args.step_speed_thresh, k=args.step_speed_k
        )
        step_frames, step_feet = detect_steps_alternating(
            rows, return_start,
            l_thresh=effective_l_thresh,
            r_thresh=effective_r_thresh,
            min_gap_frames=args.min_step_gap_frames,
            min_alt_step_spacing_frames=args.min_alt_step_spacing_frames,
            first_foot=first_foot,
            expected_steps=args.expected_steps,
        )

    pause_rows: List[Dict] = []
    if pause_start is not None and pause_end is not None:
        pause_rows = [r for r in rows if pause_start <= r["frame"] <= pause_end]

    approach_rows: List[Dict] = []
    if return_start is not None and return_end is not None:
        approach_rows = [r for r in rows if return_start <= r["frame"] <= return_end]

    ankle_samples = [r for r in approach_rows if r["left_ankle"] is not None and r["person_width"] is not None]
    ankle_xs = [r["left_ankle"][0] for r in ankle_samples]
    ankle_frames = [int(r["frame"]) for r in ankle_samples]
    widths = [r["person_width"] for r in ankle_samples]

    ankle_straightness_ratio = None
    ankle_is_straight = None
    if ankle_xs and widths and ankle_frames:
        avg_w = mean(widths)
        ankle_straightness_ratio = line_residual_ratio(ankle_xs, ankle_frames, avg_w)
        if ankle_straightness_ratio is not None:
            ankle_is_straight = ankle_straightness_ratio <= 0.10

    # Wrist should first appear around step 3 (observational check).
    first_wrist_frame = None
    if return_start is not None:
        for r in rows:
            if r["frame"] < return_start:
                continue
            if r["right_wrist"] is not None:
                first_wrist_frame = r["frame"]
                break

    first_wrist_step_index = None
    if first_wrist_frame is not None and step_frames:
        idx = sum(1 for s in step_frames if s <= first_wrist_frame)
        first_wrist_step_index = idx if idx > 0 else 1

    hand_eval_start = first_wrist_frame
    if len(step_frames) >= 3:
        hand_eval_start = step_frames[2]

    offsets = [
        r["wrist_offset_ratio"]
        for r in approach_rows
        if r["wrist_offset_ratio"] is not None and (hand_eval_start is None or r["frame"] >= hand_eval_start)
    ]
    hand_lateral_range = None
    hand_control_ok = None
    if offsets:
        hand_lateral_range = max(offsets) - min(offsets)
        hand_control_ok = hand_lateral_range <= 0.35

    step_count = len(step_frames)
    step_count_ok = step_count == args.expected_steps
    step_intervals = [step_frames[i] - step_frames[i - 1] for i in range(1, len(step_frames))]

    # Wrist-vs-ball relationship analysis.
    pickup_for_analysis = pickup_frame if pickup_frame is not None else 0
    wb_rows = []
    for r in rows:
        if r["frame"] < pickup_for_analysis:
            continue
        if r["right_wrist"] is None or r["ball_center"] is None or r["person_width"] is None:
            continue
        wrist_x = float(r["right_wrist"][0])
        ball_x = float(r["ball_center"][0])
        ball_y = float(r["ball_center"][1])
        body_mid_x = float(r["body_mid_x"]) if r["body_mid_x"] is not None else None
        person_w = float(max(1.0, r["person_width"]))
        wb_rows.append(
            {
                "frame": int(r["frame"]),
                "wrist_x": wrist_x,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "body_mid_x": body_mid_x,
                "person_w": person_w,
                "rel_x": wrist_x - ball_x,
            }
        )

    backswing_top_frame = None
    backswing_samples = 0
    backswing_wrist_behind_ratio = None
    backswing_wrist_stays_behind = None
    wrist_moves_right_frame = None
    wrist_already_right_at_analysis_start = None
    ball_returns_to_body_frame = None
    ball_returns_to_body_confident = None
    wrist_right_timing_frames_vs_body_return = None
    wrist_right_timing_seconds_vs_body_return = None
    wrist_right_before_body_return = None

    if wb_rows:
        # Top of backswing: highest ball point in image (smallest y) after pickup.
        top_row = min(wb_rows, key=lambda rr: rr["ball_y"])
        backswing_top_frame = int(top_row["frame"])

        backswing_rows = [rr for rr in wb_rows if rr["frame"] <= backswing_top_frame]
        backswing_samples = len(backswing_rows)
        if backswing_rows:
            behind_count = 0
            for rr in backswing_rows:
                # Wrist is considered "behind/not yet around-right" when not clearly right of ball.
                behind_thresh_px = 0.03 * rr["person_w"]
                if rr["rel_x"] <= behind_thresh_px:
                    behind_count += 1
            backswing_wrist_behind_ratio = behind_count / max(1, backswing_samples)
            backswing_wrist_stays_behind = backswing_wrist_behind_ratio >= 0.80

        # First frame wrist clearly moves to right of ball.
        prev_rel = None
        prev_thresh = None
        first_right_frame = None
        for rr in wb_rows:
            right_thresh_px = 0.04 * rr["person_w"]
            rel_x = rr["rel_x"]
            if first_right_frame is None and rel_x > right_thresh_px:
                first_right_frame = int(rr["frame"])
            if prev_rel is not None and prev_thresh is not None:
                if prev_rel <= prev_thresh and rel_x > right_thresh_px:
                    wrist_moves_right_frame = int(rr["frame"])
                    break
            prev_rel = rel_x
            prev_thresh = right_thresh_px

        if wrist_moves_right_frame is None and first_right_frame is not None:
            wrist_moves_right_frame = first_right_frame
            wrist_already_right_at_analysis_start = True
        elif wrist_moves_right_frame is not None:
            wrist_already_right_at_analysis_start = False

        # Ball return-to-body event: ball center reaches near body midline after backswing top.
        closest_body_row = None
        closest_body_ratio = None
        for rr in wb_rows:
            if rr["frame"] < backswing_top_frame:
                continue
            if rr["body_mid_x"] is None:
                continue
            body_dist_ratio = abs(rr["ball_x"] - rr["body_mid_x"]) / rr["person_w"]
            if closest_body_ratio is None or body_dist_ratio < closest_body_ratio:
                closest_body_ratio = body_dist_ratio
                closest_body_row = rr

            body_thresh_px = 0.12 * rr["person_w"]
            if abs(rr["ball_x"] - rr["body_mid_x"]) <= body_thresh_px:
                ball_returns_to_body_frame = int(rr["frame"])
                ball_returns_to_body_confident = True
                break

        if ball_returns_to_body_frame is None and closest_body_row is not None:
            ball_returns_to_body_frame = int(closest_body_row["frame"])
            ball_returns_to_body_confident = False

        if wrist_moves_right_frame is not None and ball_returns_to_body_frame is not None:
            wrist_right_timing_frames_vs_body_return = wrist_moves_right_frame - ball_returns_to_body_frame
            wrist_right_timing_seconds_vs_body_return = wrist_right_timing_frames_vs_body_return / max(1e-6, float(fps))
            wrist_right_before_body_return = wrist_right_timing_frames_vs_body_return < 0

    summary = {
        "frames": len(rows),
        "bowler_hand": args.bowler_hand,
        "step_sequence": "L-R-L-R-L" if args.bowler_hand == "right" else "R-L-R-L-R",
        "camera_assumption": (
            "For right-handed bowlers, place camera behind and to the right. "
            "For left-handed bowlers, place camera behind and to the left. "
            "Changing camera angle can shift wrist visibility timing and step-phase interpretation."
        ),
        "pickup_frame": pickup_frame,
        "pause_start": pause_start,
        "pause_end": pause_end,
        "pause_frames": (pause_end - pause_start + 1) if pause_start is not None and pause_end is not None else 0,
        "approach_start": return_start,
        "approach_end": return_end,
        "detected_step_frames": step_frames,
        "detected_step_feet": step_feet,
        "detected_step_count": step_count,
        "detected_step_intervals": step_intervals,
        "expected_step_count": args.expected_steps,
        "step_count_ok": step_count_ok,
        "min_alt_step_spacing_frames": args.min_alt_step_spacing_frames,
        "step_spacing_warning": (
            "Minimum inter-step spacing is enforced. This can undercount bowlers with very short steps; "
            "reduce --min-alt-step-spacing-frames if steps are getting dropped."
            if args.min_alt_step_spacing_frames > 0
            else None
        ),
        "effective_l_ankle_thresh": round(effective_l_thresh, 2),
        "effective_r_ankle_thresh": round(effective_r_thresh, 2),
        "left_ankle_straightness_ratio": ankle_straightness_ratio,
        "left_ankle_straight": ankle_is_straight,
        "first_wrist_frame_in_approach": first_wrist_frame,
        "first_wrist_step_index": first_wrist_step_index,
        "right_hand_lateral_range_ratio": hand_lateral_range,
        "right_hand_control_ok": hand_control_ok,
        "backswing_top_frame": backswing_top_frame,
        "backswing_samples": backswing_samples,
        "backswing_wrist_behind_ratio": backswing_wrist_behind_ratio,
        "backswing_wrist_stays_behind": backswing_wrist_stays_behind,
        "wrist_moves_right_around_ball_frame": wrist_moves_right_frame,
        "wrist_already_right_at_analysis_start": wrist_already_right_at_analysis_start,
        "ball_returns_to_body_frame": ball_returns_to_body_frame,
        "ball_returns_to_body_confident": ball_returns_to_body_confident,
        "wrist_right_before_body_return": wrist_right_before_body_return,
        "wrist_right_timing_frames_vs_body_return": wrist_right_timing_frames_vs_body_return,
        "wrist_right_timing_seconds_vs_body_return": wrist_right_timing_seconds_vs_body_return,
    }

    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": rows}, f, indent=2)

    print(f"frames={summary['frames']}")
    print(f"bowler_hand={summary['bowler_hand']} ({summary['step_sequence']})")
    print(f"camera_assumption={summary['camera_assumption']}")
    print(f"pickup_frame={summary['pickup_frame']}")
    print(f"pause_start={summary['pause_start']}")
    print(f"pause_end={summary['pause_end']}")
    print(f"approach_start={summary['approach_start']}")
    print(f"detected_step_count={summary['detected_step_count']}")
    print(f"detected_steps={list(zip(summary['detected_step_frames'], summary['detected_step_feet']))}")
    print(f"detected_step_intervals={summary['detected_step_intervals']}")
    print(f"step_count_ok={summary['step_count_ok']}")
    print(f"min_alt_step_spacing_frames={summary['min_alt_step_spacing_frames']}")
    if summary["step_spacing_warning"] is not None:
        print(f"step_spacing_warning={summary['step_spacing_warning']}")
    print(f"effective_l_ankle_thresh={summary['effective_l_ankle_thresh']}")
    print(f"effective_r_ankle_thresh={summary['effective_r_ankle_thresh']}")
    print(f"left_ankle_straightness_ratio={summary['left_ankle_straightness_ratio']}")
    print(f"left_ankle_straight={summary['left_ankle_straight']}")
    print(f"first_wrist_step_index={summary['first_wrist_step_index']}")
    print(f"right_hand_lateral_range_ratio={summary['right_hand_lateral_range_ratio']}")
    print(f"right_hand_control_ok={summary['right_hand_control_ok']}")
    print(f"backswing_top_frame={summary['backswing_top_frame']}")
    print(f"backswing_wrist_stays_behind={summary['backswing_wrist_stays_behind']}")
    print(f"backswing_wrist_behind_ratio={summary['backswing_wrist_behind_ratio']}")
    print(f"wrist_moves_right_around_ball_frame={summary['wrist_moves_right_around_ball_frame']}")
    print(f"wrist_already_right_at_analysis_start={summary['wrist_already_right_at_analysis_start']}")
    print(f"ball_returns_to_body_frame={summary['ball_returns_to_body_frame']}")
    print(f"ball_returns_to_body_confident={summary['ball_returns_to_body_confident']}")
    print(f"wrist_right_before_body_return={summary['wrist_right_before_body_return']}")
    print(f"wrist_right_timing_frames_vs_body_return={summary['wrist_right_timing_frames_vs_body_return']}")
    print(f"wrist_right_timing_seconds_vs_body_return={summary['wrist_right_timing_seconds_vs_body_return']}")
    if args.no_video:
        print("output=skipped (--no-video)")
    else:
        print(f"output={out_path}")
    print(f"stats={stats_path}")


if __name__ == "__main__":
    main()