import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


# COCO keypoint indices for the right arm.
RIGHT_SHOULDER_IDX = 6
RIGHT_ELBOW_IDX = 8
RIGHT_WRIST_IDX = 10
SPORTS_BALL_CLASS_ID = 32


def center_of(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_height(box: Tuple[int, int, int, int]) -> float:
    return float(max(1, box[3] - box[1]))


def choose_single_person(
    boxes: List[Tuple[int, int, int, int]],
    prev_center: Optional[Tuple[float, float]],
) -> Optional[int]:
    if not boxes:
        return None
    if prev_center is None:
        return max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))

    px, py = prev_center

    def score(i: int) -> float:
        cx, cy = center_of(boxes[i])
        return (cx - px) ** 2 + (cy - py) ** 2

    return min(range(len(boxes)), key=score)


def kp_xy_conf(kp_tensor, index: int) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    if kp_tensor is None:
        return None, None
    x = float(kp_tensor.xy[0, index, 0].item())
    y = float(kp_tensor.xy[0, index, 1].item())
    conf = float(kp_tensor.conf[0, index].item()) if kp_tensor.conf is not None else None
    if x <= 0 and y <= 0:
        return None, conf
    return (x, y), conf


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def choose_ball_near_hand(
    ball_boxes: List[Tuple[int, int, int, int]],
    hand_xy: Optional[Tuple[float, float]],
) -> Optional[int]:
    if hand_xy is None or not ball_boxes:
        return None

    return min(range(len(ball_boxes)), key=lambda i: dist(center_of(ball_boxes[i]), hand_xy))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect bowling ball pickup and backswing hand/ball positions.")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="Pose model path")
    parser.add_argument("--detect-model", default="yolov8n.pt", help="Detection model path")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--stats", required=True, help="Output JSON stats path")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--pose-conf", type=float, default=0.35, help="Pose person confidence threshold")
    parser.add_argument("--ball-conf", type=float, default=0.2, help="Ball detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--kp-conf", type=float, default=0.2, help="Minimum right wrist confidence")
    parser.add_argument(
        "--hold-dist-ratio",
        type=float,
        default=0.25,
        help="Max wrist-to-ball center distance as fraction of person height to count as holding",
    )
    parser.add_argument(
        "--backswing-rise-ratio",
        type=float,
        default=0.12,
        help="Required upward wrist motion from pickup as fraction of person height to mark backswing",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap; 0 means all")
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
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    pose_model = YOLO(args.pose_model)
    detect_model = YOLO(args.detect_model)

    rows: List[Dict] = []
    frame_idx = 0
    prev_center: Optional[Tuple[float, float]] = None

    pickup_frame: Optional[int] = None
    pickup_wrist_y: Optional[float] = None
    in_holding = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        pose_result = pose_model.predict(
            source=frame,
            classes=[0],
            conf=args.pose_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        detect_result = detect_model.predict(
            source=frame,
            classes=[SPORTS_BALL_CLASS_ID],
            conf=args.ball_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        person_boxes: List[Tuple[int, int, int, int]] = []
        person_kps: List = []
        person_confs: List[float] = []

        if pose_result.boxes is not None and len(pose_result.boxes) > 0:
            for i, b in enumerate(pose_result.boxes):
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                person_boxes.append((x1, y1, x2, y2))
                person_confs.append(float(b.conf[0].item()))
                if pose_result.keypoints is not None:
                    person_kps.append(pose_result.keypoints[i])
                else:
                    person_kps.append(None)

        ball_boxes: List[Tuple[int, int, int, int]] = []
        ball_confs: List[float] = []
        if detect_result.boxes is not None and len(detect_result.boxes) > 0:
            for b in detect_result.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                ball_boxes.append((x1, y1, x2, y2))
                ball_confs.append(float(b.conf[0].item()))

        selected_person_idx = choose_single_person(person_boxes, prev_center)
        person_box = None
        person_conf = None
        right_wrist = None
        right_wrist_conf = None
        right_elbow = None
        right_shoulder = None
        person_h = None

        if selected_person_idx is not None:
            person_box = person_boxes[selected_person_idx]
            person_conf = person_confs[selected_person_idx]
            prev_center = center_of(person_box)
            person_h = box_height(person_box)
            kp = person_kps[selected_person_idx]
            right_wrist, right_wrist_conf = kp_xy_conf(kp, RIGHT_WRIST_IDX)
            right_elbow, _ = kp_xy_conf(kp, RIGHT_ELBOW_IDX)
            right_shoulder, _ = kp_xy_conf(kp, RIGHT_SHOULDER_IDX)

            x1, y1, x2, y2 = person_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 200, 40), 2)
            cv2.putText(frame, f"bowler {person_conf:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 200, 40), 2)

        selected_ball_idx = choose_ball_near_hand(ball_boxes, right_wrist)
        ball_box = None
        ball_conf = None
        ball_center = None

        if selected_ball_idx is not None:
            ball_box = ball_boxes[selected_ball_idx]
            ball_conf = ball_confs[selected_ball_idx]
            ball_center = center_of(ball_box)
            bx1, by1, bx2, by2 = ball_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 180, 255), 2)
            cv2.putText(frame, f"ball {ball_conf:.2f}", (bx1, max(18, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)

        if right_wrist is not None and (right_wrist_conf is None or right_wrist_conf >= args.kp_conf):
            rx, ry = int(right_wrist[0]), int(right_wrist[1])
            cv2.circle(frame, (rx, ry), 7, (255, 180, 0), -1)
            cv2.putText(frame, "R wrist", (rx + 8, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
        else:
            right_wrist = None

        if right_elbow is not None:
            ex, ey = int(right_elbow[0]), int(right_elbow[1])
            cv2.circle(frame, (ex, ey), 5, (160, 160, 255), -1)
        if right_shoulder is not None:
            sx, sy = int(right_shoulder[0]), int(right_shoulder[1])
            cv2.circle(frame, (sx, sy), 5, (160, 255, 160), -1)

        wrist_ball_dist = None
        is_holding = False
        if right_wrist is not None and ball_center is not None and person_h is not None:
            wrist_ball_dist = dist(right_wrist, ball_center)
            is_holding = wrist_ball_dist <= (args.hold_dist_ratio * person_h)

        if is_holding and not in_holding and pickup_frame is None and right_wrist is not None:
            pickup_frame = frame_idx
            pickup_wrist_y = right_wrist[1]
        in_holding = is_holding

        in_backswing = False
        if (
            pickup_wrist_y is not None
            and right_wrist is not None
            and person_h is not None
            and (pickup_wrist_y - right_wrist[1]) >= (args.backswing_rise_ratio * person_h)
        ):
            in_backswing = True

        phase = "none"
        if is_holding:
            phase = "holding"
        if in_backswing and is_holding:
            phase = "backswing"

        cv2.putText(
            frame,
            f"phase: {phase}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 220, 255) if phase != "none" else (200, 200, 200),
            2,
        )

        rows.append(
            {
                "frame": frame_idx,
                "person_detected": selected_person_idx is not None,
                "person_bbox": person_box,
                "person_confidence": person_conf,
                "right_wrist": right_wrist,
                "right_wrist_confidence": right_wrist_conf,
                "ball_bbox": ball_box,
                "ball_center": ball_center,
                "ball_confidence": ball_conf,
                "wrist_ball_distance": wrist_ball_dist,
                "is_holding_ball": is_holding,
                "is_backswing": in_backswing and is_holding,
                "phase": phase,
            }
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    person_detected = sum(1 for r in rows if r["person_detected"])
    ball_detected = sum(1 for r in rows if r["ball_bbox"] is not None)
    hand_detected = sum(1 for r in rows if r["right_wrist"] is not None)
    holding = sum(1 for r in rows if r["is_holding_ball"])
    backswing = sum(1 for r in rows if r["is_backswing"])
    first_pickup = next((r["frame"] for r in rows if r["is_holding_ball"]), None)
    first_backswing = next((r["frame"] for r in rows if r["is_backswing"]), None)

    print(f"frames={len(rows)}")
    print(f"person_detected_frames={person_detected}")
    print(f"right_wrist_detected_frames={hand_detected}")
    print(f"ball_detected_frames={ball_detected}")
    print(f"holding_frames={holding}")
    print(f"backswing_frames={backswing}")
    print(f"first_pickup_frame={first_pickup}")
    print(f"first_backswing_frame={first_backswing}")
    print(f"output={out_path}")
    print(f"stats={stats_path}")


if __name__ == "__main__":
    main()