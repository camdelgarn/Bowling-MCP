import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO


# COCO keypoint order used by YOLO pose models.
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


def center_of(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def choose_single_person(boxes: List[Tuple[int, int, int, int]], prev_center: Optional[Tuple[float, float]]) -> Optional[int]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone YOLO pose feet detector for bowling videos.")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--stats", required=True, help="Output JSON stats path")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--conf", type=float, default=0.4, help="Person confidence threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU")
    parser.add_argument("--ankle-kp-conf", type=float, default=0.25, help="Minimum keypoint confidence for ankle output")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for faster iteration; 0 means all")
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

    model = YOLO(args.model)

    frame_idx = 0
    prev_center = None
    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        result = model.predict(
            source=frame,
            classes=[0],
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        boxes: List[Tuple[int, int, int, int]] = []
        kps: List = []
        confs: List[float] = []

        if result.boxes is not None and len(result.boxes) > 0:
            for i, b in enumerate(result.boxes):
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                boxes.append((x1, y1, x2, y2))
                confs.append(float(b.conf[0].item()))
                if result.keypoints is not None:
                    kps.append(result.keypoints[i])
                else:
                    kps.append(None)

        selected_idx = choose_single_person(boxes, prev_center)
        selected_box = None
        selected_conf = None
        left_ankle = None
        right_ankle = None
        left_ankle_conf = None
        right_ankle_conf = None

        if selected_idx is not None:
            selected_box = boxes[selected_idx]
            selected_conf = confs[selected_idx]
            prev_center = center_of(selected_box)
            selected_kp = kps[selected_idx]

            left_ankle, left_ankle_conf = kp_xy_conf(selected_kp, LEFT_ANKLE_IDX)
            right_ankle, right_ankle_conf = kp_xy_conf(selected_kp, RIGHT_ANKLE_IDX)

            x1, y1, x2, y2 = selected_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
            cv2.putText(frame, f"bowler {selected_conf:.2f}", (x1, max(y1 - 10, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2)

            if left_ankle is not None and (left_ankle_conf is None or left_ankle_conf >= args.ankle_kp_conf):
                lx, ly = int(left_ankle[0]), int(left_ankle[1])
                cv2.circle(frame, (lx, ly), 7, (255, 200, 0), -1)
                cv2.putText(frame, f"L {left_ankle_conf:.2f}" if left_ankle_conf is not None else "L", (lx + 8, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            else:
                left_ankle = None

            if right_ankle is not None and (right_ankle_conf is None or right_ankle_conf >= args.ankle_kp_conf):
                rx, ry = int(right_ankle[0]), int(right_ankle[1])
                cv2.circle(frame, (rx, ry), 7, (0, 220, 255), -1)
                cv2.putText(frame, f"R {right_ankle_conf:.2f}" if right_ankle_conf is not None else "R", (rx + 8, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
            else:
                right_ankle = None

        rows.append(
            {
                "frame": frame_idx,
                "person_detected": selected_idx is not None,
                "bbox": selected_box,
                "bbox_confidence": selected_conf,
                "left_ankle": left_ankle,
                "left_ankle_confidence": left_ankle_conf,
                "right_ankle": right_ankle,
                "right_ankle_confidence": right_ankle_conf,
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

    detected = sum(1 for r in rows if r["person_detected"])
    both_feet = sum(1 for r in rows if r["left_ankle"] is not None and r["right_ankle"] is not None)
    print(f"frames={len(rows)}")
    print(f"person_detected_frames={detected}")
    print(f"both_ankles_frames={both_feet}")
    print(f"output={out_path}")
    print(f"stats={stats_path}")


if __name__ == "__main__":
    main()
