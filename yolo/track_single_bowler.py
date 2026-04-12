import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO


def center_of(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def area_of(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def choose_single_bowler(
    boxes: List[Tuple[int, int, int, int]],
    confs: List[float],
    prev_center: Optional[Tuple[float, float]],
    max_center_shift: float,
) -> Optional[int]:
    if not boxes:
        return None

    if prev_center is None:
        best_idx = max(range(len(boxes)), key=lambda i: (confs[i] * 100.0 + area_of(boxes[i]) * 0.001))
        return best_idx

    px, py = prev_center

    gated = []
    for i, box in enumerate(boxes):
        cx, cy = center_of(box)
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if dist <= max_center_shift:
            gated.append(i)

    candidate_indices = gated if gated else list(range(len(boxes)))

    def score(i: int) -> float:
        cx, cy = center_of(boxes[i])
        dist2 = (cx - px) ** 2 + (cy - py) ** 2
        # Prefer temporal consistency first, then confidence and size.
        return dist2 - (confs[i] * 500.0) - (area_of(boxes[i]) * 0.002)

    return min(candidate_indices, key=score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track a single bowler from YOLO person detections.")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO weights path")
    parser.add_argument("--output", required=True, help="Annotated output video path")
    parser.add_argument("--stats", default=None, help="Optional JSON output with frame-by-frame stats")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--min-area-ratio", type=float, default=0.005, help="Minimum person box area ratio relative to frame")
    parser.add_argument("--max-center-shift-ratio", type=float, default=0.18, help="Max center shift ratio of frame diagonal before considering detection a jump")
    parser.add_argument("--max-hold-frames", type=int, default=4, help="Keep last selected bowler for this many missed frames")
    parser.add_argument("--draw-raw", action="store_true", help="Draw all raw person detections in red")
    parser.add_argument("--report-raw-count", action="store_true", help="Include raw YOLO person count in stats for debugging")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    model = YOLO(args.model)

    frame_index = 0
    prev_center = None
    prev_box = None
    prev_confidence = None
    missed_streak = 0
    frame_stats = []
    frame_area = float(width * height)
    frame_diag = float((width ** 2 + height ** 2) ** 0.5)
    min_area = args.min_area_ratio * frame_area
    max_center_shift = args.max_center_shift_ratio * frame_diag

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            classes=[0],
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        boxes: List[Tuple[int, int, int, int]] = []
        confs: List[float] = []

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                c = float(b.conf[0].item())
                candidate = (x1, y1, x2, y2)
                if area_of(candidate) < min_area:
                    continue
                boxes.append(candidate)
                confs.append(c)

        selected_idx = choose_single_bowler(boxes, confs, prev_center, max_center_shift)

        if args.draw_raw:
            # Draw all raw detections in red when debugging.
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 220), 2)
                cv2.putText(
                    frame,
                    f"person {confs[i]:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (30, 30, 220),
                    2,
                )

        selected_box = None
        selected_confidence = None
        selected_source = "none"
        if selected_idx is not None:
            selected_box = boxes[selected_idx]
            selected_confidence = confs[selected_idx]
            selected_source = "detected"
            x1, y1, x2, y2 = selected_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 3)
            cv2.putText(
                frame,
                f"selected bowler {confs[selected_idx]:.2f}",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 220, 40),
                2,
            )
            prev_center = center_of(selected_box)
            prev_box = selected_box
            prev_confidence = selected_confidence
            missed_streak = 0
        else:
            missed_streak += 1
            if prev_box is not None and missed_streak <= args.max_hold_frames:
                selected_box = prev_box
                selected_confidence = prev_confidence
                selected_source = "held"
                x1, y1, x2, y2 = selected_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 180, 230), 2)
                cv2.putText(
                    frame,
                    f"held bowler {missed_streak}/{args.max_hold_frames}",
                    (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (30, 180, 230),
                    2,
                )
                prev_center = center_of(selected_box)

        row = {
            "frame": frame_index,
            "person_count": 1 if selected_box is not None else 0,
            "selected_source": selected_source,
            "selected": selected_box,
            "selected_confidence": selected_confidence,
        }
        if args.report_raw_count:
            row["raw_person_count"] = len(boxes)
        frame_stats.append(row)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(frame_stats, f, indent=2)

    multi_count = sum(1 for row in frame_stats if row.get("raw_person_count", 0) > 1)
    print(f"frames={len(frame_stats)}")
    print(f"frames_with_multiple_person_boxes={multi_count}")
    print(f"output={output_path}")
    if args.stats:
        print(f"stats={args.stats}")


if __name__ == "__main__":
    main()
