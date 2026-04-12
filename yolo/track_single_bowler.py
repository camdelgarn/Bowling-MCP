import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO


def parse_roi(roi_text: Optional[str], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not roi_text:
        return None
    parts = [p.strip() for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    x1, y1, x2, y2 = [int(v) for v in parts]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must have x2>x1 and y2>y1")
    return (x1, y1, x2, y2)


def center_of(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def area_of(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def box_inside_roi(box: Tuple[int, int, int, int], roi: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    cx, cy = center_of(box)
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def choose_single_bowler(
    boxes: List[Tuple[int, int, int, int]],
    confs: List[float],
    prev_center: Optional[Tuple[float, float]],
) -> Optional[int]:
    if not boxes:
        return None

    if prev_center is None:
        best_idx = max(range(len(boxes)), key=lambda i: (confs[i] * 100.0 + area_of(boxes[i]) * 0.001))
        return best_idx

    px, py = prev_center

    def score(i: int) -> float:
        cx, cy = center_of(boxes[i])
        dist2 = (cx - px) ** 2 + (cy - py) ** 2
        # Prefer temporal consistency first, then confidence and size.
        return dist2 - (confs[i] * 500.0) - (area_of(boxes[i]) * 0.002)

    return min(range(len(boxes)), key=score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track a single bowler from YOLO person detections.")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO weights path")
    parser.add_argument("--output", required=True, help="Annotated output video path")
    parser.add_argument("--stats", default=None, help="Optional JSON output with frame-by-frame stats")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--roi", default=None, help="Optional ROI as x1,y1,x2,y2")
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

    roi = parse_roi(args.roi, width, height)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    model = YOLO(args.model)

    frame_index = 0
    prev_center = None
    frame_stats = []

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
                if roi is not None and not box_inside_roi(candidate, roi):
                    continue
                boxes.append(candidate)
                confs.append(c)

        selected_idx = choose_single_bowler(boxes, confs, prev_center)

        # Draw ROI for debugging if present.
        if roi is not None:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
            cv2.putText(frame, "ROI", (roi[0], max(roi[1] - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw all raw detections in red.
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
        if selected_idx is not None:
            selected_box = boxes[selected_idx]
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

        frame_stats.append(
            {
                "frame": frame_index,
                "raw_person_count": len(boxes),
                "selected": selected_box,
                "selected_confidence": None if selected_idx is None else confs[selected_idx],
            }
        )

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(frame_stats, f, indent=2)

    multi_count = sum(1 for row in frame_stats if row["raw_person_count"] > 1)
    print(f"frames={len(frame_stats)}")
    print(f"frames_with_multiple_person_boxes={multi_count}")
    print(f"output={output_path}")
    if args.stats:
        print(f"stats={args.stats}")


if __name__ == "__main__":
    main()
