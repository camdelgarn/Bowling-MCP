import argparse
import csv
import json
import random
from pathlib import Path

import cv2


def pick_frames(records, kp_conf=0.2, n_hard=120, n_pos=40, near_window=12):
    # Positive examples where the detector actually found the ball.
    pos = [r for r in records if r.get("ball_bbox") is not None]

    # Hard negatives: wrist visible but no ball detection.
    wrist_visible_miss = [
        r
        for r in records
        if r.get("person_detected")
        and r.get("right_wrist") is not None
        and (r.get("right_wrist_confidence") or 0.0) >= kp_conf
        and r.get("ball_bbox") is None
    ]

    # Prioritize misses near true ball detections (hard transitions/occlusions).
    pos_frames = [int(r["frame"]) for r in pos]
    near_pos = []
    for r in wrist_visible_miss:
        f = int(r["frame"])
        if any(abs(f - pf) <= near_window for pf in pos_frames):
            near_pos.append(r)

    near_pos_ids = {int(r["frame"]) for r in near_pos}
    far_pos = [r for r in wrist_visible_miss if int(r["frame"]) not in near_pos_ids]

    random.shuffle(near_pos)
    random.shuffle(far_pos)
    random.shuffle(pos)

    hard_pick = near_pos[: min(len(near_pos), n_hard // 2)]
    remaining = n_hard - len(hard_pick)
    hard_pick.extend(far_pos[: min(len(far_pos), remaining)])

    pos_pick = pos[: min(len(pos), n_pos)]

    chosen = {}
    for r in hard_pick:
        chosen[int(r["frame"])] = {
            "frame": int(r["frame"]),
            "reason": "hard_wrist_visible_ball_missing",
            "has_ball": 0,
            "wrist_conf": float(r.get("right_wrist_confidence") or 0.0),
        }
    for r in pos_pick:
        chosen[int(r["frame"])] = {
            "frame": int(r["frame"]),
            "reason": "positive_ball_detected",
            "has_ball": 1,
            "wrist_conf": float(r.get("right_wrist_confidence") or 0.0),
        }

    return sorted(chosen.values(), key=lambda x: x["frame"])


def extract_frames(video_path: Path, rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_to_meta = {int(r["frame"]): r for r in rows}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    target = set(frame_to_meta.keys())
    idx = 0
    written = 0
    while target:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target:
            meta = frame_to_meta[idx]
            fn = f"frame_{idx:05d}_{'pos' if meta['has_ball'] else 'hard'}.jpg"
            cv2.imwrite(str(out_dir / fn), frame)
            meta["image"] = fn
            written += 1
            target.remove(idx)
        idx += 1

    cap.release()
    return written


def main():
    parser = argparse.ArgumentParser(description="Extract hard/positive frames for bowling ball detector training.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--kp-conf", type=float, default=0.2)
    parser.add_argument("--hard-frames", type=int, default=120)
    parser.add_argument("--positive-frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    records = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
    rows = pick_frames(
        records,
        kp_conf=args.kp_conf,
        n_hard=args.hard_frames,
        n_pos=args.positive_frames,
    )

    out_dir = Path(args.out_dir)
    written = extract_frames(Path(args.video), rows, out_dir)

    csv_path = out_dir / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "image", "has_ball", "reason", "wrist_conf"])
        w.writeheader()
        w.writerows(rows)

    hard_count = sum(1 for r in rows if r["has_ball"] == 0)
    pos_count = sum(1 for r in rows if r["has_ball"] == 1)

    print(f"selected={len(rows)} written={written}")
    print(f"hard_frames={hard_count} positive_frames={pos_count}")
    print(f"out_dir={out_dir}")
    print(f"manifest={csv_path}")


if __name__ == "__main__":
    main()
