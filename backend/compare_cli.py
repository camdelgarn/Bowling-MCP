#!/usr/bin/env python3
"""Simple CLI to compare two videos using the MBP prototype.

Usage:
  python backend/compare_cli.py --file1 PATH --file2 PATH
  python backend/compare_cli.py --dir PATH   # picks first two video files
"""
import argparse
import glob
import json
import os
import sys
import uuid
import time

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from app import processing
from app.processing import compare_videos_sync


def find_two_videos_in_dir(d: str):
    patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(d, p))))
    return files[:2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file1", help="first video file")
    p.add_argument("--file2", help="second video file")
    p.add_argument("--dir", help="directory containing sample videos (will pick first two)")
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--no-normalize", dest="no_normalize", action="store_true", help="Skip ffmpeg normalization and run pipeline on original files")
    args = p.parse_args()

    if args.dir:
        files = find_two_videos_in_dir(args.dir)
        if len(files) < 2:
            print(f"Need at least two videos in directory: {args.dir}")
            sys.exit(2)
        f1, f2 = files[0], files[1]
    else:
        if not args.file1 or not args.file2:
            p.print_help()
            sys.exit(2)
        f1, f2 = args.file1, args.file2

    if not os.path.exists(f1) or not os.path.exists(f2):
        print("One or both video files not found:", f1, f2)
        sys.exit(2)

    print("Comparing:\n", f1, "\n", f2)
    job_id = str(uuid.uuid4())
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{job_id}.json")

    t0 = time.time()
    try:
        if getattr(args, "no_normalize", False):
            # Run local pipeline without calling ffmpeg/normalization
            info1 = processing.get_video_info(f1)
            info2 = processing.get_video_info(f2)

            try:
                pose1 = processing.run_pose_on_video(f1, max_frames=args.max_frames)
            except Exception as e:
                pose1 = {"error": f"pose_failed: {e}"}
            try:
                pose2 = processing.run_pose_on_video(f2, max_frames=args.max_frames)
            except Exception as e:
                pose2 = {"error": f"pose_failed: {e}"}

            try:
                det1 = processing.detect_ball_hough(f1, max_frames=args.max_frames)
                if not det1.get("detections"):
                    det1 = processing.detect_ball_color_blob(f1, max_frames=args.max_frames)
                unified1 = processing._unify_detections(det1)
                tracks1 = processing.track_detections(unified1)
                ball1 = {"raw": det1, "unified": unified1, "tracks": tracks1}
            except Exception as e:
                ball1 = {"error": f"ball_failed: {e}"}

            try:
                det2 = processing.detect_ball_hough(f2, max_frames=args.max_frames)
                if not det2.get("detections"):
                    det2 = processing.detect_ball_color_blob(f2, max_frames=args.max_frames)
                unified2 = processing._unify_detections(det2)
                tracks2 = processing.track_detections(unified2)
                ball2 = {"raw": det2, "unified": unified2, "tracks": tracks2}
            except Exception as e:
                ball2 = {"error": f"ball_failed: {e}"}

            rel1 = processing.detect_release(pose1 if isinstance(pose1, dict) else {}, ball1 if isinstance(ball1, dict) else {}, info1)
            rel2 = processing.detect_release(pose2 if isinstance(pose2, dict) else {}, ball2 if isinstance(ball2, dict) else {}, info2)

            res = {
                "status": "ok",
                "summary": "Local pipeline (no normalization)",
                "video_info": {"video1": info1, "video2": info2},
                "pose": {"video1": pose1, "video2": pose2},
                "ball": {"video1": ball1, "video2": ball2},
                "release": {"video1": rel1, "video2": rel2},
            }
        else:
            res = compare_videos_sync(f1, f2, max_frames=args.max_frames)
    except Exception as e:
        print("Error running comparison:", e)
        res = {"status": "error", "error": str(e)}
    elapsed = time.time() - t0

    wrapped = {"job_id": job_id, "elapsed_s": elapsed, "result": res}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(wrapped, fh, indent=2)

    print("Saved result to", out_path)
    print(json.dumps(wrapped, indent=2))


if __name__ == "__main__":
    main()
