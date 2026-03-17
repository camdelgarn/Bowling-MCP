#!/usr/bin/env python3
"""Detect bowling ball in a single video and save an image with the ball outlined."""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from app import processing
import cv2


def find_first_two(d):
    patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(d, p))))
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", help="path to video file")
    p.add_argument("--dir", help="directory to pick first video from")
    p.add_argument("--method", choices=["hough", "color", "combined"], default="hough")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--no-motion", action="store_true", help="prefer static/largest detections over motion-first")
    p.add_argument("--debug-frames", type=int, default=0, help="save overlay images for every Nth detected frame (0=off)")
    p.add_argument("--out", help="output image path (optional)")
    args = p.parse_args()

    if args.dir:
        files = find_first_two(args.dir)
        if not files:
            print("No videos found in", args.dir)
            return
        video = files[0]
    else:
        if not args.video:
            p.print_help()
            return
        video = args.video

    if not os.path.exists(video):
        print("Video not found:", video)
        return

    print("Processing", video)

    # Run detector
    if args.method == "hough":
        det = processing.detect_ball_hough(video, max_frames=args.max_frames)
    elif args.method == "color":
        det = processing.detect_ball_color_blob(video, max_frames=args.max_frames)
    else:
        det = processing.detect_ball_combined(video, max_frames=args.max_frames)

    # If the combined detector returned only small or high-frame detections
    # (e.g. fingers) try a permissive static color-blob pass to catch a
    # stationary ball on a rack.
    try:
        detections = det.get("detections") or []
        cap = cv2.VideoCapture(video)
        h_frame = 720
        if cap.isOpened():
            ret0, f0 = cap.read()
            if ret0:
                h_frame = f0.shape[0]
        cap.release()

        def is_good(d):
            if d.get("area", 0) > 400:
                return True
            if d.get("r", 0) > 25 and d.get("y", 0) > 0.35 * h_frame:
                return True
            if d.get("bbox"):
                x, y, w, h = d["bbox"]
                if (y + h) > int(0.35 * h_frame) and (w * h) > 300:
                    return True
            return False

        if detections and not any(is_good(d) for d in detections):
            print("No strong combined detections found — running permissive static color fallback")
            fallback = processing.detect_ball_color_blob(video, hsv_lower=(0, 0, 0), hsv_upper=(179, 255, 255), max_frames=args.max_frames)
            for fd in fallback.get("detections", []):
                # avoid duplicating frames already present
                if not any(fd.get("frame_index") == d.get("frame_index") for d in detections):
                    detections.append(fd)
            # update det to include merged set
            det["detections"] = detections
        # If still no good candidates, try a Hough-only pass (no motion requirement)
        if not any(is_good(d) for d in det.get("detections", [])):
            print("Fallback: running Hough-only pass to catch static circular objects in lower frame")
            hough = processing.detect_ball_hough(video, max_frames=args.max_frames)
            hough_dets = hough.get("detections", [])
            for hd in hough_dets:
                y = hd.get("y", 0)
                r = hd.get("r", 0)
                if y > 0.35 * h_frame and 8 < r < 150:
                    # avoid duplicate frame entries
                    if not any(hd.get("frame_index") == d.get("frame_index") for d in det.get("detections", [])):
                        det.setdefault("detections", []).append(hd)
    except Exception:
        pass

    if not det.get("detections"):
        print("No detections found with method", args.method)
        return

    # Rank detections: prefer motion, then area/radius, and prefer lower-frame vertical position
    detections = det["detections"]
    # If user asked to prefer static detections, skip motion priority
    if args.no_motion:
        def score(d):
            if d.get("area"):
                return float(d.get("area"))
            if d.get("r"):
                return float(d.get("r")) * 50.0
            return 0.0

        best = max(detections, key=score)
    else:
        # pick detection with largest motion if available
        motion_candidates = [d for d in detections if d.get("motion", 0) > 0]
        if motion_candidates:
            best = max(motion_candidates, key=lambda d: d.get("motion", 0))
        else:
            # fallback: prefer largest area (bbox) or largest radius
            def score(d):
                if d.get("area"):
                    return float(d.get("area"))
                if d.get("r"):
                    return float(d.get("r")) * 50.0
                return 0.0

            best = max(detections, key=score)

    frame_idx = best.get("frame_index")

    # read the frame
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Cannot open video")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        # try reading from start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        frame_idx = 0
        if not ret:
            print("Failed to read video frames")
            cap.release()
            return

    out = frame.copy()

    # draw detection(s) on the frame: show all detections on that frame
    for d in det.get("detections", []):
        if d.get("frame_index") != frame_idx:
            continue
        if "r" in d:
            cv2.circle(out, (int(d["x"]), int(d["y"])), int(d["r"]), (0, 255, 0), 3)
        elif "bbox" in d:
            x, y, w, h = d["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            cv2.circle(out, (int(d.get("x", 0)), int(d.get("y", 0))), 20, (0, 255, 0), 3)

    cap.release()

    base = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, f"{base}_frame{frame_idx:04d}_detected.jpg")

    cv2.imwrite(out_path, out)
    print("Wrote:", out_path)

    # Optionally save debug overlays for detected frames
    if args.debug_frames and args.debug_frames > 0:
        for d in detections:
            fi = d.get("frame_index")
            if fi is None:
                continue
            if fi % args.debug_frames != 0:
                continue
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, fimg = cap.read()
            cap.release()
            if not ret:
                continue
            dbg = fimg.copy()
            for dd in [x for x in detections if x.get("frame_index") == fi]:
                if "r" in dd:
                    cv2.circle(dbg, (int(dd["x"]), int(dd["y"])), int(dd["r"]), (0, 0, 255), 2)
                elif "bbox" in dd:
                    x, y, w, h = dd["bbox"]
                    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.circle(dbg, (int(dd.get("x", 0)), int(dd.get("y", 0))), 20, (0, 0, 255), 2)

            dbg_path = os.path.join(out_dir, f"{base}_frame{fi:04d}_debug.jpg")
            cv2.imwrite(dbg_path, dbg)
            print("Wrote debug:", dbg_path)


if __name__ == "__main__":
    main()
