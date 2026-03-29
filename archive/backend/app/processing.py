"""Processing utilities for a minimal MBP prototype (pose + ball tracking).

This module provides a self-contained prototype pipeline that:
- normalizes input videos with ffmpeg
- runs per-frame pose estimation via MediaPipe (if available)
- detects the ball using simple Hough/color heuristics
- links detections into tracks and attempts release detection

The functions are small, easily replaceable with ML-based detectors later.
"""

import os
import subprocess
import tempfile
import time
from typing import Dict, Any, Optional, List
import base64
import requests

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def normalize_video(input_path: str, output_path: Optional[str] = None, target_fps: int = 30, width: int = 1280, height: int = 720) -> str:
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"fps={target_fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        output_path,
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return output_path


def get_video_info(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    return {"path": path, "fps": fps, "frame_count": frame_count, "duration_s": duration, "width": width, "height": height}


def run_pose_on_video(path: str, max_frames: Optional[int] = None, timeout_seconds: Optional[float] = 60.0) -> Dict[str, Any]:
    if mp is None:
        raise RuntimeError("mediapipe is not installed or failed to import")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    mp_pose = mp.solutions.pose
    selected = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]

    results_list: List[Dict[str, Any]] = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        idx = 0
        t0 = time.time()
        while True:
            # timeout guard
            if timeout_seconds is not None and (time.time() - t0) > timeout_seconds:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and idx >= max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            frame_landmarks: Dict[str, Any] = {}
            if res.pose_landmarks:
                for lm in selected:
                    lm_obj = res.pose_landmarks.landmark[lm]
                    frame_landmarks[lm.name] = (lm_obj.x, lm_obj.y, lm_obj.z, lm_obj.visibility)

            results_list.append({"frame_index": idx, "landmarks": frame_landmarks})
            idx += 1

    cap.release()
    return {"path": path, "fps": fps, "frame_count": frame_count, "landmarks": results_list}


def detect_ball_hough(path: str, max_frames: Optional[int] = None, timeout_seconds: Optional[float] = 30.0) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    detections = []
    idx = 0
    t0 = time.time()
    while True:
        if timeout_seconds is not None and (time.time() - t0) > timeout_seconds:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and idx >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=6, maxRadius=200)
        if circles is not None:
            circles = circles[0]
            best = max(circles, key=lambda c: c[2])
            x, y, r = int(best[0]), int(best[1]), int(best[2])
            detections.append({"frame_index": idx, "x": x, "y": y, "r": r})
        idx += 1

    cap.release()
    return {"path": path, "detections": detections}


def detect_ball_color_blob(path: str, hsv_lower=(0, 0, 0), hsv_upper=(179, 255, 80), max_frames: Optional[int] = None, timeout_seconds: Optional[float] = 30.0) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    detections = []
    idx = 0
    t0 = time.time()
    while True:
        if timeout_seconds is not None and (time.time() - t0) > timeout_seconds:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and idx >= max_frames:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area > 50:
                x, y, w, h = cv2.boundingRect(best)
                detections.append({"frame_index": idx, "bbox": [int(x), int(y), int(w), int(h)], "area": float(area)})
        idx += 1

    cap.release()
    return {"path": path, "detections": detections}


def detect_ball_combined(path: str, max_frames: Optional[int] = None, timeout_seconds: Optional[float] = 30.0) -> Dict[str, Any]:
    """Combine color-blob and Hough circle detection for robustness.

    Strategy:
    - For each frame create a color mask tuned for dark/orange tones.
    - Find contours and pick strong candidates.
    - Run HoughCircles on the grayscale image and merge circle detections
      with contour centroids when they align.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    detections = []
    idx = 0
    t0 = time.time()
    while True:
        if timeout_seconds is not None and (time.time() - t0) > timeout_seconds:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and idx >= max_frames:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # dark mask
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 90))
        # orange-ish mask
        orange_mask = cv2.inRange(hsv, (5, 80, 80), (25, 255, 255))
        mask = cv2.bitwise_or(dark_mask, orange_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate = None
        if contours:
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area > 200:
                if len(best) >= 5:
                    ellipse = cv2.fitEllipse(best)
                    (cx, cy), (ma, mb), angle = ellipse
                    cx = int(cx)
                    cy = int(cy)
                    r_approx = int((ma + mb) / 4)  # approximate radius
                    candidate = (cx, cy, r_approx, area)
                else:
                    # fallback to bounding rect
                    x, y, w, h = cv2.boundingRect(best)
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)
                    candidate = (cx, cy, None, area)

        # Hough on grayscale to detect circular shapes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40, param1=50, param2=30, minRadius=6, maxRadius=200)
        if circles is not None:
            circles = circles[0]
            # pick largest circle
            best_c = max(circles, key=lambda c: c[2])
            cx_h, cy_h, r_h = int(best_c[0]), int(best_c[1]), int(best_c[2])
        else:
            cx_h = cy_h = r_h = None

        # Motion check: compute frame diff with previous gray frame to prefer moving objects
        motion_score = 0
        if 'prev_gray' in locals():
            diff = cv2.absdiff(gray, prev_gray)
            _, diff_t = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            # compute motion in candidate region
            if candidate:
                cx, cy, r_or_bbox, area = candidate
                if r_or_bbox is not None:  # has radius
                    r = r_or_bbox
                    x0 = max(0, cx - r)
                    y0 = max(0, cy - r)
                    x1 = min(gray.shape[1], cx + r)
                    y1 = min(gray.shape[0], cy + r)
                    roi = diff_t[y0:y1, x0:x1]
                    motion_score = int(roi.sum() / 255) if roi.size > 0 else 0
                else:
                    # fallback to full frame if no radius
                    motion_score = int(diff_t.sum() / 255)
            else:
                motion_score = int(diff_t.sum() / 255)

        # prefer detections in lower part of frame and with motion
        h_frame = frame.shape[0]
        keep_detection = False
        if candidate and cx_h is not None:
            cx, cy, r_or_bbox, area = candidate
            dist = ((cx - cx_h) ** 2 + (cy - cy_h) ** 2) ** 0.5
            if dist < max(30, int(0.5 * r_h)):
                # candidate aligns with Hough circle
                # require motion and that detection is in lower portion of frame
                if motion_score > 20 and cy > int(0.35 * h_frame):
                    detections.append({"frame_index": idx, "x": int((cx + cx_h) / 2), "y": int((cy + cy_h) / 2), "r": int(r_h), "motion": motion_score})
                    keep_detection = True
            if not keep_detection and r_or_bbox is None:
                # if contour has strong motion and is low in frame, accept as bbox (only if no radius)
                # but since we have ellipse, perhaps always use circle if possible
                pass  # skip bbox for now
        elif cx_h is not None:
            # Hough-only detection: require motion near circle center and low vertical position
            if 'prev_gray' in locals():
                # sample small window around circle center
                r = max(6, int(r_h))
                x0 = max(0, cx_h - r)
                y0 = max(0, cy_h - r)
                x1 = min(gray.shape[1], cx_h + r)
                y1 = min(gray.shape[0], cy_h + r)
                roi = diff_t[y0:y1, x0:x1]
                score = int(roi.sum() / 255) if roi.size > 0 else 0
            else:
                score = 0
            if score > 5 and cy_h > int(0.35 * h_frame):  # lower for static
                detections.append({"frame_index": idx, "x": cx_h, "y": cy_h, "r": r_h, "motion": score})
                keep_detection = True
        elif candidate:
            cx, cy, r_or_bbox, area = candidate
            if r_or_bbox is not None and motion_score > 10 and cy > int(0.35 * h_frame):  # lower motion threshold for static
                detections.append({"frame_index": idx, "x": cx, "y": cy, "r": r_or_bbox, "motion": motion_score})

        # store previous gray for next iteration
        prev_gray = gray

        idx += 1

    cap.release()
    return {"path": path, "detections": detections}


def _unify_detections(detector_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in detector_output.get("detections", []):
        if "x" in d and "y" in d:
            out.append({"frame_index": d["frame_index"], "x": d["x"], "y": d["y"]})
        elif "bbox" in d:
            x, y, w, h = d["bbox"]
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            out.append({"frame_index": d["frame_index"], "x": cx, "y": cy})
    return out


def track_detections(detections: List[Dict[str, Any]], max_distance: int = 80, max_gap: int = 3) -> List[Dict[str, Any]]:
    if not detections:
        return []

    by_frame = {}
    for d in detections:
        by_frame.setdefault(d["frame_index"], []).append(d)

    frames = sorted(by_frame.keys())
    tracks: List[Dict[str, Any]] = []
    next_id = 1
    active: List[Dict[str, Any]] = []

    for f in frames:
        items = by_frame[f]

        for item in items:
            best_track = None
            best_dist = None
            for t in active:
                if f - t["last_frame"] > max_gap:
                    continue
                dx = item["x"] - t["last_x"]
                dy = item["y"] - t["last_y"]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= max_distance and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_track = t

            if best_track is not None:
                best_track["detections"].append(item)
                best_track["last_x"] = item["x"]
                best_track["last_y"] = item["y"]
                best_track["last_frame"] = f
            else:
                t = {"track_id": next_id, "last_x": item["x"], "last_y": item["y"], "last_frame": f, "detections": [item]}
                next_id += 1
                active.append(t)

        still_active = []
        for t in active:
            if f - t["last_frame"] <= max_gap:
                still_active.append(t)
            else:
                tracks.append({"track_id": t["track_id"], "detections": t["detections"]})
        active = still_active

    for t in active:
        tracks.append({"track_id": t["track_id"], "detections": t["detections"]})

    return tracks


def _get_wrist_coords_from_pose(pose_result: Dict[str, Any], hand: str = "RIGHT_WRIST") -> Dict[int, tuple]:
    mapping = {}
    for item in pose_result.get("landmarks", []):
        fi = item.get("frame_index")
        lm = item.get("landmarks", {})
        if hand in lm:
            x, y, z, v = lm[hand]
            mapping[fi] = (x, y)
    return mapping


def _compute_wrist_speeds(wrist_map: Dict[int, tuple], fps: float, width: int, height: int) -> Dict[int, float]:
    frames = sorted(wrist_map.keys())
    speeds: Dict[int, float] = {}
    prev = None
    for f in frames:
        x_norm, y_norm = wrist_map[f]
        x = x_norm * width
        y = y_norm * height
        if prev is None:
            speeds[f] = 0.0
        else:
            px_dist = ((x - prev[0]) ** 2 + (y - prev[1]) ** 2) ** 0.5
            speeds[f] = px_dist * fps
        prev = (x, y)
    return speeds


def detect_lane(img: np.ndarray) -> Dict[str, Any]:
    """Detect lane using Hough line detection. Lanes appear as long parallel lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=20)
    lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter near-horizontal lines (lanes are mostly straight)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10:  # Near-horizontal
                lane_lines.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    return {"description": "Long, straight, brown/tan lines at bottom of image", "lines": lane_lines}


def detect_rack(img: np.ndarray) -> Dict[str, Any]:
    """Detect rack using contour analysis for rectangular shapes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    racks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # Large enough
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            aspect_ratio = rect[1][0] / rect[1][1] if rect[1][1] != 0 else 0
            if 0.5 < aspect_ratio < 2:  # Roughly square/rectangular
                racks.append({"bbox": box.tolist(), "area": area})
    return {"description": "Rectangular wooden/metal stand near lane, holding balls", "racks": racks}


def detect_approach(img: np.ndarray) -> Dict[str, Any]:
    """Detect approach as the upper textured region before lane."""
    h, w = img.shape[:2]
    # Simple: top 30% of image, assuming lane is bottom
    approach_region = img[0:int(0.3 * h), :]
    # Feedback: textured floor, often with patterns
    return {"description": "Textured floor area at top, before lane start", "region": [0, 0, w, int(0.3 * h)]}


def detect_release(pose_result: Dict[str, Any], ball_result: Dict[str, Any], video_info: Dict[str, Any]) -> Dict[str, Any]:
    try:
        fps = video_info.get("fps", 30.0)
        width = int(video_info.get("width", 1280))
        height = int(video_info.get("height", 720))

        wrist_map = _get_wrist_coords_from_pose(pose_result, hand="RIGHT_WRIST")
        if not wrist_map:
            wrist_map = _get_wrist_coords_from_pose(pose_result, hand="LEFT_WRIST")

        ball_unified = ball_result.get("unified") if isinstance(ball_result, dict) else None

        if ball_unified:
            ball_speeds = _compute_ball_speeds_from_unified(ball_unified, fps)
            tracks = ball_result.get("tracks") or []
            if tracks:
                chosen_track = max(tracks, key=lambda t: len(t.get("detections", [])))
                unified = chosen_track.get("detections", [])
            else:
                unified = ball_unified

            ball_map = {d["frame_index"]: (d["x"], d["y"]) for d in unified}

            if wrist_map:
                common_frames = sorted(set(wrist_map.keys()) & set(ball_map.keys()))
                if not common_frames:
                    candidate_frames = sorted(ball_speeds.keys())
                else:
                    dist_rates = {}
                    prev_d = None
                    for f in common_frames:
                        wx_norm, wy_norm = wrist_map[f]
                        wx = wx_norm * width
                        wy = wy_norm * height
                        bx, by = ball_map[f]
                        d = ((bx - wx) ** 2 + (by - wy) ** 2) ** 0.5
                        if prev_d is None:
                            dist_rates[f] = 0.0
                        else:
                            dist_rates[f] = d - prev_d
                        prev_d = d

                    sorted_rates = sorted(dist_rates.items(), key=lambda x: x[0])
                    candidate_frames = [f for f, r in sorted_rates if r > 5.0]
            else:
                candidate_frames = [f for f, s in ball_speeds.items() if s > 50.0]

            if candidate_frames:
                frame_idx = int(candidate_frames[0])
                return {"frame_index": frame_idx, "time_s": frame_idx / fps, "confidence": 0.8, "method": "ball_wrist"}

            if ball_speeds:
                peak = max(ball_speeds.items(), key=lambda kv: kv[1])
                return {"frame_index": int(peak[0]), "time_s": peak[0] / fps, "confidence": 0.6, "method": "ball_speed_peak"}

        if wrist_map:
            wrist_speeds = _compute_wrist_speeds(wrist_map, fps, width, height)
            if wrist_speeds:
                peak_frame = max(wrist_speeds.items(), key=lambda kv: kv[1])[0]
                return {"frame_index": int(peak_frame), "time_s": peak_frame / fps, "confidence": 0.5, "method": "wrist_speed_peak"}

        return {"error": "insufficient_data", "reason": "no wrist or ball detections"}
    except Exception as e:
        return {"error": "exception", "reason": str(e)}


def compare_videos_sync(path1: str, path2: str, max_frames: int = 500) -> Dict[str, Any]:
    start = time.time()
    norm1 = None
    norm2 = None
    try:
        norm1 = normalize_video(path1)
        norm2 = normalize_video(path2)

        info1 = get_video_info(norm1)
        info2 = get_video_info(norm2)

        try:
            pose1 = run_pose_on_video(norm1, max_frames=max_frames)
        except Exception:
            pose1 = {"error": "mediapipe unavailable or failed"}
        try:
            pose2 = run_pose_on_video(norm2, max_frames=max_frames)
        except Exception:
            pose2 = {"error": "mediapipe unavailable or failed"}

        # Ball detection + simple tracking
        try:
            det1 = detect_ball_hough(norm1, max_frames=max_frames)
            if not det1.get("detections"):
                det1 = detect_ball_color_blob(norm1, max_frames=max_frames)
            unified1 = _unify_detections(det1)
            tracks1 = track_detections(unified1)
            ball1 = {"raw": det1, "unified": unified1, "tracks": tracks1}
        except Exception:
            ball1 = {"error": "ball detection failed"}

        try:
            det2 = detect_ball_hough(norm2, max_frames=max_frames)
            if not det2.get("detections"):
                det2 = detect_ball_color_blob(norm2, max_frames=max_frames)
            unified2 = _unify_detections(det2)
            tracks2 = track_detections(unified2)
            ball2 = {"raw": det2, "unified": unified2, "tracks": tracks2}
        except Exception:
            ball2 = {"error": "ball detection failed"}

        # Attempt release detection
        rel1 = detect_release(pose1 if isinstance(pose1, dict) else {}, ball1 if isinstance(ball1, dict) else {}, info1)
        rel2 = detect_release(pose2 if isinstance(pose2, dict) else {}, ball2 if isinstance(ball2, dict) else {}, info2)

        elapsed = time.time() - start
        return {
            "status": "ok",
            "summary": "Preprocessing complete (normalization + info + pose + ball attempts).",
            "elapsed_s": elapsed,
            "video_info": {"video1": info1, "video2": info2},
            "pose": {"video1": pose1, "video2": pose2},
            "ball": {"video1": ball1, "video2": ball2},
            "release": {"video1": rel1, "video2": rel2},
        }
    finally:
        for p in (norm1, norm2):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def detect_lane(img: np.ndarray) -> Dict[str, Any]:
    """Detect lane using Hough line detection. Lanes appear as long parallel lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=20)
    lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter near-horizontal lines (lanes are mostly straight)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10:  # Near-horizontal
                lane_lines.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    return {"description": "Long, straight, brown/tan lines at bottom of image", "lines": lane_lines}


def detect_rack(img: np.ndarray) -> Dict[str, Any]:
    """Detect rack using contour analysis for rectangular shapes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    racks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # Large enough
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            aspect_ratio = rect[1][0] / rect[1][1] if rect[1][1] != 0 else 0
            if 0.5 < aspect_ratio < 2:  # Roughly square/rectangular
                racks.append({"bbox": box.tolist(), "area": area})
    return {"description": "Rectangular wooden/metal stand near lane, holding balls", "racks": racks}


def detect_approach(img: np.ndarray) -> Dict[str, Any]:
    """Detect approach as the upper textured region before lane."""
    h, w = img.shape[:2]
    # Simple: top 30% of image, assuming lane is bottom
    approach_region = img[0:int(0.3 * h), :]
    # Feedback: textured floor, often with patterns
    return {"description": "Textured floor area at top, before lane start", "region": [0, 0, w, int(0.3 * h)]}


def detect_ball_roboflow(img_path: str, api_key: str, model: str = "bowling-ball-detection", version: str = "1") -> List[Dict[str, Any]]:
    """Detect bowling balls using Roboflow API."""
    # Encode image to base64
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    url = f"https://detect.roboflow.com/{model}/{version}"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"image": img_data}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        detections = []
        for pred in result.get("predictions", []):
            detections.append({
                "x": pred["x"],
                "y": pred["y"],
                "width": pred["width"],
                "height": pred["height"],
                "confidence": pred["confidence"]
            })
        return detections
    except Exception as e:
        print(f"Roboflow API error: {e}")
        return []


def analyze_bowling_image(img_path: str, roboflow_api_key: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a bowling image for objects: ball, rack, approach, lane."""
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Could not load image"}

    # Person detection with YOLO
    person_detections = []
    if YOLO is not None:
        try:
            model = YOLO('yolov8m.pt')  # Load medium model for better accuracy
            results = model(img, conf=0.3)  # Lower confidence threshold
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        person_detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
        except Exception as e:
            print(f"YOLO detection failed: {e}")

    # Pose detection
    pose_detections = []
    if mp is not None:
        try:
            with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
                    pose_detections.append({"landmarks": landmarks})
        except Exception as e:
            print(f"Pose detection failed: {e}")
            pose_detections = []

    # For ball, use Roboflow if API key provided, else fallback to Hough
    ball_detections = []
    if roboflow_api_key:
        ball_detections = detect_ball_roboflow(img_path, roboflow_api_key)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=5, maxRadius=100)
        if circles is not None:
            circles = circles[0]
            for c in circles:
                x, y, r = int(c[0]), int(c[1]), int(c[2])
                ball_detections.append({"x": x, "y": y, "r": r})

    lane = detect_lane(img)
    rack = detect_rack(img)
    approach = detect_approach(img)

    return {
        "ball": {"description": "Dark circular sphere, often black/orange, may be on rack or moving", "detections": ball_detections},
        "rack": rack,
        "approach": approach,
        "lane": lane,
        "pose": {"description": "Human pose landmarks", "detections": pose_detections},
        "person": {"description": "Detected persons via YOLO", "detections": person_detections}
    }

