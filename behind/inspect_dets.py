from app import processing
import json

video = "../video/behind/20260123_145635.mp4"
res = processing.detect_ball_color_blob(video, hsv_lower=(0, 0, 0), hsv_upper=(179, 255, 90), max_frames=1)
print(json.dumps({"count": len(res.get('detections', [])), "detections": res.get('detections', [])}, indent=2))
