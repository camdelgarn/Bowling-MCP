from app import processing
import json

video = "../video/behind/20260123_145635.mp4"
res = processing.detect_ball_hough(video, max_frames=300)
print(json.dumps({"count": len(res.get('detections', [])), "detections": res.get('detections', [])}, indent=2))
