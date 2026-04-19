"""Find a truly clear frame (no bowler) in GX010014 for proper 2-row calibration."""
import cv2
import sys
import numpy as np

sys.path.insert(0, r"C:\Development\Bowling-MCP\backend")
from detect_lane_livestream import find_all_dot_rows

VIDEO = r"C:\video\behind\GX010014.MP4"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Check frames throughout the whole video for ones where both real dot rows
# (y~676 and y~869 based on nobowler calibration) are visible with 5 dots each
for fi in range(0, fc, 15):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame_4k = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame_4k, (1920, 1080))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = find_all_dot_rows(gray)
    
    # Look for rows with 5 dots near y~676 and y~869
    good_rows = []
    for r in rows:
        if len(r) >= 5:
            mean_y = np.mean([d["y"] for d in r])
            if 600 < mean_y < 750 or 800 < mean_y < 950:
                good_rows.append((mean_y, len(r)))
    
    if len(good_rows) >= 2:
        print(f"Frame {fi:4d} (t={fi/fps:5.1f}s): GOOD - {good_rows}")

cap.release()
