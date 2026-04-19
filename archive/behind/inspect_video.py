"""Find lane edges near the bowler's motion area."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

cap = cv2.VideoCapture(r'c:\video\GX010022.MP4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get empty frame (last frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
cap.release()

print(f'Frame: {w}x{h}')
print(f'Bowler motion is around x=1000-2200 (from heatmap)')
print()

# Search for lane edges focusing on x=800-2400 (where bowler is)
print('Brightest lane-width pairs near the motion area (center < 2000):')
for y in range(800, 2200, 50):
    row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=30)
    grad = np.gradient(row)
    pos_peaks, _ = find_peaks(grad, height=0.2, distance=30)
    neg_peaks, _ = find_peaks(-grad, height=0.2, distance=30)
    
    best = None
    for pp in pos_peaks:
        for np_val in neg_peaks:
            width = np_val - pp
            if 150 < width < 700:
                center = (pp + np_val) // 2
                if center > 2400:
                    continue  # too far right, not bowler's lane
                avg_b = gray[y, pp:np_val].mean()
                if avg_b > 100 and (best is None or avg_b > best[4]):
                    best = (pp, np_val, width, center, avg_b)
    if best:
        print(f'  y={y:4d}: left={best[0]:4d} right={best[1]:4d} w={best[2]:3d} center={best[3]:4d} bright={best[4]:.0f}')
