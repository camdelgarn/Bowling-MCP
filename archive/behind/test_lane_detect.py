"""Quick diagnostic: find gradient pairs at different y/sigma for GX010023."""
import cv2, numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

cap = cv2.VideoCapture(r"c:\video\GX010023.MP4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 118)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()
hint = 2368

for y in [1330, 1200, 1150, 1100]:
    for sigma in [30, 60, 90]:
        row = gaussian_filter1d(gray[y, :].astype(np.float64), sigma=sigma)
        grad = np.gradient(row)
        pp, _ = find_peaks(grad, height=0.3, distance=40)
        np_, _ = find_peaks(-grad, height=0.3, distance=40)
        pairs = []
        for p in pp:
            for n in np_[np_ > p][:3]:
                w = n - p
                if 150 < w < 700:
                    c = (p + n) // 2
                    b = float(np.mean(gray[y, p:n]))
                    pairs.append((p, n, w, c, b, abs(c - hint)))
        pairs.sort(key=lambda x: x[5])
        top = pairs[:2] if pairs else "none"
        print(f"y={y}, sigma={sigma:2d}: {len(pairs):2d} pairs, closest: {top}")
