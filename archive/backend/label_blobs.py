"""Generate a PNG with all detected blobs numbered for identification.

Shows TWO tiers:
  - Yellow circles + red numbers: high-contrast blobs (contrast > 25)
  - Gray circles + gray numbers: low-contrast blobs (contrast 10-25) that might be missed dots
"""
import cv2
import numpy as np

gray = cv2.cvtColor(cv2.imread("livestream_frame.jpg"), cv2.COLOR_BGR2GRAY)
out = cv2.imread("livestream_frame.jpg")
h, w = gray.shape

# Use relaxed blob detector to catch more candidates
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0
params.filterByArea = True
params.minArea = 8          # lower to catch smaller dots
params.maxArea = 600
params.filterByCircularity = True
params.minCircularity = 0.20  # slightly more relaxed
params.filterByConvexity = True
params.minConvexity = 0.30    # slightly more relaxed
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)
roi_top = int(h * 0.35)
roi = gray[roi_top:int(h * 0.85), :]
kps = detector.detect(roi)

print(f"Raw blobs from detector: {len(kps)}")

blobs = []
for kp in kps:
    x, yl = kp.pt
    ya = yl + roi_top
    ix, iy = int(x), int(ya)
    if iy < 0 or iy >= h or ix < 0 or ix >= w:
        continue
    pv = gray[iy, ix]
    nb = gray[max(0, iy - 15):iy + 15, max(0, ix - 15):ix + 15].mean()
    c = nb - pv
    # Keep anything with contrast > 10 (show low-contrast ones differently)
    if c > 10:
        blobs.append({"x": ix, "y": iy, "size": round(kp.size, 1),
                       "contrast": round(c, 1), "high": c > 25})

blobs.sort(key=lambda b: (b["y"], b["x"]))

for i, b in enumerate(blobs):
    num = i + 1
    x, y = b["x"], b["y"]
    if b["high"]:
        cv2.circle(out, (x, y), 20, (0, 255, 255), 2)
        cv2.putText(out, str(num), (x - 8, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.circle(out, (x, y), 20, (128, 128, 128), 2)
        cv2.putText(out, str(num), (x - 8, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    label = "s={:.0f} c={:.0f}".format(b["size"], b["contrast"])
    cv2.putText(out, label, (x + 25, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

n_high = sum(1 for b in blobs if b["high"])
n_low = len(blobs) - n_high
cv2.putText(out, f"{n_high} high-contrast (yellow) + {n_low} low-contrast (gray)",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(out, "s=size, c=contrast", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

cv2.imwrite("blobs_numbered.png", out)
print(f"Saved blobs_numbered.png with {len(blobs)} blobs ({n_high} high, {n_low} low)")
print()
for i, b in enumerate(blobs):
    tag = "HIGH" if b["high"] else " low"
    print(f"  #{i+1:2d}  x={b['x']:5d}, y={b['y']:4d}, size={b['size']:5.1f}, "
          f"contrast={b['contrast']:5.1f}  [{tag}]")
