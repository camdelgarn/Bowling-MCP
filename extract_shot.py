import sys
import cv2

src   = r"F:\bowlingvideos\behindlane23.MP4"
dst   = r"F:\bowlingvideos\behindlane23_shot1.mp4"
START = 580
END   = 1460

print(f"Opening {src} ...")
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    sys.exit("ERROR: could not open source video")

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source: {w}x{h} @ {fps:.3f} fps")

# Try codecs in order of preference
codecs = [("mp4v", dst), ("XVID", dst.replace(".mp4", ".avi"))]
writer = None
chosen_dst = dst
for fourcc_str, out_path in codecs:
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    w_test = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if w_test.isOpened():
        writer = w_test
        chosen_dst = out_path
        print(f"Using codec {fourcc_str} -> {out_path}")
        break
    w_test.release()

if writer is None:
    sys.exit("ERROR: no working VideoWriter codec found")

cap.set(cv2.CAP_PROP_POS_FRAMES, START)
count = 0
for fi in range(START, END + 1):
    ok, frame = cap.read()
    if not ok:
        print(f"  Read stopped at frame {START + count}")
        break
    writer.write(frame)
    count += 1
    if count % 100 == 0:
        print(f"  Written {count}/{END - START + 1} frames ...")

cap.release()
writer.release()
print(f"Done: {count} frames ({count/fps:.1f}s) -> {chosen_dst}")
