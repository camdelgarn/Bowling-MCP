#!/usr/bin/env python3
"""Set the detected circle center to given coordinates (x,y) and save an adjusted image.

Usage: python set_center.py [x y] [path_to_detected_image]
"""
import sys
import os
import glob
import cv2

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def find_image(path=None):
    if path and os.path.exists(path):
        return path
    cand = sorted(glob.glob(os.path.join(OUT_DIR, "*_frame0000_detected.jpg")))
    if cand:
        return cand[0]
    cand = sorted(glob.glob(os.path.join(OUT_DIR, "*_detected.jpg")))
    return cand[0] if cand else None


def find_green_circle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (40, 40, 40)
    upper = (80, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(best)
    return int(cx), int(cy), int(r)


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        try:
            tx = int(args[0]); ty = int(args[1])
        except Exception:
            print("Invalid coordinates. Use: x y")
            return
        img_path = args[2] if len(args) >= 3 else None
    else:
        print("Usage: python set_center.py x y [image_path]")
        return

    img_file = find_image(img_path)
    if not img_file:
        print("No detected image found in backend/outputs")
        return
    img = cv2.imread(img_file)
    if img is None:
        print("Failed to read image", img_file)
        return

    found = find_green_circle(img)
    if found:
        cx, cy, r = found
    else:
        # fallback radius
        cx, cy, r = img.shape[1]//2, img.shape[0]//2, min(img.shape[0], img.shape[1])//6

    out = img.copy()
    # draw original (green) and new (red)
    cv2.circle(out, (cx, cy), r, (0,255,0), 3)
    cv2.circle(out, (tx, ty), r, (0,0,255), 3)
    cv2.line(out, (tx-10, ty), (tx+10, ty), (0,0,255), 2)
    cv2.line(out, (tx, ty-10), (tx, ty+10), (0,0,255), 2)

    base = os.path.splitext(os.path.basename(img_file))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_set_{tx}_{ty}.jpg")
    cv2.imwrite(out_path, out)
    print("Wrote:", out_path)
    print("Original center:", (cx, cy), "r=", r)
    print("New center:", (tx, ty))

if __name__ == '__main__':
    main()
