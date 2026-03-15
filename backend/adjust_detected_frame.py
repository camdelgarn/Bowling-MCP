#!/usr/bin/env python3
"""Adjust a detected overlay image by a relative offset.

Finds the green detection overlay in a *_detected.jpg file, computes
an adjusted center by shifting left 10% of radius and down 40% of radius,
and writes an adjusted image with a red circle and crosshair.
"""
import glob
import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "outputs")


def find_detected_image(path=None):
    if path and os.path.exists(path):
        return path
    # prefer frame0000 if present
    candidates = sorted(glob.glob(os.path.join(OUT_DIR, "*_frame0000_detected.jpg")))
    if candidates:
        return candidates[0]
    # fallback to any *_detected.jpg
    candidates = sorted(glob.glob(os.path.join(OUT_DIR, "*_detected.jpg")))
    return candidates[0] if candidates else None


def find_green_circle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # loose green range to catch annotation (may be anti-aliased)
    lower = np.array([40, 40, 40])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    ((cx, cy), r) = cv2.minEnclosingCircle(best)
    return int(cx), int(cy), int(r)


def hough_fallback(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=6, maxRadius=600)
    if circles is None:
        return None
    c = circles[0]
    best = max(c, key=lambda x: x[2])
    return int(best[0]), int(best[1]), int(best[2])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    img_path = find_detected_image(arg)
    if not img_path:
        print("No detected image found in backend/outputs")
        return
    print("Using:", img_path)
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image:", img_path)
        return

    found = find_green_circle(img)
    if not found:
        print("No green overlay found, trying Hough fallback")
        found = hough_fallback(img)
        if not found:
            print("No circle found")
            return

    cx, cy, r = found
    # apply offsets: left 10% of radius, down 40% of radius
    dx = -0.10 * r
    dy = 0.40 * r
    nx = int(round(cx + dx))
    ny = int(round(cy + dy))

    out = img.copy()
    # draw original in green (if present) and adjusted in red
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 255, 0), 3)
    cv2.circle(out, (nx, ny), int(max(4, r * 0.9)), (0, 0, 255), 3)
    # crosshair
    cv2.line(out, (nx - 10, ny), (nx + 10, ny), (0, 0, 255), 2)
    cv2.line(out, (nx, ny - 10), (nx, ny + 10), (0, 0, 255), 2)

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_adjusted.jpg")
    cv2.imwrite(out_path, out)
    print(f"Wrote adjusted image: {out_path}")
    print(f"Original center: ({cx}, {cy}), r={r}")
    print(f"Adjusted center: ({nx}, {ny})")


if __name__ == '__main__':
    main()
