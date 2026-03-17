#!/usr/bin/env python3
"""Detect a circular object near a given center in an image.

Usage: python detect_by_center.py --image PATH --center X Y
"""
import argparse
import os
import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def clamp(v, a, b):
    return max(a, min(b, v))


def detect_circle_near(img, cx, cy, crop_size=600):
    h, w = img.shape[:2]
    half = crop_size // 2
    x0 = clamp(int(cx - half), 0, w - 1)
    y0 = clamp(int(cy - half), 0, h - 1)
    x1 = clamp(int(cx + half), 0, w)
    y1 = clamp(int(cy + half), 0, h)
    crop = img[y0:y1, x0:x1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Tighter Hough parameters to avoid large false positives
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=30, param1=60, param2=35, minRadius=20, maxRadius=150)
    if circles is not None:
        circles = circles[0]
        # convert circle coords back to image space
        circs = []
        for c in circles:
            ccx = int(c[0]) + x0
            ccy = int(c[1]) + y0
            r = int(c[2])
            d = ((ccx - cx) ** 2 + (ccy - cy) ** 2) ** 0.5
            circs.append((ccx, ccy, r, d))

        # Filter by size constraints and prioritize closer to center
        valid = [c for c in circs if 20 <= c[2] <= 150]
        if valid:
            # Sort by distance, pick the closest
            best = min(valid, key=lambda x: x[3])
            return (best[0], best[1], best[2]), crop, (x0, y0)

    # If no good Hough circles, use contour analysis with ellipse fit for partial circles
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 100))
    orange_mask = cv2.inRange(hsv, (5, 70, 70), (25, 255, 255))
    mask = cv2.bitwise_or(dark_mask, orange_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (mx, my), (ma, mb), angle = ellipse
                # Approximate radius as average of semi-axes
                mr = int((ma + mb) / 4)
                if not (20 <= mr <= 150):
                    continue
                mx = int(mx) + x0
                my = int(my) + y0
                d = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
                valid_contours.append((mx, my, mr, d))
            else:
                # Fallback to minEnclosingCircle if not enough points for ellipse
                (mx, my), mr = cv2.minEnclosingCircle(cnt)
                mr = int(mr)
                if not (20 <= mr <= 150):
                    continue
                mx = int(mx) + x0
                my = int(my) + y0
                d = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
                valid_contours.append((mx, my, mr, d))
        
        if valid_contours:
            # Pick the closest to center
            best = min(valid_contours, key=lambda x: x[3])
            return (best[0], best[1], best[2]), crop, (x0, y0)

    return None, crop, (x0, y0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--center", nargs=2, type=int, required=True)
    p.add_argument("--crop", type=int, default=600)
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read image:", args.image)
        return

    cx, cy = args.center
    res, crop, origin = detect_circle_near(img, cx, cy, crop_size=args.crop)

    out = img.copy()
    if res is not None:
        ccx, ccy, r = res
        cv2.circle(out, (int(ccx), int(ccy)), int(r), (0, 255, 0), 3)
        cv2.line(out, (int(ccx) - 10, int(ccy)), (int(ccx) + 10, int(ccy)), (0, 255, 0), 2)
        cv2.line(out, (int(ccx), int(ccy) - 10), (int(ccx), int(ccy) + 10), (0, 255, 0), 2)
        print(f"Detected circle: center=({ccx},{ccy}) radius={r}")
    else:
        print("No circle detected near provided center")

    base = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_center_{cx}_{cy}_detected.jpg")
    cv2.imwrite(out_path, out)
    print("Wrote:", out_path)


if __name__ == '__main__':
    main()
