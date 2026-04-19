# Approach Board Detection - Session Notes (April 4, 2026)

## Key Observations
- Canny edge detection works well for vertical lines near the camera, but fades as you move further away (due to perspective, lighting, and resolution loss).
- The lane edge lines are the most persistent and visible, extending furthest into the approach.
- Too many vertical lines are detected near the camera, but too few are detected further away.
- The space between the two most persistent vertical lines (lane edges) contains 39 boards.
- The approach for the lane to the left (off camera) quickly loses visible lines.

## Ideas for Robust Board Detection
- Use the two most persistent vertical lines as lane edges.
- Interpolate 39 board positions evenly between these two lane edges, even if some board lines are faint or missing.
- Optionally, split the ROI into near and far zones and tune edge/Hough parameters for each region.
- Consider more aggressive contrast enhancement or adaptive thresholding for the far region.

## Next Steps (when resuming)
- Decide whether to use lane edge interpolation or tune detection parameters for near/far regions.
- Implement and visualize interpolated board lines between detected lane edges.
- Continue saving debug images (Canny, detected lines) for reference.

---
_Last updated: April 4, 2026_
