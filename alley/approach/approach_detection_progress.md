# Approach Dot and Tape Measure Detection Progress

## Current Achievements
- Successfully aligned the vertical ROI to cover the area where the approach dots and tape measure appear (using pixel values 1750 to 1900).
- Implemented yellow tape measure detection using HSV color thresholding, with left and right bounds correctly found when the tape measure is in the ROI.
- Dot detection ROI is now automatically set between the detected tape measure edges, focusing on the area where the dots should be.
- The script saves debug images for each run, including:
  - `approach_boards_detected.png`: Shows detected boards (green lines), dots (red circles), and tape measure ROI (yellow rectangle).
  - `dot_detection_roi.png`: The cropped region used for dot detection.
  - `tape_measure_yellow_mask_<timestamp>.png`: The mask used for yellow tape detection.
  - `tape_measure_yellow_detected_<timestamp>.png`: The tape measure region with detected bounds drawn.

## Outstanding Issues
- The five approach dots are not reliably detected yet, possibly due to:
  - HoughCircles parameters needing further tuning.
  - Dots not being sufficiently distinct from the background or reflections.
  - Lighting or contrast issues in the video.
- The yellow tape measure is sometimes confused with reflections or the tape measure container.

## Next Steps (when resuming)
- Further tune the HoughCircles parameters (minDist, param1, param2, minRadius, maxRadius) for better dot detection.
- Consider additional preprocessing (adaptive thresholding, contrast enhancement) on the dot ROI.
- Optionally, manually annotate a frame to compare detected dot positions with ground truth.
- Explore alternative dot detection methods (template matching, blob detection) if HoughCircles remains unreliable.
- Continue saving and reviewing debug images for each run to guide adjustments.

---
_Last updated: March 29, 2026_
