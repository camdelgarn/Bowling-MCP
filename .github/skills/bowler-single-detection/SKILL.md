---
name: bowler-single-detection
description: 'Detect and track a single bowler in bowling videos using YOLO full-frame inference with temporal filtering. Use when tuning phantom bowler suppression, reducing missed frames, generating annotated videos, and parsing detection stats.'
argument-hint: '[video_path] [model_path] [output_stem]'
user-invocable: true
---

# Bowler Single Detection

Use this skill to run full-frame YOLO person detection, keep only one tracked bowler, suppress phantom detections, and measure quality from JSON stats.

## When To Use
- A bowling video shows false extra people.
- The model is mostly correct but flickers to zero on short gaps.
- You need deterministic output metrics for tuning.
- You need a repeatable command set for this repository.

## Prerequisites
1. CUDA-enabled PyTorch and Ultralytics are installed in the active Python environment.
2. Input video exists (example: `F:\bowlingvideos\GX010016.MP4`).
3. Script exists: `yolo/track_single_bowler.py`.

## Default Workflow
1. Run full-frame single-bowler tracking (no ROI).
2. Save annotated output video and JSON stats.
3. Parse stats for detection rate, misses, and confidence.
4. Tune thresholds and rerun.

## Run Command (Full Frame)
```powershell
python "F:\development\Bowling-MCP\yolo\track_single_bowler.py" \
  --source "F:\bowlingvideos\GX010016.MP4" \
  --model "F:\development\Bowling-MCP\yolov8n.pt" \
  --output "F:\development\Bowling-MCP\yolo\outputs\GX010016_single_bowler_fullframe_v3.mp4" \
  --stats "F:\development\Bowling-MCP\yolo\outputs\GX010016_single_bowler_fullframe_v3_stats.json" \
  --device 0 \
  --conf 0.58 \
  --iou 0.30 \
  --min-area-ratio 0.008 \
  --max-center-shift-ratio 0.12 \
  --max-hold-frames 6
```

## Parse Command (Quality Metrics)
```powershell
$p='F:\development\Bowling-MCP\yolo\outputs\GX010016_single_bowler_fullframe_v3_stats.json';
$d=Get-Content $p -Raw | ConvertFrom-Json;
$total=$d.Count;
$det=$d | Where-Object { $_.person_count -eq 1 };
$miss=$d | Where-Object { $_.person_count -eq 0 };
$avgConf=($det | Measure-Object -Property selected_confidence -Average).Average;
Write-Output "total_frames=$total";
Write-Output "detected_frames=$($det.Count)";
Write-Output "missed_frames=$($miss.Count)";
Write-Output "detection_rate=$([math]::Round(($det.Count/$total)*100,2))";
Write-Output "avg_confidence=$([math]::Round($avgConf,4))";
```

## Tuning Guide
- `--conf`: raise to reduce false positives; lower to reduce misses.
- `--min-area-ratio`: raise to ignore tiny ghost boxes.
- `--max-center-shift-ratio`: lower to prevent sudden tracker jumps.
- `--max-hold-frames`: increase to bridge short misses without changing person identity.
- `--iou`: lower can reduce duplicate overlap picks.

## Known Good Behavior
- `person_count` should be only `0` or `1`.
- `person_count` should never be `2+` in final stats.
- Most misses happen at clip start/end or severe motion blur.

## Debug Mode
Add these options only when diagnosing:
- `--draw-raw` to draw all raw person boxes.
- `--report-raw-count` to include raw YOLO candidate count in stats.

## Outputs
- Annotated video in `yolo/outputs/`.
- Stats JSON in `yolo/outputs/` with fields:
  - `frame`
  - `person_count`
  - `selected_source` (`detected`, `held`, `none`)
  - `selected`
  - `selected_confidence`

## Session Learnings (GX010016)
- Full-clip feet pose tracking is stable in the middle of the approach and weaker at clip edges where the bowler enters/exits frame.
- Baseline feet run (`pose_conf=0.4`, `ankle_kp_conf=0.25`) achieved about 81.10% both-ankle detection across 598 frames.
- Tuned feet run (`pose_conf=0.35`, `ankle_kp_conf=0.20`) improved both-ankle detection to about 83.44%, with largest gain in the first third.
- Generic COCO sports-ball detection (`class 32`) is sparse for this camera angle and motion blur; right hand is usually detectable but ball detections are intermittent.

## Ball Tracking Guidance
- If ball detections are fewer than about 20-30 frames per shot, do not trust backswing phase logic from COCO ball class alone.
- Use a custom bowling-ball detector trained on this lane/camera geometry for reliable pickup and hand-ball association.

## Custom Bowling-Ball Detector (Recommended)
1. Build a dataset with lane-specific frames (pickup, pushaway, backswing, release, and negatives).
2. Label one class (`bowling_ball`) with tight boxes.
3. Train at higher image size (`imgsz 1280`) for small-object recall.
4. Start from `yolov8s.pt` and fine-tune with active-learning rounds.
5. Replace `--detect-model yolov8n.pt` with trained `best.pt` in standalone scripts.

Example training command:
```powershell
yolo task=detect mode=train model=yolov8s.pt data=F:/development/Bowling-MCP/dataset/bowling_ball/bowling_ball.yaml imgsz=1280 epochs=120 batch=8 device=0 project=F:/development/Bowling-MCP/yolo/runs name=bowling_ball_v1
```
