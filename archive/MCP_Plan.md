# MCP Plan: Bowling Release & Trajectory Comparison

**Overview**
- **Goal:** Build an MCP that accepts two videos of a bowler and reports differences in release (timing, hand/arm pose, release angle/speed) and ball trajectory (initial line, skid/roll/flip, hook/curve).
- **Deliverable:** A minimal web/API service that ingests two videos, outputs a comparative analysis report (numerical metrics + annotated video/overlay + visualizations).

**Scope**
- **Supported viewpoints (MVP):** one primary camera viewpoint — either a side view (recommended) or behind/overhead; both videos should use the same viewpoint for most-accurate comparisons.
- **Video requirements:** 720p or 1080p resolution, 30–60 FPS preferred, stable mount (tripod), clear view of bowler from approach through release; good lighting and minimal occlusion of the throwing hand.
- **Supported outputs:** release frame/time, per-frame arm/hand landmarks, estimated release vector (position + angle + speed proxy), ball 2D trajectory on image plane, trajectory phase segmentation (skid/hook/roll), summary of differences.
- **Not in MVP scope:** full 3D lane/world reconstruction, spin-axis accurate estimation from a single monocular camera, multi-camera fusion, or automated coaching recommendations requiring large labeled datasets.

**Success Metrics (concrete & testable)**
- **Release detection accuracy:** >= 90% of releases detected within 50 ms on a curated validation set (50–100 clips with ground-truth release frames).
- **Pose landmark coverage:** landmarks (shoulder/elbow/wrist) detected in >= 90% of release-adjacent frames; per-landmark visibility/confidence reported.
- **Pose landmark precision:** mean Euclidean error <= 8% of shoulder-to-hip distance (normalized) versus manual annotations on a 50-clip test set.
- **Ball localization accuracy:** average reprojection error of detected ball centroid <= 20 pixels on 720p input for unobstructed sequences.
- **Trajectory consistency metric:** normalized RMS or DTW distance between matched trajectories — target: distinguish clearly different vs similar throws with at least 80% classifier separability (evaluated on labeled pairs).
- **Feature delta thresholds (for reporting):** report differences considered significant if they exceed: release time 30 ms, release angle 3 degrees, lateral displacement 10 cm (image-plane scaled), and peak speed proxy 10%.
- **Latency:** preprocessing + pose + ball-tracking for two 5–8s clips should complete under 30s on a single GPU node (target) and under 2 minutes on a modern CPU-only machine.
- **Robustness checks:** system should produce a confidence score and fail-gracefully messages (e.g., `low_fps`, `dark_video`, `pose_unstable`) when inputs don't meet requirements.

**Evaluation protocols**
- Maintain a small labeled validation set (50–150 clips) with: release frame, wrist/elbow/shoulder landmarks (sampled frames), and ball bounding boxes.
- For each PR/feature, run automated checks measuring the metrics above and store results so regressions are visible.


**High-level Architecture**
- **Ingest:** HTTP endpoint or web UI to upload two video files.
- **Preprocessing:** decode, normalize framerate, crop/resize, sync timelines.
- **Per-frame analysis:** pose estimation, hand/arm keypoint smoothing, ball detection & tracking.
- **Feature extraction:** compute release point/time, release angle, release speed, axis rotation estimate, entry angle, skid/roll transition, lateral displacement over lane.
- **Comparison engine:** align release events, compute differences and similarity scores, produce natural-language summary.
- **Visualization:** overlay keypoints and ball path on video + plots (trajectory X vs Y, speed over time, release vectors).
- **API/UI:** FastAPI backend + simple React UI (MVP) to upload and view results.
- **Storage/Deploy:** Docker + optional GPU host (NVIDIA Docker), object storage for persisted videos/results.

**Data Requirements**
- **Video specs:** 720p preferred, 30–60 FPS; consistent camera viewpoints if possible (benchmarked for front / side / behind).
- **Dataset:** Start with 100–300 paired videos across bowlers to validate algorithms; expand for ML models.
- **Annotations:** release frame/time, hand landmarks, ball bounding boxes (for tracking), lane markers (optional) — use CVAT or Labelbox for labeling.

**Tools & Libraries (recommended)**
- **Pose estimation:** MediaPipe BlazePose or OpenPose (MediaPipe for speed; OpenPose for multi-person or higher fidelity).
- **Ball detection & tracking:** OpenCV for detection, YOLOv8 or Detectron2 for robust detection (if colored ball/marker not available); DeepSORT or strong SORT for tracking.
- **Optical flow / tracking refinement:** OpenCV (Farneback / Pyramidal LK) for short-term tracking and sub-pixel motion.
- **Video I/O & processing:** FFmpeg for decoding/normalizing, OpenCV for frame processing.
- **ML frameworks (if training):** PyTorch or TensorFlow.
- **API / server:** FastAPI (Python) + Uvicorn; containerize with Docker.
- **Frontend:** React + simple visualization libs (Plotly, D3) for trajectory plots and overlays.
- **Labeling / dataset tools:** CVAT, Labelbox, or V7.
- **Visualization export:** FFmpeg to create annotated comparison videos/GIFs.
- **Infrastructure:** Optional S3-compatible storage (MinIO), GPU instance (NVIDIA) for faster inference.

**Detailed Pipeline & Tasks**
- **1) Define evaluation & scope**: specify camera angles supported, required accuracy, and user-facing outputs.
- **2) Ingest + Normalization**: use FFmpeg to normalize framerate and resolution; store metadata (timestamps, fps).
- **3) Synchronization**: support manual alignment plus automatic alignment (cross-correlation of motion energy or audio clap detection).
- **4) Pose estimation**: run per-frame pose detection, apply temporal smoothing (Kalman filter or low-pass), compute elbow/wrist/shoulder vectors.
- **5) Release detection**: detect release frame via combined cues — hand velocity sign change, finger separation (if high-fidelity), sudden ball motion independent of hand.
- **6) Ball detection & tracking**: detect ball bounding boxes, track across frames; refine with optical flow for sub-frame accuracy.
- **7) Feature computation**:
  - **Release time & frame**
  - **Release position (x,y) relative to lane**
  - **Release velocity & derived speed**
  - **Release angle (tilt and horizontal)**
  - **Axis rotation estimate (approx.)**
  - **Trajectory segmentation:** skid -> hook -> roll phases
  - **Lateral deviation / breakpoint location**
- **8) Comparison & statistics**: align releases, compute differences (delta in ms, angles, speeds), compute similarity scores, flag significant deviations.
- **9) Visualization & reporting**: annotated composite video (side-by-side or overlay), plots (trajectory traces, speed/time, angular change), natural-language summary.
- **10) Optional ML refinements**: train small models to classify release style or predict error-prone releases given labeled training data.
- **11) API & UI**: create endpoints to submit videos and fetch results; simple React UI to show overlays, plots, and plain-language insights.
- **12) Deployment and monitoring**: Docker images, GPU node for inference, logging, and performance metrics.

**MVP (Minimum Viable Product)**
- **Scope:** Single camera viewpoint (side or behind), CPU+GPU fallback, manual alignment, deterministic rule-based feature extraction (no ML training).
- **Outputs:** A web UI to upload two clips, annotated side-by-side comparison video, numeric differences for release time/angle/speed, trajectory plot, and a short text summary.
- **Estimated effort:** 4–6 weeks (1 engineer + occasional testing) for a working MVP.

**Evaluation & Metrics**
- **Unit tests:** synthetic video unit tests (known motions) to validate pose and tracker outputs.
- **Human-in-the-loop validation:** sample of 50 comparisons reviewed by an expert coach.
- **Performance tests:** measure latency on CPU and GPU.

**Risks & Mitigations**
- **Camera viewpoint mismatch:** limit initial support to 1–2 viewpoints and document constraints.
- **Low-quality video:** provide upload guidance and include automatic checks (too dark / low fps warning).
- **Pose failures for occluded hands:** supplement with ball tracking and optical-flow heuristics.

**Next Steps**
- **Short-term:** implement ingestion, normalization, and pose + ball detection prototypes.
- **Mid-term:** build comparison engine, visualization, and UI.
- **Long-term:** collect labeled dataset and optionally train ML models for release classification.

**References & Resources**
- **MediaPipe BlazePose:** https://developers.google.com/mediapipe
- **OpenPose:** https://github.com/CMU-Perceptual-Computing-Lab/openpose
- **YOLOv8 / Ultralytics:** https://github.com/ultralytics/ultralytics
- **DeepSORT:** https://github.com/nwojke/deep_sort
- **CVAT:** https://github.com/opencv/cvat

---

If you'd like, I can: (1) scaffold the FastAPI project and processing pipeline, (2) implement the MVP prototype for pose+ball tracking on sample videos, or (3) generate a minimal React UI to upload videos. Tell me which next step to start.