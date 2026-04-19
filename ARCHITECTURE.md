# Project Architecture

## cameras/
- Contains generic and specific camera/video stream handling modules.
- `camera.py` and `camera.md`: Generic camera base and documentation.
- `gopro/`: GoPro-specific connection logic and docs.
- `brio/`: Brio webcam-specific connection logic and docs.

## alley/, person/, ball/
- Structured for organizing approach, lane, behind, and side data or code for each entity.

## archive/
- Contains archived/legacy project files.

## rtmp/
- RTMP server configuration and related files.

---

This structure supports modular development and clear separation of camera and entity-specific logic.
