# gopro.md

This module provides GoPro-specific camera connection and streaming logic.

- `gopro.py`: GoPro camera connection and stream handling.
- Extend from the generic camera base if needed.

## Implemented capabilities

- Start and stop local RTMP NGINX in `rtmp/`.
- Scan for GoPro devices over BLE.
- Connect to a selected GoPro BLE address (or first discovered GoPro).
- Generate RTMP publish URL for GoPro streaming.

## CLI usage

Run from project root:

```powershell
python cameras/gopro/gopro.py start-nginx
python cameras/gopro/gopro.py scan --timeout 8
python cameras/gopro/gopro.py connect --address "XX:XX:XX:XX:XX:XX"
python cameras/gopro/gopro.py stream-url
python cameras/gopro/gopro.py stop-nginx
```

If BLE features are needed, install dependency:

```powershell
pip install bleak
```

## Three-GoPro synchronized capture app

Use this app to ingest three RTMP streams from NGINX simultaneously, show three live windows, and write timestamp-paired MP4 outputs.

Script:

`cameras/gopro/multi_stream_capture.py`

Default stream URLs expected:

- `rtmp://127.0.0.1:1935/live/gopro1`
- `rtmp://127.0.0.1:1935/live/gopro2`
- `rtmp://127.0.0.1:1935/live/gopro3`

Run:

```powershell
python cameras/gopro/multi_stream_capture.py
```

Python version note:

- Use Python 3.12.x for this workflow.
- Scripts now fail fast on Python 3.13+ to avoid BLE/OpenCV runtime issues.
- On this machine, use:

```powershell
C:/Users/grass/AppData/Local/Programs/Python/Python312/python.exe cameras/gopro/multi_stream_capture.py
```

Optional custom stream map:

```powershell
python cameras/gopro/multi_stream_capture.py --streams-file cameras/gopro/streams.example.json
```

Each session writes:

- `backend/outputs/gopro_sessions/<SESSION_ID>/<stream_name>_<SESSION_ID>.mp4`
- `backend/outputs/gopro_sessions/<SESSION_ID>/<stream_name>_<SESSION_ID>_timestamps.csv`
- `backend/outputs/gopro_sessions/<SESSION_ID>/session_manifest.json`

`timestamps.csv` includes frame index, UTC capture time, and elapsed milliseconds from the shared session start.
