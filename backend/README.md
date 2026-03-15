# Bowling MCP Backend (Scaffold)

This scaffold provides a minimal FastAPI app to accept two video uploads and run a placeholder comparison.

Quick start (local):

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

3. POST two files to `http://localhost:8000/compare` using a tool like `curl` or Postman.

Notes:
- The current `app/processing.py` contains stubs. Next steps are implementing normalization, pose estimation, ball detection/tracking, and metrics extraction.
- For GPU acceleration and production, build the Docker image and run on an NVIDIA-enabled host.
