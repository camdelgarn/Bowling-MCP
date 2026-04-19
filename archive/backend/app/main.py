from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uuid
import os
from app.processing import compare_videos_sync

app = FastAPI(title="Bowling MCP Backend")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/compare")
async def compare(video1: UploadFile = File(...), video2: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    path1 = os.path.join(UPLOAD_DIR, f"{job_id}_1_{video1.filename}")
    path2 = os.path.join(UPLOAD_DIR, f"{job_id}_2_{video2.filename}")

    with open(path1, "wb") as f:
        f.write(await video1.read())
    with open(path2, "wb") as f:
        f.write(await video2.read())

    result = compare_videos_sync(path1, path2)
    return JSONResponse(content={"job_id": job_id, "result": result})
