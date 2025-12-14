#!/usr/bin/env python3
"""
Simple API server for quick experimentation.
Allows uploading videos and processing via HTTP.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

# FastAPI for quick API
try:
    from fastapi import FastAPI, File, UploadFile, BackgroundTasks
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn
except ImportError:
    print("Install FastAPI: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import VideoIdentityTransformer

app = FastAPI(title="Video Identity Transform API")

# Global pipeline (initialized once)
pipeline = None
jobs = {}

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup():
    global pipeline
    print("Loading pipeline...")
    pipeline = VideoIdentityTransformer("configs/transform_config.yaml")
    print("Pipeline ready!")


@app.get("/")
async def root():
    return {"status": "ready", "message": "Video Identity Transform API"}


@app.get("/health")
async def health():
    import torch
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/transform")
async def transform_video(
    video: UploadFile = File(...),
    target1: UploadFile = File(None),
    target2: UploadFile = File(None),
    background_tasks: BackgroundTasks = None
):
    """
    Transform video with optional target identity images.
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded video
    video_path = UPLOAD_DIR / f"{job_id}_input.mp4"
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    # Save target images if provided
    targets = []
    for i, target in enumerate([target1, target2]):
        if target:
            target_path = UPLOAD_DIR / f"{job_id}_target{i}.jpg"
            with open(target_path, "wb") as f:
                f.write(await target.read())
            targets.append(str(target_path))
    
    output_path = OUTPUT_DIR / f"{job_id}_output.mp4"
    
    # Process (blocking for now - could be async)
    jobs[job_id] = {"status": "processing"}
    
    try:
        result = pipeline.process_video(
            str(video_path),
            str(output_path)
        )
        jobs[job_id] = {"status": "complete", "output": str(output_path)}
        return {"job_id": job_id, "status": "complete", "download": f"/download/{job_id}"}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Download processed video."""
    output_path = OUTPUT_DIR / f"{job_id}_output.mp4"
    if output_path.exists():
        return FileResponse(output_path, media_type="video/mp4", filename=f"transformed_{job_id}.mp4")
    return JSONResponse(status_code=404, content={"error": "Not found"})


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Check job status."""
    return jobs.get(job_id, {"status": "not_found"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

