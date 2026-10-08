"""사진 세트(20~100여장) 업로드 → 3D(GLB/STL/PLY) 변환 API.

영상 프레임 추출 등 전처리는 호출 측에서 끝내고, 이 API에는 바로 SfM+MVS 처리 가능한
사진 세트만 보낸다고 가정한다. 처리 후 업로드 원본·작업 폴더(work/)는 자동 삭제되고,
최종 결과 3개(glb/stl/ply)만 남는다.

실행: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from photogrammetry import PipelineConfig
from photogrammetry.config import QUALITY_PRESETS
from photogrammetry.tools import PROJECT_ROOT

from .auth import verify_api_key
from .jobs import Job, create_job, get_job, list_jobs, new_job_id

UPLOAD_ROOT = PROJECT_ROOT / "api_uploads"
OUTPUT_ROOT = PROJECT_ROOT / "api_output"
UPLOAD_ROOT.mkdir(exist_ok=True)
OUTPUT_ROOT.mkdir(exist_ok=True)

app = FastAPI(
    title="modelingengine API",
    description="영상/이미지 → 3D(GLB/STL/PLY) 변환",
    dependencies=[Depends(verify_api_key)],
)


@app.post("/jobs")
async def submit_job(
    files: list[UploadFile] = File(..., description="사진 파일들 (20~100여장)"),
    quality: str = Form("turbo"),
    formats: str = Form("glb,stl,ply"),
    scale: float = Form(1.0),
    name: str | None = Form(None),
):
    if quality not in QUALITY_PRESETS:
        raise HTTPException(400, f"quality는 {list(QUALITY_PRESETS)} 중 하나여야 합니다.")

    job_id = new_job_id()
    images_dir = UPLOAD_ROOT / job_id
    images_dir.mkdir(parents=True)
    for f in files:
        with (images_dir / Path(f.filename).name).open("wb") as out:
            shutil.copyfileobj(f.file, out)

    cfg = PipelineConfig.preset(quality)
    cfg.formats = [x.strip() for x in formats.split(",") if x.strip()]
    cfg.scale = scale
    cfg.keep_workspace = False  # 처리 후 work/ 폴더(프레임, COLMAP DB 등) 자동 삭제

    job = create_job(job_id, images_dir, OUTPUT_ROOT / job_id, name, cfg)
    return _job_summary(job)


@app.get("/jobs")
async def get_jobs():
    return [_job_summary(j) for j in list_jobs()]


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return _job_summary(job)


@app.get("/jobs/{job_id}/files/{fmt}")
async def download_file(job_id: str, fmt: str):
    job = get_job(job_id)
    if job is None or job.status != "done" or job.result is None:
        raise HTTPException(404, "완료된 결과가 없습니다.")
    path = job.result.get("mesh", {}).get("files", {}).get(fmt)
    if not path or not Path(path).exists():
        raise HTTPException(404, f"{fmt} 파일이 없습니다.")
    return FileResponse(path)


def _job_summary(job: Job) -> dict:
    return {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "result": job.result,
        "error": job.error,
    }
