"""모델링 서버: 사진 업로드 → 파이프라인(별도 프로세스) → GLB/STL/PLY 다운로드 (ModelCapture 앱용 API).

  POST /jobs                      multipart, 사진마다 `files` 파트 하나
  GET  /jobs/{job_id}             작업 상태
  GET  /jobs/{job_id}/files/{fmt} 모델 파일 (glb | stl | ply)

작업 응답: { job_id, status: queued|running|done|failed, error, result, created_at, ... }
GPU 하나를 쓰므로 작업은 한 번에 하나씩 순서대로 처리합니다.

실행: python server.py [--host 0.0.0.0] [--port 8000] [--quality turbo]
API 키: 환경 변수 MODEL_API_KEY 또는 api_key.txt (없으면 인증 없음)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from photogrammetry.config import QUALITY_PRESETS
from photogrammetry.inputs import IMAGE_EXTS
from photogrammetry.tools import PROJECT_ROOT

log = logging.getLogger("server")

JOBS_DIR = PROJECT_ROOT / "output" / "server_jobs"
FORMATS = ["glb", "stl", "ply"]
MODEL_NAME = "model"


def _load_api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    key_file = PROJECT_ROOT / "api_key.txt"
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    return key


API_KEY = _load_api_key()
QUALITY = os.environ.get("PG_SERVER_QUALITY", "turbo")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: queue.Queue[str] = queue.Queue()


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _save(job: dict) -> None:
    path = _job_dir(job["job_id"]) / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _update(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs[job_id]
        job.update(fields)
        _save(job)


def _restore_jobs() -> None:
    """재시작 전 작업 복원. 끝나지 않은 작업은 다시 큐에 넣습니다."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    pending = []
    for f in JOBS_DIR.glob("*/job.json"):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("작업 파일을 읽지 못했습니다: %s (%s)", f, e)
            continue
        _jobs[job["job_id"]] = job
        if job["status"] in ("queued", "running"):
            job["status"] = "queued"
            _save(job)
            pending.append(job)
    for job in sorted(pending, key=lambda j: j["created_at"]):
        _queue.put(job["job_id"])
    if pending:
        log.info("미완료 작업 %d개를 다시 큐에 넣었습니다.", len(pending))


def _failure_reason(log_path: Path, returncode: int) -> str:
    """CLI 로그의 오류 원인. CLI는 "ERROR 실패: <입력>" 다음 줄에 예외 메시지를 남깁니다."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for i in range(len(lines) - 1, -1, -1):
        if " ERROR " in lines[i]:
            return lines[i + 1].strip() if i + 1 < len(lines) else lines[i].split("ERROR", 1)[1].strip()
    return f"Pipeline process exited with code {returncode & 0xFFFFFFFF:#x}"


def _run(job_id: str) -> None:
    """파이프라인을 별도 프로세스로 실행 (네이티브 크래시가 서버를 죽이지 않도록)."""
    d = _job_dir(job_id)
    result_dir = d / "result"
    log_path = d / "job.log"
    _update(job_id, status="running", started_at=time.time())
    log.info("작업 시작: %s", job_id)
    cmd = [sys.executable, "-m", "photogrammetry", str(d / "input"), "-o", str(result_dir),
           "--name", MODEL_NAME, "--quality", QUALITY, "--formats", ",".join(FORMATS)]
    # mesh 환경의 MKL intel_thread 레이어는 합성 단계 splu에서 0xc06d007f로 죽음 → 순차 레이어 사용
    env = {"MKL_THREADING_LAYER": "SEQUENTIAL", **os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        with open(log_path, "wb") as out:
            rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT, env=env).returncode
        report_path = result_dir / f"{MODEL_NAME}_report.json"
        if rc != 0 or not report_path.exists():
            raise RuntimeError(_failure_reason(log_path, rc))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = {
            "formats": [f for f in FORMATS if (result_dir / f"{MODEL_NAME}.{f}").exists()],
            "images_used": report.get("images_used"),
            "images_registered": report.get("images_registered"),
            "dense_method": (report.get("dense") or {}).get("method"),
            "elapsed_sec": report.get("elapsed_sec"),
        }
        _update(job_id, status="done", result=result, finished_at=time.time())
        log.info("작업 완료: %s (%.0f초)", job_id, report.get("elapsed_sec", 0))
    except Exception as e:  # noqa: BLE001
        log.error("작업 실패: %s — %s", job_id, e)
        _update(job_id, status="failed", error=str(e) or type(e).__name__, finished_at=time.time())


def _worker() -> None:
    while True:
        job_id = _queue.get()
        try:
            _run(job_id)
        finally:
            _queue.task_done()


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _restore_jobs()
    threading.Thread(target=_worker, name="pipeline-worker", daemon=True).start()
    log.info("품질 프리셋: %s, 인증: %s", QUALITY, "API 키" if API_KEY else "없음")
    yield


app = FastAPI(title="modelingengine server", lifespan=_lifespan)


def _public(job: dict) -> dict:
    out = dict(job)
    if job["status"] == "queued":
        with _lock:
            waiting = sorted((j for j in _jobs.values() if j["status"] == "queued"), key=lambda j: j["created_at"])
        out["queue_position"] = next(i for i, j in enumerate(waiting) if j["job_id"] == job["job_id"]) + 1
    return out


def _get(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/health")
def health() -> dict:
    return {"ok": True, "queued": _queue.qsize()}


@app.post("/jobs", dependencies=[Depends(require_key)])
def create_job(files: list[UploadFile]) -> dict:
    job_id = uuid.uuid4().hex
    input_dir = _job_dir(job_id) / "input"
    input_dir.mkdir(parents=True)
    saved = 0
    for i, f in enumerate(files):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in IMAGE_EXTS:
            ext = ".jpg"
        # 업로드 순서를 파일명에 남겨 정렬이 촬영 순서와 같도록 합니다.
        with open(input_dir / f"{i:04d}{ext}", "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved += 1
    if saved == 0:
        shutil.rmtree(_job_dir(job_id), ignore_errors=True)
        raise HTTPException(status_code=400, detail="No files uploaded")
    job = {"job_id": job_id, "status": "queued", "error": None, "result": None,
           "num_photos": saved, "created_at": time.time()}
    with _lock:
        _jobs[job_id] = job
        _save(job)
    _queue.put(job_id)
    log.info("작업 접수: %s (사진 %d장)", job_id, saved)
    return _public(job)


@app.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str) -> dict:
    return _public(_get(job_id))


@app.get("/jobs/{job_id}/files/{fmt}", dependencies=[Depends(require_key)])
def get_file(job_id: str, fmt: str) -> FileResponse:
    job = _get(job_id)
    if fmt not in FORMATS:
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    path = _job_dir(job_id) / "result" / f"{MODEL_NAME}.{fmt}"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No {fmt} output")
    return FileResponse(path, filename=f"{job_id}.{fmt}", media_type="application/octet-stream")


def main() -> None:
    global QUALITY
    ap = argparse.ArgumentParser(description="modelingengine HTTP 서버")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--quality", choices=QUALITY_PRESETS, default=QUALITY)
    a = ap.parse_args()
    QUALITY = a.quality

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("trimesh").setLevel(logging.WARNING)

    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_config=None)


if __name__ == "__main__":
    main()
