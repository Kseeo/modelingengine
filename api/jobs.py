"""백그라운드 작업 큐.

GPU가 1개뿐이라 COLMAP/OpenMVS를 동시에 두 개 돌리면 서로 경합해 오히려 느려지므로,
워커를 1개로 고정해 작업을 직렬 처리한다.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from photogrammetry import PipelineConfig, reconstruct

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}


@dataclass
class Job:
    id: str
    input_path: Path
    output_dir: Path
    status: str = "queued"  # queued | running | done | failed
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def create_job(job_id: str, input_path: Path, output_dir: Path, name: str | None, config: PipelineConfig) -> Job:
    job = Job(id=job_id, input_path=input_path, output_dir=output_dir)
    with _lock:
        _jobs[job.id] = job
    _executor.submit(_run, job, name, config)
    return job


def _run(job: Job, name: str | None, config: PipelineConfig) -> None:
    job.status = "running"
    job.started_at = time.time()
    try:
        job.result = reconstruct(job.input_path, job.output_dir, name=name, config=config)
        job.status = "done"
    except Exception as e:  # noqa: BLE001
        log.exception("작업 실패: %s", job.id)
        job.error = str(e)
        job.status = "failed"
    finally:
        job.finished_at = time.time()
        # 업로드된 원본 이미지는 재구성이 끝나면 더 필요 없으므로 삭제.
        # work 폴더는 reconstruct()가 config.keep_workspace=False일 때 이미 정리한다.
        shutil.rmtree(job.input_path, ignore_errors=True)


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[Job]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
