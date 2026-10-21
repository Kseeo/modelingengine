"""2D 영상/이미지 → 3D 메쉬(GLB, STL) 전체 파이프라인."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from . import inputs
from .config import PipelineConfig
from .export import export_models
from .mvs import run_mvs
from .fuse import GsJob, fuse
from .segment import SegmentJob, undistort_masks
from .sfm import run_sfm
from .tools import find_colmap, find_openmvs_dir, fresh_dir

log = logging.getLogger(__name__)


def _is_ascii(p: Path) -> bool:
    return str(p).isascii() and " " not in str(p)


def _workspace(output_dir: Path, name: str) -> Path:
    """작업 폴더 (외부 도구용으로 ASCII·공백 없는 경로)."""
    preferred = output_dir / "work"
    if _is_ascii(preferred.resolve()):
        return preferred
    root = os.environ.get("PG_WORK_ROOT")
    candidates = [Path(root)] if root else []
    candidates += [Path(tempfile.gettempdir()) / "pg_work",
                   Path(os.environ.get("SystemDrive", "C:") + os.sep) / "pg_work"]
    for c in candidates:
        if _is_ascii(c):
            ws = c / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
            log.info("출력 경로에 비ASCII/공백 문자가 있어 작업 폴더를 %s 에 둡니다.", ws)
            return ws
    raise RuntimeError("ASCII 작업 경로를 찾을 수 없습니다. 환경변수 PG_WORK_ROOT를 지정하세요.")


def reconstruct(input_path: str | Path, output_dir: str | Path, name: str | None = None,
                config: PipelineConfig | None = None) -> dict:
    """영상 또는 이미지 폴더 → GLB/STL. 결과 요약 dict 반환 (report.json에도 저장)."""
    cfg = config or PipelineConfig()
    t0 = time.time()
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    kind, src = inputs.classify_input(input_path)
    name = name or re.sub(r"[^\w\-]+", "_", src.stem if kind == "video" else src.name)

    colmap = find_colmap()
    openmvs = find_openmvs_dir()
    log.info("COLMAP: %s", colmap)
    log.info("OpenMVS: %s", openmvs)

    work = _workspace(output_dir, name)
    work.mkdir(parents=True, exist_ok=True)
    images_dir = fresh_dir(work / "images")

    log.info("=== [1/4] 입력 준비 (%s) ===", kind)
    if kind == "video":
        imgs = inputs.extract_frames(src, images_dir, cfg.frame_interval_sec, cfg.pick_sharpest,
                                     cfg.max_image_size, cfg.video_min_sec, cfg.video_max_sec)
        same_size = True
    else:
        imgs, same_size = inputs.prepare_images(src, images_dir, cfg.max_image_size)
    if len(imgs) < cfg.min_images:
        raise inputs.InputError(f"이미지가 {len(imgs)}장입니다. 최소 {cfg.min_images}장이 필요합니다.")
    single_camera = cfg.single_camera if cfg.single_camera is not None else same_size

    timings = {"input": round(time.time() - t0, 1)}
    log.info("=== [2/4] SfM: 카메라 포즈 추정 ===")
    seg = SegmentJob(images_dir, work / "seg_raw", cfg, work / "segment.log") if cfg.seg_masks else None
    try:
        sfm = run_sfm(colmap, images_dir, work, cfg, single_camera, ordered=(kind == "video"))
        seg_raw = seg.wait() if seg else None
    except BaseException:
        if seg:
            seg.cancel()
        raise

    timings["sfm"] = round(time.time() - t0 - sum(timings.values()), 1)
    seg_dirs = undistort_masks(seg_raw, sfm.model_dir, sfm.dense_dir, work / "seg", cfg) if seg_raw else None
    gs = None
    if cfg.dense_method == "fused":
        if seg_dirs is None:
            log.warning("분할 마스크가 없어 2DGS 합성을 건너뜁니다 (MVS만 사용).")
        else:
            gs = GsJob(sfm.dense_dir, seg_dirs[1], work, cfg)
    fusion = {"method": "mvs"}

    def fuse_hook(mvs_mesh: Path) -> Path:
        res = gs.wait()
        if gs.hull_check:
            fusion["hull_check"] = gs.hull_check
        if res is None:
            return mvs_mesh
        try:
            fused, stats = fuse(Path(res["shape"]), mvs_mesh, work / "mvs" / "scene_mesh_fused.ply", cfg,
                                Path(res["hull"]) if res.get("hull") else None, res.get("hull_voxel", 0.0))
        except Exception as e:  # noqa: BLE001
            log.warning("합성 실패 → MVS 메쉬 사용: %s", e)
            return mvs_mesh
        fusion.update(method="fused", gs_sec=res.get("total_sec"), gs_timings=res.get("timings_sec"), **stats)
        return fused

    log.info("=== [3/4] Dense: MVS%s + 메쉬 + 텍스처 ===", " ∥ 2DGS 합성" if gs else "")
    try:
        mesh_path, shape_path = run_mvs(openmvs, sfm.dense_dir, work, cfg, sfm.depth_dir, seg_dirs,
                                        fuse_hook if gs else None)
    finally:
        if gs:
            gs.cancel()

    timings["mvs"] = round(time.time() - t0 - sum(timings.values()), 1)
    log.info("=== [4/4] 내보내기 ===")
    info = export_models(mesh_path, output_dir, name, cfg.formats, cfg.scale, cfg.place_on_ground,
                         shape_path=shape_path, color_source=work / "mvs" / "scene_dense.ply")

    timings["export"] = round(time.time() - t0 - sum(timings.values()), 1)
    report = {
        "name": name,
        "input": str(src),
        "input_type": kind,
        "images_used": len(imgs),
        "images_registered": sfm.num_registered,
        "sparse_points": sfm.num_points,
        "up_aligned": sfm.aligned_up,
        "seg_masks": seg_raw is not None,
        "dense": fusion,
        "mesh": info,
        "units": "임의 단위 (SfM은 절대 스케일을 모름. --scale로 보정)" if cfg.scale == 1.0 else f"scale x{cfg.scale}",
        "workspace": str(work) if cfg.keep_workspace else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "timings_sec": timings,
        "config": cfg.to_dict(),
    }
    (output_dir / f"{name}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cfg.keep_workspace:
        shutil.rmtree(work, ignore_errors=True)
    log.info("완료 (%.1f분): %s", report["elapsed_sec"] / 60, ", ".join(info["files"].values()))
    return report
