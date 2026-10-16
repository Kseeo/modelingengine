"""발 분할 마스크 (BiRefNet): SfM과 병렬 생성 → 왜곡 보정 → 깊이맵 마스크·메쉬 다중 뷰 투표."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .config import PipelineConfig
from .sfm import read_camera_models, read_images
from .tools import fresh_dir

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("segment_worker.py")


def _read_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)


def _write_png(path: Path, img: np.ndarray) -> None:
    cv2.imencode(".png", img)[1].tofile(str(path))


def gpu_python(cfg: PipelineConfig) -> str:
    """torch(CUDA)가 있는 파이썬: 설정 → PG_GPU_PYTHON (구 PG_SEG_PYTHON) → 현재 파이썬."""
    return (cfg.gpu_python or os.environ.get("PG_GPU_PYTHON") or os.environ.get("PG_SEG_PYTHON")
            or sys.executable)


class SegmentJob:
    """분할 워커 프로세스 (SfM과 병렬)."""

    def __init__(self, images_dir: Path, out_dir: Path, cfg: PipelineConfig, log_file: Path):
        self.out_dir = fresh_dir(out_dir)
        self.t0 = time.time()
        python = gpu_python(cfg)
        self.log_fh = open(log_file, "w", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                [python, str(WORKER), str(images_dir), str(out_dir), cfg.seg_model, str(cfg.seg_size)],
                stdout=self.log_fh, stderr=subprocess.STDOUT)
        except OSError as e:
            log.warning("분할 워커 실행 실패 (%s): %s", python, e)
            self.proc = None
        else:
            log.info("발 분할 시작 (BiRefNet, 백그라운드): %s", python)

    def wait(self) -> Path | None:
        """마스크 폴더. 실패 시 None."""
        if self.proc is None:
            return None
        t = time.time()
        code = self.proc.wait()
        self.log_fh.close()
        if code != 0:
            log.warning("발 분할 실패 (종료 코드 %d) → 원형 마스크 사용. 로그: %s", code, self.log_fh.name)
            return None
        log.info("발 분할 완료 (총 %.1f초, SfM 이후 대기 %.1f초)", time.time() - self.t0, time.time() - t)
        return self.out_dir

    def cancel(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.log_fh.close()


def _undistort_map(src_cam, dst_cam):
    """원본(왜곡) → 왜곡 보정(PINHOLE) remap 테이블. 미지원 모델이면 None."""
    model_id, w0, h0, p = src_cam
    _, w, h, q = dst_cam
    if model_id == 0:
        k, d = (p[0], p[0], p[1], p[2]), [0, 0, 0, 0]
    elif model_id == 1:
        k, d = p[:4], [0, 0, 0, 0]
    elif model_id == 2:
        k, d = (p[0], p[0], p[1], p[2]), [p[3], 0, 0, 0]
    elif model_id == 3:
        k, d = (p[0], p[0], p[1], p[2]), [p[3], p[4], 0, 0]
    elif model_id == 4:
        k, d = p[:4], p[4:8]
    else:
        return None
    kk = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
    kn = np.array([[q[0], 0, q[2]], [0, q[1], q[3]], [0, 0, 1]])
    return cv2.initUndistortRectifyMap(kk, np.array(d, float), None, kn, (int(w), int(h)), cv2.CV_32FC1)


def _skin_model(samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float] | None:
    """LAB 표본(N×3)들 → (a·b 평균, a·b 역공분산, 최소 밝기). 상위 80%만 남겨 반복 추정 (받침대·바지 영향 축소)."""
    if not samples:
        return None
    x = np.concatenate(samples)
    if len(x) < 500:
        return None
    ab = x[:, 1:]
    keep = np.ones(len(ab), bool)
    for _ in range(3):
        mean = ab[keep].mean(0)
        icov = np.linalg.inv(np.cov(ab[keep].T) + np.eye(2) * 1e-3)
        d = np.einsum("ni,ij,nj->n", ab - mean, icov, ab - mean)
        keep = d <= np.quantile(d, 0.8)
    return mean, icov, 0.4 * float(np.median(x[keep, 0]))


def _skin_region(lab: np.ndarray, mask: np.ndarray, model) -> np.ndarray | None:
    """마스크 안 피부색 영역(큰 덩어리만, 경계 여유). 피부가 없으면 None."""
    mean, icov, l_min = model
    ab = lab[..., 1:] - mean
    d = np.einsum("...i,ij,...j->...", ab, icov, ab)
    skin = (mask & (d < 3.5 ** 2) & (lab[..., 0] >= l_min)).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.isin(labels, 1 + np.nonzero(areas >= 0.05 * areas.max())[0])
    # 피부로 둘러싸인 안쪽(그늘진 발바닥 등)은 복원: 제거 대상은 가장자리에 붙은 물체뿐
    from scipy import ndimage

    keep = ndimage.binary_fill_holes(keep).astype(np.uint8)
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0


def undistort_masks(raw_dir: Path, model_dir: Path, dense_dir: Path, out_dir: Path,
                    cfg: PipelineConfig) -> tuple[Path, Path] | None:
    """(깊이맵용 넉넉한 마스크, 투표용 마스크) 폴더. 깊이맵용은 OpenMVS 이름 규칙(<stem>.mask.png)."""
    src_cams = read_camera_models(model_dir / "cameras.bin")
    src_img = {n: c for n, _, _, c in read_images(model_dir / "images.bin")}
    dst_cams = read_camera_models(dense_dir / "sparse" / "cameras.bin")
    loose_dir, vote_dir = fresh_dir(out_dir / "loose"), fresh_dir(out_dir / "vote")
    maps, ok, coverage = {}, 0, []
    entries = read_images(dense_dir / "sparse" / "images.bin")
    scale = 0.5  # 피부색 판정 해상도

    def remap(name, cam_id):
        raw = _read_mask(raw_dir / f"{name}.png")
        key = (src_img.get(name), cam_id)
        if raw is not None and key not in maps:
            maps[key] = _undistort_map(src_cams[key[0]], dst_cams[cam_id]) if key[0] in src_cams else None
        if raw is None or maps[key] is None:
            return None
        return cv2.remap(raw, *maps[key], cv2.INTER_LINEAR, borderValue=0)

    def small_lab(name):
        img = cv2.imdecode(np.fromfile(str(dense_dir / "images" / name), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    model = None
    if cfg.seg_skin_filter:
        # 영상 전체 피부색: 마스크 안쪽 깊은 곳 표본 (프레임별 오분할에 덜 흔들림)
        samples = []
        for name, _, _, cam_id in entries[::max(1, len(entries) // 30)]:
            prob = remap(name, cam_id)
            if prob is None:
                continue
            lab = small_lab(name)
            m = cv2.resize(prob, (lab.shape[1], lab.shape[0])) >= 128
            core = cv2.erode(m.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
            if core.sum() > 200:
                samples.append(lab[core][:: max(1, int(core.sum()) // 2000)])
        model = _skin_model(samples)
    trimmed = []
    for name, _, _, cam_id in entries:
        _, w, h, _ = dst_cams[cam_id]
        prob = remap(name, cam_id)
        if prob is None:
            # 마스크 없음: 깊이맵은 제한 없음, 투표에서는 제외
            _write_png(loose_dir / f"{Path(name).stem}.mask.png", np.full((int(h), int(w)), 255, np.uint8))
            continue
        if model is not None:
            # 발에 붙은 비피부 물체(받침대·칸막이·바지) 제거. 60% 넘게 잘리면 판정 실패로 보고 유지
            lab = small_lab(name)
            m = cv2.resize(prob, (lab.shape[1], lab.shape[0])) >= 255 * cfg.seg_threshold
            skin = _skin_region(lab, m, model)
            if skin is not None:
                keep = cv2.resize(skin.astype(np.uint8), (int(w), int(h)), interpolation=cv2.INTER_NEAREST) > 0
                before = (prob >= 128).sum()
                after = ((prob >= 128) & keep).sum()
                if before and after >= 0.4 * before:
                    prob = np.where(keep, prob, 0).astype(prob.dtype)
                    trimmed.append(1 - after / before)
        r = max(1, round(cfg.seg_dilate * max(w, h)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        loose = cv2.dilate((prob >= 255 * cfg.seg_threshold).astype(np.uint8) * 255, kernel)
        _write_png(loose_dir / f"{Path(name).stem}.mask.png", loose)
        _write_png(vote_dir / f"{name}.png", (prob >= 128).astype(np.uint8) * 255)
        coverage.append(loose.mean() / 255)
        ok += 1
    if ok == 0:
        log.warning("분할 마스크를 왜곡 보정하지 못해 원형 마스크를 사용합니다.")
        return None
    log.info("분할 마스크 %d장 (깊이맵용 화면 평균 %.0f%% 영역)", ok, 100 * np.mean(coverage))
    if trimmed:
        log.info("비피부 영역 제거: %d장, 마스크 평균 %.1f%% 감소", len(trimmed), 100 * np.mean(trimmed))
    return loose_dir, vote_dir


def carve_by_masks(src: Path, sparse_dir: Path, vote_dir: Path, cfg: PipelineConfig, dst: Path) -> Path:
    """다중 뷰 투표: 화면 안에 보이는 뷰 중 마스크 안 비율이 낮은 정점의 면 제거."""
    mesh = trimesh.load(src, process=False, force="mesh")
    v = np.asarray(mesh.vertices)
    cams = read_camera_models(sparse_dir / "cameras.bin")
    inside = np.zeros(len(v), np.int32)
    seen = np.zeros(len(v), np.int32)
    for name, r, t, cam_id in read_images(sparse_dir / "images.bin"):
        mask = _read_mask(vote_dir / f"{name}.png")
        if mask is None:
            continue
        _, w, h, (fx, fy, cx, cy) = cams[cam_id]
        # 경계 오차 허용
        mask = cv2.dilate(mask, np.ones((2 * cfg.seg_vote_margin + 1,) * 2, np.uint8))
        xc = v @ r.T + t
        z = xc[:, 2]
        front = z > 1e-9
        u = np.full(len(v), -1, np.int64)
        y = np.full(len(v), -1, np.int64)
        u[front] = np.floor(fx * xc[front, 0] / z[front] + cx).astype(np.int64)
        y[front] = np.floor(fy * xc[front, 1] / z[front] + cy).astype(np.int64)
        vis = front & (u >= 0) & (u < w) & (y >= 0) & (y < h)
        seen += vis
        inside[vis] += mask[y[vis], u[vis]] > 0
    ratio = inside / np.maximum(seen, 1)
    keep_v = (seen < cfg.seg_vote_min_views) | (ratio >= cfg.seg_vote_ratio)
    keep = keep_v[mesh.faces].all(axis=1)
    if keep.sum() < 0.05 * len(keep):
        log.warning("마스크 투표로 남는 면이 너무 적어 건너뜁니다 (%d / %d faces).", keep.sum(), len(keep))
        return src
    log.info("마스크 투표로 배경 제거: %d / %d faces 유지", keep.sum(), len(keep))
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    mesh.export(dst)
    return dst
