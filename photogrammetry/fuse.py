"""2DGS 마감 + MVS 큰 곡률 합성.

2DGS 워커를 MVS와 병렬 실행 → MVS 메쉬 기준 노멀 오프셋을 저주파 보간해 2DGS 정점 이동
→ MVS 근거 없는 영역(hull 덩어리)은 주변 곡면으로 biharmonic 재보간.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.linalg import cg, splu
from scipy.spatial import cKDTree

from .config import PipelineConfig
from .segment import gpu_python
from .sfm import camera_focus
from .tools import PROJECT_ROOT, fresh_dir

log = logging.getLogger(__name__)

FUSE_SCREEN = 0.05  # 근거 없는 정점 오프셋 감쇠 가중치
MVS_NEAR = 0.03     # 2DGS에서 이 거리(대각 비율) 안의 MVS 점만 근거로 사용
MVS_MIN_AGREE = 0.7 # MVS 점 중 근거 비율이 이 미만이면 MVS 불신 → 2DGS 단독



class GsJob:
    """2DGS 워커 프로세스 (MVS와 병렬)."""

    def __init__(self, dense_dir: Path, mask_dir: Path, work: Path, cfg: PipelineConfig):
        self.out = fresh_dir(work / "gs")
        self.cfg = cfg
        self.t0 = time.time()
        self.proc = None
        self.hull_check = None
        res = camera_focus(dense_dir / "sparse" / "images.bin")
        if res is None:
            log.warning("카메라들이 한 대상을 향하지 않아 2DGS를 건너뜁니다.")
            return
        focus, dist = res
        job = {"dense_dir": str(dense_dir), "mask_dir": str(mask_dir), "out_dir": str(self.out),
               "center": focus.tolist(), "radius": cfg.crop_radius_factor * dist,
               "iters": cfg.gs_iters, "max_views": cfg.gs_max_views, "image_size": cfg.gs_image_size}
        job_file = self.out / "job.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")
        self.log_fh = open(self.out / "gs.log", "w", encoding="utf-8")
        python = gpu_python(cfg)
        # torch의 CUDA 확장 JIT 컴파일(ninja)이 python과 같은 env의 bin/에 있으므로 PATH에 추가.
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT),
               "PATH": f"{Path(python).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        try:
            self.proc = subprocess.Popen([python, "-m", "photogrammetry.gs", str(job_file)],
                                         cwd=PROJECT_ROOT, env=env, stdout=self.log_fh, stderr=subprocess.STDOUT)
            log.info("2DGS 시작 (백그라운드, MVS와 병렬)")
        except OSError as e:
            log.warning("2DGS 워커 실행 실패: %s", e)
            self.log_fh.close()

    def wait(self) -> dict | None:
        """result.json 내용. 실패·시간 초과 시 None."""
        if self.proc is None:
            return None
        t = time.time()
        remain = max(1.0, self.cfg.gs_timeout_sec - (t - self.t0))
        try:
            code = self.proc.wait(timeout=remain)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            log.warning("2DGS 시간 초과 (%.0f초) → MVS 메쉬 사용", self.cfg.gs_timeout_sec)
            return None
        finally:
            self.log_fh.close()
        res_file = self.out / "result.json"
        if code != 0 or not res_file.is_file():
            log.warning("2DGS 실패 (종료 코드 %d) → MVS 메쉬 사용. 로그: %s", code, self.out / "gs.log")
            return None
        res = json.loads(res_file.read_text(encoding="utf-8"))
        self.hull_check = res.get("hull_check")
        if res.get("rejected"):
            log.warning("hull이 마스크와 어긋남 (%s) → MVS 메쉬 사용", res.get("hull_check"))
            return None
        log.info("2DGS 완료 (총 %.1f초, MVS 이후 대기 %.1f초)", time.time() - self.t0, time.time() - t)
        return res

    def cancel(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()


def _adjacency(mesh: trimesh.Trimesh) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """(인접 행렬, 균등 라플라시안)."""
    e = mesh.edges_unique
    n = len(mesh.vertices)
    a = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    return a, sp.diags(np.asarray(a.sum(1)).ravel()) - a


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    keep = labels == np.bincount(labels).argmax()
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def _drop_spikes(mesh: trimesh.Trimesh, k: float = 6.0) -> trimesh.Trimesh:
    """중앙값 k배 넘는 긴 엣지를 가진 면 제거 → 최대 성분 (재보간으로 늘어난 가시 정리)."""
    el = mesh.edges_unique_length
    bad = (el > k * np.median(el))[mesh.faces_unique_edges].any(1)
    if bad.any():
        mesh.update_faces(~bad)
        mesh.remove_unreferenced_vertices()
    return _largest_component(mesh)


def fuse(gs_mesh: Path, mvs_mesh: Path, dst: Path, cfg: PipelineConfig,
         hull: Path | None = None, hull_voxel: float = 0.0) -> tuple[Path, dict]:
    """2DGS 메쉬를 MVS 큰 곡률로 보정 + MVS 근거 없이 hull 표면에 붙은 영역(덩어리) 재보간."""
    t0 = time.time()
    g = _largest_component(trimesh.load(gs_mesh, process=False, force="mesh"))  # 떨어진 조각 → 가시 방지
    m = trimesh.load(mvs_mesh, process=False, force="mesh")
    diag = float(np.linalg.norm(g.extents))
    q, fid = trimesh.sample.sample_surface(m, 600_000)
    qn = m.face_normals[fid]
    gs_pts, _ = trimesh.sample.sample_surface_even(g, 100_000, seed=0)
    near = cKDTree(gs_pts).query(q, workers=-1)[0] < MVS_NEAR * diag
    mvs_agree = float(near.mean())
    if mvs_agree < MVS_MIN_AGREE:
        log.warning("MVS가 2DGS와 크게 어긋남 (일치 %.0f%%) → 2DGS 단독 사용", 100 * mvs_agree)
        g = _drop_spikes(g)
        g.export(dst)
        return dst, {"sec": round(time.time() - t0, 1), "mvs_agree": round(mvs_agree, 3), "skipped": "mvs_disagree",
                     "faces": int(len(g.faces))}
    q, qn = q[near], qn[near]  # 멀리 떨어진 MVS 판 조각 제외
    tree = cKDTree(q)
    a, lap = _adjacency(g)
    n_v = len(g.vertices)

    # 1) 저주파 오프셋: 평활 노멀 방향으로 MVS까지 부호 거리 → (W + λL) f = W·off
    smooth = g.copy()
    trimesh.smoothing.filter_taubin(smooth, iterations=30)
    nrm = np.asarray(smooth.vertex_normals, float)
    v = np.asarray(g.vertices, float)
    d, i = tree.query(v, workers=-1)
    off = ((q[i] - v) * nrm).sum(1)
    agree = np.abs((qn[i] * nrm).sum(1)) > 0.5
    valid = (d < 0.03 * diag) & agree
    # 근거 없는 정점은 이동량 0으로 감쇠 (MVS 없는 막이 벌어지는 것 방지)
    f, _ = cg(sp.diags(valid + FUSE_SCREEN * ~valid) + cfg.fuse_lambda * lap + 1e-6 * sp.eye(n_v),
              valid * off, maxiter=3000, rtol=1e-6)
    out = trimesh.Trimesh(v + f[:, None] * nrm, g.faces, process=False)
    trimesh.smoothing.filter_taubin(out, iterations=10)

    # 2) MVS 근거 없고 hull 표면에 붙은 정점(학습이 못 움직인 덩어리) → biharmonic 재보간
    #    hull 안쪽으로 들어간 2DGS 형상(발가락, 오목한 아치)은 유지
    x = np.asarray(out.vertices, float)
    d, _ = tree.query(x, workers=-1)
    sup = d < cfg.fuse_support_tol * diag
    on_hull = np.ones(len(x), bool)
    if hull is not None and Path(hull).is_file() and hull_voxel > 0:
        hp, _ = trimesh.sample.sample_surface(trimesh.load(hull, process=False, force="mesh"), 400_000)
        on_hull = cKDTree(hp).query(v, workers=-1)[0] < 2 * hull_voxel  # 보정 전 2DGS 위치 기준
    free = ~sup & on_hull
    for _ in range(2):  # 경계 2링 확장 → 이음새 완화
        free = free | ((a @ free.astype(float)) > 0)
    if free.any() and (~free).any():
        bi = (lap @ lap).tocsr()
        fi, bo = np.nonzero(free)[0], np.nonzero(~free)[0]
        x[fi] = splu(bi[fi][:, fi].tocsc()).solve(-(bi[fi][:, bo] @ x[bo]))
        out = trimesh.Trimesh(x, g.faces, process=False)
        trimesh.smoothing.filter_taubin(out, iterations=5)
    out = _drop_spikes(out)
    out.export(dst)
    stats = {"sec": round(time.time() - t0, 1), "mvs_agree": round(mvs_agree, 3), "mvs_supported": round(float(sup.mean()), 3),
             "on_hull": round(float(on_hull.mean()), 3),
             "refilled": round(float(free.mean()), 3), "faces": int(len(out.faces))}
    log.info("합성: MVS 근거 %.0f%%, 재보간 %.0f%% (%d faces, %.1fs)", 100 * stats["mvs_supported"],
             100 * stats["refilled"], stats["faces"], stats["sec"])
    return dst, stats
