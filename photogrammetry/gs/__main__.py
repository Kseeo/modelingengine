"""2DGS 워커: python -m photogrammetry.gs <job.json>

job: dense_dir, mask_dir, out_dir, center, radius, iters, max_views, image_size
결과: out_dir/result.json (shape, hull, hull_voxel, timings_sec, vram_mb, ...)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import hull as hull_mod
from . import mesh as mesh_mod
from . import priors
from .scene import Views, load_views
from .train import TrainConfig, init_splats, train

log = logging.getLogger("photogrammetry.gs")

HULL_RES = 160
HULL_RATIO = 0.8          # 화면 안 뷰 중 마스크 안 비율 (오분할 뷰에 강건)
HULL_MIN_VIEWS = 5
HULL_OFF_VIEW = 0.3       # hull 투영 중 마스크 밖 비율이 이보다 크면 어긋난 뷰
HULL_OFF_MAX = 0.15       # 어긋난 뷰 비율 상한 (넘으면 hull 불신 → MVS만 사용)
HULL_BAD_RECALL = 0.3     # 마스크를 hull이 이보다 적게 덮으면 포즈 오류 뷰 → 제외
INIT_POINTS = 100_000
TSDF_RES = 256
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _subset(views: Views, keep: np.ndarray) -> Views:
    k = torch.from_numpy(np.nonzero(keep)[0]).to(views.images.device)
    return Views([views.names[i] for i in k.tolist()], views.images[k], views.masks[k], views.Ks[k],
                 views.viewmats[k], views.width, views.height)


class StageLog:
    """단계별 시간·최대 VRAM."""

    def __init__(self):
        self.timings, self.vram = {}, {}

    def __call__(self, name: str):
        stage = self

        class _Ctx:
            def __enter__(self):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self.t = time.time()
                log.info("[2DGS] %s", name)

            def __exit__(self, *exc):
                torch.cuda.synchronize()
                stage.timings[name] = round(time.time() - self.t, 1)
                stage.vram[name] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        return _Ctx()


def run(job: dict) -> dict:
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    dense_dir, mask_dir = Path(job["dense_dir"]), Path(job["mask_dir"])
    center, radius = np.asarray(job["center"], float), float(job["radius"])
    st = StageLog()
    t0 = time.time()

    with st("load"):
        views = load_views(dense_dir, mask_dir, job["max_views"], job["image_size"])
    log.info("학습 뷰 %d장 (%dx%d)", len(views), views.width, views.height)

    with st("hull"):
        hull = hull_mod.carve(views, center, radius, HULL_RES, HULL_RATIO, HULL_MIN_VIEWS)
        bad = hull_mod.mask_recall(views, hull) < HULL_BAD_RECALL
        if bad.any() and (~bad).sum() >= 10:
            log.info("포즈 오류 뷰 %d장 제외 후 hull 재계산: %s", bad.sum(), [n for n, b in zip(views.names, bad) if b])
            views = _subset(views, ~bad)
            hull = hull_mod.carve(views, center, radius, HULL_RES, HULL_RATIO, HULL_MIN_VIEWS)
        pts, nrm = hull_mod.surface_samples(hull, INIT_POINTS)
        splats = init_splats(pts, nrm, hull.voxel * 0.7, views)
        hull_depth, hull_alpha = mesh_mod.render_depths(splats, views)
        hull_cover = hull_alpha > 0.5
        # 오분할(사람·의자) 영역은 hull 투영 밖 → 학습 마스크에서 제외
        near_hull = F.max_pool2d(hull_cover.float()[:, None], 11, 1, 5)[:, 0] > 0.5
        train_masks = views.masks * near_hull
        near_mask = F.max_pool2d(views.masks[:, None], 11, 1, 5)[:, 0] > 0.5
        off = ((hull_cover & ~near_mask).sum((1, 2)) / hull_cover.sum((1, 2)).clamp(min=1)).cpu().numpy()
    hull.mesh.export(out / "hull.ply")  # 합성 단계에서 hull 덩어리 판별에 사용
    hull_check = {"off_views": round(float((off > HULL_OFF_VIEW).mean()), 3),
                  "off_median": round(float(np.median(off)), 3), "off_max": round(float(off.max()), 3)}
    log.info("hull-마스크 불일치: 어긋난 뷰 %.0f%% (중앙값 %.2f, 최대 %.2f)",
             100 * hull_check["off_views"], hull_check["off_median"], hull_check["off_max"])
    if hull_check["off_views"] > HULL_OFF_MAX:
        res = {"rejected": "hull", "hull_check": hull_check, "timings_sec": st.timings,
               "total_sec": round(time.time() - t0, 1)}
        (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        log.warning("hull이 마스크와 크게 어긋나 2DGS를 중단합니다.")
        return res

    with st("priors"):
        normal_prior, prior_valid, prior_stats = priors.estimate(
            views, hull_depth, hull_cover & (views.masks > 0.5), DEPTH_MODEL)

    with st("train"):
        stats = train(splats, views, train_masks, radius, TrainConfig(iters=job["iters"]),
                      normal_prior, prior_valid)

    with st("mesh"):
        # 렌더는 학습 불필요 → 등록 뷰 전부로 융합 (학습 뷰에서 빠진 영역 보완)
        del views
        torch.cuda.empty_cache()
        views = load_views(dense_dir, mask_dir, 0, job["image_size"])
        good = hull_mod.mask_recall(views, hull) >= HULL_BAD_RECALL
        if (~good).any() and good.sum() >= 10:
            views = _subset(views, good)
        depths, alphas = mesh_mod.render_depths(splats, views)
        valid = (alphas > 0.5) & (F.max_pool2d(views.masks[:, None], 5, 1, 2)[:, 0] > 0.5)
        shape = mesh_mod.tsdf_mesh(depths, valid, views, center, radius, TSDF_RES, hull)
        shape = mesh_mod.clean(shape, center, radius)
        shape.export(out / "mesh.ply")
    log.info("메쉬: %d verts, %d faces (TSDF 뷰 %d장)", len(shape.vertices), len(shape.faces), len(views))

    res = {"shape": str(out / "mesh.ply"), "hull": str(out / "hull.ply"), "hull_voxel": hull.voxel,
           "views": len(views), "faces": len(shape.faces), "timings_sec": st.timings, "vram_mb": st.vram,
           "total_sec": round(time.time() - t0, 1), "train": stats, "priors": prior_stats,
           "hull_check": hull_check, "bad_pose_views": int(bad.sum())}
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("2DGS 완료 %.1fs: %s", res["total_sec"], st.timings)
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        run(job)
    except FloatingPointError as e:
        log.error("학습 발산: %s", e)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
