"""이미지별 노출·화이트밸런스 보정: SfM 트랙의 다중 뷰 색 관측으로 gain 추정.

같은 3D 점이 이미지마다 다른 색으로 찍힌 차이를 이미지별 gain으로 흡수한다 (채널별, Huber IRLS).
    log c(i, p) = log gain_i + log albedo_p
감마 인코딩(sRGB) 값에서도 노출 배율은 곱으로 남으므로 로그 공간에서 선형.
보정한 dense/images를 텍스처·2DGS·점군 색이 함께 쓴다 (OpenMVS seam leveling 대체).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .sfm import read_image_points

log = logging.getLogger(__name__)

PATCH = 5               # 관측 색 = 주변 PATCH×PATCH 평균 (노이즈 완화)
VALID_RANGE = (8, 245)  # 이 범위 밖 채널(포화·암부)이 있는 관측 제외
MIN_TRACK = 3           # 이보다 적은 이미지에서 보인 점은 제외
MIN_OBS = 30            # 관측이 이보다 적은 이미지는 보정하지 않음
MAX_CORR = 2.0          # 보정 배율 상한 (하한은 역수). 실측 영상의 역광 구간이 ~1.9배
SKIP_BELOW = 0.02       # 보정 배율이 1에서 이만큼도 안 벗어나면 이미지를 다시 쓰지 않음


def _read(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def _sample(path: Path, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """삼각화된 2D 관측 위치의 패치 평균 색 [N, 3] (BGR), 3D 점 id [N]."""
    img = _read(path)
    obs = pts[pts["id"] >= 0]
    h, w = img.shape[:2]
    x = np.clip(obs["x"].astype(np.int64), 0, w - 1)
    y = np.clip(obs["y"].astype(np.int64), 0, h - 1)
    return cv2.blur(img, (PATCH, PATCH))[y, x], obs["id"]


def _apply(path: Path, corr: np.ndarray) -> None:
    img = _read(path)
    out = np.clip(img * corr.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
    cv2.imencode(path.suffix, out, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(path))


def _track_spread(y: np.ndarray, pt_idx: np.ndarray, n_pts: int) -> float:
    """트랙 평균 대비 관측 편차의 중앙값 (로그, 채널 평균)."""
    cnt = np.bincount(pt_idx, minlength=n_pts)[:, None]
    mean = np.stack([np.bincount(pt_idx, y[:, c], n_pts) for c in range(3)], 1) / cnt
    return float(np.median(np.abs(y - mean[pt_idx])))


def _solve(img_idx: np.ndarray, pt_idx: np.ndarray, y: np.ndarray, n_img: int, n_pts: int,
           iters: int = 5) -> np.ndarray:
    """채널 하나의 log gain [n_img] (Huber IRLS, 미정 상수는 호출 측에서 고정)."""
    m = len(y)
    rows = np.r_[np.arange(m), np.arange(m)]
    a = sp.csr_matrix((np.ones(2 * m), (rows, np.r_[img_idx, n_img + pt_idx])), shape=(m, n_img + n_pts))
    w = np.ones(m)
    for _ in range(iters):
        sw = np.sqrt(w)
        x = lsqr(sp.diags(sw) @ a, sw * y, atol=1e-8, btol=1e-8)[0]
        r = np.abs(y - a @ x)
        delta = 1.345 * 1.4826 * np.median(r) + 1e-9
        w = np.minimum(1.0, delta / np.maximum(r, 1e-12))
    return x[:n_img]


def balance_colors(dense_dir: Path, workers: int = 8) -> dict | None:
    """dense_dir/images를 제자리 보정하고 통계 반환. 관측이 부족하면 None (원본 유지)."""
    t0 = time.time()
    pts = read_image_points(dense_dir / "sparse" / "images.bin")
    names = sorted(pts)
    img_dir = dense_dir / "images"
    with ThreadPoolExecutor(workers) as ex:
        samples = list(ex.map(lambda n: _sample(img_dir / n, pts[n]), names))

    img_idx = np.concatenate([np.full(len(ids), i) for i, (_, ids) in enumerate(samples)])
    pid = np.concatenate([ids for _, ids in samples])
    col = np.concatenate([c for c, _ in samples]).astype(np.float64)
    ok = ((col >= VALID_RANGE[0]) & (col <= VALID_RANGE[1])).all(1)
    img_idx, pid, col = img_idx[ok], pid[ok], col[ok]
    _, inv, cnt = np.unique(pid, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= MIN_TRACK
    _, pt_idx = np.unique(inv[keep], return_inverse=True)
    img_idx, y = img_idx[keep], np.log(col[keep])
    n_img, n_pts = len(names), int(pt_idx.max()) + 1 if len(pt_idx) else 0
    has = np.bincount(img_idx, minlength=n_img) >= MIN_OBS
    if has.sum() < 2:
        log.warning("색 보정: 다중 뷰 관측이 부족해 건너뜁니다 (%d점).", n_pts)
        return None

    g = np.stack([_solve(img_idx, pt_idx, y[:, c], n_img, n_pts) for c in range(3)], 1)  # [n_img, 3] BGR
    g -= np.median(g[has], axis=0)  # 중앙 이미지 기준 (전체 밝기 유지)
    g[~has] = 0.0
    corr = np.exp(-g).clip(1 / MAX_CORR, MAX_CORR)
    todo = [i for i in range(n_img) if np.abs(corr[i] - 1).max() >= SKIP_BELOW]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda i: _apply(img_dir / names[i], corr[i]), todo))

    before = _track_spread(y, pt_idx, n_pts)
    after = _track_spread(y + np.log(corr[img_idx]), pt_idx, n_pts)
    stats = {"sec": round(time.time() - t0, 1), "images": n_img, "corrected": len(todo),
             "points": n_pts, "obs": len(y),
             "corr_range": [round(float(corr.min()), 3), round(float(corr.max()), 3)],
             "track_spread": [round(before, 4), round(after, 4)]}
    log.info("색 보정: %d / %d장 수정 (배율 %.2f~%.2f), 트랙 색 편차 %.3f → %.3f (%.1fs)",
             len(todo), n_img, *stats["corr_range"], before, after, stats["sec"])
    return stats
