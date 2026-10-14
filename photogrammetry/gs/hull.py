"""마스크 visual hull (투표식 space carving) → 표면 서펠 초기점."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from .scene import Views

log = logging.getLogger(__name__)


@dataclass
class Hull:
    occ: np.ndarray        # [R, R, R] bool (x, y, z 순)
    origin: np.ndarray     # 복셀 (0,0,0) 중심의 월드 좌표
    voxel: float
    mesh: trimesh.Trimesh  # 표면 (월드 좌표)


def mask_votes(views: Views, pts: torch.Tensor, margin_px: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """점별 (마스크 안 뷰 수, 화면 안 뷰 수)."""
    k = 2 * margin_px + 1
    masks = F.max_pool2d(views.masks[:, None], k, 1, margin_px)[:, 0] > 0.5  # 경계 오차 허용
    inside = torch.zeros(len(pts), device=pts.device, dtype=torch.int16)
    seen = torch.zeros_like(inside)
    for i in range(len(views)):
        xc = pts @ views.viewmats[i, :3, :3].T + views.viewmats[i, :3, 3]
        uvw = xc @ views.Ks[i].T
        z = uvw[:, 2]
        u = (uvw[:, 0] / z).floor().long()
        v = (uvw[:, 1] / z).floor().long()
        vis = (z > 1e-6) & (u >= 0) & (u < views.width) & (v >= 0) & (v < views.height)
        seen += vis
        hit = torch.zeros_like(vis)
        hit[vis] = masks[i, v[vis], u[vis]]
        inside += hit
    return inside, seen


def carve(views: Views, center: np.ndarray, radius: float, res: int, ratio: float,
          min_views: int, margin_px: int = 3) -> Hull:
    """화면 안에 보이는 뷰 중 마스크 안 비율 ≥ ratio인 복셀만 유지 (오분할 뷰에 강건)."""
    from scipy import ndimage
    from skimage.measure import marching_cubes

    dev = views.images.device
    voxel = 2 * radius / res
    origin = np.asarray(center, float) - radius + voxel / 2
    ax = torch.arange(res, device=dev, dtype=torch.float32) * voxel
    gx, gy, gz = torch.meshgrid(ax + origin[0], ax + origin[1], ax + origin[2], indexing="ij")
    pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3)

    inside, seen = mask_votes(views, pts, margin_px)
    occ = (seen >= min_views) & (inside.float() >= ratio * seen.float())
    occ = occ.reshape(res, res, res).cpu().numpy()

    labels, n = ndimage.label(occ)
    if n == 0:
        raise RuntimeError("visual hull이 비었습니다 (마스크·포즈 확인).")
    occ = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    occ = ndimage.binary_fill_holes(occ)
    # 계단 완화 후 등치면
    field = ndimage.gaussian_filter(occ.astype(np.float32), 1.0)
    verts, faces, _, _ = marching_cubes(np.pad(field, 1), 0.5)
    verts = (verts - 1) * voxel + origin
    mesh = trimesh.Trimesh(verts, faces, process=False)
    if mesh.volume < 0:  # 바깥 방향 노멀로
        mesh.invert()
    log.info("visual hull: %d³ 중 %d 복셀 (%.1f%%), 표면 %d faces",
             res, occ.sum(), 100 * occ.mean(), len(mesh.faces))
    return Hull(occ, origin, voxel, mesh)


def surface_samples(hull: Hull, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """hull 표면 균등 샘플 (점, 바깥 노멀)."""
    pts, fidx = trimesh.sample.sample_surface_even(hull.mesh, n, seed=seed)
    return pts.astype(np.float32), hull.mesh.face_normals[fidx].astype(np.float32)
