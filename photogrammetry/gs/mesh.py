"""2DGS → 메쉬: 렌더 깊이 TSDF 융합 → marching cubes → 최대 성분 (텍스처는 OpenMVS가 담당)."""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from .scene import Views
from .train import render

log = logging.getLogger(__name__)


@torch.no_grad()
def render_depths(splats, views: Views) -> tuple[torch.Tensor, torch.Tensor]:
    """뷰별 (median 깊이 [C,H,W], alpha [C,H,W])."""
    ds, als = [], []
    for i in range(len(views)):
        _, a, _, _, _, med, _, _ = render(splats, views.viewmats[i:i + 1], views.Ks[i:i + 1],
                                         views.width, views.height)
        ds.append(med[0, ..., 0])
        als.append(a[0, ..., 0])
    return torch.stack(ds), torch.stack(als)


@torch.no_grad()
def tsdf_mesh(depths: torch.Tensor, valid: torch.Tensor, views: Views, center: np.ndarray,
              radius: float, res: int, hull=None, trunc_vox: float = 4.0) -> trimesh.Trimesh:
    """valid 픽셀 깊이만 융합. 미관측 복셀은 hull 안 = 내부로 채워 닫힌 표면."""
    from skimage.measure import marching_cubes

    dev = depths.device
    voxel = 2 * radius / res
    origin = torch.tensor(np.asarray(center) - radius + voxel / 2, device=dev, dtype=torch.float32)
    ax = torch.arange(res, device=dev, dtype=torch.float32) * voxel
    gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
    pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3) + origin
    tsdf = torch.ones(len(pts), device=dev)
    weight = torch.zeros(len(pts), device=dev)
    trunc = trunc_vox * voxel
    for i in range(len(views)):
        xc = pts @ views.viewmats[i, :3, :3].T + views.viewmats[i, :3, 3]
        uvw = xc @ views.Ks[i].T
        z = uvw[:, 2]
        u = (uvw[:, 0] / z).round().long()
        v = (uvw[:, 1] / z).round().long()
        inb = (z > 1e-6) & (u >= 0) & (u < views.width) & (v >= 0) & (v < views.height)
        idx = inb.nonzero()[:, 0]
        d = depths[i, v[idx], u[idx]]
        ok = valid[i, v[idx], u[idx]] & (d > 0)
        sdf = d - z[idx]
        ok &= sdf > -trunc
        idx, sdf = idx[ok], sdf[ok]
        s = (sdf / trunc).clamp(-1, 1)
        w0 = weight[idx]
        tsdf[idx] = (tsdf[idx] * w0 + s) / (w0 + 1)
        weight[idx] = w0 + 1
    unseen = weight == 0
    if hull is not None:
        # 미관측 복셀: 평활 hull 필드 보간 (계단 블록 방지)
        from scipy import ndimage

        field = ndimage.gaussian_filter(hull.occ.astype(np.float32), 1.5)
        vol_h = torch.from_numpy(field).to(dev)[None, None]  # [1,1,X,Y,Z]
        n = hull.occ.shape[0]
        g = (pts[unseen] - torch.tensor(hull.origin, device=dev, dtype=torch.float32)) / hull.voxel
        g = g / (n - 1) * 2 - 1  # [-1, 1], grid_sample은 (z, y, x) 순서 축
        f = F.grid_sample(vol_h, g[:, [2, 1, 0]].view(1, -1, 1, 1, 3), align_corners=True,
                          padding_mode="zeros").view(-1)
        tsdf[unseen] = ((0.5 - f) * 2).clamp(-1, 1)
        vol = tsdf.reshape(res, res, res).cpu().numpy()
        verts, faces, _, _ = marching_cubes(np.pad(vol, 1, constant_values=1.0), 0.0)
        verts -= 1
    else:
        vol = tsdf.reshape(res, res, res).cpu().numpy()
        seen = (~unseen).reshape(res, res, res).cpu().numpy()
        verts, faces, _, _ = marching_cubes(vol, 0.0, mask=seen)
    verts = verts * voxel + origin.cpu().numpy()
    mesh = trimesh.Trimesh(verts, faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def clean(mesh: trimesh.Trimesh, center: np.ndarray, radius: float) -> trimesh.Trimesh:
    """구 밖 제거 → 최대 연결 성분."""
    keep = np.linalg.norm(mesh.triangles_center - center, axis=1) < radius
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    big = labels == np.bincount(labels).argmax()
    mesh.update_faces(big)
    mesh.remove_unreferenced_vertices()
    return mesh
