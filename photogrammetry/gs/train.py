"""2DGS 학습 (gsplat): RGB + 실루엣 + 노멀 prior + depth-normal 일관성."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy

from .scene import Views

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    iters: int = 6000
    w_ssim: float = 0.2
    w_mask: float = 0.5
    w_normal_prior: float = 0.1
    w_depth_normal: float = 0.05
    normal_prior_from: int = 0        # 이 스텝부터 노멀 prior
    depth_normal_from: int = 1000     # 이 스텝부터 depth-normal 일관성
    refine_start: int = 300
    refine_stop: int = 3500
    refine_every: int = 100
    lr_means: float = 1.6e-4          # scene_scale 배
    seed: int = 0


def _quats_from_normals(n: np.ndarray) -> np.ndarray:
    """서펠 z축이 n을 향하는 회전 (wxyz)."""
    z = np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    axis = np.cross(z, n)
    s = np.linalg.norm(axis, axis=1)
    c = n[:, 2]
    ang = np.arctan2(s, c)
    axis = np.where(s[:, None] > 1e-8, axis / np.maximum(s, 1e-8)[:, None], [1.0, 0.0, 0.0])
    return np.column_stack([np.cos(ang / 2), axis * np.sin(ang / 2)[:, None]]).astype(np.float32)


def init_splats(points: np.ndarray, normals: np.ndarray, spacing: float, views: Views,
                device: str = "cuda") -> torch.nn.ParameterDict:
    """hull 표면 서펠. 색은 첫 보이는 뷰들의 평균."""
    pts = torch.from_numpy(points).to(device)
    rgb = torch.zeros_like(pts)
    cnt = torch.zeros(len(pts), 1, device=device)
    for i in range(len(views)):
        xc = pts @ views.viewmats[i, :3, :3].T + views.viewmats[i, :3, 3]
        uvw = xc @ views.Ks[i].T
        u = (uvw[:, 0] / uvw[:, 2]).long()
        v = (uvw[:, 1] / uvw[:, 2]).long()
        ok = (uvw[:, 2] > 0) & (u >= 0) & (u < views.width) & (v >= 0) & (v < views.height)
        rgb[ok] += views.images[i, v[ok], u[ok]]
        cnt[ok] += 1
    rgb = (rgb / cnt.clamp(min=1)).clamp(0.02, 0.98)
    scales = torch.full((len(pts), 3), math.log(spacing), device=device)
    scales[:, 2] = math.log(spacing * 1e-3)  # 2DGS는 앞 두 축만 사용
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "quats": torch.nn.Parameter(torch.from_numpy(_quats_from_normals(normals)).to(device)),
        "scales": torch.nn.Parameter(scales),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((len(pts),), 0.7, device=device))),
        "colors": torch.nn.Parameter(torch.logit(rgb)),
    })


def render(splats, viewmats: torch.Tensor, Ks: torch.Tensor, width: int, height: int, **kw):
    """(rgb, alpha, normals, surf_normals, distort, median_depth, expected_depth, meta)."""
    out = rasterization_2dgs(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]), colors=torch.sigmoid(splats["colors"]),
        viewmats=viewmats, Ks=Ks, width=width, height=height, render_mode="RGB+ED", **kw)
    colors, alphas, normals, surf_normals, distort, median, meta = out
    return colors[..., :3], alphas, normals, surf_normals, distort, median, colors[..., 3:], meta


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[1, 3, H, W] SSIM 평균 (가우시안 11)."""
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    f = lambda t: F.conv2d(t, w, padding=5, groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()


def train(splats, views: Views, train_masks: torch.Tensor, scene_scale: float, cfg: TrainConfig,
          normal_prior: torch.Tensor | None = None, prior_valid: torch.Tensor | None = None) -> dict:
    """train_masks: [C, H, W] 실루엣·RGB 대상 영역. 반환: 학습 통계."""
    torch.manual_seed(cfg.seed)
    lrs = {"means": cfg.lr_means * scene_scale, "quats": 1e-3, "scales": 5e-3,
           "opacities": 5e-2, "colors": 2.5e-3 * 4}
    optims = {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
              for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(optims["means"], gamma=0.01 ** (1.0 / cfg.iters))
    strategy = DefaultStrategy(refine_start_iter=cfg.refine_start, refine_stop_iter=cfg.refine_stop,
                               refine_every=cfg.refine_every, reset_every=10 ** 9,
                               key_for_gradient="gradient_2dgs", verbose=False)
    strategy.check_sanity(splats, optims)
    state = strategy.initialize_state(scene_scale=scene_scale)

    t0 = time.time()
    order = torch.randperm(len(views))
    hist = []
    for step in range(cfg.iters):
        i = int(order[step % len(views)])
        if step % len(views) == len(views) - 1:
            order = torch.randperm(len(views))
        rgb, alpha, nrm, surf, _, _, _, meta = render(
            splats, views.viewmats[i:i + 1], views.Ks[i:i + 1], views.width, views.height)
        strategy.step_pre_backward(splats, optims, state, step, meta)

        m = train_masks[i][None, ..., None]
        gt = views.images[i:i + 1] * m
        loss_rgb = (rgb - gt).abs().mean()
        loss = (1 - cfg.w_ssim) * loss_rgb
        if cfg.w_ssim > 0:
            loss = loss + cfg.w_ssim * (1 - _ssim(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        loss = loss + cfg.w_mask * (alpha[..., 0] - m[..., 0]).abs().mean()

        fg = (m[..., 0] > 0.5) & (alpha[..., 0].detach() > 0.5)
        if normal_prior is not None and cfg.w_normal_prior > 0 and step >= cfg.normal_prior_from:
            valid = fg & prior_valid[i:i + 1]
            if valid.any():
                cos = (F.normalize(nrm, dim=-1) * normal_prior[i:i + 1]).sum(-1)
                loss = loss + cfg.w_normal_prior * (1 - cos)[valid].mean()
        if cfg.w_depth_normal > 0 and step >= cfg.depth_normal_from and fg.any():
            cos = (F.normalize(nrm, dim=-1) * F.normalize(surf, dim=-1)).sum(-1)
            loss = loss + cfg.w_depth_normal * ((1 - cos) * alpha[..., 0].detach())[fg].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"학습 발산 (step {step})")
        loss.backward()
        strategy.step_post_backward(splats, optims, state, step, meta)
        for o in optims.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        if step % 500 == 0 or step == cfg.iters - 1:
            hist.append((step, float(loss), float(loss_rgb), len(splats["means"])))
            log.info("2DGS step %5d  loss %.4f  rgb %.4f  splats %d  (%.0fs)",
                     step, float(loss), float(loss_rgb), len(splats["means"]), time.time() - t0)
    return {"iters": cfg.iters, "sec": round(time.time() - t0, 1), "splats": len(splats["means"]),
            "history": hist}
