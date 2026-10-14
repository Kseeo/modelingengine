"""단안 노멀 prior: Depth Anything V2 깊이를 hull 깊이에 스케일·시프트 정렬 후 노멀 유도."""
from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F
from gsplat.utils import depth_to_normal

from .scene import Views

log = logging.getLogger(__name__)


@torch.no_grad()
def _relative_disparity(views: Views, model_name: str, input_size: int = 518, batch: int = 8) -> torch.Tensor:
    """[C, H, W] 상대 역깊이 (스케일·시프트 미정)."""
    from transformers import AutoModelForDepthEstimation

    dev = views.images.device
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(dev).half().eval()
    h, w = views.height, views.width
    s = input_size / max(h, w)
    size = (max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14))
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(views), batch):
        x = views.images[i:i + batch].permute(0, 3, 1, 2)
        x = F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
        d = model(pixel_values=((x - mean) / std).half()).predicted_depth.float()
        out.append(F.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0])
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


def _align(disp: torch.Tensor, ref_depth: torch.Tensor, valid: torch.Tensor, iters: int = 3):
    """disp ≈ a / ref_depth + b (견고 최소제곱) → 정렬 깊이. 실패 시 None."""
    x, y = 1.0 / ref_depth[valid], disp[valid]
    keep = torch.ones_like(x, dtype=torch.bool)
    a = b = None
    for _ in range(iters):
        if keep.sum() < 100:
            return None
        A = torch.stack([x[keep], torch.ones_like(x[keep])], 1)
        sol = torch.linalg.lstsq(A, y[keep][:, None]).solution[:, 0]
        a, b = float(sol[0]), float(sol[1])
        r = (a * x + b - y).abs()
        keep = r <= 2.5 * r[keep].median()
    if a is None or a <= 0:
        return None
    inv = (disp - b) / a
    return torch.where(inv > 1e-6, 1.0 / inv.clamp(min=1e-6), torch.zeros_like(inv))


def estimate(views: Views, ref_depth: torch.Tensor, ref_valid: torch.Tensor,
             model_name: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(월드 노멀 [C,H,W,3], 유효 [C,H,W], 통계). ref: hull 렌더 깊이."""
    t0 = time.time()
    disp = _relative_disparity(views, model_name)
    depth = torch.zeros_like(disp)
    ok = torch.zeros(len(views), dtype=torch.bool, device=disp.device)
    for i in range(len(views)):
        d = _align(disp[i], ref_depth[i], ref_valid[i])
        if d is not None:
            depth[i], ok[i] = d, True
    normals = depth_to_normal(depth[..., None], views.camtoworlds, views.Ks)
    valid = ref_valid & ok[:, None, None] & (depth > 0) & (normals.norm(dim=-1) > 0.5)
    # 깊이 불연속(경계)에서 노멀 제외
    gy, gx = torch.gradient(depth, dim=(1, 2))
    valid &= (gx.abs() + gy.abs()) < 0.05 * depth
    stats = {"sec": round(time.time() - t0, 1), "views_aligned": int(ok.sum()), "views": len(views),
             "valid_ratio": round(float(valid.float().sum() / ref_valid.float().sum().clamp(min=1)), 3)}
    log.info("노멀 prior (%s): 정렬 %d/%d 뷰, 유효 %.0f%%, %.1fs", model_name, stats["views_aligned"],
             len(views), 100 * stats["valid_ratio"], stats["sec"])
    return normals, valid, stats
