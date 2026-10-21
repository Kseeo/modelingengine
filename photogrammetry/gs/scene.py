"""학습용 뷰 로드: 왜곡 보정된 COLMAP 모델(PINHOLE) + 이미지 + 발 마스크."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..sfm import read_camera_models, read_images


@dataclass
class Views:
    names: list[str]
    images: torch.Tensor   # [C, H, W, 3] 0~1
    masks: torch.Tensor    # [C, H, W] 0/1
    Ks: torch.Tensor       # [C, 3, 3]
    viewmats: torch.Tensor # [C, 4, 4] world→cam
    width: int
    height: int

    @property
    def camtoworlds(self) -> torch.Tensor:
        return torch.linalg.inv(self.viewmats)

    def __len__(self) -> int:
        return len(self.names)


def _imread(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def load_views(dense_dir: Path, mask_dir: Path, max_views: int, long_side: int,
               device: str = "cuda") -> Views:
    """등록 이미지 중 max_views장을 균등 선택, 긴 변 long_side로 축소. 마스크 없는 뷰는 제외."""
    sparse = dense_dir / "sparse"
    cams = read_camera_models(sparse / "cameras.bin")
    entries = sorted(read_images(sparse / "images.bin"), key=lambda e: e[0])
    if max_views > 0 and len(entries) > max_views:
        entries = [entries[int(i)] for i in np.linspace(0, len(entries) - 1, max_views)]

    names, imgs, masks, Ks, vms = [], [], [], [], []
    size = None
    for name, r, t, cam_id in entries:
        _, w, h, (fx, fy, cx, cy) = cams[cam_id]
        img = _imread(dense_dir / "images" / name)
        mask = _imread(mask_dir / f"{name}.png", cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        s = long_side / max(w, h) if long_side > 0 else 1.0
        if size is None:
            size = (round(w * s), round(h * s))
        if (round(w * s), round(h * s)) != size:
            continue  # 해상도가 다른 뷰는 배치에서 제외
        sx, sy = size[0] / w, size[1] / h
        imgs.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA)[:, :, ::-1])
        masks.append(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) >= 128)
        Ks.append([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = r, t
        vms.append(vm)
        names.append(name)
    if not names:
        raise RuntimeError("학습에 쓸 뷰가 없습니다 (이미지·마스크 확인).")
    f32 = dict(dtype=torch.float32, device=device)
    return Views(
        names=names,
        images=torch.from_numpy(np.stack(imgs).copy()).to(device).float() / 255,
        masks=torch.from_numpy(np.stack(masks)).to(**f32),
        Ks=torch.tensor(np.array(Ks), **f32),
        viewmats=torch.tensor(np.array(vms), **f32),
        width=size[0], height=size[1],
    )
