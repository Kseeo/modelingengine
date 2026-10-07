"""GLB(텍스처, Y-up) / STL·PLY(형상, Z-up) 내보내기."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from .plyio import points_and_colors, read_vertex_ply

log = logging.getLogger(__name__)

# COLMAP(위 = -Y) → Z-up
_COLMAP_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
# Z-up → Y-up (glTF)
_ZUP_TO_YUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def _to_single_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    if hasattr(scene, "to_mesh"):
        return scene.to_mesh()
    return trimesh.util.concatenate(scene.dump())


def vertex_colors_from_points(vertices: np.ndarray, points: np.ndarray, colors: np.ndarray,
                              k: int = 4) -> np.ndarray:
    """가까운 k개 점 색의 역거리 가중 평균 (RGBA uint8)."""
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(points).query(vertices, k=k, workers=-1)
    w = 1.0 / np.maximum(dist, 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    rgb = (colors[idx, :3].astype(np.float64) * w[..., None]).sum(axis=1)
    return np.column_stack([np.clip(rgb.round(), 0, 255), np.full(len(rgb), 255)]).astype(np.uint8)


def textured_surface_samples(scene: trimesh.Scene, n_samples: int) -> tuple[np.ndarray, np.ndarray] | None:
    """텍스처 메쉬 표면 샘플 (점, RGB). 텍스처가 없으면 None."""
    geoms = scene.dump()
    geoms = geoms if isinstance(geoms, list) else [geoms]
    textured = []
    for g in geoms:
        mat = getattr(g.visual, "material", None)
        img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
        if getattr(g.visual, "uv", None) is not None and img is not None:
            textured.append((g, np.asarray(img.convert("RGB"))))
    if not textured:
        return None
    total = sum(g.area for g, _ in textured)
    pts_all, rgb_all = [], []
    for g, img in textured:
        pts, fidx = trimesh.sample.sample_surface(g, max(1000, int(n_samples * g.area / total)))
        bary = trimesh.triangles.points_to_barycentric(g.triangles[fidx], pts)
        uv = (g.visual.uv[g.faces[fidx]] * bary[..., None]).sum(axis=1)
        h, w = img.shape[:2]
        x = np.clip(np.round((uv[:, 0] % 1.0) * (w - 1)), 0, w - 1).astype(int)
        y = np.clip(np.round((1.0 - uv[:, 1] % 1.0) * (h - 1)), 0, h - 1).astype(int)
        pts_all.append(pts)
        rgb_all.append(img[y, x])
    return np.vstack(pts_all), np.vstack(rgb_all)


def _ply_vertex_colors(mesh: trimesh.Trimesh, visual: trimesh.Scene, color_source: Path | None,
                       xf: np.ndarray) -> np.ndarray | None:
    """텍스처 샘플 → 조밀 점군 순으로 정점 색 결정. 둘 다 없으면 None."""
    vertices = np.asarray(mesh.vertices)
    samples = textured_surface_samples(visual, int(np.clip(8 * len(vertices), 1_000_000, 4_000_000)))
    if samples is not None:
        # 원본 해상도 텍스처가 점군 색보다 선명
        log.info("PLY 정점 색: 텍스처 표면 %d점 샘플에서 매핑", len(samples[0]))
        return vertex_colors_from_points(vertices, samples[0], samples[1], k=3)
    if color_source is not None and Path(color_source).is_file():
        xyz, rgb = points_and_colors(read_vertex_ply(color_source))
        pts = trimesh.transform_points(xyz, xf)
        log.info("PLY 정점 색: 조밀 점군 %d점에서 매핑", len(pts))
        return vertex_colors_from_points(vertices, pts, rgb)
    return None


def export_models(mesh_path: Path, out_dir: Path, name: str, formats: list[str],
                  scale: float = 1.0, place_on_ground: bool = True,
                  shape_path: Path | None = None, color_source: Path | None = None) -> dict:
    """mesh_path → GLB, shape_path(없으면 mesh_path) → STL/PLY/OBJ. 두 메쉬에 같은 변환 적용."""
    visual = trimesh.load(mesh_path, force="scene")
    if not visual.geometry:
        raise ValueError(f"빈 메쉬입니다: {mesh_path}")
    shape = trimesh.load(shape_path, force="scene") if shape_path and shape_path != mesh_path else visual

    xf = _COLMAP_TO_ZUP.copy()
    xf[:3, :3] *= scale
    shape.apply_transform(xf)
    if place_on_ground:
        lo, hi = shape.bounds
        t = np.eye(4)
        t[:3, 3] = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        shape.apply_transform(t)
        xf = t @ xf
    if shape is not visual:
        visual.apply_transform(xf)

    lo, hi = shape.bounds
    merged = _to_single_mesh(shape)
    info = {
        "vertices": int(len(merged.vertices)),
        "faces": int(len(merged.faces)),
        "extents_zup": [float(v) for v in (hi - lo)],
        "watertight": bool(merged.is_watertight),
        "files": {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = out_dir / f"{name}.{fmt}"
        if fmt in ("glb", "gltf"):
            yup = visual.copy()
            yup.apply_transform(_ZUP_TO_YUP)
            yup.export(path)
        elif fmt == "ply":
            colored = merged.copy()
            colors = _ply_vertex_colors(colored, visual, color_source, xf)
            if colors is not None:
                colored.visual = trimesh.visual.ColorVisuals(colored, vertex_colors=colors)
            else:
                log.warning("색 점군이 없어 무색 PLY로 저장합니다.")
            colored.export(path)
        elif fmt in ("stl", "obj"):
            merged.export(path)
        else:
            raise ValueError(f"지원하지 않는 출력 형식: {fmt}")
        info["files"][fmt] = str(path)
        log.info("저장: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return info
