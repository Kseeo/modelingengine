"""MVS (OpenMVS): 조밀 점군 → 메쉬 → 정밀화 → 텍스처."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .config import PipelineConfig
from .plyio import points_and_colors, read_vertex_ply, write_vertex_subset
from .segment import carve_by_masks
from .sfm import camera_focus, make_focus_masks
from .tools import EXE, ToolError, run

log = logging.getLogger(__name__)

# OpenMVS TextureMesh --empty-color 기본값 (0xFF7F27, 어느 뷰에도 안 보인 면을 채우는 색).
# 뷰가 없는 면(주로 2DGS 합성으로 새로 생긴 구멍 메움 부분)이 그대로 주황색으로 튀어보이므로
# 텍스처에서 이 색을 찾아 주변 색으로 인페인팅한다.
_EMPTY_COLOR_BGR = (39, 127, 255)
_EMPTY_COLOR_INT = 0xFF7F27


def _output_mesh(mvs_file: Path, ext: str) -> Path:
    p = mvs_file.with_suffix(ext)
    if not p.is_file():
        raise ToolError(f"예상한 메쉬 파일이 생성되지 않았습니다: {p}")
    return p


def _ply_face_count(path: Path) -> int:
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"element face"):
                return int(line.split()[2])
            if line.startswith(b"end_header"):
                break
    return 0


Focus = tuple[np.ndarray, float]  # (초점, 카메라-초점 거리)


def crop_points_to_focus(src: Path, target: Focus, radius_factor: float, dst: Path) -> Path:
    """조밀 점군을 대상 주변 구 안으로 자른다 (가시성 정보 보존)."""
    focus, cam_dist = target
    ply = read_vertex_ply(src)
    xyz, _ = points_and_colors(ply)
    keep = np.linalg.norm(xyz - focus, axis=1) < radius_factor * cam_dist
    if keep.sum() < max(1000, 0.01 * len(keep)):
        log.warning("초점 주변 점이 거의 없어 점군 자르기를 건너뜁니다 (%d점).", keep.sum())
        return src
    n = write_vertex_subset(ply, keep, dst)
    log.info("점군을 촬영 대상 주변으로 자름: %d / %d points", n, len(keep))
    return dst


def crop_to_focus(src: Path, target: Focus, radius_factor: float, dst: Path) -> Path:
    """메쉬를 대상 주변 구 안의 면만 남긴다."""
    focus, cam_dist = target
    radius = radius_factor * cam_dist
    mesh = trimesh.load(src, process=False, force="mesh")
    keep = np.linalg.norm(mesh.triangles_center - focus, axis=1) < radius
    if keep.sum() < 0.01 * len(mesh.faces) or keep.sum() < 1000:
        log.warning("초점 주변에 메쉬가 거의 없어 자르기를 건너뜁니다 (%d faces).", keep.sum())
        return src
    log.info("촬영 대상 주변만 유지: 반경 %.2f (카메라 거리 %.2f × %.2f) → %d / %d faces",
             radius, cam_dist, radius_factor, keep.sum(), len(mesh.faces))
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    mesh.export(dst)
    return dst


def keep_largest_component(src: Path, dst: Path) -> Path:
    """면 수 기준 최대 연결 성분만 남긴다."""
    mesh = trimesh.load(src, process=False, force="mesh")
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    counts = np.bincount(labels)
    if len(counts) <= 1:
        log.info("메쉬 연결 성분 1개 - 정리 불필요")
        return src
    keep = labels == counts.argmax()
    log.info("메쉬 연결 성분 %d개 중 최대 성분만 유지 (%d / %d faces)", len(counts), keep.sum(), len(mesh.faces))
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    mesh.export(dst)
    return dst


def inpaint_empty_texture(obj_path: Path, tol: int = 24) -> None:
    """텍스처에서 뷰가 없어 --empty-color로 채워진 부분을 주변 색으로 인페인팅한다.

    UV 아틀라스의 미사용 배경(빈 패딩)도 같은 색이라 크기로는 "진짜 구멍"과 구분이 안 된다
    (2DGS 합성 구멍은 작지 않을 수도 있음). 대신 패딩은 항상 이미지 테두리까지 이어진
    하나의 연결 영역이라는 점을 이용 - 테두리에 닿은 성분만 배경으로 제외하고, 나머지
    (차트 안쪽에 갇힌, 테두리에 안 닿는) 성분은 크기와 무관하게 전부 인페인팅한다.
    """
    mtl_path = obj_path.with_suffix(".mtl")
    if not mtl_path.is_file():
        return
    images = [obj_path.parent / m.group(1) for m in re.finditer(r"^map_Kd\s+(.+)$", mtl_path.read_text(), re.MULTILINE)]
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dist = np.abs(img.astype(np.int16) - np.array(_EMPTY_COLOR_BGR, dtype=np.int16)).sum(axis=2)
        raw_mask = (dist < tol).astype(np.uint8)
        if raw_mask.sum() == 0:
            continue
        n, labels = cv2.connectedComponents(raw_mask, connectivity=8)
        border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
        border_labels.discard(0)
        hole_labels = [i for i in range(1, n) if i not in border_labels]
        if not hole_labels:
            log.info("텍스처 인페인팅: %s - 빈 영역이 테두리까지 이어진 배경 패딩뿐이라 건너뜀", img_path.name)
            continue
        mask = np.isin(labels, hole_labels).astype(np.uint8)
        log.info("텍스처 인페인팅: %s (%d개 구멍, %d픽셀, 전체의 %.1f%%)",
                  img_path.name, len(hole_labels), mask.sum(), 100 * mask.mean())
        filled = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(str(img_path), filled)


def run_mvs(openmvs_dir: Path, dense_dir: Path, work: Path, cfg: PipelineConfig,
            depth_dir: Path | None = None, seg: tuple[Path, Path] | None = None,
            mesh_hook=None) -> tuple[Path, Path]:
    """(텍스처 메쉬, 형상 메쉬) 반환. 형상 메쉬는 축소하지 않는다.
    seg: (깊이맵용, 투표용) 분할 마스크 폴더. mesh_hook: 텍스처 전 메쉬 교체 (2DGS 합성)."""
    mvs = work / "mvs"
    mvs.mkdir(parents=True, exist_ok=True)
    logf = work / "openmvs.log"
    # 기본 우선순위(낮음)는 다른 작업과 겹치면 크게 느려짐
    common = ["-w", mvs, "--max-threads", str(cfg.max_threads), "-v", "2", "--process-priority", "0"]

    def tool(name: str, *args):
        run([openmvs_dir / f"{name}{EXE}", *args, *common], cwd=mvs, log_file=logf, env_path=openmvs_dir)

    scene = mvs / "scene.mvs"
    log.info("[MVS 1/5] COLMAP → OpenMVS 씬 변환")
    tool("InterfaceCOLMAP", "-i", depth_dir or dense_dir, "-o", scene, "--image-folder", dense_dir / "images")

    log.info("[MVS 2/5] 조밀 점군 생성 (깊이맵 추정 + 융합)")
    mask_args = []
    if seg is not None:
        mask_args = ["--mask-path", seg[0]]
    elif cfg.crop_to_focus and cfg.mvs_mask:
        mask_dir = make_focus_masks(dense_dir / "sparse", dense_dir / "images", mvs / "masks",
                                    cfg.crop_radius_factor * 1.1, mask_name=lambda n: f"{Path(n).stem}.mask.png")
        if mask_dir is not None:
            mask_args = ["--mask-path", mask_dir]
    dense_scene = mvs / "scene_dense.mvs"
    tool("DensifyPointCloud", "-i", scene, "-o", dense_scene, *mask_args,
         "--resolution-level", str(cfg.densify_resolution_level),
         "--max-resolution", str(cfg.densify_max_resolution),
         "--number-views-fuse", str(cfg.densify_number_views_fuse),
         "--iters", str(cfg.densify_iters),
         "--geometric-iters", str(cfg.densify_geometric_iters),
         # ROI 자르기·tower 모드 끔 (발끝 잘림·크래시 방지)
         "--crop-to-roi", "0", "--tower-mode", "0")
    dense_ply = _output_mesh(dense_scene, ".ply")
    focus = camera_focus(dense_dir / "sparse" / "images.bin") if cfg.crop_to_focus else None
    if cfg.crop_to_focus and focus is None:
        log.warning("카메라들이 한 대상을 향하지 않아 초점 기준 자르기를 건너뜁니다.")
    if focus is not None:
        dense_ply = crop_points_to_focus(dense_ply, focus, cfg.crop_radius_factor, mvs / "scene_dense_crop.ply")

    log.info("[MVS 3/5] 표면 메쉬 복원 (Delaunay + graph-cut)")
    mesh_scene = mvs / "scene_mesh.mvs"
    tool("ReconstructMesh", "-i", dense_scene, "-p", dense_ply, "-o", mesh_scene,
         "--decimate", str(cfg.mesh_decimate), "--crop-to-roi", "0")
    mesh = _output_mesh(mesh_scene, ".ply")
    if seg is not None:
        mesh = carve_by_masks(mesh, dense_dir / "sparse", seg[1], cfg, mvs / "scene_mesh_carve.ply")
    elif focus is not None:
        # 분할 마스크가 없을 때만 구체 자르기로 배경 제거 (마스크 투표가 더 정밀하므로 중복 자르기는 생략)
        mesh = crop_to_focus(mesh, focus, cfg.crop_radius_factor, mvs / "scene_mesh_crop.ply")
    if cfg.keep_largest_component:
        mesh = keep_largest_component(mesh, mvs / "scene_mesh_clean.ply")

    if cfg.refine_mesh:
        log.info("[MVS 4/5] 메쉬 정밀화 (광도 일관성 기반)")
        refine_scene = mvs / "scene_refine.mvs"
        try:
            tool("RefineMesh", "-i", dense_scene, "-m", mesh, "-o", refine_scene,
                 "--resolution-level", str(cfg.refine_resolution_level))
            mesh = _output_mesh(refine_scene, ".ply")
        except ToolError as e:
            log.warning("메쉬 정밀화 실패, 정밀화 전 메쉬를 사용합니다: %s", str(e).splitlines()[0])
    else:
        log.info("[MVS 4/5] 메쉬 정밀화 건너뜀")
    if mesh_hook is not None:
        mesh = mesh_hook(mesh)

    if not cfg.texture:
        log.info("[MVS 5/5] 텍스처 매핑 건너뜀")
        return mesh, mesh

    decimate = 1.0
    faces = _ply_face_count(mesh)
    if cfg.texture_max_faces > 0 and faces > cfg.texture_max_faces:
        decimate = cfg.texture_max_faces / faces
    log.info("[MVS 5/5] 텍스처 매핑 (%d faces%s)", faces,
             f" → {cfg.texture_max_faces} 로 축소" if decimate < 1 else "")
    tex_input = dense_scene
    if depth_dir is not None:
        tex_input = mvs / "scene_all_views.mvs"
        tool("InterfaceCOLMAP", "-i", dense_dir, "-o", tex_input, "--image-folder", dense_dir / "images")
    tex_scene = mvs / "scene_texture.mvs"
    args = ["-i", tex_input, "-m", mesh, "-o", tex_scene,
            "--export-type", "obj", "--resolution-level", str(cfg.texture_resolution_level),
            "--decimate", f"{decimate:.4f}", "--cost-smoothness-ratio", str(cfg.texture_smoothness),
            "--empty-color", str(_EMPTY_COLOR_INT)]
    # seam leveling은 Windows 빌드에서 텍스처 손상·크래시 → 실패 시 끄고 재시도
    seam = ["--global-seam-leveling", "1" if cfg.texture_seam_leveling else "0",
            "--local-seam-leveling", "1" if cfg.texture_seam_leveling else "0"]
    no_seam = ["--global-seam-leveling", "0", "--local-seam-leveling", "0"]
    attempts = [seam, no_seam, no_seam]
    for i, extra in enumerate(attempts):
        try:
            tool("TextureMesh", *args, *extra)
            tex_obj = _output_mesh(tex_scene, ".obj")
            inpaint_empty_texture(tex_obj)
            return tex_obj, mesh
        except ToolError as e:
            log.warning("텍스처 매핑 실패 (%d/%d): %s", i + 1, len(attempts), str(e).splitlines()[0])
    log.warning("텍스처 매핑을 포기하고 형상만 출력합니다.")
    return mesh, mesh
