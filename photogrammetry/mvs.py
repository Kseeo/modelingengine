"""Multi-View Stereo (OpenMVS).

SfM 카메라로 깊이맵을 계산해 조밀 점군 생성 → 표면 메쉬 → 사진 기반 정밀화 → 텍스처 매핑.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from .config import PipelineConfig
from .plyio import points_and_colors, read_vertex_ply, write_vertex_subset
from .sfm import camera_focus
from .tools import EXE, ToolError, run

log = logging.getLogger(__name__)


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


def crop_points_to_focus(src: Path, images_bin: Path, radius_factor: float, dst: Path) -> Path:
    """조밀 점군에서 촬영 대상 주변 구 안의 점만 남긴다 (가시성 정보 보존)."""
    res = camera_focus(images_bin)
    if res is None:
        return src
    focus, cam_dist = res
    ply = read_vertex_ply(src)
    xyz, _ = points_and_colors(ply)
    keep = np.linalg.norm(xyz - focus, axis=1) < radius_factor * cam_dist
    if keep.sum() < max(1000, 0.01 * len(keep)):
        log.warning("초점 주변 점이 거의 없어 점군 자르기를 건너뜁니다 (%d점).", keep.sum())
        return src
    n = write_vertex_subset(ply, keep, dst)
    log.info("점군을 촬영 대상 주변으로 자름: %d / %d points", n, len(keep))
    return dst


def crop_to_focus(src: Path, images_bin: Path, radius_factor: float, dst: Path) -> Path:
    """카메라 광축이 모이는 점 주변 구(sphere) 안의 면만 남긴다."""
    res = camera_focus(images_bin)
    if res is None:
        log.warning("카메라들이 한 대상을 향하지 않아 초점 기준 자르기를 건너뜁니다.")
        return src
    focus, cam_dist = res
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
    """가장 큰 연결 성분(면 수 기준)만 남긴다. 배경 조각/부유물 제거용."""
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


def run_mvs(openmvs_dir: Path, dense_dir: Path, work: Path, cfg: PipelineConfig,
            depth_dir: Path | None = None) -> tuple[Path, Path]:
    """(시각용 메쉬, 형상용 메쉬) 반환.

    시각용: 텍스처 사용 시 .obj (texture_max_faces로 축소될 수 있음), 아니면 형상용과 동일
    형상용: 텍스처 전 최종 .ply (STL 등 치수 용도, 축소하지 않음)
    """
    mvs = work / "mvs"
    mvs.mkdir(parents=True, exist_ok=True)
    logf = work / "openmvs.log"
    # --process-priority 0: OpenMVS 기본값(-1, 낮음)이면 다른 작업과 겹칠 때 크게 느려짐
    common = ["-w", mvs, "--max-threads", str(cfg.max_threads), "-v", "2", "--process-priority", "0"]

    def tool(name: str, *args):
        # OpenMVS는 cwd에 자체 .log 파일을 남기므로 mvs/ 에서 실행
        run([openmvs_dir / f"{name}{EXE}", *args, *common], cwd=mvs, log_file=logf, env_path=openmvs_dir)

    scene = mvs / "scene.mvs"
    log.info("[MVS 1/5] COLMAP → OpenMVS 씬 변환")
    # 깊이맵은 depth_dir(부분 모델), 텍스처는 dense_dir(전체 모델). 이미지는 둘 다 dense_dir/images
    tool("InterfaceCOLMAP", "-i", depth_dir or dense_dir, "-o", scene, "--image-folder", dense_dir / "images")

    log.info("[MVS 2/5] 조밀 점군 생성 (깊이맵 추정 + 융합)")
    dense_scene = mvs / "scene_dense.mvs"
    tool("DensifyPointCloud", "-i", scene, "-o", dense_scene,
         "--resolution-level", str(cfg.densify_resolution_level),
         "--max-resolution", str(cfg.densify_max_resolution),
         "--number-views-fuse", str(cfg.densify_number_views_fuse),
         "--iters", str(cfg.densify_iters),
         "--geometric-iters", str(cfg.densify_geometric_iters),
         # OpenMVS 자동 ROI 상자가 발보다 작게 잡혀 발끝이 잘리는 경우가 있어 자르지 않음
         # (대상 분리는 crop_to_focus가 담당). ROI 추정 자체는 이웃 뷰 선택 가중에 계속 쓰인다
         "--crop-to-roi", "0",
         # 희소점이 적으면 탑 형태로 오판해 tower 경로에서 접근 위반(0xC0000005)으로 죽는 경우가 있음 (test07)
         "--tower-mode", "0")
    dense_ply = _output_mesh(dense_scene, ".ply")
    if cfg.crop_to_focus:
        # 메쉬 생성 전에 배경 점을 잘라 ReconstructMesh/텍스처 계산량을 줄인다 (메쉬 단계에서 한 번 더 자름)
        dense_ply = crop_points_to_focus(dense_ply, dense_dir / "sparse" / "images.bin",
                                         cfg.crop_radius_factor, mvs / "scene_dense_crop.ply")

    log.info("[MVS 3/5] 표면 메쉬 복원 (Delaunay + graph-cut)")
    mesh_scene = mvs / "scene_mesh.mvs"
    tool("ReconstructMesh", "-i", dense_scene, "-p", dense_ply, "-o", mesh_scene,
         "--decimate", str(cfg.mesh_decimate), "--crop-to-roi", "0")
    mesh = _output_mesh(mesh_scene, ".ply")
    if cfg.crop_to_focus:
        # 배경을 먼저 잘라내야 이후 '최대 연결 성분'이 방 전체가 아니라 대상이 된다
        mesh = crop_to_focus(mesh, dense_dir / "sparse" / "images.bin", cfg.crop_radius_factor,
                             mvs / "scene_mesh_crop.ply")
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
            "--decimate", f"{decimate:.4f}", "--cost-smoothness-ratio", str(cfg.texture_smoothness)]
    # OpenMVS 2.4 Windows 빌드의 이음새 보정(global/local seam leveling)은 텍스처를 90% 이상 검게 칠하고
    # 간헐적으로 비정상 종료(0xC0000409)까지 한다 → 기본은 끄고, 설정으로 켠 경우만 실패 시 끈 채로 재시도
    seam = ["--global-seam-leveling", "1" if cfg.texture_seam_leveling else "0",
            "--local-seam-leveling", "1" if cfg.texture_seam_leveling else "0"]
    no_seam = ["--global-seam-leveling", "0", "--local-seam-leveling", "0"]
    attempts = [seam, no_seam, no_seam]
    for i, extra in enumerate(attempts):
        try:
            tool("TextureMesh", *args, *extra)
            return _output_mesh(tex_scene, ".obj"), mesh
        except ToolError as e:
            log.warning("텍스처 매핑 실패 (%d/%d): %s", i + 1, len(attempts), str(e).splitlines()[0])
    log.warning("텍스처 매핑을 포기하고 형상만 출력합니다.")
    return mesh, mesh
