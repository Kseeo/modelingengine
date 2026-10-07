"""SfM (COLMAP): 특징점 → 매칭 → 포즈·희소 점군 → 방향 정렬 → 왜곡 보정."""
from __future__ import annotations

import logging
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .tools import ToolError, fresh_dir, run

log = logging.getLogger(__name__)


@dataclass
class SfmResult:
    dense_dir: Path
    num_input: int
    num_registered: int
    num_points: int
    aligned_up: bool
    depth_dir: Path | None = None  # 깊이맵용 부분 모델 (None이면 dense_dir)


def _count_bin(path: Path) -> int:
    """COLMAP .bin 항목 수."""
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def _best_model(sparse_root: Path) -> tuple[Path, int]:
    models = [d for d in sparse_root.iterdir() if (d / "images.bin").is_file()]
    if not models:
        raise ToolError("SfM 복원 실패: 카메라 포즈를 추정하지 못했습니다. 촬영 중첩(overlap)과 텍스처를 확인하세요.")
    scored = sorted(((_count_bin(d / "images.bin"), d) for d in models), reverse=True)
    if len(scored) > 1:
        log.warning("SfM 모델이 %d개로 분리됨 (각 등록 이미지 수: %s). 가장 큰 모델을 사용합니다.",
                    len(scored), [n for n, _ in scored])
    return scored[0][1], scored[0][0]


def _qvec_to_rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_images(images_bin: Path) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    """등록 이미지별 (파일명, R, t, camera_id). X_cam = R @ X_world + t."""
    out = []
    with open(images_bin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)
            q = struct.unpack("<4d", f.read(32))
            t = np.array(struct.unpack("<3d", f.read(24)))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = bytearray()
            while (ch := f.read(1)) != b"\0":
                name += ch
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * num_pts, 1)
            out.append((name.decode("utf-8"), _qvec_to_rot(q), t, cam_id))
    return out


def read_image_names(images_bin: Path) -> list[str]:
    return [name for name, *_ in read_images(images_bin)]


def read_camera_rays(images_bin: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """카메라별 (중심, 광축 방향), 월드 좌표."""
    return [(-r.T @ t, r[2]) for _, r, t, _ in read_images(images_bin)]


def camera_focus(images_bin: Path) -> tuple[np.ndarray, float] | None:
    """카메라 광축들이 모이는 점과 카메라-초점 거리 중앙값. 한 점을 향하지 않으면 None."""
    rays = read_camera_rays(images_bin)
    if len(rays) < 3:
        return None
    centers = np.array([c for c, _ in rays])
    dirs = np.array([d for _, d in rays])
    proj = np.eye(3)[None] - dirs[:, :, None] * dirs[:, None, :]
    w = np.ones(len(rays))
    focus = None
    # IRLS: 이상치 카메라 영향 축소
    for _ in range(10):
        a = (w[:, None, None] * proj).sum(axis=0)
        if np.linalg.cond(a) > 1e6:
            return None
        focus = np.linalg.solve(a, (w[:, None] * np.einsum("nij,nj->ni", proj, centers)).sum(axis=0))
        dist = np.linalg.norm(np.einsum("nij,nj->ni", proj, focus - centers), axis=1)
        w = 1.0 / np.maximum(dist, 0.1 * np.median(dist) + 1e-9)
    in_front = np.mean([(focus - c) @ d > 0 for c, d in rays])
    if in_front < 0.8:
        return None
    return focus, float(np.median([np.linalg.norm(c - focus) for c, _ in rays]))


_CAM_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 16}


def read_cameras(cameras_bin: Path) -> dict[int, tuple[int, int, float, float, float, float]]:
    """camera_id → (width, height, fx, fy, cx, cy)."""
    cams = {}
    with open(cameras_bin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            k = _CAM_NUM_PARAMS[model_id]
            p = struct.unpack(f"<{k}d", f.read(8 * k))
            if model_id in (0, 2, 3, 7, 8, 9):  # 단일 초점거리 모델
                cams[cam_id] = (w, h, p[0], p[0], p[1], p[2])
            else:
                cams[cam_id] = (w, h, p[0], p[1], p[2], p[3])
    return cams


def make_focus_masks(model: Path, images_dir: Path, out_dir: Path, radius_factor: float) -> Path | None:
    """촬영 대상 주변 구를 각 프레임에 투영한 원형 마스크. 미등록 프레임은 가장 가까운 프레임 것 사용."""
    import cv2
    from PIL import Image

    res = camera_focus(model / "images.bin")
    if res is None:
        log.warning("카메라들이 한 대상을 향하지 않아 대상 기준 SfM을 건너뜁니다.")
        return None
    focus, cam_dist = res
    radius = radius_factor * cam_dist
    cams = read_cameras(model / "cameras.bin")
    circles = {}  # name → (u, v, r), 정규화 좌표
    for name, r, t, cam_id in read_images(model / "images.bin"):
        w, h, fx, fy, cx, cy = cams[cam_id]
        xc = r @ focus + t
        if xc[2] <= 0:
            continue
        circles[name] = ((fx * xc[0] / xc[2] + cx) / w, (fy * xc[1] / xc[2] + cy) / h,
                         max(fx, fy) * radius / xc[2] / max(w, h))
    if not circles:
        return None

    fresh_dir(out_dir)
    names = sorted(p.name for p in images_dir.iterdir())
    reg_idx = [i for i, n in enumerate(names) if n in circles]
    coverage = []
    for i, name in enumerate(names):
        src = name if name in circles else names[min(reg_idx, key=lambda j: abs(j - i))]
        u, v, rad = circles[src]
        with Image.open(images_dir / name) as im:
            w, h = im.size
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (round(u * w), round(v * h)), round(rad * max(w, h)), 255, -1)
        coverage.append(mask.mean() / 255)
        cv2.imencode(".png", mask)[1].tofile(str(out_dir / f"{name}.png"))
    log.info("대상 마스크 %d장 생성 (반경 %.2f, 화면 평균 %.0f%% 영역)", len(names), radius, 100 * np.mean(coverage))
    return out_dir


def video_pairs(names: list[str], window: int, global_stride: int) -> list[tuple[str, str]]:
    """영상 매칭 쌍: 앞뒤 window 이웃(순환) + global_stride 간격 전역 쌍."""
    n = len(names)
    pairs = set()
    for i in range(n):
        for d in range(1, window + 1):
            pairs.add(tuple(sorted((i, (i + d) % n))))
    keys = list(range(0, n, max(1, global_stride)))
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            pairs.add((keys[a], keys[b]))
    return [(names[i], names[j]) for i, j in sorted(pairs) if i != j]


def run_sfm(colmap: Path, images_dir: Path, work: Path, cfg: PipelineConfig,
            single_camera: bool, ordered: bool = False) -> SfmResult:
    """ordered: 파일명 순서 = 촬영 순서 (영상)."""
    bin_dir = colmap.parent
    logf = work / "colmap.log"
    db = work / "database.db"
    sparse = work / "sparse"
    dense = work / "dense"
    for p in (db, sparse, dense, work / "dense_depth"):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    sparse.mkdir(parents=True)
    num_input = sum(1 for _ in images_dir.iterdir())
    gpu = "1" if cfg.use_gpu else "0"

    def colmap_run(*args):
        run([colmap, *args], cwd=work, log_file=logf, env_path=bin_dir)

    used_pairs = ordered and cfg.video_pair_matching

    def reconstruct_sparse(db: Path, out: Path, mask_dir: Path | None = None, tag: str = "") -> tuple[Path | None, int]:
        """(최대 모델, 등록 수)."""
        nonlocal gpu
        db.unlink(missing_ok=True)
        mask_args = ["--ImageReader.mask_path", mask_dir] if mask_dir else []

        def extract(use_gpu: str):
            colmap_run("feature_extractor",
                       "--database_path", db, "--image_path", images_dir,
                       "--ImageReader.camera_model", cfg.camera_model,
                       "--ImageReader.single_camera", "1" if single_camera else "0",
                       "--FeatureExtraction.use_gpu", use_gpu, *mask_args,
                       *(["--FeatureExtraction.max_image_size", str(cfg.feature_max_image_size)]
                         if cfg.feature_max_image_size > 0 else []))

        log.info("[SfM 1/5]%s 특징점 추출 (SIFT, %s)", tag, "GPU" if gpu == "1" else "CPU")
        try:
            extract(gpu)
        except ToolError as e:
            if gpu == "0":
                raise
            log.warning("GPU 특징점 추출 실패 → CPU로 재시도 (느림). GPU 드라이버 업데이트를 권장합니다.\n  %s",
                        str(e).splitlines()[-1])
            gpu = "0"
            db.unlink(missing_ok=True)
            extract(gpu)

        if used_pairs:
            names = sorted(p.name for p in images_dir.iterdir())
            pairs = video_pairs(names, cfg.video_pair_window, cfg.video_pair_global_stride)
            pair_file = work / "match_pairs.txt"
            pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
            log.info("[SfM 2/5]%s 특징점 매칭 (영상 쌍 %d개 = 전체의 %.0f%%, %d장)", tag,
                     len(pairs), 100 * len(pairs) / max(1, num_input * (num_input - 1) // 2), num_input)
            colmap_run("matches_importer", "--database_path", db, "--match_list_path", pair_file,
                       "--match_type", "pairs", "--FeatureMatching.use_gpu", gpu)
        elif num_input <= cfg.exhaustive_match_max:
            log.info("[SfM 2/5]%s 특징점 매칭 (exhaustive, %d장)", tag, num_input)
            colmap_run("exhaustive_matcher", "--database_path", db,
                       "--FeatureMatching.use_gpu", gpu)
        else:
            log.info("[SfM 2/5]%s 특징점 매칭 (sequential, %d장)", tag, num_input)
            colmap_run("sequential_matcher", "--database_path", db,
                       "--FeatureMatching.use_gpu", gpu,
                       "--SequentialMatching.overlap", str(cfg.sequential_overlap))

        def map_sparse():
            fresh_dir(out)
            colmap_run("mapper", "--database_path", db, "--image_path", images_dir,
                       "--output_path", out)
            return _best_model(out)

        log.info("[SfM 3/5]%s 증분식 복원 (카메라 포즈 + 희소 점군)", tag)
        try:
            model, n_reg = map_sparse()
        except ToolError:
            if not used_pairs:
                raise
            model, n_reg = None, 0
        if used_pairs and n_reg < 0.9 * num_input:
            # 끊긴 구간 연결: 남은 쌍 추가 매칭
            log.warning("영상 쌍 매칭으로 %d / %d장만 등록 → 나머지 쌍을 추가 매칭 후 재복원", n_reg, num_input)
            colmap_run("exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", gpu)
            model, n_reg = map_sparse()
        return model, n_reg

    model, n_reg = reconstruct_sparse(db, sparse)
    if model is None:
        raise ToolError("SfM 복원 실패: 카메라 포즈를 추정하지 못했습니다.")

    if cfg.object_centric:
        # 대상 주변 특징점만으로 포즈 재추정 (촬영 중 대상 움직임 보정)
        mask_dir = make_focus_masks(model, images_dir, work / "masks", cfg.object_mask_radius_factor)
        if mask_dir is not None:
            try:
                model2, n_reg2 = reconstruct_sparse(work / "database_obj.db", work / "sparse_obj",
                                                    mask_dir, tag=" (대상 기준)")
            except ToolError as e:
                model2, n_reg2 = None, 0
                log.warning("대상 기준 SfM 실패: %s", str(e).splitlines()[0])
            if model2 is not None and n_reg2 >= cfg.object_min_reg_ratio * n_reg:
                log.info("대상 기준 포즈 사용: 등록 %d장 (배경 기준 %d장)", n_reg2, n_reg)
                model, n_reg = model2, n_reg2
            else:
                log.warning("대상 기준 등록이 부족해 배경 기준 포즈를 유지합니다 (%d vs %d장)", n_reg2, n_reg)

    log.info("등록된 이미지: %d / %d", n_reg, num_input)
    if n_reg < 0.5 * num_input:
        log.warning("입력의 절반 미만만 등록되었습니다. 결과 품질이 낮을 수 있습니다.")

    aligned = False
    if cfg.align_up:
        log.info("[SfM 4/5] 위쪽 방향 정렬 (카메라 업벡터 기준)")
        aligned_dir = work / "sparse_aligned"
        fresh_dir(aligned_dir)
        try:
            colmap_run("model_orientation_aligner", "--image_path", images_dir,
                       "--input_path", model, "--output_path", aligned_dir,
                       "--method", "IMAGE-ORIENTATION")
            model, aligned = aligned_dir, True
        except ToolError as e:
            log.warning("방향 정렬 실패, 원래 좌표계를 사용합니다: %s", str(e).splitlines()[0])

    log.info("[SfM 5/5] 이미지 왜곡 보정 (PINHOLE)")
    colmap_run("image_undistorter", "--image_path", images_dir, "--input_path", model,
               "--output_path", dense, "--output_type", "COLMAP",
               "--max_image_size", str(cfg.undistort_max_size))

    depth_dir = None
    if cfg.mvs_view_stride > 1:
        # 깊이맵용 부분 모델 (텍스처는 전체 사용)
        reg = sorted(read_image_names(dense / "sparse" / "images.bin"))
        drop = [n for i, n in enumerate(reg) if i % cfg.mvs_view_stride]
        if len(reg) - len(drop) >= cfg.min_images:
            drop_file = work / "mvs_dropped_images.txt"
            drop_file.write_text("\n".join(drop) + "\n", encoding="utf-8")
            depth_dir = work / "dense_depth"
            (depth_dir / "sparse").mkdir(parents=True)
            colmap_run("image_deleter", "--input_path", dense / "sparse",
                       "--output_path", depth_dir / "sparse", "--image_names_path", drop_file)
            log.info("깊이맵용 이미지: 등록 %d장 중 %d장 (%d장 간격), 텍스처는 %d장 전부",
                     len(reg), len(reg) - len(drop), cfg.mvs_view_stride, len(reg))
        else:
            log.info("등록 이미지가 적어 깊이맵에 전부 사용 (%d장)", len(reg))

    n_pts = _count_bin(dense / "sparse" / "points3D.bin")
    log.info("희소 점군: %d points", n_pts)
    return SfmResult(dense, num_input, n_reg, n_pts, aligned, depth_dir)
