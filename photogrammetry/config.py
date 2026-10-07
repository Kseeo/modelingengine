"""파이프라인 설정값."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

QUALITY_PRESETS = {
    "turbo": dict(feature_max_image_size=1600, video_pair_matching=True, mvs_view_stride=2,
                  densify_resolution_level=2, densify_iters=2, densify_geometric_iters=1,
                  refine_mesh=False, texture_resolution_level=0, texture_max_faces=200_000,
                  texture_smoothness=1.0),
    "fast": dict(densify_resolution_level=2, refine_mesh=False, texture_resolution_level=1),
    "normal": dict(densify_resolution_level=1, refine_mesh=True, refine_resolution_level=1),
    "high": dict(densify_resolution_level=0, densify_max_resolution=4096, refine_mesh=True,
                 refine_resolution_level=0),
}


@dataclass
class PipelineConfig:
    # ---- 입력 ----
    frame_interval_sec: float = 0.5
    pick_sharpest: bool = True           # 구간 내 최선명 프레임 사용
    min_images: int = 20
    video_min_sec: float = 30.0          # 권장 길이 (경고만)
    video_max_sec: float = 60.0
    max_image_size: int = 3200

    # ---- SfM (COLMAP) ----
    camera_model: str = "OPENCV"
    single_camera: bool | None = None    # None: 해상도가 모두 같으면 True
    use_gpu: bool = True
    feature_max_image_size: int = 0      # 0 = 원본
    exhaustive_match_max: int = 200      # 초과 시 sequential 매칭
    sequential_overlap: int = 20
    video_pair_matching: bool = False    # 영상: 이웃·루프·희소 전역 쌍만 매칭
    video_pair_window: int = 10
    video_pair_global_stride: int = 4
    object_centric: bool = False         # 대상 주변 특징점으로 포즈 재추정
    object_mask_radius_factor: float = 0.35
    object_min_reg_ratio: float = 0.8
    align_up: bool = True
    undistort_max_size: int = 3200
    mvs_view_stride: int = 1             # 깊이맵에 N장 중 1장 사용

    # ---- MVS (OpenMVS) ----
    densify_resolution_level: int = 1    # 0=원본, 1=1/2, 2=1/4
    densify_max_resolution: int = 2560
    densify_number_views_fuse: int = 3
    densify_iters: int = 3
    densify_geometric_iters: int = 2
    mesh_decimate: float = 1.0
    crop_to_focus: bool = True           # 촬영 대상 주변만 유지
    crop_radius_factor: float = 0.5      # 반경 = 카메라-대상 거리 × 값
    keep_largest_component: bool = True
    refine_mesh: bool = True
    refine_resolution_level: int = 1
    texture: bool = True
    texture_resolution_level: int = 0
    texture_max_faces: int = 0           # 0 = 축소 안 함
    texture_smoothness: float = 0.1
    texture_seam_leveling: bool = False  # Windows 빌드에서 텍스처 손상
    max_threads: int = 0                 # 0 = 모든 코어

    # ---- 출력 ----
    formats: list[str] = field(default_factory=lambda: ["glb", "stl", "ply"])
    scale: float = 1.0
    place_on_ground: bool = True
    keep_workspace: bool = True

    @classmethod
    def preset(cls, quality: str, **overrides) -> PipelineConfig:
        return cls(**{**QUALITY_PRESETS[quality], **overrides})

    def to_dict(self) -> dict:
        return asdict(self)
