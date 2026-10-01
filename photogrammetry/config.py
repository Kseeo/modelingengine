"""파이프라인 설정값."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class PipelineConfig:
    # ---- 입력 ----
    frame_interval_sec: float = 0.5      # 영상: 0.5초당 1장
    pick_sharpest: bool = True           # 각 0.5초 구간 안에서 가장 선명한 프레임 선택 (False면 구간 시작 프레임)
    min_images: int = 20                 # 최소 입력 이미지 수
    video_min_sec: float = 30.0          # 권장 영상 길이 (벗어나면 경고만)
    video_max_sec: float = 60.0
    max_image_size: int = 3200           # 입력 이미지 긴 변 최대 픽셀 (초과 시 축소)

    # ---- SfM (COLMAP) ----
    camera_model: str = "OPENCV"         # 스마트폰 렌즈 왜곡 보정 (k1,k2,p1,p2)
    single_camera: bool | None = None    # None: 영상이거나 해상도가 모두 같으면 True
    use_gpu: bool = True
    feature_max_image_size: int = 0      # 특징점 추출 시 긴 변 최대 픽셀 (0 = 입력 그대로)
    exhaustive_match_max: int = 200      # 이미지 수가 이 이하면 exhaustive, 초과면 sequential 매칭
    sequential_overlap: int = 20
    video_pair_matching: bool = False    # 영상: 전체 대신 시간상 이웃 + 순환(루프) + 희소 전역 쌍만 매칭
    video_pair_window: int = 10          #   이웃 프레임 범위 (앞뒤 각각)
    video_pair_global_stride: int = 4    #   이 간격의 프레임끼리는 전부 매칭 (재방문·루프 검출용)
    object_centric: bool = False         # 대상 주변 특징점만으로 포즈 재추정 (촬영 중 대상이 움직인 영상 보정)
    object_mask_radius_factor: float = 0.35  # 대상 마스크 반경 = 카메라-대상 거리 × 이 값
    object_min_reg_ratio: float = 0.8    # 대상 기준 등록 수가 배경 기준의 이 비율 미만이면 배경 기준 유지
    align_up: bool = True               # 카메라 업벡터 기반으로 중력(위쪽) 방향 정렬
    undistort_max_size: int = 3200
    mvs_view_stride: int = 1             # MVS/텍스처에 등록 이미지 N장 중 1장만 사용 (SfM은 전부 사용)

    # ---- MVS (OpenMVS) ----
    densify_resolution_level: int = 1    # 0=원본, 1=1/2, 2=1/4 ...
    densify_max_resolution: int = 2560
    densify_number_views_fuse: int = 3
    densify_iters: int = 3               # patch-match 반복
    densify_geometric_iters: int = 2     # 기하 일관성 반복 (0이면 생략, 깊이맵 시간 약 절반)
    mesh_decimate: float = 1.0           # ReconstructMesh 단계 decimation (0..1]
    crop_to_focus: bool = True           # 카메라 광축이 모이는 점(촬영 대상) 주변만 남김 → 방·바닥 등 배경 제거
    crop_radius_factor: float = 0.5      # 자르기 반경 = 카메라-대상 거리 중앙값 × 이 값
    keep_largest_component: bool = True  # 떠다니는 조각 제거 (가장 큰 연결 성분만 유지)
    refine_mesh: bool = True
    refine_resolution_level: int = 1
    texture: bool = True
    texture_resolution_level: int = 0
    texture_max_faces: int = 0           # 텍스처용 메쉬 최대 면 수 (0 = 축소 안 함). STL은 축소 전 메쉬 사용
    texture_smoothness: float = 0.1      # 클수록 뷰 선택이 매끄러워져 패치 수 감소 → 아틀라스 생성 빨라짐
    texture_seam_leveling: bool = False  # OpenMVS 2.4 Windows 빌드에서 켜면 텍스처가 검게 나옴 (검증됨)
    max_threads: int = 0                # 0 = 모든 코어

    # ---- 출력 ----
    formats: list[str] = field(default_factory=lambda: ["glb", "stl", "ply"])  # ply: 정점 색 포함
    scale: float = 1.0                   # SfM 결과는 임의 스케일. 실측값이 있으면 배율 지정
    place_on_ground: bool = True         # 바운딩박스 바닥을 z=0, 중심을 원점으로
    keep_workspace: bool = True          # 중간 산출물(work/) 보존

    def to_dict(self) -> dict:
        return asdict(self)
