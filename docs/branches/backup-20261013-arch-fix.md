# `backup/20261013-arch-fix` 브랜치

| 항목 | 내용 |
|---|---|
| 최신 커밋 | `98d092b` 발 기준 SfM 카메라 값 고정 + 깊이맵 시점 각도 균등 선택 (2026-10-14) |
| 분기점 | `d08d8ee` (2차 백업) |
| 원격 | 없음 (로컬 전용) |
| 상태 | 백업용. `main`에 병합되지 않음 |
| 변경 규모 | `photogrammetry/sfm.py` 1개 파일, +44 / −7 |

## 목적

아치가 계곡처럼 끊기는 현상을 줄이기 위한 SfM 단계 실험입니다.
`feat/fused-dense`로 구조를 바꾸기 직전, **기존 MVS 단독 파이프라인**에서 시도한 수정을 보존합니다.

## 기반 구조 (`d08d8ee`)

- SfM(COLMAP) → MVS(OpenMVS) 단독, 2DGS·분할 없음
- turbo 프리셋: 영상 쌍 매칭, 특징점 1600px/4000개, `object_centric=True`
- object-centric: 전체 프레임으로 1차 복원 → 초점 주변 원형 마스크(`make_focus_masks`)로 대상 기준 2차 복원.
  2차 등록 수가 1차의 `object_min_reg_ratio`(0.6) 이상이면 2차 채택
- 작업 폴더 기본 보존(`keep_workspace=True`), CLI 옵션 다수(`--no-crop`, `--cpu`, `--object-centric` 등)

## 변경 내용

### 1. 대상 기준 SfM에서 카메라 내부 파라미터 고정

대상 영역만의 특징점으로는 초점거리·왜곡 추정이 불안정하다는 판단입니다.

- `read_camera_params()` 추가: 1차 복원의 `cameras.bin`에서 단일 카메라 파라미터를 문자열로 읽음
- 2차(대상 기준) 복원에 `--ImageReader.camera_params`로 전달하고
  `--Mapper.ba_refine_focal_length 0`, `--Mapper.ba_refine_extra_params 0`으로 번들 조정 중 고정
- `single_camera`일 때만 적용

### 2. 깊이맵 시점 각도 균등 선택

turbo의 `mvs_view_stride=2`는 기존에 파일명 순서로 2장 중 1장을 골라, 촬영 속도가 고르지 않으면 특정 방향이 비었습니다.

- `spread_views()` 추가: 초점 기준 카메라 방향 벡터에 최원점 샘플링(farthest point sampling)을 적용해
  방향이 고르게 퍼지도록 `ceil(N / stride)`장 선택
- 초점을 못 구하면 기존 등간격 선택으로 폴백. 텍스처 매핑은 여전히 전체 뷰 사용

## `main`과의 관계

| 항목 | 이 브랜치 | `main` |
|---|---|---|
| Dense | MVS 단독 | MVS ∥ 2DGS 합성 |
| SfM | 증분식 + object-centric 2차 패스 | 전역 복원 기본, object-centric 제거 |
| 마스크 | 원형 초점 마스크 | BiRefNet 분할 |
| 내부 파라미터 고정 | 있음 | 없음 (2차 패스 자체가 제거됨) |
| `spread_views` | 있음 | 없음 (등간격 선택) |

`main`은 object-centric 패스를 없앴으므로 변경 1은 그대로 옮길 대상이 없습니다.
변경 2(`spread_views`)는 `main`의 `mvs_view_stride` 처리에도 독립적으로 적용할 수 있습니다.
