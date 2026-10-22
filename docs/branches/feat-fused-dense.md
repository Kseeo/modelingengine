# `feat/fused-dense` 브랜치

| 항목 | 내용 |
|---|---|
| 최신 커밋 | `437863a` Add queued modeling API server (2026-10-21) |
| 분기점 | `d08d8ee` (2차 백업) |
| 원격 | `origin/feat/fused-dense` (동기화됨) |
| 상태 | PR #1로 `main`에 병합 완료. 이후 추가 커밋 없음 |
| 변경 규모 | 21개 파일, +1514 / −184 (분기점 대비) |

## 목적

MVS만으로는 텍스처가 없는 피부에서 구멍·잡음이 생기고, 2DGS만으로는 아치·발등 같은 큰 곡률이 무뎌집니다.
두 방식을 **병렬 실행 후 합성**해 MVS의 실제 곡률과 2DGS의 매끈하고 닫힌 표면을 함께 쓰는 것이 목적입니다.

## 커밋별 변경

### 1. `26998d0` 2DGS + MVS 합성 Dense, 전역 SfM, 발 분할 마스크 (10-14)

- **SfM** (`sfm.py`)
  - `global_mapper`를 기본으로 사용. 결과가 부족하면 증분식·초점거리 보정 전역 복원과 비교해 등록 수가 가장 많은 쪽 선택
  - 초점거리 보정은 DB 복사본에서 수행, 매칭 보강 재복원도 더 나은 쪽 선택
  - 기존 object-centric 2차 패스 제거 (`object_centric`, `scout_*` 설정 삭제)
- **분할** (`segment.py`, `segment_worker.py`)
  - BiRefNet 워커를 GPU 파이썬으로 SfM과 병렬 실행
  - 깊이맵용 마스크(넉넉한 임계값·팽창) + 메쉬 다중 뷰 투표로 발 밖 정점 제거
  - 분할 실패 시 원형 초점 마스크로 폴백
- **Dense** (`gs/`, `fuse.py`, `mvs.py`)
  - OpenMVS(CPU)와 2DGS(gsplat, GPU)를 병렬 실행
  - 2DGS: 마스크 기반 visual hull 초기화 → Depth-Anything 노멀 prior → 학습(`gs_iters` 6000) → TSDF 메쉬
  - 합성: MVS 메쉬 기준 노멀 오프셋을 저주파 보간해 2DGS 정점 이동, MVS 근거 없이 hull에 붙은 영역은 biharmonic 재보간
  - 2DGS 실패·`gs_timeout_sec`(300초) 초과 시 MVS 메쉬로 폴백, 텍스처는 OpenMVS
- **CLI·설정**: 명백히 나은 쪽으로 옵션 고정(`--no-crop`, `--cpu` 등 다수 제거), 성공 시 작업 폴더 삭제(`keep_workspace=False`)
- **환경**: `requirements-gpu.txt`, `PG_GPU_PYTHON` 추가. turbo 특징점 1600→800px, 4000→2000개

### 2. `eecf4ab` 피부색 기반 마스크 필터 (10-16)

- `seg_skin_filter` 옵션: LAB 색공간 피부 모델로 받침대·칸막이·바지 등 비피부 영역을 마스크에서 제거
- 면적 손실이 과도하면 필터를 건너뛰는 안전장치 포함
- **다음 커밋에서 제거됨** (현재 코드에 없음)

### 3. `437863a` 큐 기반 모델링 API 서버 (10-21)

- **`server.py`** (FastAPI)
  - `POST /jobs` 사진 업로드 → 작업 생성, `GET /jobs/{id}` 상태(대기 순번 포함), `GET /jobs/{id}/files/{glb|stl|ply}` 다운로드, `GET /health`
  - GPU 하나 기준으로 작업을 순서대로 하나씩 처리, 파이프라인은 별도 프로세스로 실행(네이티브 크래시 격리)
  - 작업 상태를 `output/server_jobs/<id>/job.json`에 저장, 재시작 시 미완료 작업 재큐잉
  - API 키: `MODEL_API_KEY` 환경 변수 또는 `api_key.txt` (`X-API-Key` 헤더, 없으면 인증 없음)
  - MKL 스레딩 크래시(0xc06d007f) 회피용 `MKL_THREADING_LAYER=SEQUENTIAL`
- **`run_server.bat`**: mesh conda 환경으로 서버 실행
- **안정성 개선**
  - 피부색 필터 제거
  - hull-마스크 불일치 검증: hull 투영이 마스크 밖으로 크게 벗어난 뷰가 15% 초과면 hull 불신 → 2DGS 중단, MVS만 사용 (`report.json`의 `dense.hull_check`)
  - 합성 시 MVS 근거 없는 정점의 오프셋 감쇠(`FUSE_SCREEN`)

## 파이프라인

```
입력 → [SfM · COLMAP 전역 복원] ∥ [분할 · BiRefNet]
     → 마스크 왜곡 보정
     → [MVS · OpenMVS] ∥ [2DGS · hull → 학습 → TSDF]
     → 합성 (MVS 곡률 반영 + hull 덩어리 재보간) → OpenMVS 텍스처
     → GLB / STL / PLY
```

## 주요 설정 (`config.py`)

| 설정 | 기본값 | 설명 |
|---|---|---|
| `sfm_mapper` | `global` | 부족 시 incremental과 비교 |
| `seg_masks` | `True` | BiRefNet 분할 사용 |
| `seg_vote_ratio` | 0.8 | 마스크 안 비율 미만 정점 제거 |
| `dense_method` | `fused` | `mvs`로 2DGS 끔 |
| `gs_iters` / `gs_max_views` / `gs_image_size` | 6000 / 60 / 960 | 2DGS 학습 |
| `gs_timeout_sec` | 300 | 초과 시 MVS 폴백 |
| `fuse_lambda` / `fuse_support_tol` | 30 / 0.01 | 곡률 반영 평활도 / MVS 근거 거리 |

## 성능 (turbo, i7-13700F + RTX 4070, 영상 5개)

전체 190~247초. 입력 20~45초 / SfM 32~58초 / Dense+합성+텍스처 124~152초.
가장 오래 걸리는 단계는 2DGS 학습(약 85초).

## 폴백 정리

| 상황 | 동작 |
|---|---|
| 분할 실패 | 원형 초점 마스크 사용, 2DGS 생략 |
| hull-마스크 불일치 | 2DGS 중단 → MVS 메쉬 |
| 2DGS 실패·시간 초과 | MVS 메쉬 |
| 합성 예외 | MVS 메쉬 |
