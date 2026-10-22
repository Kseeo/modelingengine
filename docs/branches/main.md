# `main` 브랜치

| 항목 | 내용 |
|---|---|
| 최신 커밋 | `60da187` Merge pull request #1 from Kseeo/feat/fused-dense (2026-10-21) |
| 원격 | `origin/main` (동기화됨) |
| 상태 | 기본 브랜치. 파일 트리가 `feat/fused-dense`(`437863a`)와 **완전히 같음** |

## 개요

2D 영상/이미지에서 3D 발 메쉬(GLB, STL, PLY)를 만드는 포토그래메트리 엔진의 배포 기준 브랜치입니다.
PR #1로 `feat/fused-dense`를 병합해 현재 파이프라인은 **SfM(COLMAP) + MVS(OpenMVS) + 2DGS(gsplat) 합성** 구조입니다.
기능별 상세는 [feat-fused-dense.md](feat-fused-dense.md)를 참고하세요.

## 커밋 이력

| 날짜 | 커밋 | 내용 |
|---|---|---|
| 10-01 | `0a6b91f` | COLMAP/OpenMVS 포토그래메트리 파이프라인 최초 추가 |
| 10-06 | `23b34c4` | `mvs.py` 버그 수정 |
| 10-07 | `3d1156e` | 프리셋·파이프라인 유틸리티 리팩터링 |
| 10-13 | `d08d8ee` | 2차 백업: turbo에 object-centric 스카우트 패스, 초점 마스크 기반 MVS |
| 10-14 | `26998d0` | 2DGS + MVS 합성 Dense, 전역 SfM, 발 분할 마스크 |
| 10-16 | `eecf4ab` | 피부색 기반 마스크 필터(옵션) 추가 |
| 10-21 | `437863a` | 큐 기반 모델링 API 서버 추가, 피부색 필터 제거, hull 검증 |
| 10-21 | `60da187` | PR #1 병합 |

`d08d8ee`는 `backup/20261013-arch-fix`와의 분기점입니다.

## 현재 구조

```
photogrammetry/
  __main__.py        CLI (python -m photogrammetry)
  pipeline.py        전체 흐름: 입력 → SfM ∥ 분할 → MVS ∥ 2DGS → 합성 → 내보내기
  inputs.py          영상 프레임 추출(0.5초당 최선명 1장) / 이미지 폴더 준비
  sfm.py             COLMAP 특징점·매칭·전역/증분식 복원·위쪽 정렬·왜곡 보정
  segment.py         BiRefNet 분할 작업 관리, 마스크 왜곡 보정
  segment_worker.py  분할 워커 (GPU 파이썬)
  mvs.py             OpenMVS 깊이맵 → 메쉬 → 텍스처
  gs/                2DGS 워커 (hull 초기화, 학습, TSDF 메쉬)
  fuse.py            2DGS 메쉬 + MVS 곡률 합성
  export.py          GLB(Y-up) / STL·PLY(Z-up) 출력
  config.py          PipelineConfig, 품질 프리셋
server.py            FastAPI 작업 큐 서버
run_server.bat       Windows 서버 실행 스크립트 (mesh conda 환경)
```

## 실행

```bash
python -m photogrammetry data/foot.mp4             # CLI
python server.py --port 8000 --quality turbo       # API 서버
```

GPU 단계(분할, 2DGS)는 `PG_GPU_PYTHON`에 지정한 CUDA torch 파이썬에서 별도 프로세스로 실행됩니다.
