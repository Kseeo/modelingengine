# modelingengine — 2D 영상/이미지 → 3D 발 메쉬 (GLB, STL)

SfM(COLMAP) + MVS(OpenMVS) + 2DGS(gsplat) 합성 파이프라인.
MVS는 사진으로 직접 삼각측량해 **큰 곡률(아치·발등)**이 실제에 가깝고, 2DGS는 텍스처 없는 피부에서도
**매끈하고 구멍 없는 표면**을 만듭니다. 두 메쉬를 합성해 양쪽 장점을 함께 씁니다.

```
영상(30~60초) ─ 0.5초당 최선명 1장 ─┐
                                     ├─ [SfM · COLMAP] SIFT → 영상 쌍 매칭 → 초점거리 보정 → 전역 복원
이미지 폴더(20장+) ──────────────────┘     (결과 부족 시 증분식과 비교) → 위쪽 정렬 → 왜곡 보정
            [분할 · BiRefNet, GPU]  SfM과 병렬로 프레임별 발 마스크
            [Dense]  OpenMVS(CPU) 깊이맵 → 메쉬  ∥  2DGS(GPU) hull 초기화 → 학습 → TSDF 메쉬
            [합성]   2DGS 메쉬에 MVS 큰 곡률 반영 → MVS 근거 없이 hull에 붙은 덩어리 재보간
                     → OpenMVS 텍스처 매핑
            [Export] GLB(텍스처, Y-up) / STL·PLY(형상, Z-up)
```

2DGS가 실패하거나 `gs_timeout_sec`(300초)를 넘기면 MVS 메쉬로, 분할이 실패하면 원형 마스크로 자동 폴백합니다.

## 설치

```bash
pip install -r requirements.txt
python scripts/install_tools.py          # COLMAP 4.2.0(CUDA) + OpenMVS 2.4.0 → tools/
python scripts/install_tools.py --nocuda # GPU 없는 PC
```

분할과 2DGS는 CUDA torch가 있는 파이썬에서 별도 프로세스로 실행합니다. 같은 환경이면 그대로 쓰고,
다른 환경(예: conda)이면 그 파이썬 경로를 `PG_GPU_PYTHON`에 지정합니다.

```bash
pip install -r requirements-gpu.txt      # GPU 파이썬 환경에서
export PG_GPU_PYTHON=/path/to/gpu-env/python
```

다른 위치에 설치된 도구를 쓰려면 `COLMAP_PATH`(colmap 실행 파일), `OPENMVS_DIR`(OpenMVS 폴더)를 지정합니다.

## 사용

```bash
python -m photogrammetry data/foot.mp4             # 영상
python -m photogrammetry data/foot_images/         # 이미지 폴더
python -m photogrammetry                           # data/ 안의 모든 영상·이미지 폴더 일괄 처리
python -m photogrammetry data/foot.mp4 --quality high --scale 1.0 -o output/foot
```

| 옵션 | 설명 |
|---|---|
| `--quality turbo/fast/normal/high` | 속도·품질 프리셋 (기본 `turbo`) |
| `--formats glb,stl,ply` | 출력 형식 (기본 3종, obj도 가능) |
| `--scale` | 출력 배율. SfM 결과는 절대 크기를 모르므로 실측 기준으로 보정 |
| `-v` | COLMAP/OpenMVS 출력까지 표시 |

`turbo` 실측 (개발 PC: i7-13700F, RTX 4070, COLMAP·OpenMVS는 CPU 경로, 영상 5개):
전체 190~247초. 입력 20~45초 / SfM(분할 대기 포함) 32~58초 / Dense+합성+텍스처 124~152초.
현재 가장 오래 걸리는 단계는 2DGS 학습(약 85초, `gs_iters` 6000)입니다.

Python에서:

```python
from photogrammetry import reconstruct, PipelineConfig
report = reconstruct("data/foot.mp4", "output/foot", config=PipelineConfig.preset("turbo", dense_method="mvs"))
```

## 출력

`output/<이름>/`
- `<이름>.glb` — 텍스처 포함, glTF 규약 Y-up
- `<이름>.stl` — 형상만, Z-up, 바닥이 z=0
- `<이름>.ply` — STL과 같은 형상 + 정점 색
- `<이름>_report.json` — 등록 이미지 수, 단계별 시간, 합성 통계(`dense`: MVS 근거·재보간 비율), 설정값

작업 폴더(프레임, COLMAP DB, 모델, 로그)는 **성공하면 삭제, 실패하면 보존**합니다(`keep_workspace`).
출력 경로에 한글·공백이 있으면 작업 폴더를 `%TEMP%\pg_work\` 아래에 만듭니다(`PG_WORK_ROOT`로 변경 가능).

## 촬영 가이드

- 발을 화면 가운데에 두고 주위를 천천히 한 바퀴 이상 돌며 촬영 (인접 프레임 간 60~80% 겹침)
- 종아리를 받치고 발 전체를 띄울 것 — 뒤꿈치를 받침대에 올리면 뒤꿈치 바닥이 복원되지 않음
- 흐림(모션 블러), 역광, 무늬 없는 배경은 SfM 실패 원인

## 한계

- **스케일**: 단안 SfM은 절대 크기를 알 수 없음 → `--scale`로 후보정
- **재보간 영역**: MVS 근거 없이 hull에 붙은 2DGS 표면은 주변 곡면으로 보간한 추정값 (`report.json`의 `dense.refilled`)
- 피부·배경이 모두 밋밋한 영상은 MVS 근거가 부족해 합성 품질이 떨어짐
