# modelingengine — 2D 영상/이미지 → 3D (GLB, STL)

SfM–MVS 포토그래메트리(Structure-from-Motion + Multi-View Stereo) 모듈.
학습 기반 생성이 아닌 고전 다중뷰 기하학으로, 형상 사전지식 없이 **사진에 찍힌 것만** 복원합니다.

```
영상(30~60초) ─ 0.5초당 1장 ─┐
                              ├─ [SfM · COLMAP]  SIFT 특징점 → 매칭 → 카메라 포즈 역산 + 희소 점군
이미지 폴더(20장+) ───────────┘        → 위쪽 방향 정렬 → 왜곡 보정(PINHOLE)
                                  [MVS · OpenMVS] 깊이맵 삼각측량 → 조밀 점군 → 표면 메쉬
                                        → 촬영 대상 주변 자르기 → 부유 조각 제거 → 광도 기반 정밀화 → 텍스처 매핑
                                  [Export] GLB(텍스처, Y-up) / STL(형상, Z-up)
```

## 설치

```bash
pip install -r requirements.txt
python scripts/install_tools.py          # COLMAP 4.2.0(CUDA) + OpenMVS 2.4.0 → tools/
python scripts/install_tools.py --nocuda # GPU 없는 PC
```

다른 위치에 설치된 도구를 쓰려면 환경변수 `COLMAP_PATH`(colmap 실행 파일), `OPENMVS_DIR`(OpenMVS 폴더)를 지정합니다.

## 사용

```bash
python -m photogrammetry data/foot.mp4             # 영상
python -m photogrammetry data/foot_images/         # 이미지 폴더
python -m photogrammetry                           # data/ 안의 모든 영상·이미지 폴더 일괄 처리
python -m photogrammetry data/foot.mp4 --quality high --scale 1.0 -o output/foot
```

| 옵션 | 설명 |
|---|---|
| `--quality turbo/fast/normal/high` | 속도·품질 프리셋 (기본 `turbo`, 아래 표) |
| `--formats glb,stl,ply` | 출력 형식 (기본 3종, obj도 가능) |
| `--scale` | 출력 배율. SfM 결과는 절대 크기를 모르므로 실측 기준으로 보정 |
| `--interval 0.5` | 영상 프레임 추출 간격(초) |
| `--no-sharpest` | 0.5초 구간 내 최선명 프레임 대신 구간 시작 프레임 사용 |
| `--no-texture` / `--no-refine` | 텍스처 매핑 / 메쉬 정밀화 생략 |
| `--no-crop` | 촬영 대상 주변 자르기 끄기 (방·바닥 등 배경까지 출력) |
| `--crop-radius 0.5` | 자르기 반경 = 카메라-대상 거리 × 값. 대상 일부가 잘리면 키움 |
| `--keep-all-components` | 가장 큰 덩어리 외의 조각도 유지 |
| `--cpu` | COLMAP GPU 사용 안 함 |
| `--clean` | 완료 후 중간 작업 폴더 삭제 |
| `-v` | COLMAP/OpenMVS 출력까지 표시 |

### 품질 프리셋 (53초 영상 → 107프레임, i7-13700F CPU 전용 실측)

| 프리셋 | 시간 | 내용 |
|---|---|---|
| `turbo` | **2.6분** | 영상 쌍 매칭(이웃+루프+희소 전역, 전체의 ~24%), 특징점 1600px, MVS는 2장 중 1장, 기하 반복 1회, 텍스처용 메쉬 20만 면 축소 (STL은 축소 전 메쉬) |
| `fast` | 18.9분 | 전체 쌍 매칭, 전체 프레임 MVS, 정밀화 없음 |
| `normal` | 더 김 | + 메쉬 정밀화 |
| `high` | 가장 김 | 원본 해상도 깊이맵 |

`turbo` 단계별: 프레임 추출 34초 / SfM 54초 / MVS 69초 / 내보내기 2초.
영상 쌍 매칭 후 등록률이 90% 미만이면 남은 쌍을 자동으로 추가 매칭해 재복원합니다 (끊긴 구간 연결).
GPU(CUDA)가 동작하면 SfM·깊이맵이 추가로 빨라집니다 (NVIDIA 드라이버가 COLMAP 빌드의 CUDA 버전을 지원해야 함).

Python에서:

```python
from photogrammetry import reconstruct, PipelineConfig
report = reconstruct("data/foot.mp4", "output/foot", config=PipelineConfig(densify_resolution_level=0))
```

## 출력

`output/<이름>/`
- `<이름>.glb` — 텍스처 포함, glTF 규약 Y-up
- `<이름>.stl` — 형상만, Z-up, 바닥이 z=0
- `<이름>.ply` — STL과 같은 형상 + 정점 색 (조밀 점군 색을 가까운 4점 역거리 가중으로 매핑)
- `<이름>_report.json` — 등록 이미지 수, 점/면 수, 크기, 설정값, 작업 폴더 경로

중간 산출물(프레임, COLMAP DB, 희소/조밀 모델, 로그)은 작업 폴더에 남습니다. 출력 경로에 한글·공백이 있으면
COLMAP/OpenMVS 호환을 위해 작업 폴더를 `%TEMP%\pg_work\` 아래에 만듭니다 (`PG_WORK_ROOT`로 변경 가능).

## 촬영 가이드

- 대상 주위를 천천히 한 바퀴 이상 돌며 촬영 (인접 프레임 간 60~80% 겹침)
- 높이를 바꿔 2바퀴(위에서/옆에서) 찍으면 윗면·옆면이 모두 복원됨
- 흐림(모션 블러), 역광, 반사·투명·무늬 없는 표면은 매칭 실패 원인
- 대상이 아닌 배경(바닥 등)도 사진에 찍힌 만큼 함께 복원됨. 기본으로 **모든 카메라 광축이 가장 가깝게 모이는 점**(= 화면 중앙에 두고 찍은 대상)
  주변만 남기므로, 대상을 화면 가운데에 두고 찍을 것. 무늬 있는 바닥 위에서 찍으면 SfM은 안정적이며,
  바닥이 대상과 붙어 복원되면 후처리로 잘라내야 함

## 한계

- **스케일**: 단안 SfM은 절대 크기를 알 수 없음 → 크기를 아는 물체(자, 마커)를 함께 찍고 `--scale`로 보정
- **위쪽 방향**: 촬영 시 카메라를 똑바로 세웠다는 가정으로 정렬 (`--no-align`으로 끔)
- STL은 스캔 메쉬 그대로라 구멍이 있을 수 있음 (`report.json`의 `watertight`)
