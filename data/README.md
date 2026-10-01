# 입력 데이터

여기에 넣으면 `python -m photogrammetry` (인자 없이) 실행 시 일괄 처리됩니다.

- 영상: `data/foot01.mp4` — 30~60초, 0.5초당 1장 추출 (60~120장)
- 이미지: `data/foot02/` 폴더에 20장 이상 (jpg/png/…)

결과는 `output/<이름>/<이름>.glb`, `.stl`, `_report.json` 으로 저장됩니다.
