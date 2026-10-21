"""CLI: python -m photogrammetry [영상|이미지 폴더] (생략 시 data/ 일괄 처리)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import QUALITY_PRESETS, PipelineConfig
from .inputs import VIDEO_EXTS, list_images
from .pipeline import reconstruct
from .tools import PROJECT_ROOT



def _batch_inputs(data_dir: Path) -> list[Path]:
    items = []
    for p in sorted(data_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            items.append(p)
        elif p.is_dir() and list_images(p):
            items.append(p)
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="photogrammetry", description="2D 영상/이미지 → 3D GLB/STL (COLMAP SfM + OpenMVS)")
    ap.add_argument("input", nargs="?", help="영상 파일 또는 이미지 폴더 (생략 시 data/ 일괄 처리)")
    ap.add_argument("-o", "--output", default=str(PROJECT_ROOT / "output"), help="출력 폴더 (기본: output/)")
    ap.add_argument("--name", help="출력 파일 이름 (기본: 입력 이름)")
    ap.add_argument("--quality", choices=QUALITY_PRESETS, default="turbo", help="기본 turbo")
    ap.add_argument("--formats", default="glb,stl,ply", help="쉼표 구분: glb,stl,ply(정점 색),obj")
    ap.add_argument("--scale", type=float, default=1.0, help="출력 배율 (SfM 결과는 임의 스케일)")
    ap.add_argument("-v", "--verbose", action="store_true", help="외부 도구 출력까지 표시")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("trimesh").setLevel(logging.WARNING)

    cfg = PipelineConfig.preset(a.quality)
    cfg.formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    cfg.scale = a.scale

    if a.input:
        targets = [Path(a.input)]
    else:
        data = PROJECT_ROOT / "data"
        targets = _batch_inputs(data)
        if not targets:
            print(f"{data} 에 처리할 영상이나 이미지 폴더가 없습니다.", file=sys.stderr)
            return 1

    failed = 0
    out_root = Path(a.output)
    for t in targets:
        out = out_root if a.input else out_root / t.stem
        try:
            reconstruct(t, out, name=a.name if a.input else None, config=cfg)
        except Exception as e:  # noqa: BLE001
            failed += 1
            logging.error("실패: %s\n%s", t, e)
            if a.verbose:
                logging.exception(e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
