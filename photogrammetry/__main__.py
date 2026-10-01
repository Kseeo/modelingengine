"""CLI.

  python -m photogrammetry data/foot.mp4            # 영상 1개
  python -m photogrammetry data/foot_images/        # 이미지 폴더
  python -m photogrammetry                          # data/ 안의 모든 영상·이미지 폴더 일괄 처리
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .inputs import VIDEO_EXTS, list_images
from .pipeline import reconstruct
from .tools import PROJECT_ROOT

QUALITY = {
    # CPU에서도 빠르게: 영상 쌍 매칭, 특징점 1600px, MVS는 2장 중 1장, 기하 반복 1회, 텍스처용 메쉬 축소
    "turbo": dict(feature_max_image_size=1600, video_pair_matching=True, mvs_view_stride=2,
                  densify_resolution_level=2, densify_iters=2, densify_geometric_iters=1,
                  refine_mesh=False, texture_resolution_level=0, texture_max_faces=200_000,
                  texture_smoothness=1.0),
    "fast": dict(densify_resolution_level=2, refine_mesh=False, texture_resolution_level=1),
    "normal": dict(densify_resolution_level=1, refine_mesh=True, refine_resolution_level=1),
    "high": dict(densify_resolution_level=0, densify_max_resolution=4096, refine_mesh=True,
                 refine_resolution_level=0),
}


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
    ap.add_argument("--quality", choices=QUALITY, default="turbo", help="기본 turbo")
    ap.add_argument("--formats", default="glb,stl,ply", help="쉼표 구분: glb,stl,ply(정점 색),obj")
    ap.add_argument("--scale", type=float, default=1.0, help="출력 배율 (SfM 결과는 임의 스케일)")
    ap.add_argument("--interval", type=float, default=0.5, help="영상 프레임 추출 간격(초)")
    ap.add_argument("--no-sharpest", action="store_true", help="구간 내 최선명 프레임 선택 대신 구간 시작 프레임 사용")
    ap.add_argument("--no-texture", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--keep-all-components", action="store_true", help="떠다니는 조각을 제거하지 않음")
    ap.add_argument("--object-centric", action="store_true",
                    help="대상 주변 특징점만으로 포즈 재추정 (촬영 중 발이 움직인 영상 보정, SfM 1회 추가)")
    ap.add_argument("--no-crop", action="store_true", help="촬영 대상 주변 자르기 끄기 (배경까지 모두 출력)")
    ap.add_argument("--crop-radius", type=float, default=0.5,
                    help="자르기 반경 = 카메라-대상 거리 × 이 값 (기본 0.5, 대상이 잘리면 키움)")
    ap.add_argument("--no-align", action="store_true", help="위쪽 방향 자동 정렬 끄기")
    ap.add_argument("--cpu", action="store_true", help="COLMAP GPU 사용 안 함")
    ap.add_argument("--clean", action="store_true", help="완료 후 중간 작업 폴더 삭제")
    ap.add_argument("-v", "--verbose", action="store_true", help="외부 도구 출력까지 표시")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("trimesh").setLevel(logging.WARNING)

    cfg = PipelineConfig(**QUALITY[a.quality])
    cfg.formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    cfg.scale = a.scale
    cfg.frame_interval_sec = a.interval
    cfg.pick_sharpest = not a.no_sharpest
    cfg.texture = not a.no_texture
    if a.no_refine:
        cfg.refine_mesh = False
    cfg.keep_largest_component = not a.keep_all_components
    cfg.object_centric = cfg.object_centric or a.object_centric
    cfg.crop_to_focus = not a.no_crop
    cfg.crop_radius_factor = a.crop_radius
    cfg.align_up = not a.no_align
    cfg.use_gpu = not a.cpu
    cfg.keep_workspace = not a.clean

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
