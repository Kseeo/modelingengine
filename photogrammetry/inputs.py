"""입력 준비: 영상 프레임 추출, 이미지 정규화."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

try:  # HEIC (선택)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}


class InputError(ValueError):
    pass


def classify_input(path: Path) -> tuple[str, Path]:
    """('video', 파일) 또는 ('images', 폴더)."""
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTS:
            return "video", path
        raise InputError(f"지원하지 않는 파일 형식: {path.suffix}")
    if path.is_dir():
        if list_images(path):
            return "images", path
        videos = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if len(videos) == 1:
            return "video", videos[0]
        if len(videos) > 1:
            raise InputError(f"폴더에 영상이 여러 개 있습니다. 하나를 직접 지정하세요: {[v.name for v in videos]}")
        raise InputError(f"폴더에 이미지나 영상이 없습니다: {path}")
    raise InputError(f"입력 경로가 존재하지 않습니다: {path}")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) > 960:
        s = 960 / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _resize_max(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max_size > 0 and max(h, w) > max_size:
        s = max_size / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _write_jpg(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise InputError(f"JPEG 인코딩 실패: {path}")
    buf.tofile(str(path))  # 비ASCII 경로


def _open_video(video: Path) -> tuple[cv2.VideoCapture, Path | None]:
    cap = cv2.VideoCapture(str(video))
    if cap.isOpened():
        return cap, None
    # 비ASCII 경로면 임시 복사본으로 재시도
    tmp = Path(tempfile.mkdtemp(prefix="pg_video_")) / ("input" + video.suffix.lower())
    shutil.copy2(video, tmp)
    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        raise InputError(f"영상을 열 수 없습니다: {video}")
    return cap, tmp


def extract_frames(video: Path, out_dir: Path, interval: float = 0.5, pick_sharpest: bool = True,
                   max_size: int = 3200, min_sec: float = 30.0, max_sec: float = 60.0) -> list[Path]:
    """interval(초) 구간마다 1장 저장. pick_sharpest면 구간 내 최선명 프레임."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap, tmp = _open_video(video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 30.0
            log.warning("FPS 정보를 읽지 못해 30fps로 가정합니다.")
        duration = n_frames / fps if n_frames > 0 else 0.0
        log.info("영상: %s (%.1f fps, %d frames, %.1f s)", video.name, fps, n_frames, duration)
        if duration and not (min_sec - 1 <= duration <= max_sec + 1):
            log.warning("영상 길이 %.1f초는 권장 범위(%.0f~%.0f초)를 벗어납니다.", duration, min_sec, max_sec)

        saved: list[Path] = []
        cur_bin, best, best_score = -1, None, -1.0

        def flush():
            if best is not None:
                p = out_dir / f"frame_{len(saved):04d}.jpg"
                _write_jpg(p, _resize_max(best, max_size))
                saved.append(p)

        idx = 0
        while True:
            t = idx / fps
            b = int(t / interval + 1e-9)
            if not pick_sharpest and b == cur_bin:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if b != cur_bin:
                flush()
                cur_bin, best, best_score = b, None, -1.0
            if pick_sharpest:
                score = _sharpness(frame)
                if score > best_score:
                    best, best_score = frame, score
            elif best is None:
                best = frame
            idx += 1
        flush()
    finally:
        cap.release()
        if tmp is not None:
            shutil.rmtree(tmp.parent, ignore_errors=True)
    log.info("프레임 %d장 추출 (%.1f초 간격)", len(saved), interval)
    return saved


def prepare_images(src_dir: Path, out_dir: Path, max_size: int = 3200) -> tuple[list[Path], bool]:
    """EXIF 회전·크기 제한 후 복사. (경로 목록, 해상도 동일 여부) 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = list_images(src_dir)
    saved, sizes = [], set()
    for i, src in enumerate(srcs):
        try:
            im = Image.open(src)
            im = ImageOps.exif_transpose(im)
        except Exception as e:  # noqa: BLE001
            log.warning("이미지 로드 실패, 건너뜀: %s (%s)", src.name, e)
            continue
        exif = im.getexif()
        im = im.convert("RGB")
        if max_size > 0 and max(im.size) > max_size:
            im.thumbnail((max_size, max_size), Image.LANCZOS)
        sizes.add(im.size)
        p = out_dir / f"img_{i:04d}.jpg"
        im.save(p, "JPEG", quality=95, exif=exif.tobytes())
        saved.append(p)
    log.info("이미지 %d장 준비 완료", len(saved))
    return saved, len(sizes) == 1
