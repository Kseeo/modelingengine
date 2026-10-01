"""COLMAP / OpenMVS Windows 바이너리를 tools/ 에 설치.

  python scripts/install_tools.py            # CUDA 버전 COLMAP
  python scripts/install_tools.py --nocuda   # GPU 없는 PC
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
COLMAP_VER = "4.2.0"
OPENMVS_VER = "v2.4.0"


def download(url: str, dst: Path) -> None:
    print(f"다운로드: {url}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def install(name: str, url: str, force: bool) -> None:
    target = TOOLS / name
    if target.exists() and not force:
        print(f"{name}: 이미 설치됨 ({target}) - 재설치는 --force")
        return
    TOOLS.mkdir(exist_ok=True)
    zpath = TOOLS / f"{name}.zip"
    download(url, zpath)
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target)
    zpath.unlink()
    print(f"{name}: 설치 완료 → {target}")


def main() -> int:
    if sys.platform != "win32":
        print("Windows 전용 스크립트입니다. Linux/macOS는 패키지 매니저로 colmap, openmvs를 설치하고 "
              "COLMAP_PATH / OPENMVS_DIR 환경변수를 지정하세요.")
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--nocuda", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    flavor = "nocuda" if a.nocuda else "cuda"
    install("colmap", f"https://github.com/colmap/colmap/releases/download/{COLMAP_VER}/colmap-x64-windows-{flavor}.zip", a.force)
    install("openmvs", f"https://github.com/cdcseacave/openMVS/releases/download/{OPENMVS_VER}/OpenMVS_Windows_x64.zip", a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
