"""외부 실행 파일(COLMAP, OpenMVS) 탐색과 실행."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
EXE = ".exe" if sys.platform == "win32" else ""


class ToolError(RuntimeError):
    pass


def find_colmap() -> Path:
    candidates = []
    if os.environ.get("COLMAP_PATH"):
        candidates.append(Path(os.environ["COLMAP_PATH"]))
    candidates += [
        TOOLS_DIR / "colmap" / "bin" / f"colmap{EXE}",
        TOOLS_DIR / "colmap" / f"colmap{EXE}",
    ]
    for c in candidates:
        if c.is_file():
            return c
    found = shutil.which("colmap")
    if found:
        return Path(found)
    raise ToolError(
        "COLMAP을 찾을 수 없습니다. `python scripts/install_tools.py`를 실행하거나 "
        "환경변수 COLMAP_PATH에 colmap 실행 파일 경로를 지정하세요."
    )


def find_openmvs_dir() -> Path:
    name = f"InterfaceCOLMAP{EXE}"
    if os.environ.get("OPENMVS_DIR"):
        d = Path(os.environ["OPENMVS_DIR"])
        if (d / name).is_file():
            return d
    if TOOLS_DIR.joinpath("openmvs").is_dir():
        for hit in TOOLS_DIR.joinpath("openmvs").rglob(name):
            return hit.parent
    found = shutil.which("InterfaceCOLMAP")
    if found:
        return Path(found).parent
    raise ToolError(
        "OpenMVS를 찾을 수 없습니다. `python scripts/install_tools.py`를 실행하거나 "
        "환경변수 OPENMVS_DIR에 OpenMVS 실행 파일 폴더를 지정하세요."
    )


def run(cmd: list, cwd: Path, log_file: Path, env_path: Path | None = None) -> None:
    """명령 실행. 출력은 로그 파일과 콘솔(DEBUG)에 기록하고 실패 시 ToolError."""
    cmd = [str(c) for c in cmd]
    env = os.environ.copy()
    if env_path is not None:
        env["PATH"] = str(env_path) + os.pathsep + env.get("PATH", "")
    log.info("$ %s", " ".join(Path(cmd[0]).name if i == 0 else c for i, c in enumerate(cmd)))
    cwd.mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            log.debug(line.rstrip())
            tail.append(line.rstrip())
            tail = tail[-30:]
        ret = proc.wait()
    if ret != 0:
        raise ToolError(
            f"{Path(cmd[0]).name} 실패 (exit {ret}). 로그: {log_file}\n" + "\n".join(tail)
        )
