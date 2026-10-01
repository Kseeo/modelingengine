"""OpenMVS 조밀 점군 PLY 읽기/자르기.

OpenMVS는 정점마다 가변 길이 리스트(view_indices, view_weights)를 붙인다. trimesh는 이 형식을 읽지 못하고,
ReconstructMesh는 이 가시성 정보를 사용하므로 자를 때도 레코드를 바이트 그대로 보존해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PLY_TYPES = {"int8": "i1", "char": "i1", "uint8": "u1", "uchar": "u1", "int16": "i2", "short": "i2",
              "uint16": "u2", "ushort": "u2", "int32": "i4", "int": "i4", "uint32": "u4", "uint": "u4",
              "float32": "f4", "float": "f4", "float64": "f8", "double": "f8"}


@dataclass
class VertexPly:
    header: list[str]       # end_header 포함 헤더 줄
    data: np.ndarray        # 헤더 뒤 바이트 (uint8)
    starts: np.ndarray      # 각 정점 레코드 시작 오프셋
    ends: np.ndarray        # 각 정점 레코드 끝 오프셋
    fixed: np.ndarray       # 고정 길이 스칼라 속성 (structured array)


def read_vertex_ply(path: Path) -> VertexPly:
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii").splitlines()
    if "binary_little_endian" not in " ".join(header):
        raise ValueError(f"binary_little_endian PLY만 지원: {path}")
    n, scalars, lists, in_vertex, other = 0, [], [], False, False
    for line in header:
        tok = line.split()
        if tok[:2] == ["element", "vertex"]:
            n, in_vertex = int(tok[2]), True
        elif tok[:1] == ["element"]:
            in_vertex, other = False, True
        elif in_vertex and tok[:1] == ["property"]:
            if tok[1] == "list":
                lists.append((np.dtype(_PLY_TYPES[tok[2]]).itemsize, np.dtype(_PLY_TYPES[tok[3]]).itemsize))
            else:
                if lists:
                    raise ValueError("리스트 뒤의 스칼라 속성은 지원하지 않음")
                scalars.append((tok[2], "<" + _PLY_TYPES[tok[1]]))
    if other:
        raise ValueError("vertex 외 element가 있는 PLY는 지원하지 않음")
    fixed_dt = np.dtype(scalars)
    data = np.frombuffer(raw, np.uint8, offset=end)
    starts = np.empty(n, np.int64)
    ends = np.empty(n, np.int64)
    if not lists:
        starts[:] = np.arange(n) * fixed_dt.itemsize
        ends[:] = starts + fixed_dt.itemsize
    else:
        off = 0  # 가변 길이라 레코드 경계를 순차 계산
        for i in range(n):
            starts[i] = off
            off += fixed_dt.itemsize
            for count_size, item_size in lists:
                c = int.from_bytes(data[off:off + count_size].tobytes(), "little")
                off += count_size + c * item_size
            ends[i] = off
    fixed = data[starts[:, None] + np.arange(fixed_dt.itemsize)].copy().view(fixed_dt).ravel()
    return VertexPly(header, data, starts, ends, fixed)


def points_and_colors(ply: VertexPly) -> tuple[np.ndarray, np.ndarray]:
    f = ply.fixed
    xyz = np.column_stack([f["x"], f["y"], f["z"]]).astype(np.float64)
    rgb = np.column_stack([f["red"], f["green"], f["blue"]]).astype(np.uint8)
    return xyz, rgb


def write_vertex_subset(ply: VertexPly, keep: np.ndarray, dst: Path) -> int:
    """keep(bool 마스크)에 해당하는 정점 레코드만 바이트 그대로 저장. 저장한 정점 수 반환."""
    idx = np.flatnonzero(keep)
    header = [f"element vertex {len(idx)}" if ln.startswith("element vertex") else ln for ln in ply.header]
    lengths = ply.ends[idx] - ply.starts[idx]
    # 레코드별 바이트 범위를 한 번에 모으기 위한 인덱스
    offsets = np.repeat(ply.starts[idx] - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
    body = ply.data[np.arange(lengths.sum()) + offsets]
    with open(dst, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(body.tobytes())
    return len(idx)
