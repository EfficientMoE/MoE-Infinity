"""Parse MOE_EXPERT_DROP_TRACE binary dumps into numpy arrays.

Binary layout (little-endian), produced by
ExpertDispatcher::DumpExpertDropTrace:

    magic    char[8]  = "MOEDROPT"
    version  int32
    num_experts (ne) int32
    num_layers  (nl) int32
    reserved int32
    nrec     int64
    byte_size int64[nl * ne]        # per-(layer, expert) fetch size, row-major
    records * nrec:
        seq        int64
        layer_idx  int32
        rows       int32
        residency_known uint8       # 0 => discard (bitmap not trustworthy)
        weights    float32[rows * ne]   # pre-drop post-softmax routing weights
        mask       uint8[rows * ne]     # routed (top-k) selection
        resident   uint8[ne]            # GPU-residency snapshot at dispatch
"""

from __future__ import annotations

import struct

import numpy as np

_MAGIC = b"MOEDROPT"


def parse_trace(path: str) -> dict:
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[:8] != _MAGIC:
        raise ValueError(f"{path}: bad magic {buf[:8]!r}")
    off = 8
    version, ne, nl, _reserved = struct.unpack_from("<iiii", buf, off)
    off += 16
    (nrec,) = struct.unpack_from("<q", buf, off)
    off += 8
    byte_size = (
        np.frombuffer(buf, dtype="<i8", count=nl * ne, offset=off)
        .reshape(nl, ne)
        .copy()
    )
    off += nl * ne * 8

    seq = np.empty(nrec, dtype=np.int64)
    layer = np.empty(nrec, dtype=np.int32)
    rows = np.empty(nrec, dtype=np.int32)
    res_known = np.empty(nrec, dtype=np.uint8)
    weights: list[np.ndarray] = []
    mask: list[np.ndarray] = []
    resident: list[np.ndarray] = []
    for i in range(nrec):
        s, = struct.unpack_from("<q", buf, off)
        off += 8
        ly, rw = struct.unpack_from("<ii", buf, off)
        off += 8
        rk = buf[off]
        off += 1
        cells = rw * ne
        w = np.frombuffer(buf, dtype="<f4", count=cells, offset=off).reshape(rw, ne)
        off += cells * 4
        m = np.frombuffer(buf, dtype=np.uint8, count=cells, offset=off).reshape(rw, ne)
        off += cells
        r = np.frombuffer(buf, dtype=np.uint8, count=ne, offset=off)
        off += ne
        seq[i] = s
        layer[i] = ly
        rows[i] = rw
        res_known[i] = rk
        weights.append(w.copy())
        mask.append(m.copy())
        resident.append(r.copy())
    return {
        "version": version,
        "num_experts": ne,
        "num_layers": nl,
        "byte_size": byte_size,
        "seq": seq,
        "layer": layer,
        "rows": rows,
        "residency_known": res_known,
        "weights": weights,
        "mask": mask,
        "resident": resident,
    }


def parse_trace_to_npz(path: str, npz_out: str, decode_only: bool = True) -> dict:
    """Flatten decode (rows==1, residency-known) records into dense arrays."""
    d = parse_trace(path)
    n = len(d["rows"])
    idx = [
        i
        for i in range(n)
        if (not decode_only or d["rows"][i] == 1) and d["residency_known"][i] == 1
    ]
    ne = d["num_experts"]
    if idx:
        weights = np.stack([d["weights"][i][0] for i in idx]).astype(np.float32)
        mask = np.stack([d["mask"][i][0] for i in idx]).astype(np.uint8)
        resident = np.stack([d["resident"][i] for i in idx]).astype(np.uint8)
    else:
        weights = np.zeros((0, ne), np.float32)
        mask = np.zeros((0, ne), np.uint8)
        resident = np.zeros((0, ne), np.uint8)
    layer = d["layer"][idx].astype(np.int32)
    seq = d["seq"][idx].astype(np.int64)
    out = {
        "weights": weights,
        "mask": mask,
        "resident": resident,
        "layer": layer,
        "seq": seq,
        "byte_size": d["byte_size"],
        "num_experts": np.int64(ne),
        "num_layers": np.int64(d["num_layers"]),
        "records_total": np.int64(n),
        "records_kept": np.int64(len(idx)),
    }
    np.savez_compressed(npz_out, **out)
    return out


if __name__ == "__main__":
    import sys

    info = parse_trace_to_npz(sys.argv[1], sys.argv[2])
    print(
        f"kept {int(info['records_kept'])}/{int(info['records_total'])} decode records; "
        f"ne={int(info['num_experts'])} nl={int(info['num_layers'])}"
    )
