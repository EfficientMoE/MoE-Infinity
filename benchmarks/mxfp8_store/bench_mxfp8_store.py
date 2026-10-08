#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Manual-result host-store microbenchmark for the MXFP8 expert store (#221 p5).

This CLI never downloads weights. It applies the pure host-byte gate
(:func:`gates.host_ratio_passes`) to numbers you measured on the rig and
prints the informational onboarding row (:func:`gates.onboarding_row`).

Manual rig run (>= 1TB RAM + NVMe, one GPU), pinned to the recorded revision
``MiniMaxAI/MiniMax-M3-MXFP8@c5454eb03678d8710e54a4e0fc681b9f3b4a3dba``:

  1. Construct ``MoE(checkpoint, config)`` once through the Phase 4 BF16 path
     and once through MXFP8, timing each constructor with
     ``time.perf_counter()`` -> ``seconds_bf16`` / ``seconds_mxfp8``.
  2. For each build, measure routed host-store bytes with
     :func:`measure_routed_host_bytes` (below): it joins the native
     ``get_canonical_tensor_index_snapshot()`` rows to ``engine.name_id_map``
     and sums ``size`` only for names that ``parse_expert_id`` maps to a
     routed ``(layer, expert)`` -- stored routed tensors, not RSS or GPU
     residency.
  3. Feed the four numbers back here, e.g.::

        python benchmarks/mxfp8_store/bench_mxfp8_store.py \\
            --enforce --host-bytes-mxfp8 <MX> --host-bytes-bf16 <BF> \\
            --n-experts <E> --seconds-mxfp8 <SMX> --seconds-bf16 <SBF>

``--enforce`` prints ``PASS host_ratio`` / ``FAIL host_ratio`` and exits 0
only when the 0.6x gate holds. ``--help`` exits 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.mxfp8_store.gates import (  # noqa: E402
    host_ratio_passes,
    onboarding_row,
)


def measure_routed_host_bytes(model) -> int:
    """Sum stored host-store bytes over routed-expert tensors of a built model.

    Rig-only: joins the native canonical tensor-index snapshot
    (``tensor_id``, ``size``) to ``engine.name_id_map`` and keeps names that
    ``parse_expert_id`` maps to a routed ``(layer, expert)``.
    """
    from moe_infinity.utils import parse_expert_id

    engine = getattr(model, "engine", model)
    snapshot = engine.archer_engine.get_canonical_tensor_index_snapshot()
    size_by_id: dict[int, int] = {}
    for row in snapshot:
        tensor_id = getattr(row, "tensor_id", None)
        size = getattr(row, "size", None)
        if tensor_id is None and isinstance(row, (tuple, list)):
            tensor_id, size = row[0], row[1]
        if tensor_id is not None and size is not None:
            size_by_id[int(tensor_id)] = int(size)

    total = 0
    for name, tensor_id in engine.name_id_map.items():
        layer_id, expert_id = parse_expert_id(name, engine.config)
        if layer_id is None or expert_id is None:
            continue
        if int(tensor_id) in size_by_id:
            total += size_by_id[int(tensor_id)]
    return total


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("--host-bytes-mxfp8", type=float, default=None)
    parser.add_argument("--host-bytes-bf16", type=float, default=None)
    parser.add_argument("--n-experts", type=int, default=None)
    parser.add_argument("--seconds-mxfp8", type=float, default=None)
    parser.add_argument("--seconds-bf16", type=float, default=None)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="assert the 0.6x host-store gate; exit non-zero on failure",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if (
        args.n_experts is not None
        and args.seconds_mxfp8 is not None
        and args.seconds_bf16 is not None
    ):
        row = onboarding_row(
            args.n_experts, args.seconds_mxfp8, args.seconds_bf16
        )
        print("onboarding_row:", row)

    if args.enforce:
        if args.host_bytes_mxfp8 is None or args.host_bytes_bf16 is None:
            print(
                "--enforce requires --host-bytes-mxfp8 and --host-bytes-bf16",
                file=sys.stderr,
            )
            return 2
        passed = host_ratio_passes(args.host_bytes_mxfp8, args.host_bytes_bf16)
        print("PASS host_ratio" if passed else "FAIL host_ratio")
        return 0 if passed else 1

    if args.host_bytes_mxfp8 is not None and args.host_bytes_bf16 is not None:
        ratio = args.host_bytes_mxfp8 / args.host_bytes_bf16
        print(f"host_ratio={ratio:.4f} (gate <= 0.6, not enforced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
