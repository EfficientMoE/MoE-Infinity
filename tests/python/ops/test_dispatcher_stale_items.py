# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Stale work from a failed dispatch must not run in a later dispatch.

``FailDispatch`` zeroes ``pending_`` while items of the failed generation can
still sit in ``exec_queue_``. A worker has to drop such an item before it waits
on the (re-recorded) input-ready event or reads the next dispatch's tensors,
and it has to give the expert node back, otherwise the next dispatch of that
expert dies in the ``exec_state`` CAS of ``Enqueue``.

The scenario runs in a subprocess: on a tree without the drop the second
dispatch of the same experts aborts the process (fatal log), which must fail
this test rather than take the test session down.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from tests.python.ops.conftest import requires_cuda

REPO_ROOT = Path(__file__).resolve().parents[3]
NUM_EXPERTS = 8
FAILED_EXPERTS = 6  # experts 0..5 are dispatched; the first one is failed
HIDDEN = 256
INTER = 64
DTYPE_BF16 = 0
DEEPSEEK_EXPERT_TYPE = 5


def _scenario():
    import torch

    from moe_infinity.runtime.model_offload import _load_prefetch_lib
    from tests.python.ops.test_grouped_moe_decode_pro6000 import (
        _per_expert_reference,
    )

    torch.cuda.set_device(0)
    prefetch_lib = _load_prefetch_lib()
    tmp = tempfile.mkdtemp()
    gen = torch.Generator().manual_seed(3)
    engine = prefetch_lib.prefetch_handle(f"{tmp}/", 0.5)
    experts, ids_per_expert = {}, []
    for e in range(NUM_EXPERTS):
        weights = tuple(
            (torch.randn(*s, generator=gen) * 0.05).bfloat16()
            for s in ((INTER, HIDDEN), (INTER, HIDDEN), (HIDDEN, INTER))
        )
        ids = [3 * e, 3 * e + 1, 3 * e + 2]
        for tensor_id, tensor in zip(ids, weights):
            engine.offload(tensor, tensor_id)
        experts[e] = tuple(t.cuda() for t in weights)
        ids_per_expert.append(ids)
    dense = list(
        range(3 * NUM_EXPERTS, 3 * NUM_EXPERTS + 2 * torch.cuda.device_count())
    )
    for tensor_id in dense:
        engine.offload(torch.zeros(1, dtype=torch.bfloat16), tensor_id)
    split = len(dense) // 2
    engine.set_topology(
        [(f"model.dense.{i}", [[t]]) for i, t in enumerate(dense[:split])]
        + [("model.layers.0.mlp.experts", ids_per_expert)]
        + [
            (f"model.dense.{i + split}", [[t]])
            for i, t in enumerate(dense[split:])
        ]
    )
    # One exec worker per device: the items behind the failing one are still
    # queued when the failure happens.
    dispatcher = prefetch_lib.expert_dispatcher(
        NUM_EXPERTS, 1, DTYPE_BF16, DEEPSEEK_EXPERT_TYPE, 1
    )
    for e, ids in enumerate(ids_per_expert):
        dispatcher.register_expert(0, e, ids, "")
    dispatcher.set_grouped_decode(False, 2)

    def mask_of(active):
        m = torch.zeros(1, NUM_EXPERTS, dtype=torch.bool)
        m[:, active] = True
        return m.cuda()

    def raw_dispatch(x, mask):
        weights = mask.float() * 0.5
        dispatcher.set_inputs(x, mask, weights)
        dispatcher.dispatch_experts(0)
        return dispatcher.wait_expert().clone(), weights

    def settle():
        torch.cuda.synchronize()
        time.sleep(0.3)  # retirement callbacks return nodes to IDLE

    def err(out, mask, weights, x):
        ref = _per_expert_reference(x, mask, weights, experts)
        return float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6))

    x = torch.randn(1, HIDDEN, generator=gen).bfloat16().cuda()
    mask_a = mask_of(list(range(FAILED_EXPERTS)))
    mask_b = mask_of([6, 7])
    # Make every expert resident (cold fetches take the per-expert path).
    raw_dispatch(x, mask_of(list(range(NUM_EXPERTS))))
    settle()

    result = {}
    dispatcher._set_dispatch_fault_for_test("exec")
    try:
        raw_dispatch(x, mask_a)
        result["failed_dispatch_raised"] = False
    except RuntimeError as exc:
        result["failed_dispatch_raised"] = "injected exec failure" in str(exc)
    # Start the next dispatch right away, with different experts and new
    # tensors, while the failed one's items are still queued.
    x2 = torch.randn(1, HIDDEN, generator=gen).bfloat16().cuda()
    out_b, w_b = raw_dispatch(x2, mask_b)
    result["next_dispatch_err"] = err(out_b, mask_b, w_b, x2)
    settle()
    stats = dispatcher.get_routing_stats()
    result["stale_exec_items_dropped"] = stats.get(
        "stale_exec_items_dropped", -1
    )
    result["pending"] = stats["pending"]
    result["num_devices"] = torch.cuda.device_count()
    # The failed dispatch's experts must be usable again (their nodes were
    # given back).
    out_c, w_c = raw_dispatch(x, mask_a)
    settle()
    result["same_experts_err"] = err(out_c, mask_a, w_c, x)
    print("\nRESULT " + json.dumps(result), flush=True)
    del dispatcher
    import gc

    gc.collect()


@requires_cuda
def test_stale_items_are_dropped_and_experts_recover():
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "scenario"],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:  # native logs share stdout and can precede the marker
        lines = [
            ln[ln.index("RESULT ") :]
            for ln in proc.stdout.splitlines()
            if "RESULT {" in ln
        ]
    assert proc.returncode == 0 and lines, (
        f"scenario exited {proc.returncode}; tail:\n" + out[-2500:]
    )
    result = json.loads(lines[-1][len("RESULT ") :])
    assert result["failed_dispatch_raised"], result
    # Every one of the FAILED_EXPERTS - 1 items queued behind the failed one
    # is accounted for: dropped, or legitimately executed because its worker
    # had already taken it when the failure was published. There is one exec
    # worker per device, and the worker running the failed item is busy
    # failing, so at most (devices - 1) items can have been taken that way
    # (none on one device: all 5 are dropped). Everything else must be
    # dropped, not executed.
    queued_behind = FAILED_EXPERTS - 1
    dropped = result["stale_exec_items_dropped"]
    assert queued_behind - (result["num_devices"] - 1) <= dropped, result
    assert dropped <= queued_behind, result
    assert result["pending"] == 0, result
    assert result["next_dispatch_err"] < 0.05, result
    assert result["same_experts_err"] < 0.05, result


if __name__ == "__main__":
    if sys.argv[1:] == ["scenario"]:
        sys.path.insert(0, str(REPO_ROOT))
        _scenario()
