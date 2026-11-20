# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Dispatcher hardening around retirement (#353 review).

* A retirement push that finds ``retirement_queue_`` closed used to be dropped
  silently by ``ThreadSafeQueue::Push``: the retire event leaked and
  ``pending_retirement_callbacks_`` stayed above zero (the destructor would then
  wait on it forever). The failure must now surface as a dispatch error and the
  count must be restored.
* The CAS retry loop in ``Enqueue()`` waits for the retirement path to publish
  ``IDLE``. It must give up after a deadline and fail the dispatch, naming the
  expert and the state it was stuck in.

Each scenario runs in a subprocess (and ends with ``os._exit``) so that a
regression that hangs or aborts fails the test instead of the session.
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
HIDDEN = 256
INTER = 64


def _setup():
    import torch

    from tests.python.ops import test_grouped_moe_decode_pro6000 as grouped

    torch.cuda.set_device(0)
    gen = torch.Generator().manual_seed(3)
    engine, dispatcher, experts = grouped._make_dispatcher(
        Path(tempfile.mkdtemp()), NUM_EXPERTS, HIDDEN, INTER, gen
    )
    # The dispatcher uses globals owned by the engine: the caller keeps it.
    return torch, grouped, gen, engine, dispatcher, experts


def _mask(torch, active):
    m = torch.zeros(1, NUM_EXPERTS, dtype=torch.bool)
    m[:, active] = True
    return m.cuda()


def _finish(result):
    """Report, then leave without running destructors: the dispatcher
    destructor waits for pending_retirement_callbacks_ == 0 and would hang the
    scenario on a tree that leaves it stuck."""
    print("\nRESULT " + json.dumps(result), flush=True)
    sys.stdout.flush()
    os._exit(0)


def _raw_dispatch(dispatcher, x, mask):
    dispatcher.set_inputs(x, mask, mask.float() * 0.5)
    dispatcher.dispatch_experts(0)
    return dispatcher.wait_expert().clone()


def _scenario_closed_retirement(grouped_decode):
    torch, grouped, gen, engine, dispatcher, experts = _setup()
    dispatcher.set_grouped_decode(grouped_decode, 4)
    x = torch.randn(1, HIDDEN, generator=gen).bfloat16().cuda()
    mask = _mask(torch, [0, 1, 2, 3])
    for _ in range(2):  # fetch, then resident (grouped path when enabled)
        grouped._dispatch(dispatcher, x, mask, mask.float() * 0.5, native=True)
    if grouped_decode:
        assert dispatcher.get_routing_stats()["grouped_batches"] >= 1
    dispatcher._close_retirement_queue_for_test()
    time.sleep(0.3)  # the retirement thread exits on a closed, empty queue
    raised = False
    try:
        _raw_dispatch(dispatcher, x, mask)
    except RuntimeError as exc:
        raised = str(exc)
    time.sleep(0.3)
    torch.cuda.synchronize()
    stats = dispatcher.get_routing_stats()
    _finish(
        {
            "raised": raised,
            "pending_retirement_callbacks": stats[
                "pending_retirement_callbacks"
            ],
            "pending": stats["pending"],
        }
    )


def _scenario_cas_deadline():
    torch, grouped, gen, engine, dispatcher, experts = _setup()
    dispatcher.set_grouped_decode(False, 4)
    x = torch.randn(1, HIDDEN, generator=gen).bfloat16().cuda()
    all_experts = _mask(torch, list(range(NUM_EXPERTS)))
    grouped._dispatch(  # make every expert resident
        dispatcher, x, all_experts, all_experts.float() * 0.5, native=True
    )
    dispatcher._set_enqueue_cas_deadline_ms_for_test(300)
    dispatcher._stick_expert_for_test(0, 2, True)  # EXECUTING, never retires
    mask = _mask(torch, [1, 2])
    dispatcher.set_inputs(x, mask, mask.float() * 0.5)
    dispatcher.set_expected_queue(2)
    dispatcher.enqueue_expert(0, 1, 0, False)
    started = time.time()
    raised = False
    try:
        dispatcher.enqueue_expert(0, 2, 0, False)
    except RuntimeError as exc:
        raised = str(exc)
    elapsed = time.time() - started
    waited = False
    try:
        dispatcher.wait_expert()
    except RuntimeError as exc:
        waited = str(exc)
    dispatcher._stick_expert_for_test(0, 2, False)
    stats = dispatcher.get_routing_stats()
    # The dispatcher is usable again (experts 4, 5 were not involved).
    time.sleep(0.3)
    mask2 = _mask(torch, [4, 5])
    out = _raw_dispatch(dispatcher, x, mask2)
    ref = grouped._per_expert_reference(x, mask2, mask2.float() * 0.5, experts)
    err = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
    _finish(
        {
            "raised": raised,
            "waited": waited,
            "elapsed": elapsed,
            "pending_retirement_callbacks": stats[
                "pending_retirement_callbacks"
            ],
            "recovered_err": err,
        }
    )


SCENARIOS = {
    "cas_deadline": _scenario_cas_deadline,
    "closed_retirement_per_expert": lambda: _scenario_closed_retirement(False),
    "closed_retirement_grouped": lambda: _scenario_closed_retirement(True),
}


def _run(name, timeout=90):
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), name],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"scenario {name} did not finish within {timeout}s (hang)")
    out = proc.stdout + proc.stderr
    lines = [
        ln[ln.index("RESULT ") + 7 :]
        for ln in proc.stdout.splitlines()
        if "RESULT {" in ln
    ]
    assert proc.returncode == 0 and lines, (
        f"scenario {name} exited {proc.returncode}; tail:\n" + out[-2500:]
    )
    return json.loads(lines[-1])


@requires_cuda
@pytest.mark.parametrize("variant", ["per_expert", "grouped"])
def test_retirement_push_to_closed_queue_fails_the_dispatch(variant):
    result = _run(f"closed_retirement_{variant}")
    assert result["raised"], result  # the failure is reported, not dropped
    assert "retirement queue is closed" in result["raised"], result
    assert result["pending_retirement_callbacks"] == 0, result
    assert result["pending"] == 0, result


@requires_cuda
def test_enqueue_cas_retry_is_bounded_and_fails_the_dispatch():
    result = _run("cas_deadline")
    msg = result["raised"]
    assert msg, result  # Enqueue gave up instead of spinning forever
    for needle in ("expert 2", "layer 0", "EXECUTING"):
        assert needle in msg, result
    assert 0.25 <= result["elapsed"] < 10, result
    assert result["waited"] and "expert 2" in result["waited"], result
    assert result["pending_retirement_callbacks"] == 0, result
    assert result["recovered_err"] < 0.05, result


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    SCENARIOS[sys.argv[1]]()
