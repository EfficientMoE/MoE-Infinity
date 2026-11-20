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
* That failure must not publish ``IDLE`` on the stuck node: its previous
  retirement is still outstanding (a slow but live GPU, not only a lost event)
  and only that retirement may release the node, otherwise a new dispatch's
  claim could be overwritten by the late retirement.
* After a failed dispatch whose queued items are dropped as stale, the
  retirement count returns to zero and the dispatcher destructs.

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


def _scenario_deadline_slow_live():
    """Deadline failure through the routing worker (dispatch_experts), then the
    stuck expert's own retirement completes late."""
    torch, grouped, gen, engine, dispatcher, experts = _setup()
    dispatcher.set_grouped_decode(False, 4)
    x = torch.randn(1, HIDDEN, generator=gen).bfloat16().cuda()
    all_experts = _mask(torch, list(range(NUM_EXPERTS)))
    grouped._dispatch(  # make every expert resident
        dispatcher, x, all_experts, all_experts.float() * 0.5, native=True
    )
    dispatcher._set_enqueue_cas_deadline_ms_for_test(300)
    state = dispatcher._expert_exec_state_for_test
    result = {"state_before": state(0, 2)}
    dispatcher._stick_expert_for_test(0, 2, True)  # EXECUTING, retire pending
    result["state_stuck"] = state(0, 2)

    # Experts 1 and 3 are claimable, expert 2 is stuck: the routing worker
    # claims 1, hits the deadline on 2 and fails the dispatch.
    mask = _mask(torch, [1, 2, 3])
    try:
        _raw_dispatch(dispatcher, x, mask)
        result["first_failed"] = False
    except RuntimeError as exc:
        result["first_failed"] = "expert 2" in str(exc)
    time.sleep(0.3)
    torch.cuda.synchronize()
    # The failure must not have published IDLE on the stuck node.
    result["state_after_failure"] = state(0, 2)

    # While the old retirement is still outstanding nobody may take the node:
    # a new dispatch of expert 2 runs into the deadline again.
    try:
        _raw_dispatch(dispatcher, x, _mask(torch, [2]))
        result["second_failed"] = False
    except RuntimeError as exc:
        result["second_failed"] = "expert 2" in str(exc)
    time.sleep(0.3)
    result["state_after_second"] = state(0, 2)
    result["retire_pending_while_stuck"] = dispatcher.get_routing_stats()[
        "pending_retirement_callbacks"
    ]

    # The old retirement completes late: it publishes IDLE itself, once.
    dispatcher._stick_expert_for_test(0, 2, False)
    result["state_after_late_retire"] = state(0, 2)
    result["retire_pending_after"] = dispatcher.get_routing_stats()[
        "pending_retirement_callbacks"
    ]

    # Now expert 2 is usable, and it retires through its own retirement.
    time.sleep(0.3)
    mask2 = _mask(torch, [2, 3])
    out = _raw_dispatch(dispatcher, x, mask2)
    ref = grouped._per_expert_reference(x, mask2, mask2.float() * 0.5, experts)
    result["recovered_err"] = float(
        (out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
    )
    deadline = time.time() + 5
    while time.time() < deadline and state(0, 2) != 0:
        time.sleep(0.01)
    result["state_final"] = state(0, 2)
    _finish(result)


def _scenario_failed_dispatch_drops_stale_items_then_destructs():
    torch, grouped, gen, engine, dispatcher, experts = _setup()
    dispatcher.set_grouped_decode(False, NUM_EXPERTS)
    x = torch.randn(1, HIDDEN, generator=gen).bfloat16().cuda()
    everyone = _mask(torch, list(range(NUM_EXPERTS)))
    grouped._dispatch(  # make every expert resident
        dispatcher, x, everyone, everyone.float() * 0.5, native=True
    )
    dispatcher._set_dispatch_fault_for_test("exec")
    try:
        _raw_dispatch(dispatcher, x, everyone)
        failed = False
    except RuntimeError as exc:
        failed = "injected exec failure" in str(exc)
    deadline = time.time() + 10
    stats = dispatcher.get_routing_stats()
    while time.time() < deadline and (
        stats["pending_retirement_callbacks"] != 0
    ):
        time.sleep(0.01)
        stats = dispatcher.get_routing_stats()
    time.sleep(0.2)
    stats = dispatcher.get_routing_stats()
    # Every expert is claimable again (dropped items were restored, executed
    # ones retired through the event-polling retirement).
    states = [dispatcher._expert_exec_state_for_test(0, e) for e in range(8)]
    out = grouped._dispatch(
        dispatcher, x, everyone, everyone.float() * 0.5, native=True
    )
    ref = grouped._per_expert_reference(
        x, everyone, everyone.float() * 0.5, experts
    )
    err = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
    before_teardown = {
        "failed": failed,
        "dropped": stats["stale_exec_items_dropped"],
        "pending_retirement_callbacks": stats["pending_retirement_callbacks"],
        "pending": stats["pending"],
        "states": states,
        "recovered_err": err,
    }
    # Destruct for real: the destructor waits for the retirement count and
    # joins its workers; a leaked count would hang here (the test's timeout).
    del dispatcher
    del engine
    import gc

    gc.collect()
    before_teardown["destructed"] = True
    _finish(before_teardown)


SCENARIOS = {
    "deadline_slow_live": _scenario_deadline_slow_live,
    "failed_dispatch_drops_stale_then_destructs": (
        _scenario_failed_dispatch_drops_stale_items_then_destructs
    ),
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


@requires_cuda
def test_deadline_failure_does_not_publish_idle_on_a_node_with_a_pending_retire():
    result = _run("deadline_slow_live")
    assert result["state_before"] == 0, result
    assert result["state_stuck"] == 2, result
    assert result["first_failed"], result
    # The failure leaves the node to its own retirement.
    assert result["state_after_failure"] == 2, result
    assert result["second_failed"], result  # nobody could take the node
    assert result["state_after_second"] == 2, result
    assert result["retire_pending_while_stuck"] == 1, result
    # The late retirement publishes IDLE once and the count returns to 0.
    assert result["state_after_late_retire"] == 0, result
    assert result["retire_pending_after"] == 0, result
    assert result["recovered_err"] < 0.05, result
    assert result["state_final"] == 0, result


@requires_cuda
def test_failed_dispatch_with_dropped_stale_items_settles_and_destructs():
    result = _run("failed_dispatch_drops_stale_then_destructs")
    assert result["failed"], result
    assert result["dropped"] >= 1, result  # stale items were really dropped
    assert result["pending_retirement_callbacks"] == 0, result
    assert result["pending"] == 0, result
    assert result["states"] == [0] * NUM_EXPERTS, result
    assert result["recovered_err"] < 0.05, result
    assert result["destructed"], result


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    SCENARIOS[sys.argv[1]]()
