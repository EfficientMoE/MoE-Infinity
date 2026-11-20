"""DistributedExpertExecutor.dispatch calls the dispatcher with its real arity.

The binding ``expert_dispatcher.set_inputs(hidden_states, router_mask,
router_weight)`` takes three arguments (``ExpertDispatcher::SetInputs``); the
legacy ``dispatch`` passed two. The fake dispatcher in test_expert_dispatch.py
accepts whatever the live path passes and hid it, so these doubles enforce
the binding's signature (pybind11 raises ``TypeError`` for a wrong arity).

Not run against the native dispatcher: the native extension is not loaded here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist  # noqa: E402
import torch.distributed.rpc as rpc  # noqa: E402

from moe_infinity.distributed import expert_executor  # noqa: E402
from moe_infinity.distributed.expert_executor import (  # noqa: E402
    DistributedExpertExecutor,
)


class _StrictDispatcher:
    """Same positional signatures as the pybind11 binding."""

    def __init__(self):
        self.set_inputs_calls = []
        self.expected = None
        self.enqueued = []

    def set_inputs(self, hidden_states, router_mask, router_weight):
        self.set_inputs_calls.append(
            (hidden_states, router_mask, router_weight)
        )

    def set_expected_queue(self, expected):
        self.expected = expected

    def enqueue_expert(self, layer_id, expert_id, gpu_id, remote, phase=2):
        self.enqueued.append((layer_id, expert_id, gpu_id, remote, phase))

    def wait_expert(self):
        return ["done"]


class _DeviceMap:
    def __init__(self, targets):
        self.targets = targets

    def get_target_device(self, expert_list):
        return [self.targets[e] for e in expert_list]


class _Future:
    def wait(self):
        return None


def _executor(dispatcher, targets):
    executor = DistributedExpertExecutor.__new__(DistributedExpertExecutor)
    executor.expert_dispatcher = dispatcher
    executor.device_map_manager = _DeviceMap(targets)
    return executor


def _inputs(mask_dtype):
    hidden = torch.randn(3, 8)
    weights = torch.tensor([[0.6, 0.4, 0.0], [0.0, 0.7, 0.3], [0.5, 0.0, 0.5]])
    mask = (weights > 0) if mask_dtype is torch.bool else weights
    return hidden, mask, weights


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_local_dispatch_passes_mask_and_weights(monkeypatch, mask_dtype):
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    dispatcher = _StrictDispatcher()
    executor = _executor(dispatcher, {0: (0, 0, 0), 1: (0, 0, 1), 2: (0, 0, 2)})
    hidden, mask, weights = _inputs(mask_dtype)

    result = executor.dispatch(hidden, mask, layer_id=0)

    assert result == ["done"]
    ((got_hidden, got_mask, got_weights),) = dispatcher.set_inputs_calls
    assert got_hidden is hidden
    # The mask is boolean, as dispatch_local passes it; the weights carry the
    # routing weights (a float mask already is the weights; a bool mask means
    # weight 1 per selected expert).
    assert got_mask.dtype == torch.bool
    assert torch.equal(got_mask, weights > 0)
    expected = weights if mask_dtype is torch.float32 else (weights > 0).float()
    assert torch.equal(got_weights.float(), expected)
    assert [e[1] for e in dispatcher.enqueued] == [0, 1, 2]


def test_remote_dispatch_sends_three_arguments(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    remote = _StrictDispatcher()
    monkeypatch.setattr(
        expert_executor, "_expert_dispatcher", remote, raising=False
    )

    def rpc_async(to, func, args=()):
        func(*args)  # _call_expert_dispatcher -> the strict double
        return _Future()

    monkeypatch.setattr(rpc, "rpc_async", rpc_async)
    monkeypatch.setattr(rpc, "rpc_sync", lambda to, func, args=(): func(*args))
    local = _StrictDispatcher()
    executor = _executor(local, {0: (0, 0, 0), 1: (1, 0, 1), 2: (1, 0, 2)})
    hidden, mask, weights = _inputs(torch.bool)

    executor.dispatch(hidden, mask, layer_id=0)

    ((r_hidden, r_mask, r_weights),) = remote.set_inputs_calls
    assert r_hidden.device.type == "cpu" and r_mask.device.type == "cpu"
    assert r_weights.device.type == "cpu"
    assert r_mask.dtype == torch.bool
    assert len(local.set_inputs_calls) == 1  # rank 0 owns expert 0
