# pyright: reportAny=false, reportImplicitOverride=false

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[3]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_EXPERT_BATCH_MODULE = _load_module(
    "moe_infinity.serving.expert_batch",
    ROOT / "moe_infinity" / "serving" / "expert_batch.py",
)

BatchedExpertDispatch = _EXPERT_BATCH_MODULE.BatchedExpertDispatch


class MockExpertExecutor:
    def __init__(self) -> None:
        self.last_hidden_states: Optional[torch.Tensor] = None
        self.last_router_mask: Optional[torch.Tensor] = None
        self.last_router_weights: Optional[torch.Tensor] = None
        self.last_router_logits: Optional[torch.Tensor] = None

    def dispatch_local(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        router_weights: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> None:
        _ = layer_id
        self.last_hidden_states = hidden_states
        self.last_router_mask = router_mask
        self.last_router_weights = router_weights
        self.last_router_logits = router_logits

    def wait_dispatch_local(self) -> torch.Tensor:
        assert self.last_hidden_states is not None
        return torch.zeros_like(self.last_hidden_states)


class IdentityMockExpertExecutor(MockExpertExecutor):
    def wait_dispatch_local(self) -> torch.Tensor:
        assert self.last_hidden_states is not None
        return self.last_hidden_states.clone()


def test_dispatch_produces_correct_output_shape() -> None:
    hidden_states = torch.randn(5, 8, dtype=torch.float32)
    router_logits = torch.randn(5, 4, dtype=torch.float32)
    executor = MockExpertExecutor()

    output = BatchedExpertDispatch.dispatch(
        expert_executor=executor,
        layer_id=2,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=2,
        token_offsets=[0, 2, 5],
    )

    assert output.shape == hidden_states.shape
    assert executor.last_router_mask is not None
    assert executor.last_router_weights is not None
    assert executor.last_router_mask.shape == (5, 4)
    assert executor.last_router_weights.shape == (5, 4)


def test_dispatch_builds_correct_router_mask() -> None:
    hidden_states = torch.randn(3, 6, dtype=torch.float32)
    router_logits = torch.tensor(
        [
            [4.0, 1.0, 3.0, 2.0],
            [0.1, 0.2, 0.3, 0.4],
            [9.0, 8.0, 7.0, 6.0],
        ],
        dtype=torch.float32,
    )
    executor = MockExpertExecutor()

    _ = BatchedExpertDispatch.dispatch(
        expert_executor=executor,
        layer_id=0,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=2,
        token_offsets=[0, 1, 3],
    )

    expected_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, False, True, True],
            [True, True, False, False],
        ]
    )
    assert executor.last_router_mask is not None
    assert torch.equal(executor.last_router_mask, expected_mask)

    assert executor.last_router_weights is not None
    row_sums = executor.last_router_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_split_output_restores_sequences() -> None:
    output = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    split = BatchedExpertDispatch.split_output(
        output=output,
        token_offsets=[0, 2, 3, 6],
        seq_lengths=[2, 1, 3],
    )

    assert len(split) == 3
    assert torch.equal(split[0], output[0:2])
    assert torch.equal(split[1], output[2:3])
    assert torch.equal(split[2], output[3:6])


def test_concatenated_tokens_preserved() -> None:
    seq_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    seq_b = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    seq_c = torch.tensor(
        [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]],
        dtype=torch.float32,
    )
    hidden_states = torch.cat([seq_a, seq_b, seq_c], dim=0)
    router_logits = torch.randn(hidden_states.size(0), 5, dtype=torch.float32)
    executor = IdentityMockExpertExecutor()

    packed_output = BatchedExpertDispatch.dispatch(
        expert_executor=executor,
        layer_id=1,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=2,
        token_offsets=[0, 2, 3, 6],
    )
    split_output = BatchedExpertDispatch.split_output(
        output=packed_output,
        token_offsets=[0, 2, 3, 6],
        seq_lengths=[2, 1, 3],
    )

    assert torch.equal(split_output[0], seq_a)
    assert torch.equal(split_output[1], seq_b)
    assert torch.equal(split_output[2], seq_c)
    round_trip = torch.cat(split_output, dim=0)
    assert torch.equal(round_trip, hidden_states)
