# pyright: reportAny=false

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_SEQUENCE_MODULE = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_KV_CACHE_MODULE = _load_module(
    "moe_infinity.serving.kv_cache",
    ROOT / "moe_infinity" / "serving" / "kv_cache.py",
)
_ = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_ = _load_module(
    "moe_infinity.serving.cp_kv_interface",
    ROOT / "moe_infinity" / "serving" / "cp_kv_interface.py",
)
_SCHEDULER_MODULE = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
)

SamplingParams = _SEQUENCE_MODULE.SamplingParams
SequenceData = _SEQUENCE_MODULE.SequenceData
SequenceGroup = _SEQUENCE_MODULE.SequenceGroup
PagedKVCache = _KV_CACHE_MODULE.PagedKVCache
Scheduler = _SCHEDULER_MODULE.Scheduler


def _make_cache(*, num_blocks: int = 8) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def _make_group(
    request_id: str, seq_id: int, prompt_tokens: list[int]
) -> SequenceGroup:
    sequence = SequenceData(
        seq_id=seq_id,
        prompt_token_ids=prompt_tokens,
        sampling_params=SamplingParams(),
    )
    return SequenceGroup(request_id=request_id, sequences=[sequence])


class _MockCPManager:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores: dict[str, float] = scores

    def predict_prefix_reuse(
        self, request_id: str, token_ids: list[int]
    ) -> float:
        _ = token_ids
        return self._scores.get(request_id, 0.0)


def test_cp_aware_ordering_schedules_high_reuse_first() -> None:
    scheduler = Scheduler(
        _make_cache(),
        max_batch_size=1,
        max_tokens_per_step=128,
    )
    req1 = _make_group("req-1", 1, [1, 2, 3])
    req2 = _make_group("req-2", 2, [1, 2, 3])

    scheduler.set_cp_kv_manager(_MockCPManager({"req-1": 0.1, "req-2": 0.9}))
    scheduler.add_request(req1)
    scheduler.add_request(req2)

    output = scheduler.schedule()

    assert output.prefill_seq_ids == [2]


def test_scheduler_works_without_cp_manager() -> None:
    scheduler = Scheduler(
        _make_cache(),
        max_batch_size=1,
        max_tokens_per_step=128,
    )
    req1 = _make_group("req-1", 1, [1, 2, 3])
    req2 = _make_group("req-2", 2, [4, 5])

    scheduler.add_request(req1)
    scheduler.add_request(req2)

    output = scheduler.schedule()

    assert output.prefill_seq_ids == [1]


def test_set_cp_kv_manager() -> None:
    scheduler = Scheduler(_make_cache())
    manager = _MockCPManager({})

    scheduler.set_cp_kv_manager(manager)

    assert scheduler._cp_kv_manager is manager
