# pyright: reportAny=false

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
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

_EVICTION_SYNC_MODULE = _load_module(
    "moe_infinity.serving.eviction_sync",
    ROOT / "moe_infinity" / "serving" / "eviction_sync.py",
)
_ = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_ = _load_module(
    "moe_infinity.serving.kv_cache",
    ROOT / "moe_infinity" / "serving" / "kv_cache.py",
)
_ = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_ = _load_module(
    "moe_infinity.serving.memory_manager",
    ROOT / "moe_infinity" / "serving" / "memory_manager.py",
)
_ = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
)
_ = _load_module(
    "moe_infinity.serving.model_runner",
    ROOT / "moe_infinity" / "serving" / "model_runner.py",
)
_ = _load_module(
    "moe_infinity.serving.sampler",
    ROOT / "moe_infinity" / "serving" / "sampler.py",
)
_ENGINE_MODULE = _load_module(
    "moe_infinity.serving.engine",
    ROOT / "moe_infinity" / "serving" / "engine.py",
)
EvictionSyncAdapter = _EVICTION_SYNC_MODULE.EvictionSyncAdapter
ContinuousBatchingEngine = _ENGINE_MODULE.ContinuousBatchingEngine
set_eviction_sync = _ENGINE_MODULE.set_eviction_sync
SamplingParams = sys.modules["moe_infinity.serving.sequence"].SamplingParams


class MockCP:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def on_request_complete(self, rid: str) -> None:
        self.removed.append(rid)


@dataclass
class _MockConfig:
    vocab_size: int = 100
    eos_token_id: int = 2


class _MockModel:
    config: _MockConfig

    def __init__(
        self,
        vocab_size: int = 100,
        eos_token_id: int = 2,
    ) -> None:
        self.config = _MockConfig(
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
        )

    def eval(self) -> None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> types.SimpleNamespace:
        _ = kwargs
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.config.vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
        )

        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                token_id = int(input_ids[batch_idx, token_idx].item())
                next_token_id = (token_id + 1) % self.config.vocab_size
                logits[batch_idx, token_idx, next_token_id] = 0.0

        return types.SimpleNamespace(logits=logits)


class _MockExpertTracer:
    _next_entry: int

    def __init__(self) -> None:
        self._next_entry = 0

    def create_entry(self) -> int:
        entry = self._next_entry
        self._next_entry += 1
        return entry


class _MockOffloadEngine:
    request_id: int
    expert_tracer: _MockExpertTracer
    expert_layer_modules: list[types.SimpleNamespace]

    def __init__(self) -> None:
        self.request_id = 0
        self.expert_tracer = _MockExpertTracer()
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]

    def _generate_request_id(self) -> int:
        request_id = self.request_id
        self.request_id += 1
        return request_id


def _make_config(num_kv_blocks: int = 4) -> dict[str, object]:
    return {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 8,
        "max_tokens_per_step": 16,
        "block_size": 4,
        "num_layers": 1,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": "float16",
        "eos_token_id": 2,
        "num_kv_blocks": num_kv_blocks,
    }


def _make_engine(num_kv_blocks: int = 4) -> ContinuousBatchingEngine:
    return ContinuousBatchingEngine(
        model=_MockModel(),
        engine=_MockOffloadEngine(),
        config=_make_config(num_kv_blocks=num_kv_blocks),
    )


def test_finished_triggers_eviction() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_finished("req-finished")

    assert cp.removed == ["req-finished"]


def test_aborted_triggers_eviction() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_aborted("req-aborted")

    assert cp.removed == ["req-aborted"]


def test_kv_blocks_freed_triggers_eviction() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_kv_blocks_freed("req-freed")

    assert cp.removed == ["req-freed"]


def test_swap_does_not_evict() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_kv_blocks_swapped("req-swapped")

    assert cp.removed == []
    assert adapter.get_counters() == {
        "evict_incoming": 0,
        "evict_removed": 0,
        "evict_not_found": 0,
    }


def test_idempotent_eviction() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_finished("req-idempotent")
    adapter.on_request_aborted("req-idempotent")

    assert cp.removed == ["req-idempotent"]


def test_counters_increment() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_finished("req-1")
    adapter.on_request_aborted("req-2")
    adapter.on_kv_blocks_freed("req-2")
    adapter.on_kv_blocks_swapped("req-3")

    assert adapter.get_counters() == {
        "evict_incoming": 3,
        "evict_removed": 2,
        "evict_not_found": 1,
    }


def test_no_middleware_is_noop() -> None:
    adapter = EvictionSyncAdapter(None)

    adapter.on_request_finished("req-finished")
    adapter.on_request_aborted("req-aborted")
    adapter.on_kv_blocks_freed("req-freed")
    adapter.on_kv_blocks_swapped("req-swapped")

    assert adapter.get_counters() == {
        "evict_incoming": 3,
        "evict_removed": 0,
        "evict_not_found": 0,
    }


def test_fires_on_engine_finish() -> None:
    set_eviction_sync(None)
    try:
        cp = MockCP()
        adapter = EvictionSyncAdapter(cp)
        set_eviction_sync(adapter)
        engine = _make_engine()

        engine.add_request(
            request_id="req-finish",
            prompt_token_ids=[10],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        )

        outputs = engine.step()

        assert len(outputs) == 1
        assert outputs[0].finished is True
        assert cp.removed == ["req-finish"]
    finally:
        set_eviction_sync(None)


def test_fires_on_engine_abort() -> None:
    set_eviction_sync(None)
    try:
        cp = MockCP()
        adapter = EvictionSyncAdapter(cp)
        set_eviction_sync(adapter)
        engine = _make_engine()

        engine.add_request(
            request_id="req-abort",
            prompt_token_ids=[10],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        )

        engine.abort_request("req-abort")

        assert cp.removed == ["req-abort"]
    finally:
        set_eviction_sync(None)


def test_swap_not_wired() -> None:
    set_eviction_sync(None)
    try:
        cp = MockCP()
        adapter = EvictionSyncAdapter(cp)
        set_eviction_sync(adapter)
        engine = _make_engine(num_kv_blocks=2)

        engine.add_request(
            request_id="req-1",
            prompt_token_ids=list(range(8)),
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        )
        _ = engine.scheduler.schedule()

        engine.add_request(
            request_id="req-2",
            prompt_token_ids=list(range(4)),
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        )
        scheduler_output = engine.scheduler.schedule()

        assert scheduler_output.preempted_seq_ids == [0]
        assert cp.removed == []
        assert adapter.get_counters() == {
            "evict_incoming": 0,
            "evict_removed": 0,
            "evict_not_found": 0,
        }
    finally:
        set_eviction_sync(None)
