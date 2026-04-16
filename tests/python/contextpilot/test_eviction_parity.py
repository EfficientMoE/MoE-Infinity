# pyright: reportAny=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

from moe_infinity.engine.scheduler import Scheduler as NativeScheduler
from moe_infinity.engine.types import (
    Request,
)
from moe_infinity.engine.types import (
    SamplingParams as NativeSamplingParams,
)
from moe_infinity.memory.kv_cache_manager import KVCacheManager

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

EvictionEvent = _EVICTION_SYNC_MODULE.EvictionEvent
EvictionSyncAdapter = _EVICTION_SYNC_MODULE.EvictionSyncAdapter
ContinuousBatchingEngine = _ENGINE_MODULE.ContinuousBatchingEngine
set_eviction_sync = _ENGINE_MODULE.set_eviction_sync
ServingSamplingParams = sys.modules[
    "moe_infinity.serving.sequence"
].SamplingParams


class _MockCP:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def on_request_complete(self, request_id: str) -> None:
        self.removed.append(request_id)


@dataclass
class _MockConfig:
    vocab_size: int = 128
    eos_token_id: int = 2


class _MockModel:
    config: _MockConfig

    def __init__(self, vocab_size: int = 128, eos_token_id: int = 2) -> None:
        self.config = _MockConfig(
            vocab_size=vocab_size, eos_token_id=eos_token_id
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


def _make_serving_engine(num_kv_blocks: int = 4) -> ContinuousBatchingEngine:
    config: dict[str, object] = {
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
    return ContinuousBatchingEngine(
        model=_MockModel(),
        engine=_MockOffloadEngine(),
        config=config,
    )


class _NativeCPBridge:
    def __init__(self, adapter: EvictionSyncAdapter) -> None:
        self._adapter = adapter

    def predict_prefix_reuse(
        self,
        request_id: str,
        token_ids: list[int],
    ) -> float:
        _ = request_id
        _ = token_ids
        return 0.0

    def notify_blocks_allocated(
        self,
        request_id: str,
        block_hashes: list[int],
    ) -> None:
        _ = request_id
        _ = block_hashes

    def notify_blocks_freed(
        self,
        request_id: str,
        block_hashes: list[int],
    ) -> None:
        _ = block_hashes
        self._adapter.on_kv_blocks_freed(request_id)


def _make_native_scheduler() -> NativeScheduler:
    kv_mgr = KVCacheManager(
        num_gpu_blocks=64,
        num_cpu_blocks=64,
        block_size=4,
    )
    return NativeScheduler(
        kv_cache_manager=kv_mgr,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
    )


def _make_native_request(request_id: str, prompt_tokens: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_tokens,
        sampling_params=NativeSamplingParams(),
        arrival_time=0.0,
    )


def test_completed_and_aborted_both_evict() -> None:
    cp = _MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_finished("req-completed")
    adapter.on_request_aborted("req-aborted")

    assert cp.removed == ["req-completed", "req-aborted"]
    assert adapter.get_event_counters() == {
        "completed": 1,
        "aborted": 1,
        "freed": 0,
        "swapped": 0,
    }


def test_freed_event_evicts() -> None:
    cp = _MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_kv_blocks_freed("req-freed")

    assert cp.removed == ["req-freed"]
    assert adapter.get_event_counters() == {
        "completed": 0,
        "aborted": 0,
        "freed": 1,
        "swapped": 0,
    }


def test_swapped_event_does_not_evict() -> None:
    cp = _MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_kv_blocks_swapped("req-swapped")

    assert cp.removed == []
    assert adapter.get_event_counters() == {
        "completed": 0,
        "aborted": 0,
        "freed": 0,
        "swapped": 1,
    }


def test_enum_event_semantics() -> None:
    assert EvictionEvent.COMPLETED.value == "completed"
    assert EvictionEvent.ABORTED.value == "aborted"
    assert EvictionEvent.FREED.value == "freed"
    assert EvictionEvent.SWAPPED.value == "swapped"


def test_cross_path_parity() -> None:
    v2_cp = _MockCP()
    v2_adapter = EvictionSyncAdapter(v2_cp)
    set_eviction_sync(None)
    try:
        set_eviction_sync(v2_adapter)
        engine = _make_serving_engine()

        engine.add_request(
            request_id="v2-complete",
            prompt_token_ids=[10],
            sampling_params=ServingSamplingParams(
                temperature=0.0, max_tokens=1
            ),
        )
        _ = engine.step()

        engine.add_request(
            request_id="v2-abort",
            prompt_token_ids=[10],
            sampling_params=ServingSamplingParams(
                temperature=0.0, max_tokens=4
            ),
        )
        engine.abort_request("v2-abort")
    finally:
        set_eviction_sync(None)

    native_cp = _MockCP()
    native_adapter = EvictionSyncAdapter(native_cp)
    native_scheduler = _make_native_scheduler()
    native_scheduler.set_cp_kv_manager(_NativeCPBridge(native_adapter))

    req_finish = _make_native_request("native-finish", list(range(8)))
    native_scheduler.add_request(req_finish)
    _ = native_scheduler.schedule()
    native_scheduler.finish_request("native-finish")

    req_abort = _make_native_request("native-abort", list(range(8, 16)))
    native_scheduler.add_request(req_abort)
    _ = native_scheduler.schedule()
    native_scheduler.abort_request("native-abort")

    assert len(v2_cp.removed) == len(native_cp.removed) == 2
    assert v2_adapter.get_counters()["evict_incoming"] == 2
    assert native_adapter.get_counters()["evict_incoming"] == 2

    assert v2_adapter.get_event_counters() == {
        "completed": 1,
        "aborted": 1,
        "freed": 0,
        "swapped": 0,
    }
    assert native_adapter.get_event_counters() == {
        "completed": 0,
        "aborted": 0,
        "freed": 2,
        "swapped": 0,
    }
