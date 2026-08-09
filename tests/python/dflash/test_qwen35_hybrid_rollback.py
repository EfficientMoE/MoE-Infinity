from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from moe_infinity.distributed.expert_executor import (
    DistributedExpertExecutor,
)
from moe_infinity.entrypoints.big_modeling import MoE
from moe_infinity.models import SyncQwen3_5MoeSparseMoeBlock
from moe_infinity.spec_decode import (
    DFlashSpeculator,
    read_dflash_config,
)
from moe_infinity.spec_decode import dflash as dflash_module
from moe_infinity.utils import ArcherConfig
from tests.python.dflash.fixtures_tiny import (
    build_tiny_drafter,
    make_tiny_drafter_config,
    plain_greedy_decode,
    set_determinism,
)


def _tiny_qwen35_target(seed: int = 0) -> Qwen3_5MoeForCausalLM:
    set_determinism(seed)
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        eos_token_id=None,
        pad_token_id=0,
        bos_token_id=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = Qwen3_5MoeForCausalLM(config)
    return target.float().eval()


def _qwen35_shell() -> tuple[MoE, MagicMock, MagicMock, MagicMock]:
    shell = MoE.__new__(MoE)
    model = MagicMock()
    model.config = SimpleNamespace(model_type="qwen3_5_moe")
    model.generate.return_value = torch.tensor([[1, 2, 9]])
    shell.model = model
    shell.use_native_engine = True
    engine = MagicMock()
    engine.generate.return_value = SimpleNamespace(output_token_ids=[8])
    shell._native_generation_engine = engine
    resolve_spec = MagicMock()
    shell._resolve_spec_strategy = resolve_spec
    shell._configure_hook = MagicMock()
    shell._cached_past_key_values = None
    shell.max_seq_length = 64
    return shell, model, engine, resolve_spec


class _LocalObservedExecutor(DistributedExpertExecutor):
    def __init__(
        self, module: SyncQwen3_5MoeSparseMoeBlock, prefetcher: object
    ) -> None:
        super().__init__(
            ArcherConfig.load_from_json(
                {"offload_path": "/tmp/moe-infinity-qwen35-dflash-test"}
            )
        )
        self.module = module
        self.prefetcher = prefetcher
        self.output: torch.Tensor | None = None

    def dispatch_local(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        router_weights: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        prefetcher: object | None = None,
    ) -> None:
        del router_logits
        self._maybe_route_ahead_prefetch(
            layer_id,
            router_mask,
            router_mask.shape[-1],
            prefetcher,
        )
        self.output = self.module._local_experts(
            hidden_states, router_mask, router_weights
        )

    def wait_dispatch_local(self) -> torch.Tensor:
        assert self.output is not None
        output = self.output
        self.output = None
        return output


def _install_observed_executor_blocks(
    target: Qwen3_5MoeForCausalLM, prefetcher: object
) -> None:
    for layer_id, layer in enumerate(target.model.layers):
        hf_block = getattr(layer, "mlp")
        sync_block = SyncQwen3_5MoeSparseMoeBlock(target.config).eval()
        state = dict(hf_block.state_dict())
        gate_up = state.pop("experts.gate_up_proj")
        down = state.pop("experts.down_proj")
        for expert_id in range(target.config.num_experts):
            gate, up = gate_up[expert_id].chunk(2, dim=0)
            state[f"experts.{expert_id}.gate_proj.weight"] = gate.contiguous()
            state[f"experts.{expert_id}.up_proj.weight"] = up.contiguous()
            state[f"experts.{expert_id}.down_proj.weight"] = down[
                expert_id
            ].contiguous()
        missing, unexpected = sync_block.load_state_dict(state, strict=False)
        assert missing == [] and unexpected == []
        sync_block.layer_id = layer_id
        setattr(
            sync_block,
            "expert_executor",
            _LocalObservedExecutor(sync_block, prefetcher),
        )
        layer.mlp = sync_block


def test_qwen35_spec_off_generate_stays_on_hf_path() -> None:
    shell, model, engine, _ = _qwen35_shell()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        actual = shell.generate(
            cast(torch.LongTensor, torch.tensor([[1, 2]])),
            do_sample=False,
            max_new_tokens=1,
        )

    assert actual.tolist() == [[1, 2, 9]]
    model.generate.assert_called_once()
    engine.generate.assert_not_called()


def test_qwen35_greedy_dflash_uses_native_path() -> None:
    shell, model, engine, resolve_spec = _qwen35_shell()
    draft = object()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        actual = shell.generate(
            cast(torch.LongTensor, torch.tensor([[1, 2]])),
            do_sample=False,
            max_new_tokens=1,
            speculative_draft=draft,
        )

    assert actual.tolist() == [[1, 2, 8]]
    resolve_spec.assert_called_once_with(draft)
    engine.generate.assert_called_once()
    model.generate.assert_not_called()


def test_qwen35_sampled_dflash_is_rejected() -> None:
    shell, _, _, _ = _qwen35_shell()

    with warnings.catch_warnings(), pytest.raises(ValueError, match="greedy"):
        warnings.simplefilter("ignore", DeprecationWarning)
        shell.generate(
            cast(torch.LongTensor, torch.tensor([[1, 2]])),
            do_sample=True,
            temperature=0.7,
            speculative_draft=object(),
        )


def test_qwen35_hybrid_dflash_is_token_identical_to_plain_greedy() -> None:
    target = _tiny_qwen35_target(seed=1)
    drafter = build_tiny_drafter(
        target,
        seed=21,
        block_size=4,
        target_layer_ids=(0, 1, 2, 3),
    )
    config = read_dflash_config(
        make_tiny_drafter_config(
            target.config,
            block_size=4,
            target_layer_ids=(0, 1, 2, 3),
        )
    )
    spec = DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )
    prompt = torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long)

    plain = plain_greedy_decode(target, prompt, max_new_tokens=32)
    actual = spec.generate(prompt, max_new_tokens=32, temperature=0.0)

    assert actual.tolist() == plain.tolist()
    assert any(
        trace.accept + 1 < config.block_size for trace in spec.step_trace
    )

    cached_length = spec.last_target_cache.get_seq_length()
    with torch.no_grad():
        expected_cache = target(
            actual[:, :cached_length], use_cache=True
        ).past_key_values
    assert torch.allclose(
        spec.last_target_cache.layers[0].conv_states,
        expected_cache.layers[0].conv_states,
    )
    assert torch.allclose(
        spec.last_target_cache.layers[0].recurrent_states,
        expected_cache.layers[0].recurrent_states,
    )
    assert torch.allclose(
        spec.last_target_cache.layers[1].keys,
        expected_cache.layers[1].keys,
        rtol=1e-5,
        atol=1e-6,
    )


def test_snapshot_target_cache_copies_qwen35_linear_attention_state() -> None:
    target = _tiny_qwen35_target()
    prompt = torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long)
    with torch.no_grad():
        cache = target(prompt, use_cache=True).past_key_values

    snapshot = dflash_module.snapshot_target_cache(cache)
    layer = cache.layers[0]
    saved = snapshot.linear[0]
    assert saved.conv_states is not None
    assert saved.recurrent_states is not None
    expected_conv = layer.conv_states.clone()
    expected_recurrent = layer.recurrent_states.clone()

    layer.conv_states.add_(1)
    layer.recurrent_states.add_(1)

    assert torch.equal(saved.conv_states, expected_conv)
    assert torch.equal(saved.recurrent_states, expected_recurrent)
    assert saved.has_previous_state is True


def test_snapshot_target_cache_rejects_partial_linear_cache_contract() -> None:
    malformed = SimpleNamespace(
        layers=[SimpleNamespace(conv_states=torch.zeros(1))]
    )

    with pytest.raises(RuntimeError, match="unsupported transformers"):
        dflash_module.snapshot_target_cache(malformed)


def test_partial_rollback_replays_exact_qwen35_hybrid_cache_state() -> None:
    target = _tiny_qwen35_target(seed=3)
    prompt = torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long)
    block = torch.tensor([[13, 17, 19, 23]], dtype=torch.long)
    committed = 2
    with torch.no_grad():
        cache = target(prompt, use_cache=True).past_key_values
        snapshot = dflash_module.snapshot_target_cache(cache)
        target(block, past_key_values=cache, use_cache=True)

        replay_calls: list[torch.Tensor] = []

        def replay(prefix: torch.Tensor, replay_cache: object) -> object:
            replay_calls.append(prefix.clone())
            return target(
                prefix, past_key_values=replay_cache, use_cache=True
            ).past_key_values

        dflash_module.rollback_target_cache(
            cache,
            snapshot,
            prev_start=prompt.shape[1],
            committed=committed,
            block_size=block.shape[1],
            block=block,
            replay=replay,
        )
        expected = target(
            torch.cat([prompt, block[:, :committed]], dim=1), use_cache=True
        ).past_key_values

    assert len(replay_calls) == 1
    assert torch.equal(replay_calls[0], block[:, :committed])
    assert cache.get_seq_length() == prompt.shape[1] + committed
    assert torch.allclose(
        cache.layers[0].conv_states, expected.layers[0].conv_states
    )
    assert torch.allclose(
        cache.layers[0].recurrent_states,
        expected.layers[0].recurrent_states,
    )
    assert torch.allclose(cache.layers[1].keys, expected.layers[1].keys)
    assert torch.allclose(cache.layers[1].values, expected.layers[1].values)


def test_full_accept_does_not_replay_hybrid_cache() -> None:
    target = _tiny_qwen35_target(seed=4)
    prompt = torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long)
    block = torch.tensor([[13, 17, 19, 23]], dtype=torch.long)
    with torch.no_grad():
        cache = target(prompt, use_cache=True).past_key_values
        snapshot = dflash_module.snapshot_target_cache(cache)
        target(block, past_key_values=cache, use_cache=True)

    replay = MagicMock()
    dflash_module.rollback_target_cache(
        cache,
        snapshot,
        prev_start=prompt.shape[1],
        committed=block.shape[1],
        block_size=block.shape[1],
        block=block,
        replay=replay,
    )

    replay.assert_not_called()
    assert cache.get_seq_length() == prompt.shape[1] + block.shape[1]


def test_from_models_rejects_target_layer_ids_outside_hybrid_depth() -> None:
    target = _tiny_qwen35_target()
    drafter = build_tiny_drafter(
        target, seed=1, block_size=4, target_layer_ids=(0, 1)
    )
    invalid = read_dflash_config(
        make_tiny_drafter_config(
            target.config,
            block_size=4,
            target_layer_ids=(0, target.config.num_hidden_layers),
        )
    )

    with pytest.raises(ValueError, match="target layer 4.*only 4 layers"):
        DFlashSpeculator.from_models(
            target, drafter, config=invalid, device="cpu"
        )


def test_qwen35_verify_records_route_ahead_stats() -> None:
    target = _tiny_qwen35_target(seed=5)
    prefetcher = MagicMock()
    _install_observed_executor_blocks(target, prefetcher)
    drafter = build_tiny_drafter(
        target,
        seed=6,
        block_size=4,
        target_layer_ids=(0, 1, 2, 3),
    )
    config = read_dflash_config(
        make_tiny_drafter_config(
            target.config,
            block_size=4,
            target_layer_ids=(0, 1, 2, 3),
        )
    )
    spec = DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )
    stats = spec.enable_route_ahead_stats()

    spec.generate(
        torch.tensor([[3, 7, 11, 2, 5]]),
        max_new_tokens=8,
        temperature=0.0,
    )

    report = stats.as_dict()
    assert report["steps"] > 0
    assert report["layers_observed"] > 0
    assert "waste_ratio" in report
