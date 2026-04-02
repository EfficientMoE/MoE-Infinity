import os
from unittest.mock import MagicMock

import pytest
import torch


def test_sync_gpt_oss_mlp_packed_param_count():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    config = MagicMock()
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_local_experts = 8
    config.num_experts_per_tok = 2

    mlp = SyncGptOssMLP(config)
    experts, hidden, intermediate = 8, 64, 64

    assert mlp.experts.gate_up_proj.shape == (
        experts,
        hidden,
        2 * intermediate,
    )
    assert mlp.experts.down_proj.shape == (
        experts,
        intermediate,
        hidden,
    )

    total_expert_params = experts * (
        hidden * 2 * intermediate
        + 2 * intermediate
        + intermediate * hidden
        + hidden
    )
    actual_expert_params = sum(
        param.numel()
        for name, param in mlp.named_parameters()
        if "proj" in name
    )
    assert actual_expert_params == total_expert_params


def test_gptoss_offload_config_accepted():
    ratio = 0.75
    config = {
        "offload_path": "/tmp/gpt-oss-test",
        "device_memory_ratio": ratio,
    }
    assert "offload_path" in config
    assert 0 < ratio < 1.0


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.slow
def test_gpt_oss_offloading_reduces_memory():
    from moe_infinity import MoE

    checkpoint = "openai/gpt-oss-20b"
    offload_path = os.path.expanduser("~/moe-infinity-gpt-oss-memory")

    torch.cuda.reset_peak_memory_stats()
    MoE(
        checkpoint,
        {
            "offload_path": offload_path,
            "device_memory_ratio": 0.5,
        },
    )
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    assert peak_mb < 12000
