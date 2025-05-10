import os

import torch
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
)
from transformers.models.nllb_moe.modeling_nllb_moe import (
    NllbMoeDenseActDense,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

# from moe_infinity.models.modeling_grok import MoeMLP as GrokMoeMLP
from moe_infinity.models.modeling_arctic import ArcticMLP
from moe_infinity.models.modeling_deepseek import DeepseekV2MLP
from moe_infinity.models.modeling_deepseek_v3 import DeepseekV3MLP

EXPERT_CLS = {
    # "grok": GrokMoeMLP,
    "arctic": ArcticMLP,
    "deepseek_v2": DeepseekV2MLP,
    "deepseek_v3": DeepseekV3MLP,
    "mixtral": MixtralBlockSparseTop2MLP,
    # "nllb_moe": NllbMoeDenseActDense,
    "qwen3_moe": Qwen3MoeMLP,
}


# compile a single expert
def script_expert(save_dir, expert_type, config, **kwargs):
    """
    Compile a single expert.
    """
    # get argument list from the expert class
    # expert_cls = EXPERT_CLS[expert_type]
    # expert_args = expert_cls.__init__.__code__.co_varnames

    expert_instance = EXPERT_CLS[expert_type](config, **kwargs)
    # compile the forward function of the expert
    module = torch.jit.script(expert_instance)
    torch.jit.save(
        module,
        os.path.join(save_dir, "expert.pt"),
    )
