import time

import torch
from transformers import AutoConfig

from moe_infinity.models.modeling_arctic.modeling_arctic import (
    ArcticAttention,
    ArcticDecoderLayer,
    ArcticMLP,
)

config = AutoConfig.from_pretrained(
    "Snowflake/snowflake-arctic-instruct", trust_remote_code=True
)

attn_layer = ArcticAttention(config, 1)
attn_layer = attn_layer.to("cuda:0")

batch_size = 1
sequence_length = 1
hidden_dim = config.hidden_size

print(batch_size, sequence_length, hidden_dim)

fake_input = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim,
    dtype=torch.bfloat16,
    device="cuda:0",
)

# compute random attention
output = attn_layer(fake_input)

start = time.time()
output = attn_layer(fake_input)
torch.cuda.synchronize()
end = time.time()

print(f"Time taken: {end - start} seconds")
