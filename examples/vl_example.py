import argparse
import os
import sys

import torch
from transformers import AutoProcessor

from moe_infinity import MoE

parser = argparse.ArgumentParser(
    description="MoE-Infinity vision-language inference example"
)
parser.add_argument(
    "--checkpoint",
    default="Qwen/Qwen3.5-35B-A3B",
    help="HuggingFace VL model checkpoint",
)
parser.add_argument(
    "--image",
    required=True,
    help="Path to the input image",
)
parser.add_argument(
    "--prompt",
    default="What animal is in the picture?",
    help="Question to ask about the image",
)
parser.add_argument(
    "--offload_dir",
    default=os.path.join(os.path.expanduser("~"), "moe-infinity-vl"),
    help="Directory for offloading expert weights",
)
parser.add_argument(
    "--device_memory_ratio",
    type=float,
    default=0.5,
    help="Fraction of device memory used for expert caching",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=64,
    help="Maximum tokens to generate",
)
parser.add_argument(
    "--expect",
    default=None,
    help="Fail unless this substring appears in the answer (case-insensitive)",
)
args = parser.parse_args()

processor = AutoProcessor.from_pretrained(
    args.checkpoint, trust_remote_code=True
)

model = MoE(
    args.checkpoint,
    {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
    },
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": args.image},
            {"type": "text", "text": args.prompt},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

input_ids = inputs["input_ids"].to("cuda:0")
generate_kwargs = {
    key: (value.to("cuda:0") if isinstance(value, torch.Tensor) else value)
    for key, value in inputs.items()
    if key != "input_ids"
}
if "pixel_values" in generate_kwargs:
    generate_kwargs["pixel_values"] = generate_kwargs["pixel_values"].to(
        torch.bfloat16
    )

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
        **generate_kwargs,
    )

answer = processor.tokenizer.decode(
    output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
).strip()
print(f"ANSWER: {answer}")

if not answer:
    print("FAIL: empty answer", file=sys.stderr)
    sys.exit(1)

routing_stats = model.engine.expert_executor.get_gpu_routing_stats()
print(f"ROUTING_STATS: {routing_stats}")
route_batches = int(routing_stats.get("route_batches", 0))
fallback_count = int(routing_stats.get("fallback_count", 0))
if route_batches <= 0 and fallback_count <= 0:
    print(
        "FAIL: no routed-expert dispatches recorded "
        f"(route_batches={route_batches}, fallback_count={fallback_count})",
        file=sys.stderr,
    )
    sys.exit(1)

if args.expect is not None and args.expect.lower() not in answer.lower():
    print(
        f"FAIL: expected {args.expect!r} in answer {answer!r}",
        file=sys.stderr,
    )
    sys.exit(1)

print("PASS")
