import time

import torch

import moe_infinity._engine as engine

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.deterministic = True

lib = engine

# B, D, H = 128, 1024, 4096
B, D, H = 128, 128, 1024
hidden = torch.randn(B, D, dtype=torch.bfloat16, device="cuda").contiguous()
gate_proj = torch.randn(H, D, dtype=torch.bfloat16, device="cuda").contiguous()
up_proj = torch.randn(H, D, dtype=torch.bfloat16, device="cuda").contiguous()
down_proj = torch.randn(D, H, dtype=torch.bfloat16, device="cuda").contiguous()

_ = torch.nn.functional.linear(hidden, gate_proj, None)
# hidden = torch.randn(B, D, dtype=torch.bfloat16, device='cuda').contiguous()
# gate_proj = torch.randn(H, D, dtype=torch.bfloat16, device='cuda').contiguous()
start = time.perf_counter()
x = lib.fused_silu_gemm(hidden, gate_proj)
end = time.perf_counter()
print(f"Time taken for fused_silu_gemm: {end - start:.6f} seconds")

# print(x.to(torch.bfloat16))
# assert torch.sum(x) == B * H, f"Sum of x is not equal to B * H {torch.sum(x)}"

# hidden = torch.randn(B, D, dtype=torch.bfloat16, device='cuda').contiguous()
# gate_proj = torch.randn(H, D, dtype=torch.bfloat16, device='cuda').contiguous()
# init compute handle first
# _ = torch.nn.functional.linear(hidden, gate_proj, None)
# hidden = torch.randn(B, D, dtype=torch.bfloat16, device='cuda').contiguous()
# gate_proj = torch.randn(H, D, dtype=torch.bfloat16, device='cuda').contiguous()
start = time.perf_counter()
y = torch.mm(hidden, gate_proj.transpose(0, 1))
end = time.perf_counter()
print(f"Time taken for torch.nn.functional.silu: {end - start:.6f} seconds")
print(x)
print(y)

is_close = torch.allclose(x, y, atol=1 / 128.0)
print(f"Is the output close? {is_close}")

diff = torch.abs(x - y)
idx = diff > 1 / 128.0
diff_atol = diff[idx]
print(f"Max difference: {len(diff_atol)} {diff_atol}")
print(f"X difference: {x[idx]}")
print(f"Y difference: {y[idx]}")
print(torch.where(idx))
# lib.benchmark_fused_vs_torch(hidden, gate_proj, up_proj, down_proj)
