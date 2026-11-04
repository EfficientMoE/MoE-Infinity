# Hardware-gated kernels

Some kernels use instructions that exist on only one GPU architecture. Others
were tuned for one card's SM count, L2 size and shared memory. These run only
on matching hardware; every other device keeps the previous implementation.

## Gates

| Gate | Matches | Used for |
| --- | --- | --- |
| `SM120` | compute capability 12.0 | Kernels with sm_120a-only instructions: block-scaled `mma.sync ... kind::mxf8f6f4`, `cvt.rn.f16x2.e2m1x2`, CUTLASS sm120 collectives. Built only with `MOE_ENABLE_SM120=1`. |
| `RTX_PRO_6000` | capability 12.0, 188 SMs, 128 MiB L2, 99 KiB shared memory per block: the RTX PRO 6000 Blackwell Workstation, Max-Q and Server editions | Tile tables, cost models, grid sizes and L2 windows tuned on that card. The RTX 5090 (170 SMs, 96 MiB L2) does not match. |

- Python: `moe_infinity.utils.hw_profile.supports(gate, device)`.
- C++: `moe_infinity::hw::Supports(Gate, device)` in `extensions/kernel/hw_profile.h`.

A kernel that uses sm_120a instructions and was also tuned on the card checks
both gates. `MOE_INFINITY_UNVALIDATED_HW=1` lifts the `RTX_PRO_6000` gate, for
re-tuning on another card. It never lifts `SM120`.

## Conventions

- Gated kernels and config tables carry `sm120` or `pro6000` in their names.
  Each starts with a comment naming the card it was tuned on and what runs on
  other devices.
- Each gated path has a CPU test that fakes another GPU (for example an H100 or
  an RTX 5090) and checks that the previous implementation is selected.
- Each PR that adds a gated kernel lists it in a "Hardware specificity" section.

## Gated kernels

Added by the sm_120 kernel-audit fix PRs (tracking issue #297):

| Area | Gates | PR |
| --- | --- | --- |
| DeepSeek-V4 FP4 routed experts (block-scaled MMA) | `SM120`, `RTX_PRO_6000` | #333 |
| FP8 experts (CUTLASS sm120 blockwise GEMM, GEMV) | `SM120`, `RTX_PRO_6000` | #338 |
| MXFP4 GEMM and small-M GEMV (E2M1 conversion) | `SM120`, `RTX_PRO_6000` | #327 |
| CUTLASS expert prefill tiles and grouped prefill | `RTX_PRO_6000` | #325 |
| GPT-OSS offloaded MXFP4 experts | `RTX_PRO_6000` | #335 |
| GPT-OSS grouped MXFP4 decode (CUDA grouped kernel, Triton GEMV with `cvt.rn.f16x2.e2m1x2`, launch tables) | `SM120`, `RTX_PRO_6000` | #329 |
| Linear-attention CUDA recurrence | `RTX_PRO_6000` | #336 |
| Split-KV paged decode attention | `RTX_PRO_6000` | #326 |
| GQA attention (sinks, sliding window) | `RTX_PRO_6000` | #334 |
| Grouped MoE decode | `RTX_PRO_6000` | #332 |
| Fused router | `RTX_PRO_6000` | #331 |
