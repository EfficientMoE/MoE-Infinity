# Hardware-gated kernels

Some kernels use instructions that exist on only one GPU architecture. Others
were tuned for one card's SM count, L2 size and shared memory. These run only
on matching hardware; every other device keeps the previous implementation.

## Gates

| Gate | Matches | Used for |
| --- | --- | --- |
| `SM120` | compute capability 12.0 | Kernels with sm_120a-only instructions: block-scaled `mma.sync ... kind::mxf8f6f4`, `cvt.rn.f16x2.e2m1x2`, CUTLASS sm120 collectives. Built only with `MOE_ENABLE_SM120=1`. |
| `RTX_PRO_6000` | capability 12.0, 188 SMs, 128 MiB L2, 99 KiB shared memory per block: the RTX PRO 6000 Blackwell Workstation, Max-Q and Server editions | Tile tables, cost models, grid sizes and L2 windows unified across both RTX PRO 6000 editions. The RTX 5090 (170 SMs, 96 MiB L2) does not match. |
| `RTX_PRO_6000_SERVER` | `RTX_PRO_6000` + Server Edition (600 W TDP, 2430 MHz max SM / 1845 MHz sustained, 25 Gbps GDDR7: 335.5 TF BF16, 656.9 TF FP8, 1214.6 TF FP4, 1530.5 GB/s DRAM, BF16 ridge 219 FLOP/B) | Edition-specific tile / split / threshold overrides where 600 W compute roof vs 300 W Max-Q diverges. |
| `RTX_PRO_6000_MAXQ` | `RTX_PRO_6000` + Max-Q Workstation Edition (300 W TDP, 3090 MHz max / ~1460 MHz sustained GEMM, 28 Gbps GDDR7: 268.8 TF BF16, 530.3 TF FP8, 965.6 TF FP4, 1641.1 GB/s DRAM, BF16 ridge 164 FLOP/B) | Edition-specific tile / split / threshold overrides for the 300 W power-capped / higher-DRAM-clock envelope. |

- Python: `moe_infinity.utils.hw_profile.supports(gate, device)` and `pro6000_edition(device)`.
- C++: `moe_infinity::hw::Supports(Gate, device)` and `Edition(device)` in `extensions/kernel/hw_profile.h`.

A kernel that uses sm_120a instructions and was also tuned on the card checks
both gates. `MOE_INFINITY_UNVALIDATED_HW=1` lifts the `RTX_PRO_6000*` tuning
gates, for re-tuning on another card. It never lifts `SM120`.
`MOE_INFINITY_PRO6000_EDITION=server|maxq` overrides edition detection on a
matching RTX PRO 6000 device.

## Fusion vs Subdivision Policy (Server 600 W vs Max-Q 300 W)

1. **Label every kernel and table**: every `pro6000` kernel and config table
   must explicitly state in its header comment whether it is:
   - `RTX_PRO_6000 (Server 600W + Max-Q 300W unified)` — validated on both SKUs,
   - `RTX_PRO_6000_SERVER (600W)` — Server Edition specific, or
   - `RTX_PRO_6000_MAXQ (300W)` — Max-Q Workstation Edition specific.
2. **Fuse when Max-Q does not regress**: any kernel or dispatch edit developed
   on the 600 W Server Edition must be A/B-checked on the 300 W Max-Q card. If
   the edit does not regress Max-Q (within noise / improves both), **fuse both
   editions into the unified `RTX_PRO_6000` path** rather than maintaining two
   separate tables. Only subdivide via `RTX_PRO_6000_SERVER` /
   `RTX_PRO_6000_MAXQ` when an edit helps one edition and measurably hurts the
   other.

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
| Linear-attention fused prefill (GDN / KDA Triton prep, recurrence + output, L2-blocked CUDA recurrence, persisting-L2 window) | `RTX_PRO_6000` | #336 |
| Split-KV paged decode attention | `RTX_PRO_6000` | #326 |
| GQA attention (sinks, sliding window) | `RTX_PRO_6000` | #334 |
| Grouped MoE decode | `RTX_PRO_6000` | #332 |
| Fused router | `RTX_PRO_6000` | #331 |
