"""Optional eval-graded RMSNorm fusion for HF RMSNorm modules (opt-in, NOT output-identical).

Measures (OLMoE-1B-7B, moe-infinity engine, RTX 4090, 65-token generation, per token):
  - eager RMSNorm path: ~327 dispatch ops/token of fp32 upcast/downcast + scale muls
  - fused (torch.rms_norm): those ops collapse to 3 dispatches/call
  - measured wall: -10 to -13.8 ms/token depending on variant (see notes below)

IMPORTANT: this fusion is NOT output-identical to the eager path. torch.rms_norm's
kernel reduction order differs from the eager (x**2).mean() computation, which shifts
sampled tokens measurably at a greedy-decode token-identity bar (pairwise token
agreement vs the eager path was below the cross-process self-agreement floor). The
shift is at the level of floating-point rounding; quality at an eval bar is not
expected to move, which is why this is offered as an eval-graded optimization only.
Keep the eager path (do NOT call this) where output identity is required, e.g. for
regression tests or reproducibility baselines.

Usage:
    from moe_infinity.runtime.rmsnorm_fusion import fuse_rmsnorm_eval_graded
    fuse_rmsnorm_eval_graded()   # before model load/generate

Variants:
    variant="weight-in-kernel"  (default) - weight multiplied inside the fused kernel;
        best measured wall (~-13.5 ms/tok vs eager).
    variant="faithful" - HF's exact multiply ordering applied outside the kernel;
        same speedup class (~-10 ms/tok) and still not output-identical (reduction order).
"""
import torch

_PATCHED = False


def fuse_rmsnorm_eval_graded(variant: str = "weight-in-kernel") -> bool:
    """Monkey-patch HF OlmoeRMSNorm.forward to a torch.rms_norm-based path.

    Returns True if the patch was applied. Eval-graded only (see module docstring).
    """
    global _PATCHED
    if variant not in ("weight-in-kernel", "faithful"):
        raise ValueError(f"unknown variant: {variant!r}")
    from transformers.models.olmoe import modeling_olmoe as _mol

    def fused_forward_weight_in_kernel(self, hidden_states):
        input_dtype = hidden_states.dtype
        w = getattr(self, "_w_fp32", None)
        if w is None:
            w = self.weight.to(torch.float32)
            self._w_fp32 = w
        y = torch.rms_norm(hidden_states.to(torch.float32),
                           (hidden_states.shape[-1],), w, self.variance_epsilon)
        return y.to(input_dtype)

    def fused_forward_faithful(self, hidden_states):
        input_dtype = hidden_states.dtype
        y = torch.rms_norm(hidden_states.to(torch.float32),
                           (hidden_states.shape[-1],), None, self.variance_epsilon)
        return (y.to(input_dtype)) * self.weight

    _mol.OlmoeRMSNorm.forward = (
        fused_forward_weight_in_kernel if variant == "weight-in-kernel"
        else fused_forward_faithful)
    _PATCHED = True
    return True


def rmsnorm_fusion_active() -> bool:
    return _PATCHED
