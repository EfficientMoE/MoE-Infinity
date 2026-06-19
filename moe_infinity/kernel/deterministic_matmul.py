# ruff: noqa: I001

import torch
import os


_STATE = None


def _capture_state():
    global _STATE
    if _STATE is not None:
        return
    _STATE = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nccl_algo": os.environ.get("NCCL_ALGO"),
    }


def enable_deterministic_mode():
    _capture_state()
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["NCCL_ALGO"] = "Tree"


def disable_deterministic_mode():
    global _STATE
    if _STATE is None:
        torch.use_deterministic_algorithms(False)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        os.environ.pop("NCCL_ALGO", None)
        return

    torch.use_deterministic_algorithms(_STATE["deterministic_algorithms"])

    if _STATE["cublas_workspace_config"] is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _STATE[
            "cublas_workspace_config"
        ]

    if _STATE["nccl_algo"] is None:
        os.environ.pop("NCCL_ALGO", None)
    else:
        os.environ["NCCL_ALGO"] = _STATE["nccl_algo"]

    _STATE = None


def deterministic_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        if input.dim() <= 1:
            output = input.matmul(weight.t())
        else:
            output = torch.stack(
                [sample.matmul(weight.t()) for sample in input.unbind(dim=0)],
                dim=0,
            )
        if bias is not None:
            output = output + bias
        return output
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


if os.environ.get("MOE_DETERMINISTIC") == "1":
    enable_deterministic_mode()
