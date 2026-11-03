#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO_ROOT / "moe_infinity" / "kernel"
DEFAULT_OUTPUT_DIR = KERNEL_DIR / "_compiled"
DEFAULT_TARGET_ARCHS = ("80", "90", "120")
KERNEL_PACKAGE_PREFIX = "moe_infinity.kernel"


@dataclass(frozen=True)
class TritonConfigSpec:
    kwargs: dict[str, Any]
    num_warps: int | None = None
    num_stages: int | None = None


@dataclass(frozen=True)
class KernelSpec:
    signature: dict[str, str]
    constants: dict[str, Any]
    configs: tuple[TritonConfigSpec, ...] = ()


@dataclass(frozen=True)
class KernelCandidate:
    module_name: str
    file_path: Path
    function_name: str


_SPEC_OVERRIDES: dict[tuple[str, str], KernelSpec] = {
    (
        "moe_infinity.kernel.router",
        "softmax_kernel",
    ): KernelSpec(
        signature={
            "output_ptr": "*bf16",
            "input_ptr": "*bf16",
            "input_row_stride": "i32",
            "output_row_stride": "i32",
            "n_rows": "i32",
            "n_cols": "i32",
        },
        constants={"BLOCK_SIZE": 128, "num_stages": 4},
    ),
    (
        "moe_infinity.kernel.router",
        "fused_softmax_topk_kernel_nobias",
    ): KernelSpec(
        signature={
            "hidden_ptr": "*bf16",
            "weight_ptr": "*bf16",
            "routing_mask_ptr": "*i1",
            "routing_weight_ptr": "*bf16",
            "B": "i32",
            "H": "i32",
            "E": "i32",
        },
        constants={"TOPK": 2, "BLOCK_E": 128, "normalize_topk": True},
    ),
    (
        "moe_infinity.kernel.fused_qkv",
        "_fused_qkv_kernel",
    ): KernelSpec(
        signature={
            "hidden_ptr": "*bf16",
            "weight_ptr": "*bf16",
            "q_ptr": "*bf16",
            "k_ptr": "*bf16",
            "v_ptr": "*bf16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "Q_DIM": "i32",
            "KV_DIM": "i32",
            "stride_hidden_m": "i32",
            "stride_hidden_k": "i32",
            "stride_weight_k": "i32",
            "stride_weight_n": "i32",
            "stride_q_m": "i32",
            "stride_q_n": "i32",
            "stride_k_m": "i32",
            "stride_k_n": "i32",
            "stride_v_m": "i32",
            "stride_v_n": "i32",
        },
        constants={"OUTPUT_IS_BF16": True},
        configs=(
            TritonConfigSpec(
                kwargs={"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
                num_warps=4,
                num_stages=3,
            ),
        ),
    ),
    (
        "moe_infinity.kernel.fused_ffn",
        "_silu",
    ): KernelSpec(signature={"x": "fp32"}, constants={}),
    (
        "moe_infinity.kernel.fused_ffn",
        "_fused_gate_up_silu_kernel",
    ): KernelSpec(
        signature={
            "x_ptr": "*bf16",
            "gate_weight_ptr": "*bf16",
            "up_weight_ptr": "*bf16",
            "intermediate_ptr": "*bf16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_x_m": "i32",
            "stride_x_k": "i32",
            "stride_gate_n": "i32",
            "stride_gate_k": "i32",
            "stride_up_n": "i32",
            "stride_up_k": "i32",
            "stride_intermediate_m": "i32",
            "stride_intermediate_n": "i32",
        },
        constants={},
        configs=(
            TritonConfigSpec(
                kwargs={"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
                num_warps=4,
                num_stages=3,
            ),
        ),
    ),
    (
        "moe_infinity.kernel.fused_decode_attn",
        "_fused_decode_attention_kernel",
    ): KernelSpec(
        signature={
            "query_ptr": "*bf16",
            "key_ptr": "*bf16",
            "value_ptr": "*bf16",
            "block_tables_ptr": "*i32",
            "seq_lens_ptr": "*i32",
            "out_ptr": "*bf16",
            "scale": "fp32",
            "head_dim": "i32",
            "block_size": "i32",
            "max_blocks_per_seq": "i32",
            "q_stride_b": "i32",
            "q_stride_t": "i32",
            "q_stride_h": "i32",
            "q_stride_d": "i32",
            "k_stride_block": "i32",
            "k_stride_token": "i32",
            "k_stride_head": "i32",
            "k_stride_d": "i32",
            "v_stride_block": "i32",
            "v_stride_token": "i32",
            "v_stride_head": "i32",
            "v_stride_d": "i32",
            "bt_stride_b": "i32",
            "bt_stride_block": "i32",
            "out_stride_b": "i32",
            "out_stride_h": "i32",
            "out_stride_d": "i32",
        },
        constants={"QUERY_GROUP_SIZE": 8, "HEAD_DIM": 128},
        configs=(
            TritonConfigSpec(
                kwargs={"BLOCK_KV": 64},
                num_warps=4,
                num_stages=2,
            ),
        ),
    ),
    (
        "moe_infinity.kernel.mxfp4_gemm",
        "_fp4_lookup",
    ): KernelSpec(signature={"idx": "i32"}, constants={}),
    (
        "moe_infinity.kernel.mxfp4_gemm",
        "_ldexp",
    ): KernelSpec(
        signature={"mantissa": "fp32", "exponent": "i32"},
        constants={},
    ),
    (
        "moe_infinity.kernel.mxfp4_gemm",
        "_fused_mxfp4_gemm_kernel",
    ): KernelSpec(
        signature={
            "lhs_ptr": "*bf16",
            "rhs_packed_ptr": "*u8",
            "rhs_scales_ptr": "*u8",
            "bias_ptr": "*bf16",
            "out_ptr": "*bf16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_lhs_m": "i32",
            "stride_lhs_k": "i32",
            "stride_rhs_n": "i32",
            "stride_rhs_k": "i32",
            "stride_scales_n": "i32",
            "stride_scales_k": "i32",
            "stride_out_m": "i32",
            "stride_out_n": "i32",
        },
        constants={"HAS_BIAS": False},
        configs=(
            TritonConfigSpec(
                kwargs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
                num_warps=4,
                num_stages=3,
            ),
        ),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compile Triton kernels for multiple GPU architectures."
    )
    parser.add_argument(
        "--target-archs",
        default=",".join(DEFAULT_TARGET_ARCHS),
        help="Comma-separated compute capabilities to compile for.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where compiled artifacts should be written.",
    )
    return parser.parse_args()


def _parse_target_archs(raw_value: str) -> list[str]:
    archs = [
        part.strip().replace("sm_", "").replace("sm", "")
        for part in raw_value.split(",")
    ]
    normalized = [arch for arch in archs if arch]
    if not normalized:
        raise ValueError("at least one target architecture must be provided")
    return normalized


def _module_imports_triton(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "triton" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "triton" or node.module.startswith("triton."):
                return True
    return False


def _is_triton_jit_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        return _is_triton_jit_decorator(node.func)
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "jit"
        and isinstance(node.value, ast.Name)
        and node.value.id == "triton"
    )


def _is_constexpr_annotation(node: ast.expr | None) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    if node.attr != "constexpr":
        return False
    return isinstance(node.value, ast.Name) and node.value.id == "tl"


def _discover_kernel_candidates() -> list[KernelCandidate]:
    candidates: list[KernelCandidate] = []
    for file_path in sorted(KERNEL_DIR.glob("*.py")):
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
        if not _module_imports_triton(tree):
            continue
        module_name = f"{KERNEL_PACKAGE_PREFIX}.{file_path.stem}"
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if any(
                _is_triton_jit_decorator(decorator)
                for decorator in node.decorator_list
            ):
                candidates.append(
                    KernelCandidate(
                        module_name=module_name,
                        file_path=file_path,
                        function_name=node.name,
                    )
                )
    return candidates


def _default_signature_type(argument_name: str) -> str:
    if argument_name.endswith("_ptr"):
        if any(token in argument_name for token in ("packed", "scales")):
            return "*u8"
        if any(token in argument_name for token in ("table", "seq_len")):
            return "*i32"
        if argument_name.endswith("mask_ptr"):
            return "*i1"
        return "*bf16"
    if argument_name == "scale":
        return "fp32"
    return "i32"


def _default_constant_value(argument_name: str) -> Any:
    defaults: dict[str, Any] = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "BLOCK_KV": 64,
        "BLOCK_E": 128,
        "BLOCK_SIZE": 128,
        "HEAD_DIM": 128,
        "QUERY_GROUP_SIZE": 8,
        "HAS_BIAS": False,
        "OUTPUT_IS_BF16": True,
        "num_stages": 4,
        "normalize_topk": True,
        "TOPK": 2,
    }
    return defaults.get(argument_name, 1)


def _build_fallback_spec(candidate: KernelCandidate) -> KernelSpec:
    tree = ast.parse(
        candidate.file_path.read_text(), filename=str(candidate.file_path)
    )
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == candidate.function_name
        ):
            signature: dict[str, str] = {}
            constants: dict[str, Any] = {}
            for argument in node.args.args:
                if _is_constexpr_annotation(argument.annotation):
                    constants[argument.arg] = _default_constant_value(
                        argument.arg
                    )
                else:
                    signature[argument.arg] = _default_signature_type(
                        argument.arg
                    )
            return KernelSpec(signature=signature, constants=constants)
    raise ValueError(
        f"failed to rebuild fallback spec for {candidate.function_name}"
    )


def _get_kernel_spec(candidate: KernelCandidate) -> KernelSpec:
    override = _SPEC_OVERRIDES.get(
        (candidate.module_name, candidate.function_name)
    )
    if override is not None:
        return override
    return _build_fallback_spec(candidate)


def _import_triton():
    try:
        import triton  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "triton must be installed to compile kernels"
        ) from exc
    return triton


def _import_symbol(
    module_names: tuple[str, ...], symbol_name: str
) -> Any | None:
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, symbol_name)
        except Exception:
            continue
    return None


def _make_target(arch: str) -> Any:
    target_arch = int(arch)
    gputarget = _import_symbol(
        ("triton.backends.compiler", "triton.compiler.compiler"),
        "GPUTarget",
    )
    if gputarget is None:
        return f"sm_{target_arch}"
    return gputarget("cuda", target_arch, 32)


def _resolve_ast_source() -> Any | None:
    return _import_symbol(
        ("triton.compiler.compiler", "triton.compiler"), "ASTSource"
    )


def _make_triton_config(triton_module: Any, spec: TritonConfigSpec) -> Any:
    return triton_module.Config(
        spec.kwargs,
        num_warps=spec.num_warps,
        num_stages=spec.num_stages,
    )


def _compile_with_best_effort(
    triton_module: Any,
    kernel_fn: Any,
    kernel_spec: KernelSpec,
    arch: str,
) -> Any:
    compile_signature = inspect.signature(triton_module.compile)
    target = _make_target(arch)
    if "signature" in compile_signature.parameters:
        kwargs: dict[str, Any] = {
            "signature": kernel_spec.signature,
            "constants": kernel_spec.constants,
            "output_name": f"{kernel_fn.__name__}_{arch}",
        }
        if kernel_spec.configs:
            kwargs["configs"] = [
                _make_triton_config(triton_module, config)
                for config in kernel_spec.configs
            ]
        if "target" in compile_signature.parameters:
            kwargs["target"] = target
        return triton_module.compile(kernel_fn, **kwargs)

    ast_source_cls = _resolve_ast_source()
    if ast_source_cls is None:
        raise RuntimeError(
            "unable to locate Triton ASTSource for AOT compilation"
        )

    constants = dict(kernel_spec.constants)
    options: dict[str, Any] = {}
    if kernel_spec.configs:
        selected_config = kernel_spec.configs[0]
        constants.update(selected_config.kwargs)
        if selected_config.num_warps is not None:
            options["num_warps"] = selected_config.num_warps
        if selected_config.num_stages is not None:
            options["num_stages"] = selected_config.num_stages

    source = ast_source_cls(
        kernel_fn, signature=kernel_spec.signature, constexprs=constants
    )
    return triton_module.compile(source, target=target, options=options)


def _extract_metadata_path(metadata_group: dict[str, str]) -> Path:
    for path in metadata_group.values():
        candidate = Path(path)
        if candidate.suffix == ".json":
            return candidate
    raise RuntimeError("compiled kernel metadata JSON was not produced")


def _extract_binary_path(metadata_group: dict[str, str]) -> Path:
    binary_candidates = [
        Path(path)
        for path in metadata_group.values()
        if Path(path).suffix != ".json"
    ]
    if not binary_candidates:
        raise RuntimeError("compiled kernel binary artifact was not produced")
    suffix_priority = (".cubin", ".hsaco", ".so", ".ptx")
    for suffix in suffix_priority:
        for candidate in reversed(binary_candidates):
            if candidate.suffix == suffix:
                return candidate
    return binary_candidates[-1]


def _write_manifest(
    manifest_path: Path,
    candidate: KernelCandidate,
    kernel_spec: KernelSpec,
) -> None:
    manifest = {
        "module": candidate.module_name,
        "function": candidate.function_name,
        "signature": kernel_spec.signature,
        "constants": kernel_spec.constants,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _persist_compiled_artifact(
    compiled_kernel: Any,
    output_dir: Path,
    candidate: KernelCandidate,
    kernel_spec: KernelSpec,
    arch: str,
) -> None:
    metadata_group = getattr(compiled_kernel, "metadata_group", None)
    if not isinstance(metadata_group, dict) or not metadata_group:
        raise RuntimeError("compiled kernel did not expose metadata_group")

    metadata_path = _extract_metadata_path(metadata_group)
    binary_path = _extract_binary_path(metadata_group)
    stem = f"{candidate.function_name}_{arch}"
    shutil.copyfile(binary_path, output_dir / f"{stem}.so")
    shutil.copyfile(metadata_path, output_dir / f"{stem}.json")
    _write_manifest(output_dir / f"{stem}.manifest", candidate, kernel_spec)


def _compile_candidate(
    triton_module: Any,
    candidate: KernelCandidate,
    arch: str,
    output_dir: Path,
) -> tuple[bool, str]:
    try:
        module = importlib.import_module(candidate.module_name)
        kernel_fn = getattr(module, candidate.function_name)
        kernel_spec = _get_kernel_spec(candidate)
        compiled_kernel = _compile_with_best_effort(
            triton_module,
            kernel_fn,
            kernel_spec,
            arch,
        )
        _persist_compiled_artifact(
            compiled_kernel,
            output_dir,
            candidate,
            kernel_spec,
            arch,
        )
        return True, "compiled"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "__init__.py").touch(exist_ok=True)

    try:
        archs = _parse_target_archs(args.target_archs)
        triton_module = _import_triton()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    candidates = _discover_kernel_candidates()
    compiled = 0
    failed = 0

    for candidate in candidates:
        for arch in archs:
            success, message = _compile_candidate(
                triton_module,
                candidate,
                arch,
                output_dir,
            )
            status = "ok" if success else "skip"
            print(f"[{status}] {candidate.function_name} sm_{arch}: {message}")
            if success:
                compiled += 1
            else:
                failed += 1

    print(
        "summary: "
        f"files={len({candidate.file_path for candidate in candidates})} "
        f"kernels={len(candidates)} compiled={compiled} skipped={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
