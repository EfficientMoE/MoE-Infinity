from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any, Optional

import torch


def _import_triton_symbol(
    module_names: tuple[str, ...], symbol_name: str
) -> Any | None:
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, symbol_name)
        except Exception:
            continue
    return None


ASTSource = _import_triton_symbol(
    ("triton.compiler.compiler", "triton.compiler"), "ASTSource"
)
CompiledKernel = _import_triton_symbol(
    ("triton.compiler.compiler", "triton.compiler"),
    "CompiledKernel",
)
if CompiledKernel is None:  # pragma: no cover - Triton not installed
    CompiledKernel = Any
make_backend = _import_triton_symbol(
    ("triton.compiler.compiler", "triton.compiler"),
    "make_backend",
)
GPUTarget = _import_triton_symbol(
    ("triton.backends.compiler", "triton.compiler.compiler"),
    "GPUTarget",
)

_KERNEL_CACHE: dict[tuple[str, str], Optional[CompiledKernel]] = {}
_KERNEL_DIR = Path(__file__).resolve().parent
_COMPILED_DIR = _KERNEL_DIR / "_compiled"
_RUNTIME_DIR = _COMPILED_DIR / ".runtime"


def _current_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}"


def _make_target(arch: str) -> Any | None:
    if GPUTarget is None:
        return None
    return GPUTarget("cuda", int(arch), 32)


def _runtime_binary_ext(arch: str) -> str:
    if make_backend is None:
        return "so"
    target = _make_target(arch)
    if target is None:
        return "so"
    try:
        return str(make_backend(target).binary_ext)
    except Exception:
        return "so"


def _artifact_paths(kernel_name: str, arch: str) -> tuple[Path, Path, Path]:
    stem = f"{kernel_name}_{arch}"
    return (
        _COMPILED_DIR / f"{stem}.so",
        _COMPILED_DIR / f"{stem}.json",
        _COMPILED_DIR / f"{stem}.manifest",
    )


def _load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return None


def _build_ast_source(manifest: dict[str, Any]) -> Any | None:
    if ASTSource is None:
        return None
    module = importlib.import_module(manifest["module"])
    kernel_fn = getattr(module, manifest["function"])
    signature = dict(manifest.get("signature", {}))
    constants = dict(manifest.get("constants", {}))
    return ASTSource(kernel_fn, signature=signature, constexprs=constants)


def _prepare_runtime_artifacts(
    kernel_name: str,
    arch: str,
    so_path: Path,
    metadata_path: Path,
) -> dict[str, str]:
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    binary_ext = _runtime_binary_ext(arch)
    stem = f"{kernel_name}_{arch}"
    runtime_binary_path = _RUNTIME_DIR / f"{stem}.{binary_ext}"
    runtime_metadata_path = _RUNTIME_DIR / f"{stem}.json"

    if binary_ext == "so":
        runtime_binary_path = so_path
    elif not runtime_binary_path.exists():
        shutil.copyfile(so_path, runtime_binary_path)

    if (
        runtime_metadata_path != metadata_path
        and not runtime_metadata_path.exists()
    ):
        shutil.copyfile(metadata_path, runtime_metadata_path)

    return {
        runtime_metadata_path.name: str(runtime_metadata_path),
        runtime_binary_path.name: str(runtime_binary_path),
    }


def load_compiled_kernel(kernel_name: str) -> Optional[CompiledKernel]:
    arch = _current_arch()
    if arch is None:
        return None

    cache_key = (kernel_name, arch)
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    so_path, metadata_path, manifest_path = _artifact_paths(kernel_name, arch)
    if not so_path.exists() or not metadata_path.exists():
        _KERNEL_CACHE[cache_key] = None
        return None

    manifest = _load_manifest(manifest_path)
    if manifest is None:
        _KERNEL_CACHE[cache_key] = None
        return None

    try:
        metadata = json.loads(metadata_path.read_text())
        source = _build_ast_source(manifest)
        if source is None or CompiledKernel is Any:
            kernel = None
        else:
            metadata_group = _prepare_runtime_artifacts(
                kernel_name,
                arch,
                so_path,
                metadata_path,
            )
            kernel = CompiledKernel(
                source, metadata_group, metadata.get("hash", kernel_name)
            )
    except Exception:
        kernel = None

    _KERNEL_CACHE[cache_key] = kernel
    return kernel


__all__ = ["load_compiled_kernel"]
