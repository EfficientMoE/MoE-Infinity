"""Convert SGLang ``per_token`` expert-distribution dumps to simulator inputs.

SGLang's ``--expert-distribution-recorder-mode per_token`` recorder
(``sglang/srt/eplb/expert_distribution.py``) dumps ``.pt`` files shaped::

    {
        "records": [record, ...],                     # one per forward pass
        "last_physical_to_logical_map": LongTensor[num_layers, num_physical],
    }
    record = {
        "topk_ids_of_layer": IntTensor[num_layers, num_tokens, top_k],  # -1 = unused
        "forward_mode": int,
        "extend_seq_lens": list[int] | None,          # set on prefill/extend passes
        ...                                           # input_ids, positions, ...
    }

This already matches the ``.pt`` schema consumed by
``belady_cache_simulator.py``. This module does the two adaptations that are
still needed:

1. **Physical -> logical remap.** With expert parallelism / EPLB redundant
   experts, ``topk_ids_of_layer`` holds *physical* ids and replicas of one
   logical expert get distinct ids, which would inflate the cache-key space.
   ``remap_to_logical`` collapses them via ``last_physical_to_logical_map``.
   With no redundant experts the map is identity and this is a no-op.
2. **``.pkl`` export** for ``cache_strategy.py``: turn one single-request dump
   (capture with batch size 1) into the ``prefilling``/``decoding`` arrays its
   ``AccessSequence`` parser expects.

For the Belady path, ``--to belady`` just remaps + trims each dump (optionally
merging many dumps into one access stream); no reshaping is required.

Usage::

    # Belady inputs (near pass-through; remap + trim, one file per dump):
    python -m benchmarks.cache_simulator.convert_sglang_traces \
        --inputs "traces_raw/*.pt" --to belady --out data/belady

    # cache_strategy inputs (one dump == one request, batch size 1 capture):
    python -m benchmarks.cache_simulator.convert_sglang_traces \
        --inputs "traces_raw/req_*.pt" --to pkl --name qwen3-235b_mmlu --out data
"""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def expand_inputs(patterns: Sequence[str]) -> List[str]:
    """Expand globs, preserving order and dropping duplicates."""
    seen = set()
    out: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for match in matches:
            if match not in seen:
                seen.add(match)
                out.append(match)
    return out


def load_dump(path: str) -> dict:
    """Load one SGLang dump and validate it carries ``records``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"{path} is not an SGLang expert-distribution dump.")
    return payload


def is_prefill_record(record: dict) -> bool:
    """Classify a forward-pass record as prefill (extend) vs decode.

    Primary signal is ``extend_seq_lens`` (present and non-empty on prefill
    passes). When that field is absent we fall back to the token count: a
    prefill pass carries more than one token.
    """
    extend_seq_lens = record.get("extend_seq_lens")
    if extend_seq_lens is not None:
        if torch.is_tensor(extend_seq_lens):
            return int(extend_seq_lens.numel()) > 0
        return len(extend_seq_lens) > 0
    return int(record["topk_ids_of_layer"].shape[1]) > 1


def remap_to_logical(
    topk_ids_of_layer: torch.Tensor,
    physical_to_logical_map: Optional[torch.Tensor],
) -> torch.Tensor:
    """Map physical expert ids to logical ids, preserving ``-1`` padding.

    ``topk_ids_of_layer`` is ``[num_layers, num_tokens, top_k]``;
    ``physical_to_logical_map`` is ``[num_layers, num_physical]``.
    """
    if physical_to_logical_map is None:
        return topk_ids_of_layer
    mapping = physical_to_logical_map.to(torch.long)
    out = topk_ids_of_layer.to(torch.long)
    num_layers = out.shape[0]
    result = out.clone()
    for layer_idx in range(num_layers):
        row = out[layer_idx]
        valid = row >= 0
        mapped = mapping[layer_idx][row.clamp_min(0)]
        result[layer_idx] = torch.where(valid, mapped, torch.full_like(row, -1))
    return result.to(torch.int32)


def _logical_records(payload: dict, *, remap: bool) -> List[dict]:
    """Return records carrying only logical ``topk_ids_of_layer`` + mode."""
    mapping = payload.get("last_physical_to_logical_map") if remap else None
    records = []
    for record in payload["records"]:
        topk = remap_to_logical(record["topk_ids_of_layer"], mapping)
        records.append(
            {
                "topk_ids_of_layer": topk.to(torch.int32),
                "forward_mode": record.get("forward_mode"),
            }
        )
    return records


def iter_access_records(payload: dict, *, remap: bool = True) -> List[dict]:
    """Belady input: logical records in forward-pass order (pass-through)."""
    return _logical_records(payload, remap=remap)


def _fill_padding(ids: torch.Tensor) -> torch.Tensor:
    """Replace ``-1`` in each layer row with an existing valid id in that row.

    Keeps the array rectangular without introducing a spurious ``-1`` expert
    key, which ``cache_strategy.AccessSequence`` (``np.unique``) would otherwise
    pick up. No-op when there is no padding (e.g. top_k with no empty slots).
    """
    out = ids.to(torch.long).clone()
    flat = out.reshape(out.shape[0], -1)
    for layer_idx in range(flat.shape[0]):
        row = flat[layer_idx]
        valid = row[row >= 0]
        fill = (
            valid[0] if valid.numel() > 0 else torch.zeros((), dtype=row.dtype)
        )
        flat[layer_idx] = torch.where(row >= 0, row, fill)
    return flat.reshape(ids.shape)


def to_pkl_request(
    payload: dict, *, remap: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Convert one single-request dump to ``(prefilling, decoding)`` arrays.

    - ``prefilling`` is ``[num_layers, num_prompt_tokens * top_k]`` (all prefill
      passes concatenated; chunked prefill yields several passes).
    - ``decoding`` is a list of ``[num_layers, 1, top_k]`` arrays, one per decode
      step.

    Requires a batch-size-1 capture: decode passes must carry exactly one token
    so each step maps to this request. Multi-token decode passes (batched
    serving) cannot be attributed to a request without request ids and raise.
    """
    mapping = payload.get("last_physical_to_logical_map") if remap else None
    prefill_chunks: List[torch.Tensor] = []
    decode_steps: List[np.ndarray] = []
    for record in payload["records"]:
        topk = remap_to_logical(record["topk_ids_of_layer"], mapping)
        extend_seq_lens = record.get("extend_seq_lens")
        has_extend = extend_seq_lens is not None and (
            int(extend_seq_lens.numel()) > 0
            if torch.is_tensor(extend_seq_lens)
            else len(extend_seq_lens) > 0
        )
        if has_extend:
            prefill_chunks.append(topk)
        elif int(topk.shape[1]) == 1:
            decode_steps.append(_fill_padding(topk).to(torch.int32).numpy())
        else:
            raise ValueError(
                "Decode pass carries more than one token; capture with batch "
                "size 1 to export the .pkl format (no request ids in the dump)."
            )

    if not prefill_chunks:
        raise ValueError("Dump has no prefill record; cannot build prefilling.")
    prefill = torch.cat(prefill_chunks, dim=1)
    num_layers = prefill.shape[0]
    prefill_2d = _fill_padding(prefill.reshape(num_layers, -1))
    return prefill_2d.to(torch.int32).numpy(), decode_steps


def write_belady(
    paths: Sequence[str], out: str, *, remap: bool, merge: bool
) -> List[str]:
    """Write Belady ``.pt`` inputs. Returns the list of files written."""
    out_path = Path(out)
    written: List[str] = []
    if merge:
        records: List[dict] = []
        for path in paths:
            records.extend(iter_access_records(load_dump(path), remap=remap))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"records": records}, str(out_path))
        written.append(str(out_path))
        return written

    out_path.mkdir(parents=True, exist_ok=True)
    for path in paths:
        records = iter_access_records(load_dump(path), remap=remap)
        dest = out_path / (Path(path).stem + ".pt")
        torch.save({"records": records}, str(dest))
        written.append(str(dest))
    return written


def write_pkl(
    paths: Sequence[str], out_dir: str, name: str, *, remap: bool
) -> Tuple[str, str]:
    """Write ``prefilling_<name>.pkl`` / ``decoding_<name>.pkl``.

    Each input dump becomes one request keyed by its index.
    """
    prefilling: Dict[int, np.ndarray] = {}
    decoding: Dict[int, List[np.ndarray]] = {}
    for rid, path in enumerate(paths):
        prefill, decode_steps = to_pkl_request(load_dump(path), remap=remap)
        prefilling[rid] = prefill
        decoding[rid] = decode_steps

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefill_path = out / f"prefilling_{name}.pkl"
    decode_path = out / f"decoding_{name}.pkl"
    with open(prefill_path, "wb") as handle:
        pickle.dump(prefilling, handle)
    with open(decode_path, "wb") as handle:
        pickle.dump(decoding, handle)
    return str(prefill_path), str(decode_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SGLang per_token dumps to simulator inputs.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="SGLang dump .pt files or glob patterns.",
    )
    parser.add_argument(
        "--to",
        choices=("belady", "pkl"),
        required=True,
        help="belady: trimmed .pt records; pkl: cache_strategy dicts.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory (or output file when --to belady --merge).",
    )
    parser.add_argument(
        "--name",
        default="sglang",
        help="Tag used in prefilling_<name>.pkl / decoding_<name>.pkl.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="(belady) merge all dumps into one access-stream .pt at --out.",
    )
    parser.add_argument(
        "--no-remap",
        action="store_true",
        help="Skip physical->logical remap (ids are already logical).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise ValueError("No input files found.")
    remap = not args.no_remap

    if args.to == "belady":
        written = write_belady(paths, args.out, remap=remap, merge=args.merge)
        print(f"Wrote {len(written)} Belady file(s):")
        for path in written:
            print(f"  {path}")
    else:
        prefill_path, decode_path = write_pkl(
            paths, args.out, args.name, remap=remap
        )
        print(f"Wrote {len(paths)} request(s) to:")
        print(f"  {prefill_path}")
        print(f"  {decode_path}")


if __name__ == "__main__":
    main()
