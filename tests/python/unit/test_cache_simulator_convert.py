"""Behavioral tests for the SGLang -> simulator trace converter.

These build synthetic SGLang ``per_token`` dumps (plain torch tensors) and
verify the conversion logic end-to-end, including feeding the converted output
back into the simulator. No SGLang or GPU is required.
"""

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.cache_simulator.belady_cache_simulator import (  # noqa: E402
    extract_access_sequence,
)
from benchmarks.cache_simulator.cache_strategy import (  # noqa: E402
    AccessSequence,
    ExpertIndex,
)
from benchmarks.cache_simulator.convert_sglang_traces import (  # noqa: E402
    is_prefill_record,
    iter_access_records,
    remap_to_logical,
    to_pkl_request,
    write_belady,
    write_pkl,
)


def _record(topk_ids, *, extend_seq_lens=None, forward_mode=0):
    return {
        "topk_ids_of_layer": torch.tensor(topk_ids, dtype=torch.int32),
        "extend_seq_lens": extend_seq_lens,
        "forward_mode": forward_mode,
    }


def _payload(records, physical_to_logical=None):
    dump = {"records": records}
    if physical_to_logical is not None:
        dump["last_physical_to_logical_map"] = torch.tensor(
            physical_to_logical, dtype=torch.long
        )
    return dump


def _batch1_payload(physical_to_logical=None):
    """A one-request dump: one prefill pass (3 tokens) + two decode steps."""
    prefill = _record(
        [[[0, 1], [1, -1], [2, 2]], [[3, 3], [3, 3], [3, 3]]],
        extend_seq_lens=[3],
    )
    decode0 = _record([[[0, -1]], [[3, 3]]], extend_seq_lens=None)
    decode1 = _record([[[1, 2]], [[3, 3]]], extend_seq_lens=None)
    return _payload([prefill, decode0, decode1], physical_to_logical)


# --------------------------------------------------------------------------- #
# is_prefill_record
# --------------------------------------------------------------------------- #
def test_is_prefill_record_uses_extend_seq_lens():
    assert is_prefill_record(_record([[[0, 1]]], extend_seq_lens=[1]))
    assert is_prefill_record(
        _record([[[0, 1]]], extend_seq_lens=torch.tensor([1]))
    )
    assert not is_prefill_record(_record([[[0, 1]]], extend_seq_lens=[]))


def test_is_prefill_record_fallback_on_token_count():
    # No extend_seq_lens: >1 token is prefill, 1 token is decode.
    assert is_prefill_record(_record([[[0, 1], [2, 3]]], extend_seq_lens=None))
    assert not is_prefill_record(_record([[[0, 1]]], extend_seq_lens=None))


# --------------------------------------------------------------------------- #
# remap_to_logical
# --------------------------------------------------------------------------- #
def test_remap_collapses_physical_replicas_and_keeps_padding():
    topk = torch.tensor([[[2, -1]], [[3, 0]]], dtype=torch.int32)
    # layer0: phys 2,3 -> logical 1; layer1: phys 0,2 -> 0, phys 1,3 -> 1
    mapping = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    out = remap_to_logical(topk, mapping)
    assert out.tolist() == [[[1, -1]], [[1, 0]]]
    assert out.dtype == torch.int32


def test_remap_identity_and_none_are_noops():
    topk = torch.tensor([[[2, -1]], [[3, 0]]], dtype=torch.int32)
    identity = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    assert remap_to_logical(topk, identity).tolist() == topk.tolist()
    assert remap_to_logical(topk, None).tolist() == topk.tolist()


# --------------------------------------------------------------------------- #
# iter_access_records (Belady path)
# --------------------------------------------------------------------------- #
def test_iter_access_records_preserves_order_and_remaps():
    mapping = [[0, 0, 1, 1], [0, 1, 0, 1]]
    records = iter_access_records(_batch1_payload(mapping), remap=True)
    assert len(records) == 3  # prefill + 2 decode, in order
    assert records[0]["topk_ids_of_layer"].dtype == torch.int32
    # Physical ids remapped: layer0 of prefill token0 [0,1] -> logical [0,0].
    assert records[0]["topk_ids_of_layer"][0, 0].tolist() == [0, 0]


def test_iter_access_records_feeds_belady():
    records = iter_access_records(_batch1_payload(), remap=False)
    accesses = extract_access_sequence(
        records, ignore_layers=0, include_prefill=True
    )
    # Every (layer, expert) access is a valid non-negative pair.
    assert accesses
    assert all(layer >= 0 and expert >= 0 for layer, expert in accesses)


# --------------------------------------------------------------------------- #
# to_pkl_request (cache_strategy path)
# --------------------------------------------------------------------------- #
def test_to_pkl_request_shapes_and_padding_removed():
    prefill, decode_steps = to_pkl_request(_batch1_payload(), remap=False)
    assert prefill.shape == (2, 6)  # [num_layers, num_prompt_tokens * top_k]
    assert (prefill >= 0).all()  # padding filled, no -1
    assert len(decode_steps) == 2
    for step in decode_steps:
        assert step.shape == (2, 1, 2)
        assert (step >= 0).all()


def test_to_pkl_request_concatenates_chunked_prefill():
    # Two prefill passes (chunked prefill) + one decode step.
    chunk0 = _record([[[0, 1]], [[2, 3]]], extend_seq_lens=[1])
    chunk1 = _record([[[4, 5]], [[6, 7]]], extend_seq_lens=[1])
    decode = _record([[[0, 1]], [[2, 3]]], extend_seq_lens=None)
    prefill, decode_steps = to_pkl_request(
        _payload([chunk0, chunk1, decode]), remap=False
    )
    assert prefill.shape == (2, 4)  # two prefill tokens * top_k=2
    assert len(decode_steps) == 1


def test_to_pkl_request_rejects_batched_decode():
    prefill = _record([[[0, 1]], [[2, 3]]], extend_seq_lens=[1])
    batched_decode = _record(
        [[[0, 1], [2, 3]], [[0, 1], [2, 3]]], extend_seq_lens=None
    )  # 2 tokens, no extend -> batched decode
    with pytest.raises(ValueError, match="batch"):
        to_pkl_request(_payload([prefill, batched_decode]), remap=False)


def test_to_pkl_request_output_parses_in_access_sequence():
    prefill, decode_steps = to_pkl_request(_batch1_payload(), remap=False)
    seq = AccessSequence({0: prefill}, {0: decode_steps})
    assert seq.get_request_count() == 1
    request = seq.requests[0]
    # Prefill layer 0 experts are the unique ids {0,1,2}; layer 1 is {3}.
    prefill_pairs = {
        (idx.expert_idx, idx.layer_idx) for idx in request.prefillAccessSequence
    }
    assert (3, 1) in prefill_pairs
    assert {(0, 0), (1, 0), (2, 0)} <= prefill_pairs
    assert len(request.decodeAccessSequence) > 0
    assert all(
        isinstance(idx, ExpertIndex) for idx in request.decodeAccessSequence
    )


# --------------------------------------------------------------------------- #
# File writers round-trip through the simulator
# --------------------------------------------------------------------------- #
def test_write_belady_per_file_and_merge(tmp_path):
    p0 = tmp_path / "a.pt"
    p1 = tmp_path / "b.pt"
    torch.save(_batch1_payload(), str(p0))
    torch.save(_batch1_payload(), str(p1))

    out_dir = tmp_path / "belady"
    written = write_belady(
        [str(p0), str(p1)], str(out_dir), remap=True, merge=False
    )
    assert len(written) == 2
    loaded = torch.load(written[0], map_location="cpu", weights_only=False)
    assert "records" in loaded and len(loaded["records"]) == 3

    merged = tmp_path / "stream.pt"
    write_belady([str(p0), str(p1)], str(merged), remap=True, merge=True)
    loaded_merged = torch.load(
        str(merged), map_location="cpu", weights_only=False
    )
    assert len(loaded_merged["records"]) == 6  # 3 + 3


def test_write_pkl_round_trip(tmp_path):
    p0 = tmp_path / "req0.pt"
    p1 = tmp_path / "req1.pt"
    torch.save(_batch1_payload(), str(p0))
    torch.save(_batch1_payload(), str(p1))

    prefill_path, decode_path = write_pkl(
        [str(p0), str(p1)], str(tmp_path), "unit", remap=True
    )
    with open(prefill_path, "rb") as handle:
        prefilling = pickle.load(handle)
    with open(decode_path, "rb") as handle:
        decoding = pickle.load(handle)
    assert set(prefilling.keys()) == {0, 1}
    assert isinstance(prefilling[0], np.ndarray)
    assert isinstance(decoding[0], list)

    seq = AccessSequence(prefilling, decoding)
    assert seq.get_request_count() == 2
