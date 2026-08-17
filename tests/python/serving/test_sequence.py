# pyright: reportAny=false

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "moe_infinity"
    / "serving"
    / "sequence.py"
)
_SPEC = spec_from_file_location("sequence_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SamplingParams = _MODULE.SamplingParams
SequenceData = _MODULE.SequenceData
SequenceGroup = _MODULE.SequenceGroup
SequenceStatus = _MODULE.SequenceStatus


def test_sequence_status_transitions() -> None:
    sequence = SequenceData(
        seq_id=1,
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(),
    )

    assert sequence.status is SequenceStatus.WAITING

    sequence.set_status(SequenceStatus.PREFILL)
    sequence.set_status(SequenceStatus.DECODE)
    sequence.set_status(SequenceStatus.FINISHED)

    assert sequence.status is SequenceStatus.FINISHED


def test_sequence_append_output_token() -> None:
    sequence = SequenceData(
        seq_id=2,
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(),
    )

    sequence.set_status(SequenceStatus.PREFILL)
    sequence.set_status(SequenceStatus.DECODE)
    sequence.append_output_token(12)
    sequence.append_output_token(13)

    assert sequence.output_token_ids == [12, 13]
    assert sequence.num_computed_tokens == 4


def test_sequence_group_creation() -> None:
    sequence = SequenceData(
        seq_id=3,
        prompt_token_ids=[7],
        sampling_params=SamplingParams(),
    )
    group = SequenceGroup(request_id="req-1", sequences=[sequence])

    assert group.request_id == "req-1"
    assert group.sequence_ids == [3]
    assert group.get_sequence(3) is sequence


def test_sampling_params_defaults() -> None:
    params = SamplingParams()

    assert params.temperature == 1.0
    assert params.top_k == -1
    assert params.top_p == 1.0
    assert params.max_tokens == 256
    assert params.stop == []
    assert params.repetition_penalty == 1.0


def _fresh_sequence(seq_id: int = 100) -> object:
    return SequenceData(
        seq_id=seq_id,
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(),
    )


def test_dflash_draft_verify_round_loop() -> None:
    sequence = _fresh_sequence()

    sequence.set_status(SequenceStatus.PREFILL)
    sequence.set_status(SequenceStatus.DRAFT)
    sequence.set_status(SequenceStatus.VERIFY)
    sequence.set_status(SequenceStatus.DRAFT)
    sequence.set_status(SequenceStatus.VERIFY)

    assert SequenceStatus.DRAFT.value == "draft"
    assert SequenceStatus.VERIFY.value == "verify"
    assert sequence.status is SequenceStatus.VERIFY


def test_verify_can_finish() -> None:
    sequence = _fresh_sequence()

    sequence.set_status(SequenceStatus.PREFILL)
    sequence.set_status(SequenceStatus.DRAFT)
    sequence.set_status(SequenceStatus.VERIFY)
    sequence.set_status(SequenceStatus.FINISHED)

    assert sequence.status is SequenceStatus.FINISHED


def test_draft_and_verify_swap_then_recover_to_draft() -> None:
    sequence = _fresh_sequence()

    sequence.set_status(SequenceStatus.PREFILL)
    sequence.set_status(SequenceStatus.DRAFT)
    sequence.set_status(SequenceStatus.SWAPPED)
    sequence.set_status(SequenceStatus.DRAFT)
    sequence.set_status(SequenceStatus.VERIFY)
    sequence.set_status(SequenceStatus.SWAPPED)

    assert sequence.status is SequenceStatus.SWAPPED


def test_draft_and_verify_can_cancel() -> None:
    draft_seq = _fresh_sequence(seq_id=101)
    draft_seq.set_status(SequenceStatus.PREFILL)
    draft_seq.set_status(SequenceStatus.DRAFT)
    draft_seq.set_status(SequenceStatus.CANCELLED)
    assert draft_seq.status is SequenceStatus.CANCELLED

    verify_seq = _fresh_sequence(seq_id=102)
    verify_seq.set_status(SequenceStatus.PREFILL)
    verify_seq.set_status(SequenceStatus.DRAFT)
    verify_seq.set_status(SequenceStatus.VERIFY)
    verify_seq.set_status(SequenceStatus.CANCELLED)
    assert verify_seq.status is SequenceStatus.CANCELLED


def test_ordinary_prefill_still_decodes() -> None:
    sequence = _fresh_sequence()

    sequence.set_status(SequenceStatus.PREFILL)
    sequence.set_status(SequenceStatus.DECODE)

    assert sequence.status is SequenceStatus.DECODE


def test_waiting_cannot_enter_verify() -> None:
    sequence = _fresh_sequence()

    with pytest.raises(ValueError, match="invalid transition"):
        sequence.set_status(SequenceStatus.VERIFY)


def test_waiting_cannot_enter_draft() -> None:
    sequence = _fresh_sequence()

    with pytest.raises(ValueError, match="invalid transition"):
        sequence.set_status(SequenceStatus.DRAFT)


def test_ordinary_decode_cannot_enter_verify() -> None:
    sequence = _fresh_sequence()

    sequence.set_status(SequenceStatus.PREFILL)
    sequence.set_status(SequenceStatus.DECODE)

    with pytest.raises(ValueError, match="invalid transition"):
        sequence.set_status(SequenceStatus.VERIFY)
