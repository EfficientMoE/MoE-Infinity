import numpy as np

from moe_infinity.memory.expert_entry import ExpertCacheEntry, ExpertTraceEntry
from moe_infinity.memory.expert_priority_score import (
    lfu_score,
    lru_score,
    priority_score,
)


def _trace_entry(matrix, access=1):
    return ExpertTraceEntry(seq_id="s", matrix=matrix, access=access)


def test_lru_score_prioritize():
    entries = {
        ExpertCacheEntry(expert_idx=0, layer_idx=0, timestamp=10),
        ExpertCacheEntry(expert_idx=1, layer_idx=2, timestamp=5),
    }
    lru = lru_score(entries)
    assert {e.r for e in lru} == {10, 5}


def test_lfu_score_normalizes():
    freq = {(0, 0): 2, (1, 0): 0}
    scores = lfu_score(freq)
    mapping = {(e.layer_idx, e.expert_idx): e.r for e in scores}

    assert mapping[(0, 0)] == 1.0
    assert mapping[(0, 1)] == 0.0


def test_priority_score_outputs_entries_for_matrix():
    decoder_entry = _trace_entry(np.ones((2, 2)))
    cache_entries = {ExpertCacheEntry(expert_idx=0, layer_idx=0, timestamp=1)}
    trace_entries = {decoder_entry}
    expert_freq = {(0, 0): 1, (1, 1): 2}

    scores = priority_score(
        expert_freq,
        cache_entries,
        trace_entries,
        decoder_entry,
        current_layer=0,
        total_layer=2,
    )

    assert len(scores) == 4
    assert all(isinstance(score, ExpertCacheEntry) for score in scores)
