"""Expert-cache strategies driven by pickled expert-activation traces.

This module replays ``(prefill, decode)`` expert-activation traces stored as
pickled dicts and measures GPU expert-cache hit rate under Belady's optimal
replacement policy. Two implementations are provided:

* ``ExpertCache``     - reference Belady eviction via a linear scan.
* ``HeapExpertCache`` - the same policy backed by a lazy-deletion max-heap for
  large traces.

Both share the ``ExpertIndex`` cache key and the ``AccessSequence`` trace
parser. Trace files are resolved against ``--data-dir`` (see the module README
for where to obtain them); nothing trace-shaped is committed to the repo.
"""

from __future__ import annotations

import argparse
import heapq
import os
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:  # Progress bars are optional.
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else []


@dataclass
class ExpertIndex:
    """Cache key identifying one expert within one layer."""

    expert_idx: int
    layer_idx: int

    def __hash__(self):
        """Generate a stable hash based on expert and layer."""
        return hash((self.expert_idx, self.layer_idx))

    # Define ordering so heap tie-breaks are deterministic even when the
    # next-use position is ``inf``.
    def __lt__(self, other):
        if self.expert_idx == other.expert_idx:
            return self.layer_idx < other.layer_idx
        return self.expert_idx < other.expert_idx

    def __gt__(self, other):
        if self.expert_idx == other.expert_idx:
            return self.layer_idx > other.layer_idx
        return self.expert_idx > other.expert_idx

    def __eq__(self, other):
        return (
            self.expert_idx == other.expert_idx
            and self.layer_idx == other.layer_idx
        )


class HeapExpertCache:
    """Belady optimal cache using a lazy-deletion max-heap for eviction."""

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.cache = {}
        self.access_sequence = []
        self.access_sequence_dict = {}
        self.current_index = 0
        self.hits = 0
        self.misses = 0
        self.decode_hits = 0
        self.decode_misses = 0

        # Python's heapq is a min-heap; store negated next-use indices to get
        # max-heap behaviour. Items are ``(-next_use_index, key)``.
        self.max_heap = []
        # Each key's currently valid next-use position.
        self.cache_next_use = {}

    def set_access_sequence(self, sequence: List[ExpertIndex]):
        self.access_sequence = sequence
        for idx, expert in tqdm(
            enumerate(self.access_sequence),
            total=len(self.access_sequence),
            desc="Preprocessing skip list",
        ):
            if expert not in self.access_sequence_dict:
                self.access_sequence_dict[expert] = deque([])
            self.access_sequence_dict[expert].append(idx)
        print(f"Access sequence length: {len(self.access_sequence)}")

    def access(self, key: ExpertIndex, value=None, decode=False):
        self.current_index += 1
        if key in self.access_sequence_dict and self.access_sequence_dict[key]:
            self.access_sequence_dict[key].popleft()

        if key in self.cache:
            if decode:
                self.decode_hits += 1
            self.hits += 1
            self._push_new_next_use(key)
            return self.cache[key]
        else:
            if decode:
                self.decode_misses += 1
            self.misses += 1
            if value is not None:
                if len(self.cache) >= self.cache_size:
                    self.evict()
                self.cache[key] = value
                self._push_new_next_use(key)
            return None

    def _push_new_next_use(self, key):
        """Record the next use of ``key`` on the heap."""
        next_use = float("inf")
        if key in self.access_sequence_dict and self.access_sequence_dict[key]:
            next_use = self.access_sequence_dict[key][0]

        self.cache_next_use[key] = next_use
        heapq.heappush(self.max_heap, (-next_use, key))

    def evict(self):
        """Evict the resident key with the farthest next use."""
        while True:
            if not self.max_heap:
                return

            neg_next_use, key_to_evict = self.max_heap[0]
            if (
                key_to_evict in self.cache
                and self.cache_next_use[key_to_evict] == -neg_next_use
            ):
                heapq.heappop(self.max_heap)
                del self.cache[key_to_evict]
                del self.cache_next_use[key_to_evict]
                return
            else:
                heapq.heappop(self.max_heap)

    def get_hit_rate(self) -> float:
        total_accesses = self.hits + self.misses
        return self.hits / total_accesses if total_accesses > 0 else 0.0

    def get_decode_hit_rate(self) -> float:
        total_accesses = self.decode_hits + self.decode_misses
        if total_accesses > 0:
            return self.decode_hits / total_accesses
        return 0.0


class ExpertCache:
    """Reference Belady optimal cache using a linear eviction scan."""

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.cache = {}
        self.access_sequence = []
        self.current_index = 0
        self.hits = 0
        self.decode_hits = 0
        self.decode_misses = 0
        self.misses = 0
        self.access_sequence_dict = {}  # skip list for access sequence

    def set_access_sequence(self, sequence: List[ExpertIndex]):
        """Set the access sequence for Belady's optimal strategy."""
        self.access_sequence = sequence
        for idx, expert in tqdm(
            enumerate(self.access_sequence),
            desc="Preprocessing skip list",
            total=len(self.access_sequence),
        ):
            if expert not in self.access_sequence_dict:
                self.access_sequence_dict[expert] = deque([])
            self.access_sequence_dict[expert].append(idx)

        print(f"Access sequence length: {len(self.access_sequence)}")

    def access(self, key: ExpertIndex, value=None, decode=False):
        """Access or insert an item in the cache.

        A present key is a hit. A missing key with a provided ``value`` is
        inserted, evicting the Belady victim first when the cache is full.
        """
        self.current_index += 1
        if (
            key in self.access_sequence_dict
            and len(self.access_sequence_dict[key]) > 0
        ):
            self.access_sequence_dict[key].popleft()
        if key in self.cache:
            if decode:
                self.decode_hits += 1
            self.hits += 1
            return self.cache[key]
        else:
            if decode:
                self.decode_misses += 1
            self.misses += 1
            if value is not None:
                if len(self.cache) >= self.cache_size:
                    self.evict()
                self.cache[key] = value
            return None

    def evict(self):
        """Evict an item based on Belady's optimal strategy."""
        farthest_use = -1
        key_to_evict = None
        for key in self.cache:
            try:
                next_use = self.access_sequence_dict[key][0]
            except (ValueError, IndexError):
                next_use = float("inf")  # Key is not used again.
            if next_use > farthest_use:
                farthest_use = next_use
                key_to_evict = key
        if key_to_evict is not None:
            del self.cache[key_to_evict]

    def get_hit_rate(self) -> float:
        """Calculate the cache hit rate."""
        total_accesses = self.hits + self.misses
        return self.hits / total_accesses if total_accesses > 0 else 0.0

    def get_decode_hit_rate(self) -> float:
        total_accesses = self.decode_hits + self.decode_misses
        if total_accesses > 0:
            return self.decode_hits / total_accesses
        return 0.0


class AccessSequence:
    """Parse pickled prefill/decode traces into expert access sequences."""

    @dataclass
    class Request:
        prefillAccessSequence: List[ExpertIndex]
        decodeAccessSequence: List[ExpertIndex]

    def __init__(
        self,
        prefilling_trace: Dict[int, np.ndarray],
        decoding_trace: Dict[int, List[np.ndarray]],
        sequence_num: Optional[int] = None,
    ):
        self.current_request = 0
        self.prefilling_trace = prefilling_trace
        self.decoding_trace = decoding_trace
        self.requests = self.parse_sequence(sequence_num)

    def parse_sequence(
        self, count: Optional[int] = None, enable_encoder=True
    ) -> List[Request]:
        iteration_order = sorted(self.decoding_trace.keys())

        results = []
        sequence_num = 0
        for i in iteration_order:
            if count is not None and sequence_num >= count:
                break
            sequence_num += 1
            if i not in self.decoding_trace.keys():
                continue

            prefilling_trace = self.prefilling_trace[i]
            decoding_trace = self.decoding_trace[i]
            prefillAccessSequence = []
            decodeAccessSequence = []
            layer_num = prefilling_trace.shape[0]
            for layer_idx in range(layer_num):
                token_trace_i = prefilling_trace[layer_idx]
                unique_vectors = np.unique(token_trace_i)
                prefillAccessSequence.extend(
                    [
                        ExpertIndex(expert_idx, layer_idx)
                        for expert_idx in unique_vectors
                    ]
                )

            iteration_num = len(decoding_trace)
            for iteration_idx in range(iteration_num):
                layer_num = 0
                activated_expert_num = 0
                if len(decoding_trace[iteration_idx].shape) == 3:
                    layer_num, _, activated_expert_num = decoding_trace[
                        iteration_idx
                    ].shape
                elif len(decoding_trace[iteration_idx].shape) == 2:
                    layer_num, activated_expert_num = decoding_trace[
                        iteration_idx
                    ].shape
                else:
                    raise ValueError(
                        "decoding_trace shape is not correct, got "
                        f"{decoding_trace[iteration_idx].shape}"
                    )
                for layer_idx in range(layer_num):
                    current_iteration_trace = None
                    if len(decoding_trace[iteration_idx].shape) == 3:
                        current_iteration_trace = [
                            ExpertIndex(expert_idx, layer_idx)
                            for expert_idx in decoding_trace[iteration_idx][
                                layer_idx
                            ][0]
                        ]
                    elif len(decoding_trace[iteration_idx].shape) == 2:
                        current_iteration_trace = [
                            ExpertIndex(expert_idx, layer_idx)
                            for expert_idx in decoding_trace[iteration_idx][
                                layer_idx
                            ]
                        ]
                    decodeAccessSequence.extend(current_iteration_trace)

            results.append(
                self.Request(prefillAccessSequence, decodeAccessSequence)
            )
        return results

    def get_next_request(self) -> Optional[Request]:
        if self.current_request >= len(self.requests):
            return None
        request = self.requests[self.current_request]
        self.current_request += 1
        return request

    def get_access_sequence(
        self, count: Optional[int] = None
    ) -> List[ExpertIndex]:
        access_sequence = []
        count_len = 0
        for request in self.requests:
            if count is not None and count_len >= count:
                break
            access_sequence += request.prefillAccessSequence
            access_sequence += request.decodeAccessSequence
            count_len += 1
        return access_sequence

    def get_request_count(self) -> int:
        return len(self.requests)


# Default cache size and trace filenames per model. Paths are relative to
# ``--data-dir`` (see README.md "Trace data" for the canonical location).
MODEL_CONFIG = {
    "DeepSeek-V2-Lite-Chat": {
        "cache_size": 360,
        "prefilling_trace": "prefilling_chat.pkl",
        "decoding_trace": "decoding_chat.pkl",
    },
    "SwitchTransformers": {
        "cache_size": 953,
        "prefilling_trace": "measurement_results/"
        "flan_inputs_switch_google_switch-large-128_"
        "encoder_expert_activation.pkl",
        "decoding_trace": "measurement_results/"
        "flan_inputs_switch_google_switch-large-128_"
        "decoder_expert_activation.pkl",
    },
    "NLLB": {
        "cache_size": 52,
        "prefilling_trace": "measurement_results/"
        "flan_inputs_nllb_facebook_nllb-moe-54b_"
        "encoder_expert_activation.pkl",
        "decoding_trace": "measurement_results/"
        "flan_inputs_nllb_facebook_nllb-moe-54b_"
        "decoder_expert_activation.pkl",
    },
    "Arctic": {
        "cache_size": 76,
        "prefilling_trace": "../trace/measurement_snowflake/"
        "mmlu_snowflake_snowflake_snowflake-arctic-instruct_"
        "encoder_expert_activation.pkl",
        "decoding_trace": "../trace/measurement_snowflake/"
        "mmlu_snowflake_snowflake_snowflake-arctic-instruct_"
        "decoder_expert_activation.pkl",
    },
    "Mixtral": {
        "cache_size": 39,
        "prefilling_trace": "../trace/measurement_mixtral/"
        "mmlu_mixtral_mistralai_mixtral-8x7b-instruct-v0.1_"
        "encoder_expert_activation.pkl",
        "decoding_trace": "../trace/measurement_mixtral/"
        "mmlu_mixtral_mistralai_mixtral-8x7b-instruct-v0.1_"
        "decoder_expert_activation.pkl",
    },
}


def _load_trace(path: Optional[str]):
    if path is None:
        return {}
    with open(path, "rb") as handle:
        return pickle.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay pickled expert-activation traces through a "
        "Belady optimal expert cache.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIG.keys()),
        default="DeepSeek-V2-Lite-Chat",
        help="Model preset selecting default cache size and trace files.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MOE_SIM_DATA_DIR", "."),
        help="Directory holding the .pkl trace files "
        "(default: $MOE_SIM_DATA_DIR or current directory).",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=None,
        help="Override the preset cache size (entries).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of requests to replay (default: 100).",
    )
    parser.add_argument(
        "--strategy",
        choices=("heap", "scan"),
        default="heap",
        help="Belady implementation: heap (fast) or scan (reference).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIG[args.model]
    cache_size = args.cache_size or config["cache_size"]

    prefilling_path = config.get("prefilling_trace")
    if prefilling_path is not None:
        prefilling_path = os.path.join(args.data_dir, prefilling_path)
    decoding_path = os.path.join(args.data_dir, config["decoding_trace"])

    prefilling_trace = _load_trace(prefilling_path)
    decoding_trace = _load_trace(decoding_path)

    access_sequence = AccessSequence(
        prefilling_trace, decoding_trace, args.max_samples
    )

    cache_cls = HeapExpertCache if args.strategy == "heap" else ExpertCache
    expert_cache = cache_cls(cache_size)
    expert_cache.set_access_sequence(access_sequence.get_access_sequence())

    request_count = access_sequence.get_request_count()
    for _ in tqdm(range(request_count), desc="Simulation"):
        request = access_sequence.get_next_request()
        if request is None:
            break
        for expert in request.prefillAccessSequence:
            expert_cache.access(expert, value=1)
        for expert in request.decodeAccessSequence:
            expert_cache.access(expert, value=1, decode=True)

    print(f"Model: {args.model} (cache_size={cache_size})")
    print(f"Cache hit rate: {expert_cache.get_hit_rate()}")
    print(f"Decode hit rate: {expert_cache.get_decode_hit_rate()}")
    print(f"Cache hits: {expert_cache.hits}")
    print(f"Cache misses: {expert_cache.misses}")
    print(f"Total accesses: {expert_cache.hits + expert_cache.misses}")


if __name__ == "__main__":
    main()
