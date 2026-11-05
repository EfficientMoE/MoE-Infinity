"""C++ engine eviction policies == the Python cache-simulator reference.

``core/prefetch/eviction_policy.cpp`` ports the simulator's on-demand
policies so the engine can select them. It has no CUDA / torch dependency, so
this test compiles it with the host C++ compiler together with
``tests/cpp/unit/prefetch/eviction_policy_driver.cpp`` and checks that every
policy produces exactly the simulator's hit count on fuzzed traces (random
layer order and decode-like cyclic traces). Skipped when no compiler exists.
"""

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.cache_simulator.policies import build_policy, simulate

ROOT = Path(__file__).resolve().parents[3]
POLICIES = [
    "lru",
    "lfu",
    "lfu_k",
    "lhu",
    "lru_k",
    "arc",
    "2q",
    "lirs",
    "s3fifo",
    "sieve",
    "w_tinylfu",
    "layer_lru",
    "lcnu",
]


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("eviction") / "driver"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            f"-I{ROOT / 'core'}",
            str(ROOT / "core/prefetch/eviction_policy.cpp"),
            str(ROOT / "tests/cpp/unit/prefetch/eviction_policy_driver.cpp"),
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def cpp_hits(driver, accesses, queries):
    lines = [str(len(accesses))]
    lines += [f"{layer} {expert}" for layer, expert in accesses]
    lines.append(str(len(queries)))
    for name, cap, num_layers, alpha, beta, prior in queries:
        lines.append(f"{name} {cap} {num_layers} {alpha!r} {beta!r} {prior!r}")
    result = subprocess.run(
        [str(driver)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return [int(x) for x in result.stdout.split()]


def python_hits(name, cap, accesses, num_layers, alpha, beta, prior):
    kwargs = {}
    if name == "layer_lru":
        kwargs = {"num_layers": num_layers or None}
    elif name == "lcnu":
        kwargs = {
            "num_layers": num_layers or None,
            "alpha": alpha,
            "beta": beta,
            "prior": prior,
        }
    hits, _, _ = simulate(build_policy(name, cap, **kwargs), accesses)
    return hits


def random_trace(rng, length, layers, experts):
    return [
        (rng.randrange(layers), rng.randrange(experts)) for _ in range(length)
    ]


def decode_trace(rng, steps, layers, experts, topk, keep=0.4):
    """Layer-cyclic decode trace with skew and token-to-token reuse."""
    weights = [1.0 / (rank + 1) for rank in range(experts)]
    current = [rng.sample(range(experts), topk) for _ in range(layers)]
    accesses = []
    for _ in range(steps):
        for layer in range(layers):
            picks = [e for e in current[layer] if rng.random() < keep]
            pool = [e for e in range(experts) if e not in picks]
            while len(picks) < topk:
                expert = rng.choices(pool, [weights[e] for e in pool])[0]
                picks.append(expert)
                pool.remove(expert)
            current[layer] = picks
            accesses.extend((layer, e) for e in picks)
    return accesses


def check(driver, accesses, caps, num_layers, rng):
    queries = []
    for cap in caps:
        for name in POLICIES:
            alpha = rng.choice((0.3, 0.2, 0.5))
            beta = rng.choice((0.5, 0.0, 1.0, 0.7))
            prior = rng.choice((1 / 16, 0.25))
            queries.append((name, cap, num_layers, alpha, beta, prior))
    got = cpp_hits(driver, accesses, queries)
    for query, hits in zip(queries, got):
        name, cap, layers, alpha, beta, prior = query
        want = python_hits(name, cap, accesses, layers, alpha, beta, prior)
        assert hits == want, (query, hits, want)


def test_policy_names_match_simulator(driver):
    names = subprocess.run(
        [str(driver), "--names"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert names == POLICIES
    accesses = [(0, 0), (1, 1), (0, 0)]
    queries = [(name, 2, 2, 0.3, 0.5, 0.0625) for name in POLICIES]
    queries.append(("legacy", 2, 2, 0.3, 0.5, 0.0625))
    got = cpp_hits(driver, accesses, queries)
    assert all(h >= 0 for h in got[:-1])
    assert got[-1] == -1


def test_config_accepts_every_native_policy():
    from moe_infinity.utils.config import EXPERT_EVICTION_POLICIES

    assert EXPERT_EVICTION_POLICIES == ("legacy", *POLICIES)


def test_parity_on_random_traces(driver):
    rng = random.Random(1234)
    for _ in range(30):
        layers = rng.randrange(1, 5)
        experts = rng.randrange(2, 9)
        accesses = random_trace(rng, rng.randrange(1, 200), layers, experts)
        num_layers = rng.choice((0, layers))
        check(driver, accesses, (1, 2, 3, 5, 8, 13), num_layers, rng)


def test_parity_on_decode_traces(driver):
    rng = random.Random(99)
    for _ in range(12):
        layers = rng.randrange(2, 9)
        experts = rng.randrange(6, 33)
        topk = rng.randrange(1, min(5, experts))
        accesses = decode_trace(
            rng, rng.randrange(20, 120), layers, experts, topk
        )
        working_set = layers * topk
        caps = sorted(
            {
                1,
                max(1, working_set // 2),
                max(1, working_set - topk),
                working_set + 1,
                2 * working_set,
                layers * experts // 2,
            }
        )
        check(driver, accesses, caps, layers, rng)
