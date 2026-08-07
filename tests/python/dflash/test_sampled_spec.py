"""Track B: sampled (non-greedy) DFlash -- lossless speculative sampling.

Three autonomous gates, all CPU-only on the tiny fixtures:

(a) Hand-checked unit tests for the pure ops in ``_dflash_sample_ops`` --
    warp order (temperature -> top-k -> top-p, mirroring the engine sampler),
    residual renorm, accept/reject boundaries, and seed determinism.
(b) Distributional parity: over many seeds, the sampled-DFlash token
    histograms (per position and pooled) match a plain sampled target driven
    by an INDEPENDENT reference sampler, within KL/TVD tolerance. The seeds
    are fixed, so the measured values -- and hence the gates -- are
    deterministic; tolerances carry ~2x headroom over the measured values.
(c) Greedy regression: ``temperature == 0`` stays token-identical to plain
    greedy (the v1 contract), even when top_k/top_p are set.

The accept rule under test is per-slot rejection sampling with residual
correction against the target's true conditionals (see the
``_dflash_sample_ops`` module docstring for the losslessness proof): the
block-parallel proposal only needs each draft to be a genuine draw from its
own warped slot distribution ``Q_i``; the lemma of Leviathan et al. then
makes every committed token an exact draw from the warped target ``P_i``.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    TinyDFlashDrafter,
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
    plain_greedy_decode,
    set_determinism,
)

from moe_infinity.spec_decode import (  # noqa: E402
    DFlashSpeculator,
    read_dflash_config,
)
from moe_infinity.spec_decode._dflash_sample_ops import (  # noqa: E402
    acceptance_sampled,
    committed_tokens_sampled,
    residual_distribution,
    warped_probs,
)

PROMPT = torch.tensor([[3, 7, 11, 2, 5]])
PROMPT_LEN = int(PROMPT.shape[1])

# Parity fixtures use a smaller vocab than the TINY_VOCAB=64 default: the
# histogram-TV error of a multinomial empirical distribution scales like
# O(sqrt(vocab / n)), so 16 ids with n~thousands gives sharp tolerances at
# CPU runtime cost. The mask id must then live inside the small vocab.
PARITY_VOCAB = 16
PARITY_MASK_ID = PARITY_VOCAB - 1
PARITY_MAX_NEW = 8


# ---------------------------------------------------------------------------
# (a) Hand-checked unit tests for the pure ops
# ---------------------------------------------------------------------------


def test_warped_probs_temperature_scales_before_softmax():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    probs = warped_probs(logits, temperature=2.0)
    expected = F.softmax(logits / 2.0, dim=-1)
    assert torch.allclose(probs, expected, atol=1e-7)
    assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-6)


def test_warped_probs_top_k_keeps_exactly_k():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    probs = warped_probs(logits, temperature=1.0, top_k=2)
    nz = torch.nonzero(probs[0] > 0).flatten().tolist()
    assert nz == [0, 1]
    expected = F.softmax(torch.tensor([4.0, 3.0]), dim=-1)
    assert torch.allclose(probs[0, :2], expected, atol=1e-7)
    assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-6)


def test_warped_probs_top_p_engine_convention_hand_checked():
    # Engine convention (``GenerationEngine._sample``): drop every token whose
    # cumulative mass EXCEEDS top_p, always keeping the top token.
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    z = math.exp(4) + math.exp(3) + math.exp(2) + math.exp(1) + math.exp(0)
    p0, p1 = math.exp(4) / z, math.exp(3) / z  # 0.6365, 0.2341

    # top_p=0.5: cumsum[0] = 0.6365 > 0.5 already, but the first token is
    # always kept -> degenerate one-hot on token 0.
    probs = warped_probs(logits, top_p=0.5)
    assert probs[0, 0].item() == 1.0
    assert float(probs[0, 1:].sum()) == 0.0

    # top_p=0.9: cumsum = [0.6365, 0.8706, 0.9567, ...] -> keep {0, 1},
    # renormalized to p0/(p0+p1), p1/(p0+p1).
    probs = warped_probs(logits, top_p=0.9)
    nz = torch.nonzero(probs[0] > 0).flatten().tolist()
    assert nz == [0, 1]
    total = p0 + p1
    assert math.isclose(probs[0, 0].item(), p0 / total, rel_tol=1e-5)
    assert math.isclose(probs[0, 1].item(), p1 / total, rel_tol=1e-5)


def test_warped_probs_temperature_applies_before_top_k_top_p():
    # Two-stage check vs. explicitly staged math: T=2 softens the gaps, then
    # top_k=2 keeps the same two ids but with temperature-warped mass.
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    probs = warped_probs(logits, temperature=2.0, top_k=2)
    expected = F.softmax(torch.tensor([2.0, 1.5]), dim=-1)
    assert torch.allclose(probs[0, :2], expected, atol=1e-7)
    assert float(probs[0, 2:].sum()) == 0.0


def test_warped_probs_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature > 0"):
        warped_probs(torch.zeros(1, 4), temperature=0.0)


def test_warped_probs_matches_engine_sampler_warp():
    # Token-level equivalence with the production sampler's warp: same seed,
    # same logits, same params -> same draw. Pins warp ORDER against
    # ``GenerationEngine._sample`` on a case where all three stages bite.
    from moe_infinity.engine.generation_loop import GenerationEngine
    from moe_infinity.engine.types import SamplingParams
    from moe_infinity.memory.kv_cache_manager import KVCacheManager
    from moe_infinity.runtime.attention_types import KVCacheSpec

    engine = GenerationEngine(
        kv_cache_manager=KVCacheManager(
            num_gpu_blocks=8, num_cpu_blocks=2, block_size=4
        ),
        kv_spec=KVCacheSpec(
            num_kv_heads=1, head_dim=8, dtype=torch.float32, block_size=4
        ),
        num_layers=1,
        vocab_size=8,
    )
    torch.manual_seed(7)
    logits = torch.randn(8) * 3.0
    params = SamplingParams(temperature=0.7, top_k=4, top_p=0.85)

    for seed in range(20):
        torch.manual_seed(seed)
        via_engine = engine._sample(logits.unsqueeze(0), params)
        torch.manual_seed(seed)
        via_ops = int(
            torch.multinomial(
                warped_probs(logits, 0.7, top_k=4, top_p=0.85), 1
            ).item()
        )
        assert via_engine == via_ops


def test_residual_distribution_hand_checked():
    p = torch.tensor([0.5, 0.3, 0.1, 0.1])
    q = torch.tensor([0.25, 0.5, 0.0, 0.25])
    # max(0, p - q) = [0.25, 0, 0.1, 0]; mass 0.35 -> [5/7, 0, 2/7, 0].
    r = residual_distribution(p, q)
    assert math.isclose(r[0].item(), 5 / 7, rel_tol=1e-5)
    assert r[1].item() == 0.0
    assert math.isclose(r[2].item(), 2 / 7, rel_tol=1e-5)
    assert r[3].item() == 0.0
    assert math.isclose(float(r.sum()), 1.0, abs_tol=1e-6)


def test_residual_distribution_sums_to_one_randomized():
    torch.manual_seed(0)
    for _ in range(32):
        p = F.softmax(torch.randn(16) * 4, dim=-1)
        q = F.softmax(torch.randn(16) * 4, dim=-1)
        r = residual_distribution(p, q)
        assert math.isclose(float(r.sum()), 1.0, abs_tol=1e-6)
        assert float(r.min()) >= 0.0


def test_residual_distribution_falls_back_to_p_when_equal():
    p = torch.tensor([0.2, 0.3, 0.5])
    r = residual_distribution(p, p.clone())
    assert torch.equal(r, p)


def _one_hot(idx: int, vocab: int) -> torch.Tensor:
    row = torch.zeros(vocab)
    row[idx] = 1.0
    return row


def test_acceptance_sampled_full_accept_draws_bonus_from_last_target_row():
    # Q_i == P_i everywhere -> ratio 1 -> every draft accepted; the bonus is
    # drawn from P_B, made one-hot here so the outcome is exact.
    vocab, num_drafts = 4, 3
    draft_probs = torch.full((num_drafts, vocab), 0.25)
    target_probs = torch.full((num_drafts + 1, vocab), 0.25)
    target_probs[-1] = _one_hot(2, vocab)
    drafts = torch.tensor([1, 3, 0])

    decision = acceptance_sampled(
        draft_probs, target_probs, drafts, generator=torch.Generator().manual_seed(0)
    )
    assert decision.accept == num_drafts
    assert decision.final_token == 2


def test_acceptance_sampled_immediate_reject_emits_residual_correction():
    # Slot 1: P is one-hot on 1, Q is one-hot on 0 -> the drafted 0 has
    # p/q == 0 -> always rejected; residual == P -> correction is exactly 1.
    vocab = 4
    draft_probs = torch.stack([_one_hot(0, vocab), torch.full((vocab,), 0.25)])
    target_probs = torch.stack(
        [_one_hot(1, vocab), torch.full((vocab,), 0.25), _one_hot(2, vocab)]
    )
    drafts = torch.tensor([0, 3])

    decision = acceptance_sampled(
        draft_probs, target_probs, drafts, generator=torch.Generator().manual_seed(0)
    )
    assert decision.accept == 0
    assert decision.final_token == 1


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_acceptance_sampled_boundary_at_k(k: int):
    # Slots 1..k accept (Q == P), slot k+1 rejects on a one-hot mismatch;
    # k == 3 = full accept (bonus from the one-hot last row).
    vocab, num_drafts = 4, 3
    draft_rows = [torch.full((vocab,), 0.25) for _ in range(num_drafts)]
    target_rows = [torch.full((vocab,), 0.25) for _ in range(num_drafts + 1)]
    drafts = torch.zeros(num_drafts, dtype=torch.long)
    if k < num_drafts:
        draft_rows[k] = _one_hot(0, vocab)  # drafts[k] == 0 -> p/q == 0
        target_rows[k] = _one_hot(3, vocab)  # residual one-hot -> final 3
        expected = (k, 3)
    else:
        target_rows[-1] = _one_hot(2, vocab)
        expected = (num_drafts, 2)

    decision = acceptance_sampled(
        torch.stack(draft_rows),
        torch.stack(target_rows),
        drafts,
        generator=torch.Generator().manual_seed(0),
    )
    assert (decision.accept, decision.final_token) == expected


def test_acceptance_sampled_accepts_when_p_exceeds_q():
    # p/q == 1.8 -> min(1, .) == 1 -> deterministic accept of the draft.
    draft_probs = torch.tensor([[0.5, 0.5]])
    target_probs = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
    decision = acceptance_sampled(
        draft_probs,
        target_probs,
        torch.tensor([0]),
        generator=torch.Generator().manual_seed(0),
    )
    assert decision.accept == 1
    assert decision.final_token == 1  # bonus row is one-hot on 1


def test_acceptance_sampled_seed_determinism():
    # A genuinely stochastic case (0 < p/q < 1) replayed under equal seeds
    # must reproduce the exact outcome sequence.
    draft_probs = torch.tensor([[0.5, 0.5]])
    target_probs = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
    drafts = torch.tensor([1])  # p/q == 0.8

    def draw(seed: int, n: int):
        gen = torch.Generator().manual_seed(seed)
        return [
            acceptance_sampled(draft_probs, target_probs, drafts, generator=gen)
            for _ in range(n)
        ]

    assert draw(1234, 64) == draw(1234, 64)
    # Both outcomes occur -> the determinism above is not a degenerate gate.
    outcomes = {(d.accept, d.final_token) for d in draw(1234, 64)}
    assert len(outcomes) > 1


def test_committed_tokens_sampled_split_matches_greedy_layout():
    block = torch.tensor([[100, 11, 12, 13]])
    res = committed_tokens_sampled(block, accept=1, final_token=42)
    assert res.emitted[0].tolist() == [11, 42]
    assert res.block_prefix[0].tolist() == [100, 11]
    assert res.bonus[0].tolist() == [42]

    res = committed_tokens_sampled(block, accept=3, final_token=42)
    assert res.emitted[0].tolist() == [11, 12, 13, 42]
    assert res.block_prefix[0].tolist() == [100, 11, 12, 13]

    res = committed_tokens_sampled(block, accept=0, final_token=42)
    assert res.emitted[0].tolist() == [42]
    assert res.block_prefix[0].tolist() == [100]


# ---------------------------------------------------------------------------
# Shared tiny-model helpers for (b) and (c)
# ---------------------------------------------------------------------------


def _reference_sample(logits: torch.Tensor, temperature, top_k, top_p) -> int:
    """Plain-sampler reference: same warp spec as the engine, independent
    implementation (threshold-free top-k via index scatter, nucleus via
    cumulative mask) -- deliberately NOT importing ``warped_probs`` so the
    parity gate compares two separate code paths."""
    x = logits
    if temperature != 1.0:
        x = x / float(temperature)
    if int(top_k) > 0:
        k = min(int(top_k), int(x.shape[-1]))
        idx = torch.topk(x, k).indices
        kept = torch.full_like(x, float("-inf"))
        kept[idx] = x[idx]
        x = kept
    if float(top_p) < 1.0:
        s, order = torch.sort(x, descending=True)
        c = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
        drop = c > float(top_p)
        drop[0] = False
        s = s.masked_fill(drop, float("-inf"))
        x = torch.full_like(x, float("-inf")).scatter(-1, order, s)
    return int(torch.multinomial(F.softmax(x, dim=-1), 1).item())


@torch.no_grad()
def _plain_sampled_decode(
    model, input_ids, max_new_tokens, temperature, top_k=0, top_p=1.0
):
    """Autoregressive sampled baseline (mirror of ``plain_greedy_decode``)."""
    model.eval()
    out = model(input_ids.clone(), use_cache=True)
    past = out.past_key_values
    nxt = _reference_sample(out.logits[0, -1], temperature, top_k, top_p)
    tokens = [nxt]
    for _ in range(max_new_tokens - 1):
        out = model(
            torch.tensor([[nxt]]), past_key_values=past, use_cache=True
        )
        past = out.past_key_values
        nxt = _reference_sample(out.logits[0, -1], temperature, top_k, top_p)
        tokens.append(nxt)
    return tokens


def _build_spec(vocab_size=None, mask_token_id=None):
    set_determinism(0)
    cfg_kwargs = {} if vocab_size is None else {"vocab_size": vocab_size}
    target = build_tiny_target(seed=0, **cfg_kwargs)
    if mask_token_id is None:
        drafter = build_tiny_drafter(target, seed=1)
        config = read_dflash_config(make_tiny_drafter_config(target.config))
    else:
        # Small-vocab parity fixtures: the default mask id (63) is outside
        # the vocab, so the drafter is built with an explicit in-vocab mask.
        ns = make_tiny_drafter_config(target.config, mask_token_id=mask_token_id)
        config = read_dflash_config(ns)
        set_determinism(1)
        drafter = TinyDFlashDrafter(
            config, target.get_input_embeddings(), target.get_output_embeddings()
        ).to(torch.float32)
        drafter.eval()
    spec = DFlashSpeculator.from_models(target, drafter, config=config, device="cpu")
    return spec, target


# ---------------------------------------------------------------------------
# (b) Distributional parity: sampled DFlash vs. plain sampled target
# ---------------------------------------------------------------------------

PARITY_CONFIGS = [
    {"name": "temp0.8", "temperature": 0.8, "top_k": 0, "top_p": 1.0, "runs": 500},
    {"name": "top_p0.9", "temperature": 1.0, "top_k": 0, "top_p": 0.9, "runs": 500},
    {"name": "top_k5", "temperature": 1.2, "top_k": 5, "top_p": 1.0, "runs": 500},
]


def _tvd(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * float((p - q).abs().sum())


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> float:
    p = p / p.sum() + eps
    q = q / q.sum() + eps
    return float((p * (p / q).log()).sum())


def _run_parity(cfg):
    spec, target = _build_spec(
        vocab_size=PARITY_VOCAB, mask_token_id=PARITY_MASK_ID
    )
    plain_tokens, spec_tokens = [], []
    total_accept = 0
    for run in range(cfg["runs"]):
        torch.manual_seed(10_000 + run)
        plain_tokens.append(
            _plain_sampled_decode(
                target,
                PROMPT,
                PARITY_MAX_NEW,
                cfg["temperature"],
                top_k=cfg["top_k"],
                top_p=cfg["top_p"],
            )
        )
        torch.manual_seed(20_000 + run)
        out = spec.generate(
            PROMPT,
            max_new_tokens=PARITY_MAX_NEW,
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
        )
        spec_new = out[0, PROMPT_LEN:].tolist()
        spec_tokens.append(spec_new)
        total_accept += sum(rec.accept for rec in spec.step_trace)
        # The greedy-path cache accounting must hold in sampled mode too:
        # start advances by accept+1 and the cache ends at start.
        for rec in spec.step_trace:
            assert rec.start == rec.prev_start + rec.accept + 1
            assert rec.target_cache_len == rec.start
    return {
        "plain": plain_tokens,
        "spec": spec_tokens,
        "total_accept": total_accept,
    }


@pytest.fixture(scope="module")
def parity():
    return {cfg["name"]: _run_parity(cfg) for cfg in PARITY_CONFIGS}


def _position_histograms(tokens, vocab):
    n_pos = len(tokens[0])
    h = torch.zeros(n_pos, vocab)
    for row in tokens:
        for j, tok in enumerate(row):
            h[j, tok] += 1
    return h / len(tokens)


def _pooled_histogram(tokens, vocab):
    h = torch.zeros(vocab)
    for row in tokens:
        for tok in row:
            h[tok] += 1
    return h / h.sum()


def test_sampled_parity_pooled_tvd_and_kl(parity):
    """Pooled-over-positions histogram: sharpest statistic (n = runs x 8)."""
    lines = []
    for cfg in PARITY_CONFIGS:
        data = parity[cfg["name"]]
        hp = _pooled_histogram(data["plain"], PARITY_VOCAB)
        hs = _pooled_histogram(data["spec"], PARITY_VOCAB)
        tvd, kl = _tvd(hp, hs), _kl(hp, hs)
        lines.append(
            f"{cfg['name']}: pooled TVD={tvd:.4f} KL={kl:.4f} "
            f"runs={cfg['runs']} total_accept={data['total_accept']}"
        )
        # Deterministic at fixed seeds. Measured pooled TVD at 500 runs is
        # 0.035-0.054 across the configs, matching the multinomial noise
        # floor 0.5*sum(sqrt(2 p (1-p) / n)) ~= 0.036-0.043; the gate carries
        # ~2.3x headroom over that floor while remaining far below what a
        # broken accept rule would show (always-accept / always-reject shift
        # the pooled TVD well above 0.15 with this weak tiny drafter).
        assert tvd <= 0.10, f"{cfg['name']} pooled TVD {tvd:.4f} > 0.10"
        assert kl <= 0.05, f"{cfg['name']} pooled KL {kl:.4f} > 0.05"
    print("\n".join(lines))


def test_sampled_parity_per_position_tvd(parity):
    """Per-position histograms (n = runs): looser, but every position must
    match -- a position-localized bug cannot hide in the pooled average.
    Measured worst-position TVD at 500 runs: 0.124-0.152 (noise floor
    ~= 0.10-0.12); the 0.25 gate is ~2x that floor."""
    for cfg in PARITY_CONFIGS:
        data = parity[cfg["name"]]
        hp = _position_histograms(data["plain"], PARITY_VOCAB)
        hs = _position_histograms(data["spec"], PARITY_VOCAB)
        worst = max(_tvd(hp[j], hs[j]) for j in range(PARITY_MAX_NEW))
        print(f"{cfg['name']}: worst per-position TVD={worst:.4f}")
        assert worst <= 0.25, f"{cfg['name']} position TVD {worst:.4f} > 0.25"


def test_sampled_parity_drafter_is_exercised(parity):
    """Non-degeneracy: drafts are accepted somewhere (the parity above is a
    real draft->verify->accept path, not an accept-0 fallback)."""
    for cfg in PARITY_CONFIGS:
        assert parity[cfg["name"]]["total_accept"] > 0


def test_sampled_anchor_matches_plain_first_token_exactly():
    """max_new_tokens=1: only the anchor is drawn, from the same prefill
    logits and one RNG draw on both paths -> exact token equality per seed."""
    spec, target = _build_spec(
        vocab_size=PARITY_VOCAB, mask_token_id=PARITY_MASK_ID
    )
    for seed in range(16):
        torch.manual_seed(seed)
        plain = _plain_sampled_decode(target, PROMPT, 1, 0.8)
        torch.manual_seed(seed)
        out = spec.generate(PROMPT, max_new_tokens=1, temperature=0.8)
        assert out[0, PROMPT_LEN:].tolist() == plain


# ---------------------------------------------------------------------------
# (c) Greedy regression + sampled determinism
# ---------------------------------------------------------------------------


def test_greedy_path_still_token_identical_to_plain_greedy():
    spec, target = _build_spec()
    max_new = 24
    out = spec.generate(PROMPT, max_new_tokens=max_new, temperature=0.0)
    plain = plain_greedy_decode(target, PROMPT, max_new_tokens=max_new)
    assert torch.equal(out, plain)


def test_greedy_path_ignores_top_k_top_p_like_engine_sampler():
    # Engine semantics (``GenerationEngine._sample``): temperature == 0 is
    # argmax regardless of top_k/top_p; the speculator must match.
    spec, target = _build_spec()
    max_new = 24
    out = spec.generate(
        PROMPT, max_new_tokens=max_new, temperature=0.0, top_k=5, top_p=0.7
    )
    plain = plain_greedy_decode(target, PROMPT, max_new_tokens=max_new)
    assert torch.equal(out, plain)


def test_sampled_generate_is_seed_deterministic():
    spec, _ = _build_spec(vocab_size=PARITY_VOCAB, mask_token_id=PARITY_MASK_ID)
    torch.manual_seed(99)
    first = spec.generate(PROMPT, max_new_tokens=16, temperature=0.8, top_p=0.9)
    torch.manual_seed(99)
    second = spec.generate(PROMPT, max_new_tokens=16, temperature=0.8, top_p=0.9)
    assert torch.equal(first, second)


def test_negative_temperature_rejected():
    spec, _ = _build_spec()
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        spec.generate(PROMPT, max_new_tokens=4, temperature=-0.5)
