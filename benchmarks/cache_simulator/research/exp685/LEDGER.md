# LEDGER — warp experiment chain (protocol v2: safe settle)

Safe Settle Pattern (Michael, 2026-11-03): results pass through
environment variables into `python3 -c` (inert bytes — no shell parsing,
no sed delimiters); every git operation is path-scoped (shared worktree:
never `-am`, never repo-wide diff); leak scan runs BEFORE commit. One
entry per experiment; entry frozen by the commit hash of this file
BEFORE launch; results APPEND only.

---

## exp685: Bounded nursery ring — admission-free probation (partition dose)

- Question: exp683's decomposition says wrapper filters pay 1/k of the
  burst reuse window at the door (k>=2 admission), while the recovery win
  came from one-timers never displacing the hot set. exp684 (ring-less,
  unbounded probation) collapses in-burst because each admission evicts
  the previous one. Does a BOUNDED FIFO nursery ring — ring drains only
  when FULL; citizens evicted by LCNU scoring otherwise — hold the burst
  reuse window (in-burst) AND keep the hot set protected (recovery)?
- Design: RingProbationLCNU (subclass of sealed LCNU; miss admits
  immediately + joins ring; rehit promotes out; _victim drains oldest
  ring member ONLY when len(ring) >= ring_cap, else base scoring).
  ring_frac 0.10 / 0.25 standalone + 0.25 as the composite's lcnu inner.
- Testbed: exp681's EXACT crowd streams; 8 cells; caps {512, 2048}.
- Arms: lcnu | center (REPL1 vs exp681) | door4096 (REPL2 vs exp683) |
  ring10 | ring25 | center_ring25.
- Bars (pre-committed, dual-Pareto per Michael's ladder):
  - REPL1 center == exp681 costfit per-cell EXACT (exit 4)
  - REPL2 door4096 == exp683 published per-cell EXACT (exit 5)
  - BURST center_ring25 burst >= lcnu - 0.5pp ALL cells AND
    burst_def <= center's burst_def ALL cells (erase door's delay penalty)
  - RECOV center_ring25 rec_def <= 2.5pp at d=4096/cap=512 (retain ~60%
    of door's 6.98 -> -1.0pp swing) AND ring25 rec >= lcnu rec - 0.5pp ALL
  - SAFETY ring arms loop >= lcnu loop - 0.5pp ALL; acc (exit 3); det (exit 2)
  - Verdicts: BOTH_PASS > IN_BURST_FIXED_ONLY > RECOVERY_KEPT_ONLY > RING-HARM
- Pre-freeze smoke (disclosed): unit PASS (6 misses into cap4/ring2 -> the
  2 overflow evictions drain ring-FIFO oldest-first; rehit promotes out).
  Dual anchors EXACT at d=1024/cap=512: center == exp681 costfit (loop
  0.67998, burst 0.02832, rec 0.66357) AND door4096 == exp683 (loop
  0.69186, burst 0.02051, rec 0.69336) — the earlier "0.0234375 @ d512"
  anchor note was a garble (no d=512 cell exists; BURSTS start at 256).
  One-cell mini (d1024/cap512): ring25 burst 0.0508 vs lcnu 0.0820
  (partial fix), rec +1.1pp win, center_ring25 == center at cap512
  (inner invisible, matches exp684 attribution); mini adjudication
  RECOVERY_KEPT_ONLY pre-fix, BOTH_PASS under the corrected primary bar
  for this cell. Bars above decide on the full 8-cell grid.
- Bar correction pre-freeze (disclosed): initial code gated BOTH_PASS on
  the stretch flag (cr >= lcnu-0.5pp) INSTEAD of Michael's ladder bar
  (deficit <= center's). Fixed to gate on burst_def; stretch recorded.
- Branches (frozen, Michael's ladder):
  - BOTH_PASS -> exp686: partition sweep (ring_frac {0.15, 0.35, 0.50})
    + promotion threshold under shifted burst skew; upstream candidate
    package if Pareto holds.
  - IN_BURST_FIXED_ONLY -> ring overflows into protected on promotion;
    test lazy promotion / ghost-eviction history (exp686a).
  - RECOVERY_KEPT_ONLY -> ring too small for burst reuse stride; size
    ring >= measured reuse stride (exp686b: stride-adaptive ring).
  - RING-HARM -> partition axis dead; return to arbiter-speed lever.
- Run (Safe Settle Pattern — env vars, python3 -c, path-scoped git,
  leak scan before commit):
```bash
set -e
REPO="$HOME/warp-work/moei-granite3b"
EXP="$HOME/warp-work/exp685-moei-ring"
RELP="benchmarks/cache_simulator/research/exp685"
cd "$REPO" || exit 1
git diff --quiet -- "$RELP" || exit 1        # scoped freeze: exp path clean
cd "$EXP" || exit 1
export OUT VERDICT
if OUT=$(PYTHONDONTWRITEBYTECODE=1 "$HOME/warp-work/exp650-moei-granite3b-trace-pr/.venv/bin/python" -m run_ring 2>&1); then
  VERDICT="DONE (exit 0)"
else
  VERDICT="BLOCKED (exit $?)"
fi
python3 -c '
import os, pathlib
p = pathlib.Path("LEDGER.md")
text = p.read_text()
target = "- STATUS: NEXT  # (marker INSIDE the fenced example; tail marker below is authoritative)"
assert target in text, "no STATUS: NEXT marker"
rep = f"- STATUS: {os.environ[\"VERDICT\"]}\n- **Result:**\n```\n{os.environ[\"OUT\"][-4000:]}\n```"
p.write_text(text.replace(target, rep, 1))
'
cd "$REPO" || exit 1
mkdir -p "$RELP"
cp -p "$EXP/run_ring.py" "$EXP/LEDGER.md" "$EXP/exp685-ring.json" "$RELP/" 2>/dev/null || true
git add -- "$RELP" || true
bash benchmarks/cache_simulator/research/scripts/leak_scan.sh --staged || { git reset -- "$RELP"; exit 4; }
git commit -q -m "auto: settle exp685 (ring nursery, safe-settle v2)" || true
```
- STATUS: DONE (exit 0)
- **Result:**
```
VERDICT: IN_BURST_FIXED_ONLY | det: True | repl: exp681+exp683 exact (24 cells) | in-burst deficit <= center ALL cells (0.0 at cap2048; worst 5.37pp) | ring25 rec >= lcnu-0.5 ALL | composite rec_def +6.982pp == center at d4096/cap512 (inner invisible) | standalone ring10 = Pareto point between door and lcnu | elapsed 974s | freeze 63d4e09
```
