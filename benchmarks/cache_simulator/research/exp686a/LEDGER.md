# LEDGER — warp experiment chain (protocol v2: safe settle)

Safe Settle Pattern (Michael, 2026-11-03): results pass through
environment variables into `python3 -c` (inert bytes — no shell parsing,
no sed delimiters); every git operation is path-scoped (shared worktree:
never `-am`, never repo-wide diff); leak scan runs BEFORE commit. One
entry per experiment; entry frozen by the commit hash of this file
BEFORE launch; results APPEND only.

---

## exp686a: Ring as a FIRST-CLASS arbitration candidate (visibility test)

- Question: exp685 settled IN_BURST_FIXED_ONLY — the bounded nursery
  ring fixes victim selection standalone (Pareto point), but the
  composite's rec_def at d4096/cap512 (+6.982pp) equals center's
  EXACTLY: center_ring25 is byte-equal to center, i.e. the ring inner
  is INVISIBLE to 2-way arbitration at small cap. Probe (2026-11-03,
  instrumented choice timeline on the exp681 stream): center's choice
  at cap512 flips lcnu->s3fifo @8191 and stays s3fifo for the WHOLE
  stream (burst and recovery included); at cap2048 it returns to lcnu
  @135167 before the burst. Does making the ring a THIRD CANDIDATE of
  the sealed CostFit3 barrier (exp673 machinery, decayed cumulative-rate
  scoreboard + margin/gamma switch barrier) let arbitration see and
  adopt the ring where it wins?
- Design: RingFit3 — CostFit3Policy subclass; EVERY sealed barrier
  parameter inherited unedited (HALF_LIFE 32768, MARGIN 0.005,
  GAMMA 0.0025, RESEGMENT 4096, initial lcnu); only the candidate set
  changes: {lcnu, s3fifo, ring(->RingProbationLCNU)}. Secondary (frozen
  branch text): lazy promotion dose — promotion deferred to 2nd rehit
  (k=2 crowd keys never become citizens) standalone AND inside the
  composite. Stride diagnostic (pure stream stats, no policy behavior)
  recorded for exp686b ring sizing.
- Testbed: exp681's EXACT crowd streams (seed 68120261103 spawn
  scheme, FRESH draw); 8 cells; caps {512, 2048}.
- Arms: lcnu | s3fifo | center (REPL1 vs exp681 costfit) | ring10 |
  ring25 (REPL2 vs exp685) | lazy2 | ringfit3 (ring .25) |
  ringfit3_10 (ring .10) | ringfit3_lazy2 (ring .25 + lazy).
- Bars (pre-committed):
  - REPL1 center == exp681 costfit per-cell EXACT (exit 4)
  - REPL2 ring10+ring25 == exp685 published per-cell EXACT (exit 5)
  - H1 (primary) ringfit3 rec_def <= 2.5pp at d=4096/cap=512
  - H2 (ladder) ringfit3 burst_def <= center burst_def ALL cells
  - H3 (safety) new arms (lazy2, ringfit3*) loop >= lcnu loop - 0.5pp
    ALL cells
  - H4 churn/acc ringfit3* switches <= 6 ALL cells; credited == loop
    every composite cell; center double-sim identical (exit 3)
  - det byte-identical double grid (exit 2)
  - Verdicts: RING-FIRST-CLASS > IN_BURST_KEPT_VISIBLE > HARM
- Pre-registered prediction (frozen with this file, before the grid):
  at the PRIMARY cell d4096/cap512 the ring CANNOT displace s3fifo —
  its decayed cumulative rate tracks lcnu's, which already fails to
  overcome s3fifo's base lead; needed lead ~0.0075 vs a ~0.006
  one-window recovery boost -> H1 FAILS -> IN_BURST_KEPT_VISIBLE.
  Genuinely uncertain (and why the grid runs): ring10's whole-stream
  loop at cap512 LEADS s3fifo's (0.6805 vs 0.6768 per exp685); smoke
  (below) already shows ring-fit adoption at d1024/cap512.
- Pre-freeze smoke (disclosed; runner sha256
  1da6a3fd76bf3f77c74cfdb1fc0d80d820e8d07d703948ccdacdaaba5c1e3c4e):
  unit PASS (eager promotes on 1st rehit, lazy keeps / promotes on
  2nd; readmission restarts probation; drain is eviction-coupled per
  sealed semantics). Wiring PASS (3-candidate scoreboard {lcnu, s3fifo,
  ring}; sealed MARGIN/GAMMA/HALF_LIFE/RESEGMENT; ring_cap 128 @512;
  rothead absent). Dual anchors EXACT at d=1024/cap=512: center ==
  exp681 costfit (loop 0.679977 / burst 0.028320 / rec 0.663574),
  ring10 == exp685 (0.683653 / 0.030273 / 0.673828), ring25 == exp685
  (0.678824 / 0.050781 / 0.664307). One-cell mini (d1024/cap512):
  ringfit3_10 ADOPTS the ring (sw 3, sw_by ring 2; rec 0.673828 ==
  ring10's standalone rec exactly) while ringfit3 (sw 2, ring 1) and
  ringfit3_lazy2 (sw 1, ring 0) ride s3fifo at recovery (rec ==
  center's 0.663574); acc credited==loop ALL. Smoke defect disclosed:
  first two smoke scripts asserted admission-coupled ring drain and
  standalone-arm `.switches` — both wrong expectations about the
  sealed code, fixed in the smoke only (runner guarded already).
- Branches (frozen):
  - RING-FIRST-CLASS -> exp686b: partition sweep (ring_frac {0.15,
    0.35, 0.50}) + lazy dose grid on shifted burst skew; upstream
    candidate package (policies file + tests) if Pareto holds.
  - IN_BURST_KEPT_VISIBLE -> exp687 (new prereg): arbitration-layer
    redesign — windowed challenger scoreboard (challenger judged on
    last-4096 rate vs incumbent decayed cumulative, same barrier
    family), sealed victim-selection policies untouched; carry stride
    diagnostic into exp686b sizing if adopted anywhere.
  - HARM -> partition+visibility axis closed; return to exp680
    arbiter-speed lever.
- Run (Safe Settle Pattern — env vars, python3 -c, path-scoped git,
  leak scan before commit):
```bash
set -e
REPO="$HOME/warp-work/moei-granite3b"
EXP="$HOME/warp-work/exp686a-moei-ringfit3"
RELP="benchmarks/cache_simulator/research/exp686a"
cd "$REPO" || exit 1
git diff --quiet -- "$RELP" || exit 1        # scoped freeze: exp path clean
cd "$EXP" || exit 1
export OUT VERDICT
if OUT=$(PYTHONDONTWRITEBYTECODE=1 "$HOME/warp-work/exp650-moei-granite3b-trace-pr/.venv/bin/python" -m run_ringfit3 2>&1); then
  VERDICT="DONE (exit 0)"
else
  VERDICT="BLOCKED (exit $?)"
fi
python3 -c '
import os, pathlib
p = pathlib.Path("LEDGER.md")
text = p.read_text()
target = "- STATUS: DONE (exit 0; settle re-executed with quoting fix — disclosed)
- **Result:**
```
VERDICT: IN_BURST_KEPT_VISIBLE | det: True | repl: exp681 costfit + exp685 ring10/ring25 exact | bars: h1 False, h2/h3/h4/acc True | elapsed 1471.5s | freeze 2735a8e
(reconstructed from exp686a-ringfit3.json; original stdout capture lost to the settle-script quoting bug AFTER the run completed — measurement untouched; settle re-executed with the fix, disclosed)
cell d256/c512: cf bdef +2.73 rdef -0.22 | rf3 bdef +2.73 rdef -0.22 loop_vs_l +0.66 | rf3_10 bdef +2.73 rdef -0.71 loop_vs_l +0.93 adopt burst=ring rec=ring sw=3 | lazy2 rdef +0.51 loop_vs_l +0.34
cell d256/c2048: cf bdef +0.00 rdef +0.00 | rf3 bdef +0.00 rdef +0.00 loop_vs_l +0.04 | rf3_10 bdef +0.78 rdef -0.17 loop_vs_l +0.14 adopt burst=ring rec=ring sw=1 | lazy2 rdef +0.00 loop_vs_l +0.02
cell d1024/c512: cf bdef +5.37 rdef -0.07 | rf3 bdef +5.37 rdef -0.07 loop_vs_l +0.66 | rf3_10 bdef +5.18 rdef -1.10 loop_vs_l +0.92 adopt burst=ring rec=ring sw=3 | lazy2 rdef +0.15 loop_vs_l +0.34
cell d1024/c2048: cf bdef +0.00 rdef +0.00 | rf3 bdef +0.00 rdef +0.00 loop_vs_l +0.04 | rf3_10 bdef +5.57 rdef -1.15 loop_vs_l +0.15 adopt burst=ring rec=ring sw=1 | lazy2 rdef -0.39 loop_vs_l +0.03
cell d2048/c512: cf bdef +4.05 rdef +1.29 | rf3 bdef +4.05 rdef +1.29 loop_vs_l +0.64 | rf3_10 bdef +5.57 rdef -1.37 loop_vs_l +0.92 adopt burst=ring rec=ring sw=3 | lazy2 rdef -0.02 loop_vs_l +0.33
cell d2048/c2048: cf bdef +0.00 rdef +0.00 | rf3 bdef +0.00 rdef +0.00 loop_vs_l +0.04 | rf3_10 bdef +7.67 rdef -1.71 loop_vs_l +0.13 adopt burst=ring rec=ring sw=1 | lazy2 rdef -0.56 loop_vs_l +0.03
cell d4096/c512: cf bdef +1.07 rdef +6.98 | rf3 bdef +1.07 rdef +6.98 loop_vs_l +0.60 | rf3_10 bdef +5.96 rdef -0.73 loop_vs_l +0.89 adopt burst=ring rec=ring sw=3 | lazy2 rdef +0.64 loop_vs_l +0.32
cell d4096/c2048: cf bdef +0.00 rdef +0.00 | rf3 bdef +0.00 rdef +0.00 loop_vs_l +0.04 | rf3_10 bdef +9.23 rdef +0.00 loop_vs_l +0.08 adopt burst=ring rec=lcnu sw=2 | lazy2 rdef -0.41 loop_vs_l +0.02
```"
assert target in text, "no STATUS: NEXT marker"
rep = f"- STATUS: {os.environ[\"VERDICT\"]}\n- **Result:**\n```\n{os.environ[\"OUT\"][-4000:]}\n```"
p.write_text(text.replace(target, rep, 1))
'
cd "$REPO" || exit 1
mkdir -p "$RELP"
cp -p "$EXP/run_ringfit3.py" "$EXP/LEDGER.md" "$EXP/exp686a-ringfit3.json" "$RELP/" 2>/dev/null || true
git add -- "$RELP" || true
bash benchmarks/cache_simulator/research/scripts/leak_scan.sh --staged || { git reset -- "$RELP"; exit 4; }
git commit -q -m "auto: settle exp686a (ring first-class candidacy, safe-settle v2)" || true
```
- STATUS: NEXT  # (marker INSIDE the fenced example; tail marker below is authoritative)
