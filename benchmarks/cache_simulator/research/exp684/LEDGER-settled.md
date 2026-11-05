# LEDGER — warp experiment chain (pilot: hybrid single-file protocol)

One entry per experiment. Entry = prereg + gates + branches, frozen by the
commit hash of this file BEFORE launch. Results APPENDED only (no sed
injection). Publication (addendum, leak scan, PR readback) stays agent-side.

---

## exp684: Inner-admission probation (admit fast, demote never-rehit first)

- Question: exp683 closed wrapper-level admission filtering (door delays
  admission -> in-burst worse; hot-set protection works -> recovery flips).
  Does INNER probation (admit on first sight, evict oldest never-rehit
  resident first, re-hit promotes) keep lcnu's in-burst speed AND the
  door's hot-set protection? Frozen subclass of sealed LCNU (_victim/
  _on_miss/_on_hit hooks) + composite with ProbationLCNU as lcnu inner.
- Testbed: exp681's EXACT crowd streams (seeds 68120261103, fresh spawn in
  run_all), 8 cells, caps {512,2048}, d {256,1024,2048,4096}.
- Arms: lcnu | center (frozen composite, replication vs exp681) |
  prob_lcnu | center_prob.
- Bars: REPL center loop/burst/rec == exp681 costfit per-cell EXACT (exit
  4). H1 no-loss: prob_lcnu burst&loop >= lcnu-0.5pp ALL. H2 protection:
  prob_lcnu rec & center_prob rec >= refs-0.5pp ALL. H3 prize:
  center_prob loop >= max(lcnu,center) in >=4/8 AND >= lcnu-0.5pp ALL.
  acc center double-sim (exit 3). det byte-identical double grid (exit 2).
  Verdict: PROB-WINS > PROB-MECH > PROB-NULL > PROB-HARM.
- Pre-freeze smoke (disclosed): unit PASS (rehit citizen survives, never-
  rehit evicted first). exp681 anchor EXACT (0.0234375). Mini-signal
  (d512/cap512, reduced N): PROB-HARM lean — prob_lcnu burst -6.84pp vs
  lcnu (probation evicts crowd keys BEFORE their rehit at small cap);
  rec +0.93pp (protection real); center_prob == center at that cell =>
  composite's live inner there is s3fifo (burst failure attribution
  evidence). Full run decides; cap2048 cells are the mechanism's best case.
- Branches (frozen):
  - PROB-WINS -> exp685 upstream candidate package (prob inner + arbiter)
  - PROB-MECH/NULL -> exp685 probation GRACE dose-response (evict
    never-rehit only after G accesses; G in {1,2,4,8} = dose axis)
  - PROB-HARM -> exp685 prob-s3fifo inner (attribution: if composite
    burst failure is the s3fifo inner, fix admission THERE)
- Run (gates as exit codes; append-only result; agent publishes):
```bash
cd "$HOME/warp-work/moei-granite3b" || exit 1
git diff --quiet -- benchmarks/cache_simulator || exit 1   # scoped freeze: policy sources clean (shared worktree)
cd "$HOME/warp-work/exp684-moei-prob" || exit 1
PYTHONDONTWRITEBYTECODE=1 "$HOME/warp-work/exp650-moei-granite3b-trace-pr/.venv/bin/python" -m run_prob > run.log 2>&1
rc=$?
[ $rc -eq 0 ] || { printf 'STATUS: BLOCKED (exit %d)\n' "$rc" >> LEDGER.md; exit "$rc"; }
printf 'STATUS: DONE (exit 0)\n- Result: %s\n- Freeze: %s\n' \
  "$(tail -1 run.log)" "$(git -C "$HOME/warp-work/moei-granite3b" rev-parse --short HEAD)" >> LEDGER.md
```
- STATUS: DONE (exit 0)
- **Result:**
```
VERDICT: PROB-HARM | det: True | repl: True (exp681+exp683) | h3 4/8 | burst deficit -3.6..-8.7pp ALL 8 cells | center_prob rec -2.2..-5.4pp cap2048 | elapsed 684s | freeze fe1d97f
```
