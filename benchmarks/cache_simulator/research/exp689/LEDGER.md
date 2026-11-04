# exp689 — ring visibility: inner variants + zero-barrier arbitration

## Preregistration (frozen before run)

RENUMBER DISCLOSURE: originally drafted as "exp686a" in-chat; the
warp-exp-followup wake chain independently built+settled exp686a
(ringfit3 3-way arbitration, IN_BURST_KEPT_VISIBLE, ADDENDUM32, #210)
and exp687 (windowed challenger scoreboard, HARM/THRASH, #211) overnight.
This experiment is therefore exp689. It complements both: exp686a made
the ring a first-class CANDIDATE (adoption map); exp687 showed a
single-window scoreboard thrashes at sealed margin; exp689 tests the
remaining simple levers — barrier removal (center_zb), lazy promotion,
ghost history — as pre-registered arms with per-arm conjunctive bars.
A first cp mistakenly overwrote research/exp686a/LEDGER.md (restored
from b6db6a9 in the same commit that freezes exp689).

HYPOTHESIS: exp685's only failed gate (composite rec_def +6.982pp ==
exp681 center at d4096/cap512, center_ring25 byte-equal to center) is an
ARBITRATION property, not an inner property. Pre-run smoke evidence
(mini 61,440 acc, d1024/cap512): center_ring10 == center byte-equal even
with the ring10 inner; center_lazy/center_ghost/center_zb win arbitration
in loop segments only (loop +0.05..+0.58pp vs center, burst/rec identical).
Two registered mechanisms:
  M1 (inner): lazy promotion (k=2) / ghost eviction history (G=4096)
              change WHEN keys become citizens.
  M2 (arbiter): the barrier (MARGIN+GAMMA*sw_mass) and/or the decay
              window (HALF_LIFE=32768) prevent the ring inner from being
              chosen during recovery. center_zb removes the barrier.

STREAMS: exp681 exact (seed 68120261103 spawn(2), fresh); BURSTS
[256,1024,2048,4096] x CAPS [512,2048]; 8 cells; N=638,976.

ARMS: lcnu | center (frozen composite, REPL1 vs exp681) | ring10
(control, REPL3 vs exp685) | center_ring10 | center_lazy (k=2) |
center_ghost (G=4096) | center_zb (MARGIN=GAMMA=0).

GATES (exit codes): det=2, acc=3, repl1=4, repl3=6; settle=0.

BARS (per-arm conjunction — a verdict arm must ITSELF pass all four; no
mixing across arms):
  VIS     arm != center byte-equal at cap512 cells
  RECOV   arm rec_def <= 3.5pp at d4096/cap512 (>= half of 6.98pp gap)
  BURST   arm burst_def <= center's deficit ALL 8 cells
  SAFETY  arm loop >= lcnu - 0.5pp ALL cells
  PARTIAL rec_def <= 5.5pp; STRETCH burst_def <= 1.5pp at home cell.

VERDICT LADDER: BOTH_PASS (some arm passes all 4) > VIS_ONLY (vis +
partial + safety) > IN_BURST_ONLY (vis + burst + safety) > NO-VIS (all
arms invisible, safety holds) > HARM.

BRANCHES:
  BOTH_PASS -> exp687: dose-response on the winning mechanism + upstream
               package path (PR #342 artifacts).
  NO-VIS    -> exp687: arbitration WINDOW is the wall — regime-responsive
               arbitration (shorter decay for arbitration mass, or
               segment-aware re-arbitration). Barrier removal already
               refuted by center_zb result.
  HARM      -> exp687: revert to exp683 door composite with bounded ring
               as outer, new failure analysis.

## Run

```bash
cd ~/warp-work/exp689-moei-ringvis
PYTHONDONTWRITEBYTECODE=1 ~/warp-work/exp650-moei-granite3b-trace-pr/.venv/bin/python run_ringvis.py
```

Settle (safe-settle v2; NOTE: example marker below is QUEUED so the
fenced block can never collide with the authoritative tail marker):

```bash
export OUT="VERDICT ... | det ... | repl ... | elapsed ... | freeze <hash>"
python3 -c '
import os, pathlib
p = pathlib.Path("LEDGER.md"); t = p.read_text()
target = "- STATUS: QUEUED"
assert target in t
p.write_text(t.replace(target, "- STATUS: DONE (exit 0)\n- **Result:**\n```\n" + os.environ["OUT"] + "\n```", 1))'
git add -- LEDGER.md && bash <leak_scan> --staged && git commit -m "exp686a settle"
```

- STATUS: NEXT
