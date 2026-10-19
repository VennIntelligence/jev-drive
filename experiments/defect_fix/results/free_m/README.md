# NR1FREE-M: the one-sided longitudinal term on moving states only (2026-10-19; NR1 prereg supplement 16)

Question: can the speed ratchet of decisions 372 / 375 be fixed on the training side? In closed loop each plan asks 2-4 % less than the
current speed, and the frames lock that in. The fix tried here keeps decision 365's term (penalise the plan only when it is short of the
human and the way ahead is free, no target speed) but moves it off the low-speed wait / launch axis that broke waiting there:

- Rows: S rows where the ego moves (t0 speed >= 3 m/s) and the human neither stops nor brakes (every 0.5 s segment of his logged 4 s
  >= max(3 m/s, 0.8 x his first segment)).
- Band: 0.95 x the reference (decision 365: 0.8). The reference is min(the human's logged arc, the same speed profile reachable from the
  visited state).
- Dose: W 0.004, as in decision 365. The free check (logged boxes, 2 m + 0.5 s x v ahead) is unchanged.

Code: [lib/nr1_free.py](../../lib/nr1_free.py) (`configure`, `moving`), `rm_train --on-free W --on-free-mode move --on-free-frac 0.95`
(defaults reproduce supplement 15), [scripts/free_m.py](../../scripts/free_m.py) (gate, extract, arena, pilot rule),
[scripts/free_m.sh](../../scripts/free_m.sh) (chain). Box outputs: `$DATA_DIR/runs/defect_fix/free_m/`.

## Offline gate: passes ([gate.json](gate.json))

| check | set | result | line |
|:--|:--|--:|:--|
| false fires on the human's own paths | on1_JT S rows on moving states (44 881 of 87 012), plan = `fut` / `fut_reg` | 0 / 0 | <= 1 % |
| fires on ratchet decisions | JTN-L18-s1 native arena held, non-static windows, k >= 2, ego >= 3 m/s, plan 4 s arc < 0.95 x ref | **2 317 / 3 559 (65.1 %)**, 495 windows | >= 50 % |
| fires on AF-front windows | JTN-L18-s1 / s0 native + the four JT reads, 3 s before the first front collision | **0 / 264 windows** | <= 5 % |
| fires on static windows | JTN-L18-s1 native, 878 static windows, every decision | **0 / 8 780** | <= 1 % |
| fires on the decision 365 failure set | launch_diag dumps (JTN-L18-s0, RTJA0), wait set v2 b, lead classes, every decision | **2 / 1 660 (0.12 %)** | <= 1 % |
| switch off = bit-identical | JTN's command, 60 steps eager, 524f6a32 against the new code | identical (275 scalars, weights, dev ADE, draw hash) ([identity.json](identity.json)) | identical |

How to read it:
- Ratchet population: 7 275 moving decisions. Their median plan / reference ratio is 0.953, and 49 % are below the 0.95 band. Of those
  short decisions, 1 114 are blocked because the human brakes or nears a stop, and 128 because the way is not free. The term fires on 32 %
  of all moving decisions.
- The human check passes by construction (the reference is the smaller of the two human readings). With one reference alone the other
  human path fires often: `fut` against the raw log fires 24 %, because a slower ego cannot reach the human's arc within 2 s; `fut_reg`
  against `fut` fires 3.6 %. Without the minimum, the term would ask for more than is reachable.
- The AF-front windows are trivial again: of 186 moving decisions in those windows, none is short of 0.95 x the reference.
