# round6: lateral recovery for the PAI-human-row recipe

status: live
decisions: 433, 434, 437, 438
index: A DAgger: PAI 281 0.775 / 0.815, iter 2 0.789 / 0.794 vs NTV0 0.743; not adopted

**Question.** The PAI-human-row recipe (MV-LC, d430) fixes route-turn zeros but adds 9-12 offroad zeros on PAI 281 over NTV0, mostly on scenes NTV0
solves: can rows on the student's own drifted states, labelled with a way back, remove them without losing the turn fixes?

**Conclusion.** Line B (REC, perturbation-recovery rows): not adopted; PAI 281 0.735 / 0.727 vs MV-LC 0.746 / 0.725 and NTV0 0.743, offroad 26 / 22 (MV-LC 21 / 24, NTV0 12), ruler -288 / -242 (decision 437).

**Next.**
- Line A, DAGGER-MAP ([plans/2026-10-21-dagger-map.md](plans/2026-10-21-dagger-map.md); code [scripts/dag.sh](scripts/dag.sh) /
  [scripts/dag.py](scripts/dag.py); results [results/dagger/](results/dagger/)). The student R5LC-L18-s0 is rolled under the submission serve on the
  719 non-held curated train scenes with frame dumps. A privileged map teacher labels its own states (fed >= 2 m/s):
  - route = AlpaSim's MAP route rebuilt on the log (the polyline the driver's route messages come from);
  - target = the route lane centre + the human's own lateral offset from it (sigma 10 m, saturated at +-0.5 m; amendments 1-2);
    reference smoothed at sigma 3 m; quintic merge over clip(3 s x v0, 6, 80) m (amendment 3), then along it;
  - label points sit at the arc lengths of the student's own served plan (lateral only, no speed change).
  - Pool `on1_PAID` 110 981 rows (718 scenes), 90 300 drawn (fed >= 2 m/s, teacher ok, 1 902 over-line rows at weight 0); merged `on1_JTPHD`;
    arm R6DAG = MV-LC's argv + 8 DAgger rows a batch, seeds 0 / 1. Student |d0| median 0.31 m, > 1 m on 18 452 rows (10 511 left, 7 941 right).
  - Gates ([results/dagger/](results/dagger/): t1.md, t2.md, rows.md, identity.json; earlier teachers in gates_b0 / gates_b2):
    - T1 lane-centre check 0.299 / 0.369 m at 2 / 4 s (b0: 0.365, failed; the human sits left of the NuRec centre).
    - T2 in-canvas 62 / 65 (raw 74 / 85).
    - T3 bit-identical.
    - T4 position read p99 4.88, over-line share 2.1 % -> fallback.
  - **Read** ([results/dagger/pai281_DAG_ag.md](results/dagger/pai281_DAG_ag.md), 279 scenes, PAI 281 ACC + GAP):

    | read | mean | zeros | route-turn | offroad | zero-other | ruler PCS vs NTV0 |
    |:--|--:|--:|--:|--:|--:|--:|
    | NTV0 | 0.743 | 59 | 15 | 12 | 6 | 0 |
    | MV-LC s0 / s1 | 0.746 / 0.725 | 62 / 68 | 3 / 4 | 21 / 24 | 10 / 11 | -264 / -291 |
    | DAG s0 / s1 | 0.775 / 0.815 | 53 / 43 | 2 / 2 | 20 / 12 | 5 / 4 | -213 [-305, -57] / -155 [-260, +36] |

    - Mean vs NTV0: +0.032 [-0.011, +0.076] / +0.072 [+0.034, +0.111].
    - Mean vs same-seed MV-LC: +0.029 / +0.091 [+0.053, +0.130].
    - New zeros / partials on the 183 scenes NTV0 solves: 12 / 11 and 5 / 7 (MV-LC 19 / 8 and 23 / 10).
    - **Not an adopt-candidate.** s0 fails the offroad + zero-other cap (25 > 21). The ruler fails for s0 (CI upper -57); s1 passes it (+36).
    - Not "clearly better than MV-LC" on both seeds: s0's ruler gain over MV-LC is +51 against the +100 line.
    - Seed spread is large: s1 - s0 is +0.040, with 8 fewer offroad zeros.
  - **Iteration 2** (plan amendments 4 / 4b; code [scripts/dag2.sh](scripts/dag2.sh); results [results/dagger2/](results/dagger2/)):
    - Students R6DAG s0 / s1 rolled again under the same serve (pools on1_PAID2A, 710 scenes; on1_PAID2B, 718 scenes).
    - Teacher b4c = b3 lateral; the label's arc lengths become min(student, human from the matched station) only on curve entry (route curvature
      > 0.02 1/m) or over-speed (> 3 m/s over the human). This slows 16 % of rows; 4 s arc cut p50 6 m.
    - Union pool on1_JTPHD2 = iteration-1 states relabelled (on1_PAIDL) + PAID2A + PAID2B: 277 652 drawable rows, about 2.6 / 2.7 / 2.7 a batch (8 total).
    - Arm DAG2 = R5LC's argv + the union, from NTV0. Gates: T3 bit-identical; T2 in-canvas 66 / 69; T4 fallback 1.3-1.6 % of rows.
    - **Read** ([results/dagger2/pai281_DAG2_ag.md](results/dagger2/pai281_DAG2_ag.md)):

      | read | mean | zeros | offroad | zero-other | rear-hit | launched / 37 | ruler PCS vs NTV0 |
      |:--|--:|--:|--:|--:|--:|--:|--:|
      | DAG2 s0 / s1 | 0.789 / 0.794 | 50 / 50 | 18 / 18 | 5 / 7 | 21 / 17 | 34 / 34 | -187 [-277, -83] / -224 [-326, -105] |

    - vs NTV0 +0.045 [+0.005, +0.088] / +0.051 [+0.007, +0.095]; vs same-seed R6DAG +0.014 / -0.021 [-0.049, +0.007].
    - Offroad + zero-other 23 / 25 (cap 21). 4 / 3 of the 9 stable offroad scenes solved (R6DAG 4 / 6).
    - **Not an adopt-candidate** (both rules fail on both seeds; decisions 438; iteration 1 is decisions 434). The second iteration narrows the seed spread (s1 - s0 +0.005, against +0.040)
      but does not move the mean of the pair (0.792 against R6DAG's 0.795), and the ruler stays well below NTV0.
- Line B, REC: perturbation-recovery rows, ChauffeurNet style ([plans/2026-10-21-recovery-rows.md](plans/2026-10-21-recovery-rows.md); code
  [scripts/rec.sh](scripts/rec.sh) / [scripts/rec.py](scripts/rec.py), [lib/rec_driver.py](lib/rec_driver.py), [lib/rec_geom.py](lib/rec_geom.py); results
  [results/rec/](results/rec/)). The on1_PAIH rollouts again (NTV0 in the loop, the log driver), but the reference handed to the MPC is the log plus a
  lateral offset ramped in over 2 s (|e0| 0.5-1.5 m and / or |psi0| 3-10 deg), so frames and ego history come from an ego that really drove off the
  human's path; label = a 2 s Hermite merge back onto the human's path at the human's speed.
  - 3 rollout passes over 717 train scenes, 3 815 episodes, 9 rows each (about 34 k rows); rejected draws: added lateral acceleration > 3 m/s^2, t0
    footprint off the road_pai raster, actor overlap.
  - Gates: G1 zero offset = on1_PAIH bit for bit, G2 label inside the raster / lateral acceleration, G3 R1 + NTV0 plans not degenerate, G4 weight-0 identity.
  - Arm R6REC = MV-LC's argv + 8 recovery rows a batch (on1_JTPHREC), seeds 0 / 1; exam PAI 281 ACC + GAP + the local PCS ruler.
  - Result (decision 437): gates G1 / G3 / G4 pass, G2 passes on the probe (full pools: lateral acceleration 98.6-98.8 % vs the 99 % line, report only);
    33 250 rows, 3 692 episodes. PAI 281: not adopted, offroad not reduced ([results/rec/pai281_REC_ag.md](results/rec/pai281_REC_ag.md)). Closed.
- Line C, FIX: one joint data-filter arm from the OFFROAD-MV anatomy (d431; plan [plans/2026-10-21-fix-rows.md](plans/2026-10-21-fix-rows.md); code
  [scripts/fix.sh](scripts/fix.sh) / [scripts/fix.py](scripts/fix.py); results [results/fix/](results/fix/)). Arm R6FIX = MV-LC's recorded argv with
  one more draw-file change: R5LC with two row sets at weight 0, each PAI part renormalised to its R5LC share (still 32 PAI human rows a batch).
  - F1: ph15 stretch rows whose label peak lateral acceleration is above 3 m/s^2 or whose fed speed is above 30 m/s.
  - F2: PAI human rows that slow down with no lead and do not stop.
  - Seeds 0 / 1 from NTV0-L18-s0, 9 500 steps; controls = stored R5LC s0 / s1; exam PAI 281 ACC + GAP + the local PCS ruler + the 9 stable offroad scenes.
  - Read (10-21, [results/fix/pai281_FIX_ag.md](results/fix/pai281_FIX_ag.md)): **not an adopt-candidate**. Offroad zeros 21 / 20 against R5LC's
    21 / 24 and NTV0's 12. Offroad + zero-other 39 / 31 against a cap of 21. Mean vs NTV0 -0.031 / -0.005. Ruler -280 / -265.
    Route-turn zeros are kept (4 / 3). On the 9 stable scenes only 94c1d722 recovers (in both seeds). Removing the out-of-envelope stretch rows and the
    no-lead slow-downs does not remove the MV offroad excess.

**ROUTE-CURV (diagnosis, 2026-10-21, 0 card-h).** Question: is the route input a disguised lateral intent? (user review of 596af0c7 / 6e7b42a3.)
Results: [results/route_mem_curv.md](results/route_mem_curv.md) + `route_mem_curv_*.csv`; code [scripts/rmc.py](scripts/rmc.py) / [scripts/rmc.sh](scripts/rmc.sh)
(pool CPU, PRIO 15; box `$DATA_DIR/runs/round6/route_curv/`); sketch [plans/2026-10-21-route-frenet-sketch.md](plans/2026-10-21-route-frenet-sketch.md).
- The fed route memory (projected layout, 40 points at -6..+72 m in the rear-axle frame) is lane-centred on the ego's lane: NuRec lane offset p50 0.00 m.
- AlpaSim's generator puts it in an adjacent lane on 10-14 % of straight human states.
- The PAI command rule (the route's ~42 m point beyond +-2 m) turns curvature, pose and that artefact into L / R on 23 % of non-turn PAI human
  rows, against 3 % in navtrain.
- Before the MV family's 92 offroad / zero-other zeros the route is lane-centred as often as on clean scenes or for NTV0 (82-88 %). Zero-others are
  leftward lane departures (37 / 41).
- A misplaced route precedes and is followed in 5 / 92 events (3 scenes). The hypothesis is not the main cause; Lines A-C stay first.

**Read more.** [plans/](plans/) (Chinese working notes).

<!-- files:begin -->
## Files

- `2026-10-21-dagger-map.md` (plans): 第 6 轮 A 线：DAgger + …
- `dag.sh` (scripts): the student R5LC-L18-s0 rolled under …
- `dag.py` (scripts): DAgger rows for the LATERAL part of the …

[results/](results/) 97 result files · [plans/](plans/) 4 live plans · [lib/](lib/) 2 library · [scripts/](scripts/) 9 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `results/` small result files, `plans/` live plan notes.
