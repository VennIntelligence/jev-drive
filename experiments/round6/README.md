# round6: lateral recovery for the PAI-human-row recipe

status: live
decisions: none yet
index: line A DAgger map teacher on MV-LC's own states (rows building)

**Question.** The PAI-human-row recipe (MV-LC, d430) fixes route-turn zeros but adds 9-12 offroad zeros on PAI 281 over NTV0, mostly on scenes NTV0
solves: can rows on the student's own drifted states, labelled with a way back, remove them without losing the turn fixes?

**Conclusion.** Not read yet.

**Next.**
- Line A, DAGGER-MAP ([plans/2026-10-21-dagger-map.md](plans/2026-10-21-dagger-map.md); code [scripts/dag.sh](scripts/dag.sh) /
  [scripts/dag.py](scripts/dag.py); results [results/dagger/](results/dagger/)). The student R5LC-L18-s0 is rolled under the submission serve on the
  719 non-held curated train scenes with frame dumps. A privileged map teacher labels its own states (fed >= 2 m/s):
  - route = AlpaSim's MAP route rebuilt on the log (the polyline the driver's route messages come from);
  - path = a quintic merge over clip(2 s x v0, 6, 80) m onto the route lane, then along it;
  - label points sit at the arc lengths of the student's own served plan (lateral only, no speed change).
  - Pool `on1_PAID`, merged `on1_JTPHD`; arm R6DAG = MV-LC's argv + 8 DAgger rows a batch, seeds 0 / 1; exam PAI 281 ACC + GAP + the local PCS ruler.
  - Gates T1-T4 (teacher on human states, teacher inside the road before the MV offroad events, weight-0 identity, label lateral acceleration) are in the plan.
- Line B, REC: perturbation-recovery rows, ChauffeurNet style ([plans/2026-10-21-recovery-rows.md](plans/2026-10-21-recovery-rows.md); code
  [scripts/rec.sh](scripts/rec.sh) / [scripts/rec.py](scripts/rec.py), [lib/rec_driver.py](lib/rec_driver.py), [lib/rec_geom.py](lib/rec_geom.py); results
  [results/rec/](results/rec/)). The on1_PAIH rollouts again (NTV0 in the loop, the log driver), but the reference handed to the MPC is the log plus a
  lateral offset ramped in over 2 s (|e0| 0.5-1.5 m and / or |psi0| 3-10 deg), so frames and ego history come from an ego that really drove off the
  human's path; label = a 2 s Hermite merge back onto the human's path at the human's speed.
  - 3 rollout passes over 717 train scenes, 3 815 episodes, 9 rows each (about 34 k rows); rejected draws: added lateral acceleration > 3 m/s^2, t0
    footprint off the road_pai raster, actor overlap.
  - Gates: G1 zero offset = on1_PAIH bit for bit, G2 label inside the raster / lateral acceleration, G3 R1 + NTV0 plans not degenerate, G4 weight-0 identity.
  - Arm R6REC = MV-LC's argv + 8 recovery rows a batch (on1_JTPHREC), seeds 0 / 1; exam PAI 281 ACC + GAP + the local PCS ruler.
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

[plans/](plans/) 1 live plans · [lib/](lib/) 2 library · [scripts/](scripts/) 4 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `results/` small result files, `plans/` live plan notes.
