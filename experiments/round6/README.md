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

**Read more.** [plans/](plans/) (Chinese working notes).

<!-- files:begin -->
## Files

- `2026-10-21-dagger-map.md` (plans): 第 6 轮 A 线：DAgger + …
- `dag.sh` (scripts): the student R5LC-L18-s0 rolled under …
- `dag.py` (scripts): DAgger rows for the LATERAL part of the …

[plans/](plans/) 1 live plans · [lib/](lib/) 2 library · [scripts/](scripts/) 4 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `results/` small result files, `plans/` live plan notes.
