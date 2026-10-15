# INTENT HOLD (serve-side diagnostic, not adopted)

Prereg `plans/2026-10-15-intent-hold-prereg.md` (Amendments 1-3). RSG-L18-s0 on PAI, gains 20, `JEV_VCONT=1.0 JEV_LEAD=1 JEV_TRACK=0`
(the no-gate path of the identity) + `JEV_INTENT_HOLD=1 JEV_IH_A=<a>` (experiments/alpasim/lib/intent_hold.py), 49 scenes
([scenes49.txt](scenes49.txt)), Tokyo card 0, one stack. Identity = the stored r281/rsg sessions. Runs `/data/runs/route_night/intent_hold/`
(Tokyo): `id`, `smoke`, `a0.3`, `a0.15`.

## Checks

- Driver without the switch: S1 2 / 2 sessions bit-identical to r281/rsg (plans, tracks, scores) ([smoke_identity.txt](smoke_identity.txt)).
- `ihcheck` (every decision's branch recomputed from the logs, the intent carried along the logged branches; served vs recomputed; plans
  up to each session's first hold vs r281/rsg): smoke 398 decisions, 0 flips, 1.6e-3 m; a 0.3 9 735 decisions, 1 flip (fed speed 1.0 at
  the 1 m/s threshold, slow vs noturn: both resets), 3.8e-3 m; a 0.15 9 735, 2 flips (that one and a 2 s distance of 1.5002 vs 1.4995 m at
  the 1.5 m far threshold), 3.9e-3 m; 873 pre-hold plans bit-identical to r281/rsg in both runs.
- The hold was on at 7 500-7 700 of 9 735 decisions (resets: no turn, standing, far); first hold typically 6-40 m before the entry.

## Lines (registered; per variant)

| line | identity | a 0.3 | a 0.15 |
|:--|--:|:--|:--|
| strict 36 completed (>= 10) | 5 | 8: FAIL (+a1b166cc, ce161092, e848c843, ea468790; -8a143cb2) | 7: FAIL (+a1b166cc, ddc3e8df, e848c843, ea468790; -382a7247, 8a143cb2) |
| runway 22 completed (>= 6) | 3 | 5: FAIL | 3: FAIL |
| the identity's 5 completed kept | 5 | lost 1: FAIL | lost 2: FAIL |
| new zeros outside the turns (<= 2) | - | 0: PASS | 0: PASS |
| spins (<= 2) | 0 | 0: PASS | 0: PASS |
| stop rule (>= 3 new offroad zeros before the entry) | - | 0: not fired | 0: not fired |
| d293 shape gate (reported) | 1 | 3 | 4 |
| exam turn scenes completed (reported) | 2 / 26 | 5 / 26 | 3 / 26 |
| zeros on the 49; paired mean (reported) | 33 | 30; +0.060 [-0.021, +0.147] | 27; +0.107 [+0.011, +0.210] |
| MECHANISM: served-plan creep, completed turns (<= 0.03 m / m) | -0.015 (5) | -0.027 (8): PASS | -0.033 (7): FAIL |

The over-turn PAI zeros of d299 (2466266e, 300465f7, d8bfaaf2) have no strict turn on the AlpaSim route by rf_pai_follow's rule, so no
completion or creep is defined; they stay zeros under a 0.3 (offroad x3) and two of three under a 0.15 (d8bfaaf2 scores 1.0).

## Creep (rf_switching's measure, median per turn; m per m of travel; - = outward)

| run | plans | completed | failed | all | summed j c / f (m) | exec - plan at 2 s c / f (deg) |
|:--|:--|--:|--:|--:|:--|:--|
| identity | (served = model) | -0.015 (5) | -0.063 (31) | -0.055 | -0.40 / -2.10 | +1.6 / -2.5 |
| a 0.3 | model | -0.047 (8) | -0.043 (28) | -0.043 | -1.18 / -1.43 | +0.3 / -1.3 |
| a 0.3 | served | -0.027 (8) | -0.038 (28) | -0.035 | -0.80 / -1.62 | +0.3 / -1.5 |
| a 0.15 | model | -0.044 (7) | -0.034 (29) | -0.037 | -2.06 / -1.44 | -0.7 / -0.7 |
| a 0.15 | served | -0.033 (7) | -0.029 (29) | -0.030 | -2.33 / -1.37 | -1.3 / -0.9 |

Per turn: [a0.3.creep.md](a0.3.creep.md), [a0.15.creep.md](a0.15.creep.md). The gained turns still creep (served -0.05 to -0.09 m / m in
a1b166cc, e848c843, ea468790): they complete while the plan keeps widening.

## Reading

- Holding the policy to its own earlier path is not enough to complete the turns. Both variants fail the strict, runway and keep-5 lines.
  The gains (4 per variant, 3 of them shared: a1b166cc, e848c843, ea468790; shape gate 3-4 vs 1) are offset by losses (8a143cb2 in both:
  the track is nearly the identity's but ends past the 2.5 m route band; 382a7247 under a 0.15). The hold is harmless off the turns: no new
  non-turn zero, no spin, zeros 33 -> 30 / 27.
- Mechanism: the creep halves at best (failed turns -0.063 -> -0.038 / -0.029 served) but does not go to the completed-turn level. Most
  telling is the model plan: with the car held on its intent, the plan itself still widens by -0.034 to -0.047 m per m, and in the turns
  that complete it widens more than the identity's completed turns did (-0.044 / -0.047 vs -0.015). So decision 308's creep is mostly
  the plan widening relative to its own previous plan, not only accepting the car's shortfall. Any hold with a > 0 is a low-pass filter on
  that drift: it lags it by about (1 - a) / a decisions and keeps its slope, which is what the served numbers show. The executed-vs-planned
  heading shortfall shrinks (failed -2.5 -> -1.3 / -0.7 deg at 2 s), so the hold does keep the car on the plan; the plan is what drifts.
- Consequence for the training version (turn-night.md section 9, mechanism 3): "return to your own previous path" as a serve-side
  correction does not remove the creep. The model's per-step outward widening has to be removed from the plan itself.

Figure: [figs/intent_hold_bev.png](../../figs/intent_hold_bev.png), a 0.3, three turns (a1b166cc gained, 8a143cb2 lost, ddc3e8df failed).
Every 1 s while holding: the intent carried from the previous decision (dashed), the model plan (grey), the served plan (pink) and the
executed tracks of the identity (orange) and of the hold (green). Look at how the dashed intent and the served plan lie on top of the model
plan in every snapshot: the hold rarely pulls far, because the plan drifts every step. In a1b166cc the hold's track turns inside the identity's wide one and completes; in ddc3e8df both run wide past the entry.
