# JTB / JTN (NR1 prereg supplement 10): readout, 10-18 05:40 CST

Follow-up to the failed JT round (d352). Both arms: one L18 stage from RB0-F-s0, 9 500 steps, seed 0, pool on1_JT S rows (4 logs held out),
JT00 draw (trainval rows without standing / launch), JT schedule and lateral terms. JTB adds the arc term with `--on-arc-rows brake`
(stopping and launching rows labelled too; only the still-all-horizon rows excluded: 76 442 -> 85 523 labelled rows) at W 0.0012.
JTN has no arc term (labels loaded for monitoring only). Code 11bd069c, prereg 2c55cef1. Identity (`identity_jtb.json`): JT00's command, 60 steps
eager, f81ea48b vs the new code with the flag off: bit-identical (258 scalars, all weights, on_hash_60). Smoke gate: `smoke_gate.json`.

## Smoke and arc weight

JTBSM (1 000 steps, W 0.002), trunk-gradient norms over steps 500-1 000: imit 9.72, lon 1.58 (ratio 0.162), pool hinges 3.92, lat 0.61, nav hinge
1.38; cos(imit, pool) -0.07; the 1.0 clip fires every step. Rule: W = 0.002 x clip(0.1 x 9.72 / 1.58, 0.5, 2) = **0.0012**. Over the full JTB run
the lon / imit ratio was 0.108 (0.98 / 9.00). Gate passed (dev ADE 0.528 -> 0.553).

## Offline: 4 s arc bias (plan minus target, m) by row kind, held-out pool slice (2 048 rows)

| kind (held rows) | step 0 | JTB 9 500 | JTN 9 500 |
|---|--:|--:|--:|
| cruise | -4.61 | -2.23 | -4.56 |
| brake (decelerating, no stop) | -1.35 | **+1.61** | -1.37 |
| stop (comes to < 0.5 m/s) | +3.52 | **+5.49** | +3.08 |
| launch | +1.18 | +0.91 | +0.27 |
| still (excluded) | +6.31 | +6.55 | +5.58 |

**The check failed.** The braking rows did not pull the plan's arc down where the human brakes. The plan already overshot the human on stop rows,
and JTB makes it overshoot more (+3.5 -> +5.5 m). On brake rows the bias turns from -1.4 to +1.6 m. Cruise rows are 73 % of the labelled rows.
The model does not separate them from the stop rows, so the term mostly acts as one global speed-up.

## Reads (paired vs RSG-L18-s0 under the same serve, log-clustered 95 % CI)

| read | RSG ref | JTB | JTN |
|---|--:|--:|--:|
| (a) arena held 2 228, stop3 | 0.944 | 0.941, -0.003 [-0.009, +0.004] | 0.943, -0.001 [-0.009, +0.010] |
| (b) arena held 2 228, no hold | 0.786 | 0.822, **+0.036 [+0.012, +0.063]** | 0.874, **+0.089 [+0.050, +0.135]** |
| (c) held 460 @18 zeros (guard 39) | 34 | **48** | 34 |
| held 460 mean / progress_rel / front-rear coll | 0.866 / 0.788 / 4-38 | 0.882 / 0.893 / 14-6 | 0.864 / 0.792 / 5-35 |
| (a) AF-front / offroad / corridor zeros | 12 / 43 / 8 | 14 / 52 / 15 | 2 / 49 / 9 |
| (a) braking windows: AF-front / offroad / corridor | 0 / 7 / 5 | 3 / 11 / 10 | 2 / 10 / 6 |
| (a) "slow" partial scores | 185 | 52 | 156 |

Contributions under stop3 (JTB / JTN): static 0 / +0.006, launch -0.002 / -0.003, braking **-0.005** (14 new zeros, 1 fixed) / -0.003 (8 / 2),
turn +0.001 / 0, cruise +0.001 / 0, other +0.002 / -0.002. With no hold: JTB static +0.042 (101 fixed corridor zeros), braking -0.007;
JTN static +0.096 (213 fixed, 0 new), launch -0.006, braking -0.001.

**Success line (JTB): not met.** (b) is met (+0.036, CI > 0), but (c) fails: 48 zeros against a guard of 39, with front collisions 4 -> 14.
(a) is a tie. JTN, read descriptively, ties RSG under stop3 and in held 460 (34 zeros). Natively it is +0.089, almost all of it from standing
still in static windows, and stop3 already serves that at deploy.

## What it says about the JT failure

- The arc term is what hurt JT, not the pool or the schedule. JTN equals JT00 minus the arc term: natively JT00 was -0.034 and JTN is +0.089.
  JTN also has no front-collision rise (AF-front 2 vs 12) and keeps RSG's slow-but-safe profile (progress_rel 0.79, rear hits 35).
  The three runs order cleanly by arc dose: no arc > W 0.0012 with brake rows > W 0.0039 with moving rows only.
- Labelling the braking rows did not fix it. Offline the plan overshoots the stops more, and in closed loop JTB again drives faster (slow
  partials 185 -> 52) and hits more in braking windows. The row exclusion was not the mechanism, or not the main one. The per-row arc target is
  a time-based speed label the model cannot condition on: it cannot see why the human slows. The model learns the pool-average "faster" and
  applies it in the braking windows too. A longitudinal target would need a cue visible to the model, such as the gap to the lead, the stop
  line or the light; the human's own future speed is not enough.
- The bigger pool and schedule are neutral against RSG at deploy (stop3 tie, held tie). They help only in native static windows, which the
  serve hold already covers.

One seed per arm. The JT reference used leash3, while JTB / JTN were read under stop3 (the deployed rule), so the like-for-like JT comparison is
the no-hold read.
