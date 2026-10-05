# rc-*-pre: turn-in action supervision at approach poses (decision 128 suspect a)

Pre-registration: [plans/2026-10-05-route-ft-prereg.md](../plans/2026-10-05-route-ft-prereg.md), section "2026-10-06" (written before training; pilot recorded there).
Code: `scripts/rft.py` (`act_target_pre`, arms `rc-bear-pre` / `rc-ctl-pre`), `scripts/rft_eval.py` (`turnin` readout), `scripts/pre_chain.sh`,
[`scripts/pre_lane.py`](https://github.com/VennIntelligence/jev-drive/blob/901d1612dbbd0bb253dd00e5f536defe15900b0d/experiments/op_route_ft/scripts/pre_lane.py) (B2D desire off / on), `scripts/pre_report.py` (tables, split, panels), `scripts/pre_fig.py` (turn-in timing figure).

**One change from rc-bear / rc-ctl** (same rows, data, losses, 4000 steps, hyper-parameters, seed 0): on P rows (real and CARLA exit pairs) whose
target path is straight up to v0 * 0.2 s + 2 m and whose next maneuver (|kappa| > 0.02) starts within W = v0 * 1.5 s + R / 2, the action[0] target
is the commanded path's curvature averaged over [v0 * 0.2 s, v0 * 0.2 s + W] (a linear turn-in ramp proportional to the command), as
-0.45 * kappa * max(1, v0)^2. Why this lookahead: action[0] is the desired lateral acceleration at t + lateralDelay; a plan-tracking car's desired
curvature is the curvature of the driven path at the car, and a driven path starts curving before the lane-centre arc (clothoid entry of about R
centred on the tangent point, plus the driver's preview). Rows affected: 80-99% of the CARLA turn rows at brake d 10 / 20, cruise d 20, creep d 10;
<= 2% of straight-command rows; real approach frames nav 467 / 1957, wod 359 / 1938.

## Results

### Open loop (CARLA dev junctions, moving poses; rft_eval.py)

The action head learned the turn-in target, and only with the command. Desired curvature towards the commanded exit (1/m), mean [95% cluster CI], target in the last column:

| poses | shipped | rc-ctl | rc-bear | rc-ctl-pre | **rc-bear-pre** | target |
|---|--:|--:|--:|--:|--:|--:|
| brake, d 10 m | 0.0001 | 0.0000 | 0.0017 [0.0006, 0.0033] | 0.0030 [0.0019, 0.0042] | **0.0190 [0.0182, 0.0197]** | 0.0189 |
| brake, d 20 m | 0.0000 | 0.0001 | 0.0010 | 0.0009 | **0.0065 [0.0062, 0.0068]** | 0.0068 |
| cruise, d 20 m | -0.0003 | -0.0001 | 0.0005 | 0.0002 | **0.0038 [0.0036, 0.0041]** | 0.0042 |
| creep, d 10 m | -0.0001 | -0.0002 | 0.0021 | 0.0003 | **0.0114 [0.0106, 0.0122]** | 0.0122 |
| d 30 m (all profiles, target about 0) | -0.0001 | -0.0001 | 0.0002 | 0.0001 | 0.0003-0.0004 | 0.0002-0.0011 |
| approach rows (n 479): fit to target, corr | 0.11 | 0.09 | 0.23 | 0.35 | **0.96** | |
| action sign = command side, turn rows | 0.50 | 0.48 | 0.53 | 0.55 | **0.75** (0.97-1.00 at approach poses) | |
| left minus right command, same pose (1/m) | 0 | 0 | 0.0023 [0.0012, 0.0035] | 0 | **0.0106 [0.0093, 0.0118]** | |

Other open-loop lines (prereg):

| readout | line | rc-bear | rc-bear-pre | rc-ctl | rc-ctl-pre |
|---|---|--:|--:|--:|--:|
| CARLA dev exits correct per row | >= 0.8 | 0.583 [0.555, 0.613] | 0.583 [0.554, 0.612] FAIL | 0.291 | 0.291 |
|   among rows reaching d + 12 m / share short | | 0.881 / 0.34 | 0.876 / 0.33 | 0.372 / 0.22 | 0.373 / 0.22 |
|   left / straight / right commands | | 0.49 / 0.79 / 0.47 | 0.47 / 0.80 / 0.48 | 0.27 / 0.47 / 0.14 | 0.27 / 0.47 / 0.14 |
|   3-exit poses all right | | 0.375 | 0.361 | 0 | 0 |
| no-command drift, median (CARLA / nav jct / nav str / wod jct / wod str, m) | <= 0.10 | 0.037 / 0.012 / 0.009 / 0.009 / 0.010 | 0.036 / 0.015 / 0.010 / 0.011 / 0.012 pass | 0.063 / ... | 0.063 / 0.015 / 0.010 / 0.018 / 0.014 pass |
| negatives: CARLA N1 / navtrain screened / WOD online (mean, m) | <= 0.3 | 0.472 / 0.206 / 0.238 | 0.462 FAIL / 0.192 / 0.248 | 0.288 / 0.042 / 0.079 | 0.287 / 0.042 / 0.079 |

Reading: the target change moved the action head and nothing else. The plan head is unchanged (exits correct, drift and negatives within 0.01-0.02
of rc-bear / rc-ctl); rc-ctl-pre, without the command, cannot fit the exit-dependent target and stays at 0.003 at brake d 10 (a sixth of the
target), so the turn-in in rc-bear-pre comes from the command. Serving ONNX equivalence (rc-bear-pre, WOD reference streams): plan xy mean 0.010 / 0.012 m, max 0.13 / 0.19 m (pass).

### Closed loop (B2D 25 turns, zones off, seed 2)

Primary setting = desire off (`results/desire_off.md`: desire on / off changes nothing measurable, training fed desire 0; the prereg default). Desire on is the secondary row. Full tables: [pre_turns.md](pre_turns.md), failure split [pre_split_doff.md](pre_split_doff.md) / [pre_split_don.md](pre_split_don.md). One seed, one run per cell, 25 turns (13 choice, 12 forced); closed loop is not bitwise deterministic (1-2 turns are noise). shipped desire off has 24 turns (route 26365 failed in that lane).

| arm | desire off: took (choice / forced) | desire on: took | entered (off / on) | leaves lane (off) | window collisions (off / on) | turn-in median m after the turn start (n steered), off / on | "late" turns, off / on |
|---|--:|--:|--:|--:|--:|--:|--:|
| shipped | 3 / 24 (1 / 2) | 1 / 25 | 21 / 20 | 17 / 21 | 7 / 5 | 3.0 (10) / 3.1 (8) | 4 / 4 |
| rc-ctl | 5 / 25 (3 / 2) | 6 / 25 | 21 / 21 | 18 / 21 | 3 / 7 | 2.9 (20) / 3.5 (19) | 8 / 9 |
| rc-bear | 6 / 25 (1 / 5) | 4 / 25 | 22 / 21 | 18 / 22 | 11 / 5 | 4.5 (22) / 2.5 (17) | 11 / 6 |
| rc-ctl-pre | 3 / 25 (2 / 1) | 1 / 25 | 20 / 20 | 16 / 20 | 7 / 7 | 4.0 (7) / 0.6 (6) | 3 / 2 |
| **rc-bear-pre** | **1 / 25 (1 / 0)** | **0 / 25** | 19 / 19 | 18 / 19 | 6 / 6 | 1.7 (10) / 5.6 (12) | 4 / 7 |

Paired took-rate differences (route-cluster bootstrap, 95%, n 25), desire off | on:
- rc-bear-pre - rc-ctl-pre (primary): -0.08 [-0.24, +0.08] | -0.04 [-0.12, +0.00].
- rc-bear-pre - rc-bear (effect of the change): -0.20 [-0.41, +0.00] | -0.16 [-0.33, +0.00].
- rc-ctl-pre - rc-ctl: -0.08 [-0.24, +0.08] | -0.20 [-0.39, -0.04].

Lines (prereg): primary (rc-bear-pre >= 13 / 25 and the paired CI lower bound above 0) **FAILS** (1 / 25; CI contains 0 and is negative-leaning). Turn-in median <= 0 m **FAILS** (1.7 m desire off, 5.6 m on); "late" fewer than rc-bear (6): desire off 4 vs rc-bear 11 (pass in count) but only because turns are lost earlier in the split (not chosen 6, never entered 6 vs 0 and 3 for rc-bear), desire on 7 vs 4 (fail). Open-loop transmission: the 4.6x larger left-minus-right gap (0.0106 vs 0.0023 1/m) did not reach the closed loop.

Reading:
1. The turn-in target did not make the closed loop turn earlier or take more exits. It made it worse: rc-bear-pre takes 1 / 25 (desire off), the fewest of all arms, and rc-ctl-pre drops from rc-ctl's 6 to 1 (desire on, CI excludes 0). The cause split moves from "late" (rc-bear 11 of 25) to "not chosen" (6), "crawl / stop" (5) and "never entered" (6): the car no longer steers at all in a third of the turns (peak steer / needed 0.70 desire off, 0.81 on, vs rc-bear 2.27 / 1.34).
2. The speed profile is slower: median speed in the turn window 0.4 m/s (desire off) vs 1.6 rc-bear, stopped share 0.48 vs 0.37 (rc-ctl-pre 0.5 m/s, 0.48). The model drives the junction approach with a lower curvature gain (see the diagnosis below), the harness stops longer, and the longer stop starves the window; the open-loop head is right, the closed loop does not use it.
3. rc-ctl-pre loses the side bit it had as rc-ctl: steered to the commanded side in 5 / 10 entered choice turns and 6 / 10 forced (desire off; rc-ctl 9 / 11 and 9 / 10), peak steer / needed 0.40. The target that conflicts across left / right commands of one pose pushes the command-less head to a near-zero action at junctions (the diagnosis below).
4. Command-flip control (rc-bear-pre, desire off, mirrored navigation polyline, the 13 routes with a choice turn): [pre_flip.md](pre_flip.md). Correct command: 10 entered, steered to the true side 5, to the other side 3, neither 2, took 1; flipped command: 10 entered, steered to the true side 1, to the flipped side 2, neither 7, took 0. Mean peak curvature / needed towards the true side falls 1.11 -> 0.20; towards the other side it does not rise (0.36 -> 0.41). So the command does steer rc-bear-pre (true-side steering collapses in 8 of the 10 entered turns when the command is mirrored), but a flipped command produces no steering to the other side, it removes steering: the head follows the command only as a gain on a turn the scene already suggests, it does not choose the exit. Caveat: the mirror also flips forced-curve geometry on those routes (not scored) and 10 entered turns is small.

Diagnosis (open loop, real dev P rows, [`scripts/pre_diag.py`](../scripts/pre_diag.py); gain = least-squares slope of action[0] on the old target through the origin, 1 = the old scale; WOD in-turn n = 7 only):

| gain on the old target | O | rc-ctl | rc-bear | rc-ctl-pre | rc-bear-pre |
|---|--:|--:|--:|--:|--:|
| nav approach (n 56) | 1.19 | 0.97 | 0.96 | 0.47 | 0.50 |
| wod approach (n 60) | 0.91 | 0.96 | 0.97 | 0.22 | 0.22 |
| wod in turn (n 7) | 1.21 | 0.95 | 0.98 | 0.30 | 0.31 |
| nav / wod straight | 1.60 / 1.08 | 1.00 / 0.79 | 0.95 / 0.82 | 0.87 / 0.66 | 0.88 / 0.73 |

On the turn-in target the bear-pre fit is 0.85 (nav) / 0.73 (wod) with correlation 0.93 / 0.82, rc-ctl-pre only 0.65 / 0.42. So the supervision change reduced the action gain on in-turn and approach poses to a third (WOD in turn 0.98 -> 0.31) in exchange for the ramp shape on approach poses: the head follows the small turn-in ramp (0.01-0.02 1/m at 10-20 m) and under-commands the actual turn (needs 0.1-0.17 1/m), exactly the gap named in Doubts. This explains the closed loop (less steering, more crawl) better than "not enough supervision before the turn".

### Guard subset

`python experiments/op_guard/scripts/guard.py --candidate rc-bear-pre-s0` (subset, 2026-10-05): **FAIL, 7 pass / 3 fail** (rc-bear: 8 pass / 2 fail), copy in [experiments/op_guard/results/rc-bear-pre-s0/subset/guard.md](../../op_guard/results/rc-bear-pre-s0/subset/guard.md).

| line | rc-bear-pre | rc-bear | shipped | rule | result |
|---|--:|--:|--:|---|---|
| navtest PDMS (subset) | 84.34 (+0.31 [-0.06, +0.73]) | 84.26 | 84.03 | >= shipped - 0.3 | pass |
| navhard EPDMS two-stage | 32.87 (-0.47 [-1.67, +0.74]) | 32.90 | 33.33 | no drop | FAIL (as rc-bear) |
| navhard early-turn set | 43.66 (-0.23 [-0.58, +0.13]) | 43.59 | 43.89 | no drop | FAIL (as rc-bear) |
| WOD RFS / false start | 8.064 (+0.06) / 0.043 | 8.070 | 8.005 | no drop / <= +2 pp | pass / pass |
| HUGSIM spins / HD-Score | 0 / 0.430 (-0.113) | 0 / 0.479 | 0.544 | spins not up | pass |
| B2D DS (19 routes) | 67.98 (-1.28 [-13.5, +10.5]) | 66.82 | 69.27 | >= -7.6 | pass |
| B2D turns: took all / window collisions | 0 / 25 / 6 | 4 / 25 / 5 | 1 / 25 / 5 | collisions not up vs shipped | **FAIL** (6 vs 5, new) |
| drift (max 4 s lateral, m) | 0.038 | 0.038 | 0 | <= 0.10 | pass |
| negatives (offset vs shipped, m) | 0.745 (-0.035) | | 0.779 | <= +0.3 | pass |

The b2d_turns line uses the desire-on units (0 / 25 took). The two navhard failures are rc-bear's and are unchanged by the action-target change (-0.47 vs -0.43); the new failure is one window collision more than shipped.

## Figures

- [figs/pre_turnin_timing.png](../figs/pre_turnin_timing.png): left three panels = open-loop turn-in by profile and d (dashed = target; rc-bear-pre pink follows it, every other arm stays flat near 0); right two = closed-loop turn-in arc after the turn start per arm (desire off, desire on; filled = took the exit, thick bar = median, dotted = shipped). Look at: the pink open-loop curve on target vs the closed-loop dots, which sit no earlier than the other arms (median 1.7 / 5.6 m) and are fewer (10 / 19 steered).
- [figs/pre_b2d_turn_panels_doff.png](../figs/pre_b2d_turn_panels_doff.png) (primary), [figs/pre_b2d_turn_panels_don.png](../figs/pre_b2d_turn_panels_don.png): BEV tracks of four choice turns (shipped grey, rc-ctl-pre blue, rc-bear-pre red; shaded = route lane). Look at: route 334 (rc-bear-pre red takes the left turn, the others stop at the corner) and 15102 / 10255 (red begins the turn only after stopping, short of the corner; blue and grey drive straight on or sit still).

## Doubts

- One seed, one run per cell; the CARLA exit set has poses only at d 10 / 20 / 30 m, so the head is never supervised closer than 10 m to the junction; the closed loop reaches the turn at 1-3 m/s, where the creep-d10 target is about 0.012 1/m, a tenth of the 0.1-0.17 1/m the turn needs.
- v0 < 1 m/s rows stay unsupervised (unchanged); the B2D car is stopped about 40% of the turn window.
- Real approach frames get the synthetic ramp instead of the logged 1 s point (which already holds the human turn-in); correlation of the two 0.78 (nav), 0.48 (wod).
