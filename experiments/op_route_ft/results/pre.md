# rc-*-pre: turn-in action supervision at approach poses (decision 128 suspect a)

Pre-registration: [plans/2026-10-05-route-ft-prereg.md](../plans/2026-10-05-route-ft-prereg.md), section "2026-10-06" (written before training; pilot recorded there).
Code: `scripts/rft.py` (`act_target_pre`, arms `rc-bear-pre` / `rc-ctl-pre`), `scripts/rft_eval.py` (`turnin` readout), `scripts/pre_chain.sh`,
`scripts/pre_lane.py` (B2D desire off / on), `scripts/pre_report.py` (tables, split, panels), `scripts/pre_fig.py` (turn-in timing figure).

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

PENDING: lane `rft-pre` (desire off and desire on, both arms) running on card 2; see tmp/2026-10-06-pre-handoff.md.

### Guard subset

PENDING: guard.py rc-bear-pre-s0 (lines navtest, navhard, hugsim, b2d_ds, then drift, negatives, wod; b2d_turns from the lane's desire-on units).

## Figures

- figs/pre_turnin_timing.png (after the closed loop): open-loop turn-in by profile and d (left three panels; dashed = target) and closed-loop turn-in arc relative to the turn start per arm (filled = took the exit).
- figs/pre_b2d_turn_panels*.png: BEV tracks of four turns.

## Doubts

- One seed, one run per cell; the CARLA exit set has poses only at d 10 / 20 / 30 m, so the head is never supervised closer than 10 m to the junction; the closed loop reaches the turn at 1-3 m/s, where the creep-d10 target is about 0.012 1/m, a tenth of the 0.1-0.17 1/m the turn needs.
- v0 < 1 m/s rows stay unsupervised (unchanged); the B2D car is stopped about 40% of the turn window.
- Real approach frames get the synthetic ramp instead of the logged 1 s point (which already holds the human turn-in); correlation of the two 0.78 (nav), 0.48 (wod).
