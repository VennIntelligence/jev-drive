# DIAG1: the nuPlan adapter's speed profile is a function of the ego state only; it replaces the base model's vision-driven brake initiation and standstill hold with the logged human's timing, which is worth score on navtest and costs it on WOD and in closed loop

Written 2026-10-10. Pre-registration [plans/2026-10-09-diag1-prereg.md](../plans/2026-10-09-diag1-prereg.md), pushed before any read. Measurement only: existing
checkpoints, no training, no simulator run, nothing submitted. Code `scripts/diag1.py`, `scripts/diag1_lead.py`, `scripts/diag1_chain.sh`, `scripts/diag1_score.sh`,
`experiments/alpasim/scripts/diag1_alp.py`. Tables [diag1/](diag1/), figures `figs/diag1/`. Lane FIX1's law is used as committed
(`experiments/alpasim/lib/serve_fix.py`, 8f78938d / 0763ea89). Decision 218.

Arms: shipped = shipped Cinque policy weights, no adapter, same vision tokens; P2H10-F and SH30-F (2 seeds each); AP2H10-AB / APY10m10-AB where their input
standard is served by existing code. Domains: navtest (12 146 tokens, 136 logs), WOD val (1 437 frames, 479 rater frames), AlpaSim nuPlan rollouts COL1
extracted (2 000 decisions of 200 P2H10-F-s0 control rollouts; 620 decisions of collision rollouts), AlpaSim PAI rollouts (3 693 decisions of 20 baseline
P2H10-F-s0 rollouts, cold-start decisions dropped).

## Units and frames check (first, as registered)

No frame, unit or scaling fault: fed speed = ego speed in every domain (ratio 1.000), history poses consistent with the fed speed (0.94 - 1.02), rear-axle
frames and waypoint grids as documented ([diag1/check_units.md](diag1/check_units.md)). Two things that are not faults but matter below:
- The fed longitudinal acceleration is much wider in closed loop: std 0.76 m/s^2 on navtest, 0.27 on WOD, 0.94 on the AlpaSim nuPlan track, 2.2 on PAI
  (1 / 99 pct -6.5 / +4.6; it agrees with the 0.5 s difference of the fed speed, corr 0.79: the rollout's own motion, not a scale error).
- On PAI the first decision of 8 of 20 rollouts is fed all-zero dynamics while the ego moves (cold start, one frame each; dropped here).
- The shipped model's own speed estimate (plan velocity at t = 0) is 0.94 x (navtest) and 0.91 x (WOD) the ego speed, and the shipped plan's first 0.5 s is
  3 - 13% below the ego speed in every domain. This is the base model, not adaptation.

## Job 1: FIX1's two fixes on WOD val, open loop

Stored predictions (decision 155 harness), post-processed. (a) = the form of `col1_pai_driver.py::retime` (= `serve_fix.join` + `place`), tau 1.0 s, waypoint grid
20 x 0.25 s, v0 = the fed speed. (b) = `serve_fix.LeadPlanner` called as committed (radard vision lead, openpilot's lead MPC), cap rule of `Serve.__call__`;
ours only: the WOD grid and, having no session, a fresh planner with one reset tick per frame (2 / 20 more held ticks change no number by more than 0.01).
RFS on 479 rater frames, paired bootstrap over sequences (B 4 000); ADE@3s on 1 437 frames. G0: 7.708 / 7.734 / 8.005 reproduce.

| arm | fix | RFS | d RFS [95% CI] | seeds | d ADE@3s (m) | registered label |
|:--|:--|:--|:--|:--|:--|:--|
| P2H10-F | (a) | 7.708 -> 7.767 | +0.058 [+0.027, +0.092] | +0.054 / +0.063 | -0.154 [-0.183, -0.129] | helps |
| P2H10-F | (b) | 7.788 | +0.080 [+0.010, +0.157] | +0.088 / +0.072 | -0.014 | (not a primary row) |
| P2H10-F | (a) + (b) | 7.844 | +0.136 [+0.059, +0.220] | +0.140 / +0.132 | -0.166 [-0.195, -0.141] | helps |
| SH30-F | (a) | 7.734 -> 7.803 | +0.070 [+0.039, +0.104] | +0.068 / +0.071 | -0.165 [-0.194, -0.140] | helps |
| SH30-F | (b) | 7.802 | +0.069 [+0.012, +0.129] | +0.064 / +0.073 | -0.016 | (not a primary row) |
| SH30-F | (a) + (b) | 7.870 | +0.136 [+0.072, +0.206] | +0.130 / +0.143 | -0.180 [-0.209, -0.154] | helps |
| shipped (control) | (a) | 8.005 -> 8.052 | +0.047 [+0.013, +0.082] | | -0.138 | |
| shipped (control) | (b) | 7.998 | -0.007 [-0.070, +0.044] | | +0.004 | |
| shipped (control) | (a) + (b) | 8.044 | +0.040 [-0.029, +0.101] | | -0.134 | |

Strata of (a) + (b), d RFS (secondary, no label; full table [diag1/wod_fix_rfs_lead.md](diag1/wod_fix_rfs_lead.md), (a) alone [diag1/wod_fix_rfs.md](diag1/wod_fix_rfs.md)):

| stratum | n | P2H10 | SH30 | shipped |
|:--|--:|:--|:--|:--|
| standstill (v0 < 0.5) | 120 | +0.329 [+0.139, +0.568] | +0.284 [+0.130, +0.484] | +0.015 [-0.103, +0.126] |
| moving | 359 | +0.062 [-0.009, +0.132] | +0.083 [+0.014, +0.149] | +0.044 [-0.031, +0.108] |
| lead present | 165 | +0.269 [+0.080, +0.466] | +0.248 [+0.073, +0.418] | +0.051 [-0.065, +0.175] |
| no lead | 314 | +0.068 [+0.011, +0.166] | +0.083 [+0.022, +0.170] | +0.071 [+0.008, +0.152] |
| standstill, lead | 35 | +0.769 [+0.338, +1.221] | +0.609 [+0.231, +0.990] | +0.066 [-0.317, +0.258] |

Reading:
- (a) is not adapter-specific: shipped gains too. On WOD every arm's plan starts below the ego speed (first 0.25 s: shipped 0.93, P2H10 0.90, SH30 0.89 of
  v0 at 5 - 12 m/s), so re-timing lengthens the plan, the direction raters prefer (decision 164). It is the opposite sign of the PAI finding.
- (b) acts where the adapted plan creeps toward a stopped lead (35 frames, +0.6 to +0.8); it does nothing to shipped, which does not creep. The feared
  cost of a cap on RFS did not appear. It moves 14 - 16% of frames (1.8 - 2.4 m shorter on those).
- (a) + (b) closes about half of the gap to shipped (7.84 / 7.87 vs 8.005); both arms stay 0.17 - 0.20 below shipped with the same fixes.
- Sensitivity of (a): tau 0.5 / 2.0 gives +0.029 / +0.108 (P2H10), +0.038 / +0.111 (SH30); v0 = `init_speed` changes nothing.
- Deviation: the registered gate G1 (v0 := the plan's first-segment speed is an identity) is wrong for any plan that accelerates (0.29 - 0.39 m by construction);
  replaced by the tau -> 0 operator identity, error 1e-13 m.

The same fixes on the nuPlan board (post hoc, `score-poses` non-reactive, no-EC EPDMS, s0 seeds; [diag1/nav_retime_a.md](diag1/nav_retime_a.md), [diag1/nav_fix_b.md](diag1/nav_fix_b.md)):
(a) costs nothing (P2H10 +0.14 [+0.05, +0.23], SH30 +0.09 [+0.00, +0.18]); (b) applied open loop costs -2.88 [-3.37, -2.42] / -2.99 [-3.52, -2.48], almost all
on standstill tokens with a lead (648 tokens, -26; EP -50, NC -13). This is an upper bound on its cost, not its closed-loop cost: one stateless tick, a 4 s
open-loop score against a log in which the lead pulls away, and navtest standstill tokens are launches by selection (the log moves 2.1 m in 2 s on them).

## Job 2: what adaptation changed longitudinally

### 1. The adapted speed profile is carried entirely by the ego-input bias, and inside it by speed, acceleration and a constant

Switches on identical tokens ([diag1/d5_switches.md](diag1/d5_switches.md)). Share of the adapted - shipped difference that remains:

| switch | navtest: first-segment ratio / 2 s arc / standstill 2 s arc | WOD | AlpaSim nuPlan control rollouts |
|:--|:--|:--|:--|
| adapter off (`nobias`: fine-tuned plan weights on the vision tokens) | 0.00 / 0.00 / 0.00 | -0.33 / 0.65 / 0.00 | 0.02 / 0.0 / 0.00 |
| straight constant-speed history (`posecv`) | 1.00 / 0.98 / 1.00 | 0.99 / 1.01 / 1.00 | 1.00 / 1.0 / 1.00 |
| adapter off below 0.5 m/s (`gate`) | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.0 / 0.00 |
| constant part of the bias only (`const`) | 0.87 / 3.0 / 1.73 | 0.50 / 3.1 / 3.1 | not run |
| bias minus its constant (`resid`) | 0.18 / -1.4 / -0.07 | 0.07 / -0.8 / -0.12 | not run |

- With the adapter off the fine-tuned plan pathway is the shipped plan (navtest standstill arc 0.66 m vs shipped 0.66 m, adapted 2.05 m). The vision tokens
  contribute nothing to the difference: the adapter reads 20 ego numbers and no image, and the plan weights were anchored to shipped.
  (On WOD the 2 s arc share of 0.65 is a 1.3% arc difference of the plan weights on real frames; the standstill and first-segment differences are 0.)
- The 4-pose history carries nothing beyond the speed.
- No arm has a memory channel (all are arm P2), so "memory-mask" does not apply.
- The acceleration input is a strong lever (post hoc probe on navtest tokens, [diag1/acc_probe_nav.md](diag1/acc_probe_nav.md)): +2 m/s^2 on the fed ax raises the first-segment
  speed by 14.5% of ego speed at 2 - 5 m/s and 4.8% at 5 - 12 m/s and the 4 s arc by 43% / 18%; -2 m/s^2 shortens the 4 s arc by 36% / 24%. Slope
  0.15 m/s of first-segment speed per m/s^2 of injected ax. Shipped has no such input.

### 2. First segment and arc length by speed and domain

Adapted minus shipped first-segment ratio (plan speed over the first 0.5 s / ego speed; [diag1/d1_first_segment.md](diag1/d1_first_segment.md), figure
`figs/diag1/first_segment.png`: look at the adapted lines crossing from above shipped at low speed to below it at high speed in the open-loop panels, and
staying above it in the PAI panel):

| ego speed | navtest | WOD | AlpaSim nuPlan control | PAI |
|:--|:--|:--|:--|:--|
| 2 - 5 m/s | +0.065 [+0.051, +0.078] | +0.065 [+0.051, +0.078] | +0.087 [+0.072, +0.102] | +0.106 [+0.032, +0.176] |
| 5 - 12 m/s | +0.006 [-0.001, +0.014] | +0.004 [-0.003, +0.013] | +0.012 [-0.001, +0.026] | +0.091 [+0.064, +0.118] |
| 12 - 16 m/s | -0.063 [-0.081, -0.044] | -0.027 [-0.040, -0.014] | -0.046 [-0.068, -0.015] | +0.067 [+0.054, +0.076] |

- On logged frames (navtest, WOD) and in the nuPlan control rollouts the adapted plan is not above the ego speed at 5 - 12 m/s (-3.5% on navtest); COL1's
  "5 - 28% above ego at 5 - 12 m/s" is specific to PAI.
- On PAI the excess is the acceleration input ([diag1/acc_probe_pai.md](diag1/acc_probe_pai.md)): adapted - shipped first-segment ratio is -0.03 / -0.05 at fed ax < -2,
  +0.05 at |ax| < 0.5, +0.13 at 0.5 - 2 and +0.26 [+0.19, +0.33] / +0.24 [+0.18, +0.31] at ax > 2 m/s^2 (two scene sets); corr 0.52 / 0.55, slope 0.23 m/s
  per m/s^2. The shipped plan is 9 - 12% below ego speed whatever the ax. A plan above ego speed makes the tracker accelerate, which raises the fed ax:
  the longitudinal twin of the yaw-rate continuation of decisions 205 / 213.
- Arc length vs the log ([diag1/d2_arc_ratio.md](diag1/d2_arc_ratio.md)): on navtest the adapted plan matches the log within 1% at every speed and horizon (shipped 0.79 - 1.07);
  on WOD it is 6 - 7% short at 4 s for 5 - 12 m/s and 12% short above 16 m/s (shipped 1 - 6% short).
- AP2H10 / APY10m10 on navtest equal P2H10 / SH30 in every read (their input standard changes nothing here).

### 3. Lead response

Closing-lead frames as registered (shipped lead head p > 0.5, v0 >= 2, closing >= 1 m/s, time to contact <= 8 s). "Slows" = plan acceleration to 1.5 - 2 s
<= -0.3 m/s^2 ([diag1/d3_lead_share.md](diag1/d3_lead_share.md), [diag1/d3_lead_cells.md](diag1/d3_lead_cells.md), [diag1/acc_probe_lead.md](diag1/acc_probe_lead.md); figure `figs/diag1/lead_response.png`: look at the gap between the
shipped and adapted lines opening from navtest / WOD to the control rollouts to PAI).

| domain | closing-lead frames (units) | shipped slows | adapted slows | shipped slows, adapted not | adapted slows, shipped not | slope of plan a on required decel: shipped / adapted |
|:--|:--|:--|:--|:--|:--|:--|
| navtest (P2H10; SH30, AP arms equal) | 3 134 (107 logs) | 0.943 | 0.912 (log 0.889) | 0.060 [0.046, 0.073] | 0.029 [0.020, 0.038] | -0.79 / -0.78 |
| WOD val | 58 (41 seq) | 0.948 | 0.974 | 0.009 [0, 0.028] | 0.034 [0, 0.086] | -0.87 / -0.66 |
| AlpaSim nuPlan control (P2H10-F-s0) | 454 (83 scenes) | 0.879 | 0.793 | 0.137 [0.088, 0.193] | 0.051 [0.023, 0.083] | -0.84 / -0.41 |
| AlpaSim PAI (P2H10-F-s0) | 314 (11 scenes) | 0.959 | 0.627 | 0.331 [0.132, 0.523] | 0.000 | -0.44 / -0.15 |

Split by whether the ego is already braking (fed ax < -0.3 m/s^2):

| domain | frames | n | shipped slows (mean a) | adapted slows (mean a) | log |
|:--|:--|--:|:--|:--|:--|
| navtest | closing, ego already braking | 2 393 | 0.980 (-1.07) | 0.997 (-1.10) | (-1.09) |
| navtest | closing, ego not braking | 741 | 0.822 (-0.70) | 0.638 (-0.36) | (-0.33) |
| AlpaSim nuPlan control | closing, ego already braking | 306 | 0.938 (-0.89) | 0.958 (-0.95) | |
| AlpaSim nuPlan control | closing, ego not braking | 148 | 0.757 (-0.75) | 0.453 (-0.24) | |
| PAI | closing, ego already braking | 80 | 0.975 (-2.01) | 0.950 (-1.23) | |
| PAI | closing, ego not braking | 234 | 0.953 (-1.67) | 0.517 (-0.24) | |

- On navtest the adapted plan's response to a closing lead looks intact (0.91 vs 0.94), but 76% of those frames are fed an ego that is already braking,
  because the logged human is. With the acceleration input zeroed the adapted plan's mean deceleration on closing-lead frames drops from -0.93 to -0.65
  (shipped -0.98) and the slope on required deceleration from -0.78 to -0.53.
- Where the ego is not yet braking, the adapted plan starts to slow in 64% (navtest), 45% (nuPlan control rollouts), 52% (PAI) of closing-lead frames;
  the shipped plan on the same tokens in 82%, 76%, 95%. The adapted plan equals the logged human there (-0.36 vs -0.33 m/s^2): it learned when people
  start braking, and expresses it through the ego state, not through the image.
- At time to contact < 2 s on PAI (99 frames) the shipped plan asks -2.7 m/s^2, the adapted one -0.8 (difference +1.90 [+1.27, +2.47]).
- Sensitivity (p > 0.7 and a <= -0.5; closing speed from the model's own speed as radard does) keeps the navtest sign and size (+0.037 / +0.013 difference).
  WOD has 58 frames: no statement.

### 4. Standstill

Plan arc at 2 s from v0 < 0.5 m/s ([diag1/d4_standstill.md](diag1/d4_standstill.md); figure `figs/diag1/standstill.png`: look at the adapted bars equal with and without a stopped
lead, and at the log bar next to them per domain). Share of frames whose plan moves more than 0.5 m in 2 s in brackets.

| domain | stopped lead ahead: shipped | adapted | log | no lead: shipped | adapted | log |
|:--|:--|:--|:--|:--|:--|:--|
| navtest | 0.44 m (0.33) | 1.73 m (0.99) | 1.74 m (0.90) | 0.83 (0.52) | 2.25 (1.00) | 2.33 (0.96) |
| WOD val | 0.23 m (0.18) | 0.68 - 0.74 m (0.55 - 0.65) | 0.33 m (0.23) | 0.55 (0.30) | 1.07 - 1.17 (0.86 - 0.94) | 0.87 (0.48) |
| AlpaSim nuPlan control | 0.24 m (0.15) | 1.04 m (0.93) | (1.17, recorded ego, loose) | 0.65 (0.47) | 1.69 (1.00) | |
| AlpaSim PAI | 0.18 m (0.15) | 1.19 m (0.90) | | 0.55 (0.36) | 1.24 (0.83) | |

The adapted plan launches from standstill whether or not the model's own lead head reports a stopped vehicle within 15 m (90 - 99% of stopped-lead
frames in the nuPlan-derived domains and PAI, 55 - 65% on WOD); the shipped plan on the same tokens holds in 67% (navtest) to 82 - 85% of them. On navtest that launch is correct by selection (standstill tokens are
launches: the log moves in 90 - 96% of them, the "stopped" lead is about to go); on WOD the log holds in 77% of stopped-lead frames. The switch table
shows the launch is the bias constant (decisions 162 / 167), unchanged by SH30's stronger hinge or the AP input standard.

### 5. How much of the board gap is longitudinal (swaps)

AS = adapted path + shipped speed profile, SA = shipped path + adapted speed profile ([diag1/swap_shares.md](diag1/swap_shares.md), [diag1/swap_nav_subscores.md](diag1/swap_nav_subscores.md),
[diag1/swap_composites.md](diag1/swap_composites.md)). navtest: `score-poses --traffic non_reactive`, no-EC EPDMS, all 12 146 tokens, CI over logs; the unswapped keys reproduce
the stored bench sub-scores on every token (0 tokens differ). WOD: RFS; the P2H10 row reproduces decision 162 (+0.062 / -0.464).

| board | arm | shipped | adapted | A - S | AS - S (path) | SA - S (speed) | path share | longitudinal share |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| navtest | P2H10 | 82.39 | 89.20 | +6.81 [+5.89, +7.76] | +3.52 [+2.64, +4.46] | +1.81 [+1.19, +2.49] | 0.52 | 0.27 |
| navtest | SH30 | 82.39 | 90.07 | +7.68 [+6.71, +8.69] | +4.32 [+3.39, +5.34] | +1.78 [+1.17, +2.46] | 0.56 | 0.23 |
| WOD val | P2H10 | 8.005 | 7.708 | -0.297 [-0.501, -0.094] | +0.062 [-0.023, +0.152] | -0.463 [-0.671, -0.274] | -0.21 | 1.56 |
| WOD val | SH30 | 8.005 | 7.734 | -0.271 [-0.477, -0.069] | +0.072 [-0.013, +0.163] | -0.462 [-0.671, -0.273] | -0.27 | 1.71 |

- navtest: about half of the adaptation gain is path (DAC 93.2 -> 95.4 / 96.4 with the shipped speed), about a quarter is speed alone and the rest needs
  both. The speed profile's own part is all standstill launch (SA - S +18.5 on 848 standstill tokens, +0.56 moving, 0.0 at 5 - 12 m/s). Removing the adapted
  speed from the adapted path costs AS - A = -3.29 [-3.87, -2.77] / -3.35 [-3.91, -2.86]: EP 87.2 -> 82.9, NC 98.6 -> 97.7, TTC 98.0 -> 97.5.
- WOD: the whole loss is the speed profile (more than the whole: the adapted path alone is +0.06 / +0.07 above shipped). Standstill SA - S -0.91 / -0.97,
  >= 12 m/s -0.83 / -0.69, lead frames -0.48. SH30 = P2H10.
- What the acceleration input is worth on navtest (post hoc, [diag1/acc_probe_nav_score.md](diag1/acc_probe_nav_score.md)): fed ax := 0 costs -3.28 [-4.15, -2.50] (P2H10-F-s0) / -3.27 (SH30-F-s0),
  NC -2.3, TTC -2.8; ax x 3 costs -1.05; ax + N(0, 4.3 m/s^2) costs -9.45. That is 43 - 48% of the size of the adaptation gain on the nuPlan board, and more than its whole
  NC / TTC gain (A - S: NC +1.7, TTC +1.1); it rides on an input that in the log is the human's own reaction and in closed loop is the driver's own output.

## Answer

1. **Is the adapter's speed behaviour a nuPlan prior that overrides the base model's lead-aware longitudinal behaviour?** Yes, in a specific form. The
   adapted speed profile is a function of ego speed, ego acceleration, command and a constant; no image enters it. It is three things: a launch at
   standstill that ignores a stopped lead; a continuation of the fed acceleration; arc scaling by speed that matches the nuPlan log. Once the ego is braking
   the adapted plan brakes like shipped. What is overridden is brake initiation and the standstill hold: on the same tokens the shipped plan starts slowing
   for a closing lead in 76 - 95% of frames where the ego is not yet braking, the adapted plan in 45 - 64%, the nuPlan human at the adapted rate.
2. **Where does it cost score?** Not on navtest: there the speed profile is worth +1.8 by itself and +3.3 on the adapted path, and the acceleration input
   another 3.3, because the log's human has already reacted. On WOD all of the -0.27 / -0.30 is the speed profile. In the AlpaSim rollouts there is no
   offline board metric; the mechanism matches COL1's rear-ends: the share "shipped slows, adapted does not" rises from 6% (navtest) to 14% (nuPlan control
   rollouts) to 33% (PAI), and the PAI first-segment excess is the acceleration continuation.
3. **Cheapest change that keeps the nuPlan-board gain without exporting the prior.** Measured ceilings, both boards (per-frame switch between A and AS):

| change (training-free) | navtest no-EC EPDMS vs adapted | WOD RFS vs adapted |
|:--|:--|:--|
| FIX1 (a) speed-continuous serving | +0.14 [+0.05, +0.23] / +0.09 | +0.058 / +0.070 |
| FIX1 (a) + (b), (b) open-loop upper bound of cost | -2.77 / -2.93 | +0.136 / +0.136 |
| adapter path + base speed profile everywhere (AS) | -3.29 / -3.35 | +0.359 [+0.180, +0.546] / +0.344 (above shipped: 8.07 / 8.08) |
| base speed when moving, adapter launch at standstill | -1.96 / -1.97 | +0.199 / +0.188 |
| base speed on lead frames (p > 0.5) | -2.14 / -2.20 | +0.150 / +0.147 |
| base speed on closing-lead frames only | -0.24 [-0.45, -0.06] / -0.24 | -0.004 / -0.003 |
| fed acceleration := 0 at serving | -3.28 / -3.27 | within 0.07 (decision 162) |
| per-frame best of A and AS (privileged upper bound) | +2.10 / +2.01 | +0.735 / +0.735 |

   No training-free switch keeps the navtest gain and removes the export: every one that removes the adapter's speed costs 2 - 3 points on navtest, because
   the navtest gain is partly that prior. Per board, the cheapest are: WOD = serve the adapter's path with the base speed profile (8.07 / 8.08, above shipped,
   two policy-head passes on the same tokens, not yet with (a), which is additive in form); AlpaSim = FIX1's (a) + (b), which is what this audit says is
   missing (brake initiation from the lead outputs; the standstill hold), with its cost to be read in closed loop, not from the navtest number above.
   Training-side, in order of cost: (i) drop or heavily noise the acceleration input (and train so), since it is the closed-loop feedback path and the
   leak of the human's reaction; its navtest cost is at most the 3.3 above before retraining; (ii) keep the anchor on the base speed profile for moving rows
   and let the adapter learn path only, with launch conditioned on the base model's lead output (lead prob / gap as adapter inputs or a lead-gated speed
   residual); the per-frame oracle (+2.1 navtest, +0.74 WOD) says a per-frame choice between the two speed profiles is worth more than either.

## Limits

- Open loop everywhere; the AlpaSim reads are plans on logged decisions of our own rollouts, with correlated frames (PAI: 20 rollouts, closing-lead
  frames from 11; 10 Hz). No AlpaSim score is computed here. PAI covers the baseline driver only (the re-timed arm had no replay yet).
- Lead frames and gaps come from the shipped model's lead head (near-range bias, decision 157); the WOD front-bumper offset is approximate (3.9 m, +- 0.2).
- navtest swaps and probes are no-EC EPDMS from `score-poses`; the retime / cap / acceleration rows use seed 0 only.
- (b) open loop is a stateless single tick; its navtest cost is an upper bound and its WOD gain is not a closed-loop statement.
- The acceleration probe, the brake-initiation split, the navtest scoring of (a) / (b) and the cold-start drop on PAI are post hoc (added after the first read).
- `const` / `resid` were not run on the AlpaSim replays; AP arms were not run on WOD (no input mapping, decision 217).
- The logged-future columns in the AlpaSim tables are the recorded ego from the same sim time, not a label for the rollout's state; not used in any claim.

## Box notes

- `git pull` on the GPU box was blocked by untracked copies of `results/wod1_recipes/*` (generated there by lane WOD1 and committed from the Mac); moved to
  `$DATA_DIR/runs/op_parity/diag1/box_untracked_backup/` (identical to the committed files) and the box brought to `main`.
- No input file was missing this time (`processed/wod_zeroshot/sets.json` and `navsim_zs` indices present). The box's inbound link ran at 15 - 60 kB/s
  for part of the session (scp / fetch stalls); the AlpaSim PAI table was built on the Mac and uploaded in pieces (sha1 checked).
- Cost: about 0.6 card-hours (navtest extraction 4 min, two WOD serving jobs 16 min each, one replay 8 min, one probe 2 min), about 12 core-hours of
  pose scoring; outputs 0.3 GB under `$DATA_DIR/runs/op_parity/diag1/` and `runs/bench/poses/{swap,cap,acc}_poses-*`.
