# Replay hinge: the drivable hinge on the devkit tracker's replayed footprint removes replay-only DAC failures, by pre-compensating the tracker

Written 2026-10-07. Pre-registration [plans/2026-10-07-replay-hinge-prereg.md](../plans/2026-10-07-replay-hinge-prereg.md) (user-approved; declarations
1-5 written before any score). Question: does moving the P2H drivable hinge from the 8 raw plan poses onto the footprint the devkit's LQR tracker actually
drives (R), plus a 0.5 m front-corner margin on turning tokens (RM), recover the inside corner cuts and replay-only grazes of
[four_dirs.md](four_dirs.md) directions 1-2? Code: `lib/lqr_proxy.py` (differentiable tracker proxy), `lib/replay_hinge.py` (the loss),
`pp_train.py --hinge-replay --hinge-front-turn-margin`, `scripts/rh.py` (proxy / train / merge / report / pilot / geomtab), `scripts/rh_chain.sh`
(GPU stages). Tables in [replay_hinge/](replay_hinge/). Run data `$DATA_DIR/runs/op_parity/replay_hinge/`.

**Answer.** Yes on the scorer, and the mechanism is the one the pre-registration warned about. Every stage passed its gate: the proxy reproduces the devkit
replay to 1 cm (p95) with 100% DAC agreement; the offline thin decoder with RM cuts turning-token DAC failures by 4.6 pp (gate 0.4); the P2 pilot
(RMP-F-s0) cuts all-navtest DAC failures by 0.40 pp and raises EPDMS by +0.43 over HP-F-s0 (gates 0.3 / +0.2). The whole pilot gain is replay-only
departures (-0.41 pp), while raw-plan departures rise (+0.37 pp): the plan learns to aim where the devkit's tracker will cut, i.e. it adapts to the
scorer's execution layer, not to the road. At the full P2H10 recipe (RMH10-F, 2 seeds) the
same holds: navtest EPDMS **89.19 vs 88.67 (+0.52 [+0.38, +0.67])**, DAC failures -0.51 / -0.52 pp per seed, the WA-JEPA gap 3.04 -> 2.52; navhard
+0.25 [-0.62, +1.23] (n.s.); HUGSIM 64 `spec_plan_smooth` HD 0.435 vs 0.433 (+0.002 [-0.019, +0.027]), turning routes -0.016 [-0.040, -0.001].

## 1. Tracker proxy (validation gate: footprint p95 <= 0.1 m, DAC agreement >= 99%)

`lib/lqr_proxy.replay` re-implements, step for step in torch, what `pdm_score` does before any sub-score: linear interpolation of [ego, 8 poses] to 41
states, the two regularised least-squares fits (velocity / acceleration with jerk 1e-4, curvature / curvature rate with 1e-2), 40 closed-loop steps of
the one-step longitudinal LQR (or the stopping P controller) and the one-step lateral LQR over the 10-step linearised error dynamics, then the
kinematic bicycle with first-order acceleration / steering lag (Pacifica, wheel base 3.089 m). All gains are constants, so the proxy is the devkit
computation; gradients pass through `torch.linalg.solve` and the rollout. Inputs: v0 / a0 from the op_parity cache tab (the ego status), steering 0
(as navsim's `ego_status_to_ego_state`).

P2H10-F-s0's stored navtest plans, proxy vs the devkit `PDMSimulator` on the same plan ([proxy.json](replay_hinge/proxy.json)):

| token set | n | DAC failures | corner error median / p95 / p99 / max (m) | proxy DAC = devkit DAC |
|:--|--:|--:|:--|--:|
| **gate: 300 random (rng 0)** | 300 | 19 | 0.005 / **0.010** / 0.012 / 0.013 | **100%** |
| all navtest | 12 146 | 469 | 0.005 / 0.010 / 0.013 / 0.033 | 100% |
| DAC failures only | 469 | 469 | 0.003 / 0.009 / 0.011 / 0.012 | 100% |

Corner error = largest displacement of any of the 4 footprint corners over the 41 states. Cache-tab v0 / a0 equal the metric cache's ego state exactly
(max difference 0), and the metric cache's steering angle is 0 on every token. The residual centimetre is the devkit's own microsecond timestamp rounding
amplified by its velocity fit (proposal states differ by 6 um, velocity profiles by up to 1 cm/s), not a modelling difference. On HP-F-s0's navtest plans
(an unseen model) the proxy footprint leaves the polygons on 4.36% of tokens vs the devkit's 4.37%. **Gate passed.**

## 2. Offline gate: thin decoders on P2's plan-head hidden (gate: turning-token DAC failures -0.4 pp vs the current hinge, overall score not lower)

Decision 147 addendum 1's decoder: MLP [P2-F-s0 plan-head hidden H, ego E] -> 8 poses, navtrain s2-s4 minus dev logs (25 415 rows, 27.7% turning),
imitation + lambda 10 hinge, 4000 steps, the same seed and batch sequence for the three arms; CPU. ctrl = current hinge (8 raw poses linearly
interpolated, margin 0.3 m), R = hinge on the proxy replay (0.3 m), RM = R + 0.5 m on the two front corners of tokens whose logged 4 s heading change
exceeds 20 deg. Scored on all 12 146 navtest tokens with the devkit `pdm_score` (`opb_score.py`, reactive traffic, per-token EPDMS without EC).

| arm | DAC fail all | DAC fail > 20 deg | > 45 deg | score x100 all | score > 20 deg | LK x100 all | replay-only fail > 20 deg | raw plan out > 20 deg |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| ctrl | 3.89% | 8.85% | 10.94% | 88.85 | 82.05 | 97.31 | 3.58% | 7.93% |
| R | 2.45% | 4.69% | 6.79% | 89.21 | 84.56 | 92.98 | 0.67% | 13.00% |
| RM | **2.29%** | **4.22%** | 6.26% | **89.33** | **85.10** | 92.71 | 0.54% | 13.22% |

Paired vs ctrl, cluster bootstrap over 136 logs (108 with turning tokens), [gate_paired.md](replay_hinge/gate_paired.md), [gate.json](replay_hinge/gate.json):

| contrast | turning DAC failures (pp) | overall score x100 | gate |
|:--|:--|:--|:--|
| R - ctrl | **-4.15 [-5.32, -2.92]** | +0.36 [-0.14, +0.86] | pass |
| RM - ctrl | **-4.63 [-5.86, -3.44]** | +0.48 [-0.03, +1.01] | pass, **winner** |

Both arms pass (declared reading: DAC on the point estimate, score "not lower" = upper CI bound >= 0); RM wins by the larger turning DAC drop and goes
to the pilot. Two costs show already here: EP -0.47 [-0.61, -0.32] and lane keeping, whose failure rate rises from 2.7% to 7.0-7.3% of tokens (straight
tokens < 5 deg: score -0.94 [-1.43, -0.47], mostly LK: 98.3 -> 93.2). Raw-plan departures rise (turning tokens 7.9% -> 13.2%) while replay departures fall: the
decoder already shapes its 8 poses so that the tracker, not the poses, stays on the road.

## 3. Pilot: HP-F recipe, seed 0, drivable hinge -> RM (gate: DAC failures -0.3 pp and EPDMS >= +0.2 vs HP-F-s0)

RMP-F-s0 = the turn-train lane's HP-F pilot recipe (P2, W frames, navtrain s0-s1, 3000 steps, batch 64, lambda 10) with `--hinge-replay
--hinge-front-turn-margin 0.5`; 5.7 it/s vs 7.6 for HP (the replay costs 25% of the step). navtest through `jevdrive.bench` (W frames, the official
devkit run), paired by token, log-clustered ([pilot_paired.md](replay_hinge/pilot_paired.md), [pilot.json](replay_hinge/pilot.json)):

| RMP-F-s0 - HP-F-s0 | all (12 146) | > 20 deg (3 154) | > 45 deg (1 517) | < 5 deg (6 400) |
|:--|:--|:--|:--|:--|
| EPDMS | **+0.43 [+0.30, +0.56]** (87.71 -> 88.15) | +0.89 [+0.51, +1.29] | +0.98 [+0.43, +1.60] | +0.15 [+0.03, +0.27] |
| DAC failures (pp) | **-0.40 [-0.54, -0.25]** (4.37% -> 3.98%) | -0.86 [-1.30, -0.41] | -0.99 [-1.66, -0.38] | -0.11 [-0.22, 0.00] |
| EP x100 | -0.05 [-0.09, -0.02] | -0.21 | -0.22 | +0.03 |
| LK x100 | +0.03 [-0.10, +0.16] | +0.10 | +0.33 | -0.02 |
| TTC failures (pp) | -0.14 [-0.22, -0.05] | -0.22 | -0.26 | -0.06 |

**Gate passed** (DAC -0.40 pp, EPDMS +0.43). The offline LK cost does not carry over to the fine-tune (LK +0.03): there the hinge acts on a model
that also distils and anchors on the shipped plan, which keeps the plan's shape. Departure anatomy against the scorer's polygons, computed from both
models' stored plans with the devkit replay and the raw plan ([geom_pilot.md](replay_hinge/geom_pilot.md)):

| RMP - HP (pp) | replay out (= DAC failure) | raw plan out | replay only | raw only |
|:--|:--|:--|:--|:--|
| all | -0.40 [-0.54, -0.25] | **+0.37 [+0.15, +0.61]** | **-0.41 [-0.59, -0.24]** | +0.35 [+0.18, +0.55] |
| > 20 deg | -0.86 [-1.30, -0.41] | +0.89 [+0.25, +1.52] | -0.98 [-1.48, -0.48] | +0.76 [+0.23, +1.27] |

The gain is entirely replay-only failures; the raw plans leave the road more often, not less. The 0.5 m front-corner margin and the hinge do not move
the inside corner cut of the plan itself (raw-out rises on > 45 deg tokens too, +0.79). This is the execution-layer adaptation the pre-registration's
limits section named: the plan pre-compensates the devkit's LQR (lag and corner cutting of a 10-step-lookahead tracker) and buys DAC on this scorer.

## 4. Full stage: P2H10 recipe x 2 seeds (RMH10-F-s0 / s1), navtest / navhard G / HUGSIM 64 `spec_plan_smooth`

P2H10's recipe (all 12 navtrain shards, 10 000 steps, batch 128, lambda 10) with the RM hinge; 3.6-4.0 it/s. Reference P2H10-F-s0 / s1 (decision
148), same frames per board. Reports via `jevdrive.bench report` ([navtest_arms.md](replay_hinge/navtest_arms.md),
[navtest_paired.md](replay_hinge/navtest_paired.md), [navtest_strata.md](replay_hinge/navtest_strata.md), [navhard_paired.md](replay_hinge/navhard_paired.md),
[hugsim_spec_plan_smooth_arms.md](replay_hinge/hugsim_spec_plan_smooth_arms.md), [hugsim_spec_plan_smooth_strata.md](replay_hinge/hugsim_spec_plan_smooth_strata.md));
per-seed paired navtest in [full_seedpaired_paired.md](replay_hinge/full_seedpaired_paired.md) / [_s1](replay_hinge/full_seedpaired_s1_paired.md).

| board | RMH10 (s0 / s1) | P2H10 (s0 / s1) | RMH - P2H [95% CI] | WA-JEPA |
|:--|:--|:--|:--|:--|
| navtest EPDMS | **89.19** (89.15 / 89.24) | 88.67 (88.58 / 88.77) | **+0.52 [+0.38, +0.67]** (dDAC term +0.51, dEP -0.06) | 91.71 |
| navtest DAC | 96.69 | 96.17 | per seed: failures -0.51 [-0.71, -0.32] / -0.52 [-0.69, -0.36] pp | |
| navtest > 20 deg EPDMS | 81.3 / 81.4 | 80.3 / 80.3 | +1.05 [+0.53, +1.52] / +1.07 [+0.62, +1.55] (DAC failures -1.17 / -1.08 pp) | |
| navtest > 45 deg EPDMS | 79.49 | 78.61 | +0.88 [+0.31, +1.38] | 87.41 |
| navhard combined (G) | 32.09 (32.22 / 31.96) | 31.84 (31.40 / 32.28) | +0.25 [-0.62, +1.23]; stage 1 -0.21, stage 2 +0.97 [-0.10, +2.17] | 35.41 |
| HUGSIM 64 HD, `spec_plan_smooth` | 0.435 (0.437 / 0.433) | 0.433 (0.418 / 0.448) | +0.002 [-0.019, +0.027] | (0.451, own client) |
| HUGSIM turning routes (23) | 0.336 | 0.352 | -0.016 [-0.040, -0.001] | |

navtest gains by stratum (seed means): < 5 deg +0.21, 5-20 deg +0.63, 20-45 deg +1.23, > 45 deg +0.88, right turns +2.23 [+1.10, +3.45], left turns
+0.71; launch -0.09. NC / TTC failures also fall slightly (TTC -0.14 pp s0), LK unchanged (-0.07 / -0.03). The navtest gap to WA-JEPA closes by 0.52 of 3.04
(17%), about 40% of the ~1.3-point ceiling four_dirs put on inside cuts plus replay-only grazes; the gap on > 45 deg tokens stays 7.9 points.

Departure anatomy, seed 0 ([geom_full.md](replay_hinge/geom_full.md); seed 1 replicates: replay-only -0.49 [-0.64, -0.36] pp, raw plan out +0.28 [+0.11, +0.47]):

| RMH10 - P2H10, s0 (pp) | replay out (= DAC failure) | raw plan out | replay only | raw only |
|:--|:--|:--|:--|:--|
| all | -0.51 [-0.71, -0.32] | **+0.34 [+0.16, +0.52]** | **-0.49 [-0.68, -0.32]** | +0.35 [+0.18, +0.54] |
| > 20 deg | -1.17 [-1.67, -0.63] | +0.98 [+0.45, +1.52] | -1.20 [-1.69, -0.71] | +0.95 [+0.49, +1.40] |
| > 45 deg | -0.99 [-1.72, -0.15] | +1.12 [+0.36, +1.92] | -0.92 [-1.51, -0.32] | +1.19 [+0.47, +1.92] |

Again the whole DAC gain is replay-only departures; raw-plan departures rise by about as much as replay departures fall. On P2H10 about a third of the
DAC failures on > 20 deg tokens were replay-only (3.20 of 8.81%); RMH10 removes 38% of those and none of the raw-plan ones, which grow from 8.05% to
9.04%. The inside corner cut of the plan (four_dirs direction 1: 66% of sharp-turn failures leave in the raw plan) is not fixed by this loss.

HUGSIM does not move, as the pre-registration expected (its sharp-turn failures are fast junction entries, its tracker is HUGSIM's own); the small
loss on the 23 turning routes is within the between-run noise of a single run per scenario (P2H's two seeds differ by 0.030 HD) but its sign agrees
with plans that now aim for the devkit tracker's path rather than the road. navhard moves only in stage 2 and not significantly.

## Deviations and caveats

- Declared before scoring (prereg section "执行声明与偏差"): validation set and metrics, decoder features (P2-F-s0 H as addendum 1, not P2H's hidden),
  turning = logged |dpsi| > 20 deg, gate readings (point estimate for DAC; score "not lower" = upper 95% CI bound >= 0), all-navtest evaluation,
  `spec_plan_smooth` on HUGSIM.
- Not declared in advance but procedural: decoder scoring ran as 4 token shards (pool CPU budget); `rh.py report` concatenates them, the same 12 146
  tokens x 3 arms.
- The offline decoders' gain (-1.6 pp overall) is four times the fine-tune's (-0.40 pp); the decoder gate is a lever check, not a size estimate.
- The proxy reproduces the NAVSIM devkit's tracker; HUGSIM's controller and openpilot's own control stack are different. The replay-only gain is a
  scorer-specific gain by construction, and raw-plan departures went up.
- P2H pilot reference is HP-F-s0 (one seed); the pilot gate reads one seed by registration.
