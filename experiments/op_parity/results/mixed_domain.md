# Standstill decided by the scene instead of the adapter constant: on the navtrain driver the trained-in rule costs navtest 0.21 and HUGSIM 0.020 HD; the mixed navtrain + WOD adapter matches both open-loop boards and loses 0.096 HD on HUGSIM

Written 2026-10-08. Pre-registration and its two addenda: [../plans/2026-10-08-mixed-domain-prereg.md](../plans/2026-10-08-mixed-domain-prereg.md) (plan before any arm was
scored; addendum 1 before any arm was trained; addendum 2 after the P2HG pilot and before P2HGA was trained). Code: `scripts/pp_train.py` (`--stop-gate-free`, `--wod-mass`,
`--wod-slots`, all off by default), `scripts/mixed_domain.py` (teacher8, bias decomposition, gates, tables), `scripts/mixed_domain_chain.sh` (the lane). Tables:
[mixed_domain/](mixed_domain/). The question was reframed by the coordinator before any scoring: not "one driver for all boards" but whether one principle (standstill /
launch decided by the scene, not by a per-domain constant of the adapter bias) helps each board in that board's own recipe.

## Answer

Arms (2 seeds each, seed means; all read with the conventions of the stored references):

- **P2HGA** = the P2H10 recipe (navtrain, W frames, hinge 10 / 0.3, 10 000 x 128) with the standstill rule trained in (`--stop-gate 0.5`: the ego row is zeroed, so the
  adapter bias is exactly 0, on rows fed a speed below 0.5 m/s) and no anchor rows on the gated rows (`--stop-gate-free`). Served with the same rule.
- **MX** = the P2H recipe on navtrain + WOD r2-train, every batch 128 + 128 rows, each domain with its own frames, inputs, split and teacher; hinge on navtrain rows only;
  WOD rows keep their 9 real slots (the slot rule of addendum 1 fired, see Deviations).
- References, stored: P2H10 (navtrain driver), WP2 (WOD driver), shipped. WLG (WP2 + `--stop-gate 0.5`) is the wod-launch lane's arm: its WOD numbers below are recomputed
  from its stored predictions with this lane's tables and only cited; [wod_launch.md](wod_launch.md) is the authority.

| board | shipped | P2H10 | WP2 | **P2HGA** | **MX** | WLG (cited) |
|:--|--:|--:|--:|:--|:--|:--|
| navtest EPDMS (12 146 tokens, W frames) | 80.51 | 88.67 | not run | 88.46, **-0.21 [-0.34, -0.08]** vs P2H10 (seeds 88.40 / 88.52) | 88.79, **+0.12 [-0.09, +0.32]** vs P2H10 (88.79 / 88.79) | not run |
| HUGSIM 64 HD (`spec_plan_smooth`) | not run | 0.431 | not run | 0.411, **-0.020 [-0.043, -0.001]** vs P2H10 (0.400 / 0.423) | 0.335, **-0.096 [-0.158, -0.040]** vs P2H10 (0.343 / 0.328) | not run |
| WOD val RFS (479 rater frames) | 8.005 | 7.708 | 8.111 | 7.868, +0.159 [+0.047, +0.274] vs P2H10; -0.137 [-0.313, +0.040] vs shipped | 8.103, **-0.008 [-0.067, +0.055]** vs WP2; +0.099 [-0.067, +0.265] vs shipped | 8.187, +0.076 [-0.016, +0.180] vs WP2 |
| WOD val ADE@3s / @5s (m, 1 437 frames) | 1.041 / 2.117 | 1.324 / 2.668 | 0.574 / 1.457 | 1.307 / 2.614 | 0.590 / 1.487 (+0.016 [+0.008, +0.024] / +0.031 vs WP2) | 0.580 / 1.483 |
| navhard (G frames, 225 groups, no CI) | | 31.84 | | 31.05 (30.94 / 31.16) | 30.80 (30.86 / 30.74) | |

Labels by the pre-registered rules, each board against its current driver:

| principle applied in the board's own recipe | navtest | HUGSIM | WOD |
|:--|:--|:--|:--|
| navtrain driver: P2HGA vs P2H10 | **hurts** (CI upper bound < 0), small | **hurts** (CI upper bound -0.001), small; stuck 0 -> 0.5 | helps (+0.159, CI excludes 0), still below shipped |
| WOD driver: WLG vs WP2 (cited) | | | inconclusive overall (+0.076), standstill +0.393 [+0.096, +0.745], moving -0.022 [-0.062, +0.016] |
| MX vs each board's driver (one arm, not the goal) | inconclusive, not below (+0.12) | **hurts** (-0.096) | inconclusive, equal to WP2 (-0.008) |

- **The principle holds on WOD and does not pay on the navtrain boards.** On the navtrain driver the standstill rule can be trained in without the collapse of the serving
  gate (decision 167: navtest -1.50, HUGSIM -0.084 and 11.5 stuck), but it does not beat the constant: navtest -0.21, v0 < 0.5 -1.39 [-2.18, -0.41], launch -1.07, EP -0.22,
  EC -0.64; HUGSIM -0.020. By the rule this is "hurts" on both navtrain boards (point estimates below the -0.10 EPDMS floor; HUGSIM inside the -0.03 floor but the CI excludes
  0). P2H10 should keep its constant on its own boards.
- **What the trained-in rule does learn**: with the adapter off, the plan pathway itself launches. On the 852 gate-touched navtest tokens P2HGA's 4 s displacement is 7.8 m
  (P2H10 8.2, shipped 3.2, log 8.7). It is still mostly a prior, not the scene: on the 101 touched tokens where the log stays (<= 5 m in 4 s) P2HGA drives 5.0 m (P2H10 5.1,
  log 3.7), and the correlation of the plan's distance with the log's over the touched tokens is 0.72 (P2H10 0.81, shipped 0.57). The launch prior moved from the adapter
  constant into the plan weights; it did not become scene-dependent.
- **MX is two per-domain drivers in one set of weights, not a domain-neutral one.** It equals WP2 on WOD and P2H10 on navtest, including each one's standstill behaviour
  (WOD standstill 7.614 vs WP2 7.598, both -0.34 below shipped; navtest v0 < 0.5 93.40 vs 92.80), and its adapter constant is as large and as load-bearing as before
  (below). On the board that is in neither training domain it is worse than P2H10: HUGSIM launch stall 1 -> 2.5, stuck 0 -> 2, complete 25 -> 15, fg collisions 30 -> 33.

## Pilot gates

| gate | read | result |
|:--|:--|:--|
| G-NG, P2HG-P-s0 (stop gate trained in, anchors unchanged) vs RH0-F-s0, navtest | EPDMS -1.37 [-1.69, -1.07]; v0 < 0.5 -16.8 [-19.1, -14.7]; launch -7.6; EP -2.67; EC -1.70 | **fail** (thresholds -0.50 / -5.0): the same loss as the serving gate |
| diagnosis (`mixed_domain/pilot_ng_diag.md`) | on the 852 gate-touched tokens the gated plan is 0.19 m from shipped's (ungated arm: 4.60 m); 4 s displacement 3.5 m (shipped 3.2, ungated 8.1, log 8.7) | with the adapter off the plan is shipped: the anchor rows ("inputs zeroed -> shipped") train the same no-bias path |
| G-NG again, P2HGA-P-s0 (no anchor rows on gated rows; addendum 2) | EPDMS -0.27 [-0.43, -0.12]; v0 < 0.5 -2.1 [-3.0, -1.3]; launch -1.2; moving -0.13 [-0.28, +0.01]; touched tokens 4 s displacement 7.5 m | **pass** -> full. The anchors, not the plan pathway's ability, blocked the launch |
| G-MX, MX-P-s0 (8 + zero slots on WOD rows) | WOD RFS 8.016 (WP2-pilot 8.032, P2-W-s0 8.090, shipped 8.005); navtest 85.40 (P2-W-s0 85.67, WP2-pilot 75.02) | **pass** (not below both single-domain pilots on both boards) |
| MX9-P-s0 (9 real WOD slots), reported | WOD RFS 8.125, +0.109 [+0.002, +0.234] vs MX-P; navtest 85.68, +0.01 [-0.32, +0.35] vs P2-W-s0 | slot rule of addendum 1 fires (MX-P is 0.109 > 0.10 below MX9-P): the full arm uses 9 real WOD slots |

Cross reads of the single-domain pilots (inference only): the WOD pilot on navtest scores 75.02 (below shipped's 80.51, v0 < 0.5 71.9); the navtrain pilot on WOD scores
8.090 (+0.085 [-0.037, +0.216] vs shipped), i.e. the WOD loss of the navtrain driver (decision 155) is not there at pilot scale without the hinge.

## navtest (12 146 tokens, W frames, paired cluster bootstrap over 136 logs; difference vs P2H10)

| stratum | n | P2H10 | P2HGA | P2HGA - P2H10 | MX | MX - P2H10 | MX, navtest-mean bias subtracted (`:dn`) |
|:--|--:|--:|--:|:--|--:|:--|:--|
| all | 12 146 | 88.67 | 88.46 | -0.21 [-0.34, -0.08] | 88.79 | +0.12 [-0.09, +0.32] | 81.39, -7.29 [-8.10, -6.55] |
| v0 < 0.5 m/s | 848 | 92.80 | 91.41 | -1.39 [-2.18, -0.41] | 93.40 | +0.61 [-0.06, +1.53] | 72.92, -19.87 [-22.41, -17.21] |
| launch (v0 < 2, logged 4 s > 5 m) | 1 902 | 93.20 | 92.13 | -1.07 [-1.50, -0.58] | 93.47 | +0.27 [-0.09, +0.73] | 78.85, -14.35 |
| stay (v0 < 0.5, logged 4 s <= 5 m) | 98 | 85.57 | 85.06 | -0.52 [-4.74, +3.55] | 87.24 | +1.67 [-1.90, +5.52] | 71.28, -14.29 |
| moving (v0 >= 0.5) | 11 298 | 88.36 | 88.24 | -0.12 [-0.25, 0.00] | 88.45 | +0.08 [-0.13, +0.29] | 82.02, -6.34 |
| turn > 20 deg | 3 154 | 80.28 | 79.80 | -0.48 [-0.87, -0.09] | 80.65 | +0.37 [-0.17, +0.89] | 69.25, -11.03 |

Sub-scores vs P2H10: P2HGA EP -0.22 [-0.30, -0.15], EC -0.64 [-0.88, -0.41], DAC -0.12 [-0.24, -0.01], the rest within 0.05; MX DAC +0.20 [+0.01, +0.38], EC +0.13
[-0.07, +0.34], EP -0.02, the rest within 0.1 (`mixed_domain/full_navtest_subscores.md`). Shipped (P0, W) is 80.51; every arm is 8 points above it.

## HUGSIM 64 (`spec_plan_smooth`, one run per scenario and seed, paired over scenarios; P2H10 = the stored rr1 / rr2 repeats)

| arm | HD | HD - P2H10 [95% CI] | launch stall | stuck | spin | fg coll. | bg coll. | off route | complete |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| P2H10 | 0.431 | | 1 | 0 | 0 | 30 | 7 | 2 | 25 |
| P2HGA | 0.411 | -0.020 [-0.043, -0.001] | 1 | 0.5 | 0.5 | 28.5 | 10 | 2 | 22.5 |
| MX | 0.335 | -0.096 [-0.158, -0.040] | 2.5 | 2 | 0.5 | 33 | 10.5 | 3 | 15 |

P2HGA's gate in closed loop (`mixed_domain/full_hugsim_gate.md`, per-step logs): it fires in 14 / 16 scenarios (s0 / s1); in 12 / 13 of them the car relaunches and the
gate switches off again; 2 / 3 stay gated to the end and 0 / 1 of the gated runs end stuck (the serving gate on P2H10 latched in all 29 and 23 ended stuck). The
pre-registered oscillation definition is met in 6 / 5 scenarios (up to 16 flips): the car hovers around 0.5 m/s. Mean HD of the gated scenarios 0.71.

## WOD val (decision 155 harness, cluster-mean RFS, paired bootstrap over sequences, B 4 000)

| stratum | n | shipped | P2H10 | WP2 | WLG (cited) | P2HGA | P2HGA - P2H10 | MX | MX - WP2 | MX - shipped |
|:--|--:|--:|--:|--:|--:|--:|:--|--:|:--|:--|
| all | 479 | 8.005 | 7.708 | 8.111 | 8.187 | 7.868 | +0.159 [+0.047, +0.274] | 8.103 | -0.008 [-0.067, +0.055] | +0.099 [-0.067, +0.265] |
| standstill (v0 < 0.5) | 120 | 7.954 | 7.258 | 7.598 | 7.991 | 7.842 | +0.585 [+0.231, +0.958] | 7.614 | +0.016 [-0.062, +0.090] | -0.339 [-0.680, -0.050] |
| SS (standstill, log stays < 1 m) | 25 | 8.380 | 6.656 | 7.983 | 8.369 | 7.909 | +1.253 [+0.248, +2.101] | 8.032 | +0.049 [-0.016, +0.162] | -0.348 [-1.106, +0.006] |
| SL (standstill, log moves) | 95 | 7.705 | 7.210 | 7.362 | 7.759 | 7.709 | +0.499 [+0.040, +0.914] | 7.369 | +0.007 [-0.082, +0.098] | -0.335 [-0.730, +0.010] |
| launch (v0 < 2, log 5 s > 5 m) | 111 | 7.317 | 7.349 | 6.983 | 7.225 | 7.552 | +0.203 [-0.102, +0.542] | 7.110 | +0.127 [-0.027, +0.269] | -0.207 [-0.516, +0.084] |
| moving (v0 >= 0.5) | 359 | 7.981 | 7.790 | 8.260 | 8.239 | 7.837 | +0.046 [-0.029, +0.130] | 8.233 | -0.027 [-0.100, +0.053] | +0.252 [+0.065, +0.459] |
| turn intent | 52 | 6.756 | 6.211 | 6.107 | 6.685 | 6.639 | +0.428 [-0.015, +0.891] | 5.968 | -0.139 [-0.306, +0.080] | -0.789 [-1.387, -0.197] |

5 s displacement, median, SS / SL frames (log 0.0 / 9.0 m): shipped 0.6 / 4.3, P2H10 4.9 / 10.0, WP2 1.0 / 4.8, WLG 0.6 / 4.1, P2HGA 4.6 / 7.6, MX 0.8 / 5.6.
P2HGA still creeps where the log stays (4.6 m): the navtrain launch prior is now in the plan weights, and it transfers to WOD frames as the constant did; its standstill RFS
is nevertheless 0.59 above P2H10's. MX does not creep on WOD (0.8 m) while it launches on navtest standstill tokens as P2H10 does.

## Adapter bias: constant vs ego-dependent part (decision 162 method; rms over 32 x 512, seed means)

| arm | domain | bias rms | constant (mean over frames) | residual | standstill frames: rms | standstill constant | standstill residual |
|:--|:--|--:|--:|--:|--:|--:|--:|
| P2H10 | WOD val / navtest | 1.11 / 1.13 | 1.01 / 1.03 | 0.46 / 0.47 | 1.02 / 1.04 | 0.99 / 0.99 | 0.24 / 0.34 |
| WP2 | WOD val / navtest | 1.10 / 1.11 | 0.97 / 0.87 | 0.53 / 0.68 | 1.10 / 1.11 | 1.02 / 0.92 | 0.41 / 0.61 |
| **MX** | WOD val / navtest | 1.08 / 1.11 | 0.96 / 0.93 | 0.51 / 0.62 | 1.05 / 1.07 | 1.00 / 0.92 | 0.32 / 0.54 |
| P2HGA | WOD val / navtest | 1.06 / 1.16 | 0.88 / 1.03 | 0.59 / 0.51 | 0 | 0 | 0 |
| WLG | WOD val / navtest | 0.99 / 1.09 | 0.78 / 0.84 | 0.61 / 0.69 | 0 | 0 | 0 |

- **MX's constant did not shrink**: 0.96 / 0.93 of a 1.08 / 1.11 rms, the same as the single-domain adapters; at standstill 1.00 of 1.05 on WOD frames is one fixed vector.
- **Domain-neutral in the adapter, by construction, and not more than before**: the standstill constants of the two domains differ by 0.39 rms (cosine 0.92) for MX,
  0.44 (0.90) for WP2, 0.16 (0.99) for P2H10 (`mixed_domain/bias_cross_domain.md`); the difference comes from the inputs (command mix, residual acceleration), the adapter
  reads only the ego state.
- **The constant is still load-bearing on both boards.** WOD serving variants of MX (`mixed_domain/full_wod_biasvars.md`, difference vs MX): bias zeroed, all -0.090
  [-0.256, +0.077], standstill **+0.355 [+0.054, +0.719]**, moving -0.242 [-0.450, -0.055]; constant only, all -0.939 [-1.226, -0.664]; residual only, all -0.185
  [-0.369, -0.019], standstill +0.232 [+0.030, +0.461]. The standstill gain of zeroing is WP2's (+0.357, wod-launch Step 1). navtest with the navtest-mean bias subtracted:
  -7.29, v0 < 0.5 -19.9 (P2H10: -6.71 / -21.1, decision 167). So MX kept, in one adapter, the constant that launches on navtest and costs at standstill on WOD; which of the
  two behaviours it shows is decided downstream of the adapter, from the frames (and possibly the slot count).

## Slot count (found by the teacher8 smoke, addendum 1)

Zeroing the oldest of the 9 policy slots moves shipped's plan by 1.09 m on average on the 188 883 WOD r2 rows (1.17 m at 4 s, p99 of the per-row mean 5.5 m;
`runs/op_parity/mixed/teacher8.json`; the same path on 9 slots reproduces the stored teacher to 4e-6 m). navtrain rows are 8 + zero, the WOD harness and HUGSIM serve 9 real
slots. At pilot scale training the WOD rows on 8 + zero costs 0.109 RFS on WOD val against 9 real slots (CI [+0.002, +0.234]) and nothing on navtest. Not broken down by
speed, and not tested as a source of P2H10's WOD loss.

## Verified vs not

Verified: the tables' code reproduces the stored numbers before any new arm was read (shipped RFS 8.005, WP2-pilot 8.032, P0 80.51, P2-W-s0 85.67, P2H10 88.67, P2H10
HUGSIM 0.431 and the serving gate's -0.084 [-0.141, -0.035]); P2H10 / WP2 / shipped rows equal decisions 148 / 155 / 163; training sanity (`pp_full_check.py train`) passes
for the four full runs (dev ADE P2HGA 0.55, MX 0.63 with WOD 0.64 / navtrain 0.55; MX's inputs-off drift 0.04 m; P2HGA's 0.19 m, expected: its gated rows have no anchor);
64 / 64 scenarios in every HUGSIM run.
Not verified / not done: the default P2H / WP2 code paths were not re-run for bit-equality after the `pp_train.py` edits (the edits are behind flags; the ungated pilots of
other lanes ran on the same file tonight); WLG's numbers are this lane's recomputation from the stored predictions, not checked against the wod-launch tables; navhard has no
CI; WP2 / WLG were not run on navtest or HUGSIM, shipped not on HUGSIM `spec_plan_smooth`; why MX loses on HUGSIM was not diagnosed
(slot count as a domain cue, the missing navtrain-only schedule, or batch 256 are untested candidates); no second MX arm (domain flag, MX + stop gate) was run.

## Deviations

- The lane's question and success rule were changed by the coordinator before scoring (per-board principle; MX one arm among others); the prereg was rewritten accordingly.
- P2HG (anchors unchanged) failed its pilot gate; P2HGA (`--stop-gate-free`) was added in addendum 2 after that read and is the arm reported at full scale. It is a recipe
  chosen after seeing a navtest pilot.
- MX full run: WOD rows with 9 real slots, not the pre-registered 8 + zero, by the addendum 1 rule (pilot difference 0.109 against a 0.10 threshold, one seed, CI lower
  bound +0.002). The number of real slots is therefore a domain cue available to the plan pathway.
- MX batch 256 (128 per domain) at the single-domain learning rate; distillation scale computed over both domains' train rows.
- MX pilot navtrain rows use the G-protocol teacher on W frames, as the stored P2-W-s0 pilot did.
- One WOD serving job (MX-F-s1) was killed (rc 137, cause not found) and rerun by resuming the chain; a first teacher8 job ran out of card memory and was rerun at batch 128.
- Cost: about 3.3 card-hours of training (P2HG / P2HGA pilots 4 min each, P2HGA 2 x 33 min, MX pilots 2 x 2 min, MX 2 x 52-60 min on shared cards), 8 WOD serving jobs (one a rerun) of
  13-25 min at 8 GB, 4 HUGSIM 64 runs; not metered exactly.

## Caveats

- WOD is open loop on 479 rater frames (standstill 120, SS 25); CI half-width about 0.17 overall. HUGSIM is one preset, one run per scenario.
- "Decided by the scene" cannot be separated from "decided by the domain's appearance" here: warp-synthesised navtrain frames and real WOD frames look different to the
  frozen encoder. HUGSIM is the only read outside both training domains, and it is where MX loses.
- P2HGA's verdict rests on small effects (-0.21 EPDMS, -0.020 HD) that are consistent over two seeds and exclude 0; the HUGSIM bound (-0.001) is at the edge.
- Threshold 0.5 m/s fixed, never tuned; the HUGSIM gate reads the model-clock speed fed to the adapter.
