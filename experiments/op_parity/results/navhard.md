# navhard two-stage: input parity lifts stage 1 by 10 points but loses 4 in stage 2; combined P2 29.3 vs WA-JEPA 35.4 (same harness)

Written 2026-10-06. Chain: `scripts/pp_navhard_chain.sh` (our arms: `pp_prep.py --data lb_navhard` + `--frames warp`, `pp_eval.py plans`,
op_interp nav-export, `experiments/op_guard/scripts/nav_harness.py`; WA-JEPA: `scripts/pp_navhard_wajepa.sh`, its own runner
`experiments/top10/lib/top10_t2/wajepa_run.py` fp32 on requests built from the navhard index). Report: `scripts/pp_navhard.py report` ->
[navhard_arms.md](navhard_arms.md), [navhard_paired.md](navhard_paired.md).

Setup: navhard_two_stage, all 5 912 tokens (both stages), devkit navsim main @ 0a380a9 v2 EPDMS (the navtest readout's devkit), the
devkit's two-stage aggregation; the official number is the mean over the 225 scene-mapping groups, so paired CIs resample groups clustered
by the log of the group's stage-1 token (76 logs, B 10 000). Our arms are the full-run checkpoints under protocol W (CPU-warped lattice
frames on the 2 Hz keys, as on navtest); P1-P3 are seed means.

## Numbers

| arm | combined | stage 1 | stage 2 | s0 / s1 |
|:--|--:|--:|--:|:--|
| P0 shipped, W frames | 28.05 | 66.26 | 42.58 | |
| P1 fine-tune, inputs zeroed | 27.60 | 67.28 | 41.15 | 27.22 / 27.98 |
| P2 + ego / pose history / command | **29.31** | **76.13** | 38.33 | 29.21 / 29.40 |
| P3 + side / rear cameras | 28.56 | 74.96 | 38.05 | 28.11 / 29.01 |
| WA-JEPA (released checkpoint, our harness) | **35.41** | 81.90 | 43.53 | |
| reference: shipped Cinque, GIMM frames (op_guard `shipped`; decision 37: 33.3, decision 143 S0 33.33) | 33.33 | 71.70 | 46.90 | |
| reference: factor_wm S3 (decision 143) | 35.77 | 72.05 | 49.72 | |

| contrast | combined [95% CI] | stage 1 | stage 2 |
|:--|:--|:--|:--|
| P2 - P1 | +1.71 [-2.32, +5.67] | **+8.85 [+5.11, +12.65]** | -2.82 [-7.87, +1.94] |
| P3 - P1 | +0.96 [-2.73, +4.48] | +7.68 [+3.95, +11.46] | -3.10 [-7.87, +1.31] |
| P3 - P2 | -0.75 [-2.88, +1.29] | -1.17 [-2.53, +0.12] | -0.28 [-2.64, +2.03] |
| P1 - P0 | -0.45 [-1.81, +0.85] | +1.02 [-0.48, +2.67] | -1.44 [-2.94, -0.05] |
| P2 - P0 | +1.26 [-2.86, +5.27] | +9.87 [+5.91, +14.11] | -4.25 [-9.17, +0.33] |
| **P2 - WA-JEPA** | **-6.10 [-10.04, -2.30]** | -5.77 [-9.96, -1.77] | -5.20 [-9.18, -1.18] |
| P0 - WA-JEPA | -7.36 [-10.94, -3.67] | -15.64 [-20.35, -11.56] | -0.94 [-4.82, +3.31] |
| P0 (W) - shipped (GIMM) | -5.28 [-9.52, -1.03] | -5.44 [-8.98, -2.18] | -4.31 [-9.54, +1.15] |
| P2 - shipped (GIMM) | -4.02 [-7.93, +0.05] | +4.43 [+0.80, +7.90] | -8.57 [-12.69, -4.31] |

WA-JEPA same-harness check (the requests are built from our index, not by its SceneLoader): on 14 whole navtest logs (1 021 tokens) our
request path scores 90.19 against its stored navtest export's 90.64 on the same tokens, -0.45 [-1.15, +0.01] (pass line |mean| < 0.5;
inputs verified identical to its loader on 12 tokens; the trajectories differ by the flow sampler's float noise across hosts, ADE 0.07 m on
64 tokens). Its paper reports navtest only, so the 35.41 has no published counterpart.

## What this says

- **Stage 1 (real scenes) behaves like navtest.** The ego / pose / command inputs add 8.9 points over P1 (CI excludes 0), and P2 closes
  10 of P0's 15.6-point stage-1 gap to WA-JEPA (66.3 -> 76.1 vs 81.9).
- **Stage 2 (3DGS-rendered follow-up scenes) goes the other way.** P2 loses 2.8 to P1 and 4.3 to P0 (CIs reach 0), so the combined
  score gains only +1.7 (n.s.). P0 already matches WA-JEPA in stage 2 (-0.9); after fine-tuning our arms trail it by 5.2. [I] The ego
  inputs in stage 2 are the synthetic trajectory's state while the 4 rendered frames carry rendering artefacts; the fine-tuned arms lean
  on both, and the W warp of rendered keys is a further step off the training distribution. Not tested here.
- **Protocol W costs 5.3 points on navhard for the shipped model** (P0 W 28.05 vs GIMM 33.33), against 0.6 on navtest (80.51 vs 81.11):
  the warp of 2 Hz keys is much worse than GIMM on navhard, mostly in stage 1 (-5.4). So on navhard the W-protocol arms are handicapped
  relative to every GIMM reference in the table (shipped 33.33, factor_wm 35.77); P2 under GIMM frames was not run.
- Side / rear cameras add nothing here either (P3 - P2 -0.75).

## Caveats

- One run per checkpoint; seed spread up to 0.9 combined (P3). WA-JEPA is a single run through its sampler (stochastic across hosts,
  check above).
- The reference rows come from other harness runs (op_guard), same devkit, harness and token set, GIMM frames.
- The navhard groups are few (225, 76 logs), so CIs are wide (about +-4 on the combined score).
