# NTV1 final: S-PAI seed 1 and the S-PAI 2-seed mean (round 9, 2026-10-24, decision 479)

Pre-registration (Chinese): [../../plans/2026-10-24-ntv1-final.md](../../plans/2026-10-24-ntv1-final.md), with amendment 1.
- Lane: [../../scripts/spai.sh](../../scripts/spai.sh) `final` with `SPAI_SEED=1` (job bodies `fident`, `train`, `post`, `freport`).
- Checks and verdict: [../../scripts/spai.py](../../scripts/spai.py) `fdiff`, `fverdict`.
- Box: `$DATA_DIR/runs/round9/ntv1_final/` (attempt 1 in `attempt1/`).

Full tables: [verdict.md](verdict.md) (PAI 281 and val160, every pair, zero classes, flips, highway, gates, cost). Gate records: [fdiff.json](fdiff.json),
[fident.json](fident.json), [ratio_SP.json](ratio_SP.json), [post_SP.json](post_SP.json), [lr_full_SP.json](lr_full_SP.json).

**Result.** The S-PAI 2-seed mean is not distinguishable from R8K-L18-s2, INTP75-NR or D3's 2-seed mean on either exam. Every 95 % CI includes 0.
- Seed 1 reads 0.851 on PAI 281, 0.033 under seed 0 (0.883). The paired CI of that gap excludes 0, so S-PAI's single-seed 0.883 (d477) was the upper end of the seed spread.
- Seed spread here is 0.03-0.04, so no claim is made beyond the CIs.

## Why this arm

The rule was agreed with the user before S-NU's result:
- If S-NU helped, it would join S-PAI and both would be trained at 2 seeds.
- Otherwise S-PAI gets its seed 1.

S-NU hurt: −0.037 [−0.071, −0.004] vs D3-s0 (d478). The final NTV1 is therefore S-PAI, and seed 1 is the only new run.

## Arm

NT5SP-F-s1 is NT5SP-F-s0's stored argv with two tokens changed, `--tag` and `--seed 0 -> 1` (fdiff).
- It starts from the shipped weights and runs 19 500 steps.
- Pool, draw file and raster are S-PAI-s0's: `on1_JTDC2S`, `NTV1D2S.npz` and `road_spai.npz`.
- None of the three was rebuilt. Their mtimes predate S-PAI-s0's training start, and their sha256 are in fdiff.json.

## Gates (all pass)

| gate | result |
|:--|:--|
| fdiff | s0's built argv = stored; s1 differs only in `--tag`, `--seed`; files untouched since s0's start |
| fident (HEAD + files inert for the s0 config) | s0's argv on HEAD to step 125 vs stored NT5SP-F-s0: 85 / 85 scalars bit-identical (monitor slices 46 / 46) |
| S / imit gradient ratio, 500 steps after T | 0.598 (band 0.21-1.04); whole post-T 0.418 (s0 0.389) |
| post | DONE, dev ADE 0.5319 (<= 0.566), lr check over the run |

Amendment 1 records a fault in my lane script. Attempt 1 did not forward `SPAI_SEED` into the pool job env, so the job bodies ran as the s0 lane.
- The training job built NT5SP-F-s0's argv. The trainer's never-overwrite guard stopped it, and S-PAI-s0's checkpoint and run dir were not touched.
- fident had already passed, but its record landed in the s0 lane's dir. I moved that record and the 2 stray s0 stage lines to `attempt1/`.
- The fix was to forward the env in `DENV`. The whole chain was then resubmitted. No gate or line changed.

## PAI 281 (ACC + GAP, g10, ALPASIM_FAST=1, Harmonizer off; 277 common scenes)

| read | new | old | zeros new / old | lateral | offroad | route-turn | AF-front | overshoot |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **S-PAI 2-seed mean** | **0.867** | **0.791** | 26.5 / 47.5 | 13.5 | 8 | 6 | 5.5 | 22.5 |
| S-PAI-s1 | 0.851 | 0.775 | 31 / 52 | 17 | 10 | 6 | 6 | 23 |
| S-PAI-s0 | 0.883 | 0.808 | 22 / 43 | 10 | 6 | 6 | 5 | 22 |
| D3 2-seed mean | 0.860 | 0.784 | 27.5 / 48.5 | 18.5 | 15 | 3.5 | 4.5 | 22 |
| R8K-L18-s2 | 0.872 | 0.803 | 28 / 47 | 20 | 14 | 2 | 5 | 20 |
| INTP75-NR (ref., not clean) | 0.879 | 0.810 | 26 / 45 | 17 | 10 | 3 | 5 | 20 |

Paired differences of the S-PAI 2-seed mean (95 % bootstrap CI, scenes as units):

| S-PAI 2-seed mean − | new | old |
|:--|--:|--:|
| R8K-s2 | −0.005 [−0.038, +0.029] | −0.012 [−0.047, +0.023] |
| INTP75-NR | −0.012 [−0.042, +0.019] | −0.019 [−0.050, +0.012] |
| D3 2-seed mean | +0.007 [−0.018, +0.033] | +0.007 [−0.018, +0.034] |

Seed 1 alone vs the same references (new scorer):

| S-PAI-s1 − | new |
|:--|--:|
| S-PAI-s0 | −0.033 [−0.058, −0.008] |
| R8K-s2 | −0.021 [−0.059, +0.018] |
| INTP75-NR | −0.028 [−0.063, +0.007] |
| D3-s0 | −0.029 [−0.063, +0.006] |
| D3-s1 | +0.010 [−0.023, +0.045] |

Zero classes:
- **Offroad.** The lower offroad count holds in both seeds: 6 and 10, against D3's 13 and 17 and R8K-s2's 14. Lateral zeros (offroad + zero-other + side) are 10 and 17, against D3's 15 and 22.
- **Route-turn.** Route-turn zeros are higher in both seeds: 6 and 6, against D3's 3 and 4 and R8K-s2's 2.
- **Seed 1's losses** are mostly offroad (+4 [+1, +8] vs s0) and the zeros it adds overall (+9 old).
- **Negative flips** on NTV0's 201 full-score scenes: 12 (s0) and 17 (s1), against 14 for R8K-s2 and 8 for INTP75.

## val160 (151 common scenes)

| read | new | old | zeros new / old | lateral | offroad |
|:--|--:|--:|--:|--:|--:|
| **S-PAI 2-seed mean** | **0.862** | **0.776** | 16 / 29 | 3.5 | 2 |
| S-PAI-s1 | 0.857 | 0.771 | 17 / 30 | 5 | 3 |
| S-PAI-s0 | 0.867 | 0.781 | 15 / 28 | 2 | 1 |
| D3 2-seed mean | 0.853 | 0.767 | 17 / 30 | 6 | 5 |
| R8K-L18-s2 | 0.853 | 0.767 | 18 / 31 | 9 | 7 |
| INTP75-NR | 0.863 | 0.770 | 17 / 31 | 5 | 3 |

| S-PAI 2-seed mean − | new | old |
|:--|--:|--:|
| R8K-s2 | +0.009 [−0.029, +0.047] | +0.009 [−0.029, +0.047] |
| INTP75-NR | −0.001 [−0.029, +0.030] | +0.006 [−0.025, +0.039] |
| D3 2-seed mean | +0.009 [−0.025, +0.043] | +0.009 [−0.025, +0.043] |

## Highway (behavior_stats Q1, PAI 281)

S-PAI-s1 is not slow on highway:
- median v_ego / v_human 1.01, p10 0.90, 8 % of steps below 0.85;
- slow + rear-hit points 1.20, 1 of 33 scenes below 0.90.

These match S-PAI-s0 (1.01, 8 %, 0.94) and D3-s0 (1.01, 8 %, 0.98).

## Cost

6.0 GPU job-hours, 1.6 h wall:
- gates 0.03;
- train 1.11;
- PAI 281 reads 3.07;
- val160 reads 1.74.

Attempt 1 adds about 0.05 (fident, plus the training job that stopped at once). Disk: 7.1 GB lane plus an 87 MB checkpoint.
