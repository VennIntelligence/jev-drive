# op_parity off-track rows: the SH30 recipe with ~10 % statically perturbed (re-projected) navtrain rows

Written 2026-10-09. Pre-registration: [plans/2026-10-09-offtrack-rows-prereg.md](../plans/2026-10-09-offtrack-rows-prereg.md) (committed before any OT score; amendment 1 changes only the pre-training sign gate).
Code: `scripts/ot_rows.py` (prep / probe / train / gate / report), `scripts/ot_rows_chain.sh`. Tables / CSVs: [ot_rows/](ot_rows/). Decision 198.
Arms: `OTP-F-s0` (pilot recipe, 3 000 x 64, shards 2-4, 6 of 64 batch rows off-track), `OTC-F-s0` (same loop, 0 %), `OT30-F-s0 / s1` (10 000 x 128, 12 shards, 13 of 128 batch rows off-track).
References are the stored `SHP-F-s0 / s1` and `SH30-F-s0 / s1`; nothing re-trained.

## Answer

**Not a recipe candidate by the registered line: navhard +1.51 [+0.14, +3.10] against a line of +2.0 (the CI lower bound is above 0); navtest +0.22 [+0.03, +0.42] passes its line of -0.2.**
The gain is real and sits where it was expected (stage 2), it costs nothing on navtest, and the HUGSIM guardrail holds; it is three quarters of the registered size.

| read-out (OT30 vs SH30, 2 seeds each) | OT30 | SH30 | OT30 - SH30 [95% CI] | line |
|:--|--:|--:|:--|:--|
| navhard combined (225 groups, 76 logs, G frames) | **35.18** (34.83 / 35.53) | 33.67 (33.47 / 33.86) | **+1.51 [+0.14, +3.10]** | >= +2.0 and lower bound > 0: **not passed** |
| navhard stage 1 | 75.96 | 75.90 | +0.05 [-1.43, +1.53] | |
| navhard stage 2 | 45.86 | 44.68 | +1.18 [-0.09, +2.53] | |
| navtest EPDMS (12 146 tokens, 136 logs) | 89.77 (89.75 / 89.78) | 89.55 (89.47 / 89.63) | +0.22 [+0.03, +0.42] | >= -0.2: passed |
| navtest NC / DAC / EP / TTC / LK | 98.68 / 97.29 / 87.13 / 98.11 / 97.58 | 98.60 / 97.13 / 87.16 / 98.00 / 97.37 | +0.08 / +0.16 / -0.04 / +0.11 / +0.21 | |
| HUGSIM 64 HD (`spec_plan_smooth`, one run per scenario and seed) | 0.422 (0.417 / 0.428) | 0.439 (0.443 / 0.434) | -0.016 [-0.070, +0.031] | >= -0.03: holds |
| HUGSIM launch stalls / stuck / spin (seed means) | 1.0 / 0 / 0 | 1.0 / 0 / 0 | | launch stalls not up: holds |

Read it as: both seeds of OT30 are above both seeds of SH30 on navhard (34.83, 35.53 vs 33.47, 33.86); navhard is now level with WA-JEPA (35.18 vs 35.41, -0.23 [-4.40, +3.91]), with stage 2 above it by +2.33 [-1.25, +6.20] and stage 1 still 6 behind.
On navtest the turn strata all move up without a significant one (> 45 deg +0.36 [-0.07, +0.80]); EP is -0.04 [-0.06, -0.01]. Full tables: [verdict.md](ot_rows/full/verdict.md), [navhard_paired.md](ot_rows/full/navhard_paired.md), [navtest_paired.md](ot_rows/full/navtest_paired.md), [navtest_strata.md](ot_rows/full/navtest_strata.md), [hugsim arms](ot_rows/full/hugsim_spec_plan_smooth_arms.md), [hugsim strata](ot_rows/full/hugsim_spec_plan_smooth_strata.md).

HUGSIM detail that the guardrail does not cover: background collisions 11.5 vs 8.5 and completed scenarios 24 vs 26 (seed means); the HD point estimate is carried by pandaset / `hard` scenarios (-0.063 [-0.244, +0.114] / -0.104 [-0.280, +0.017]), turning routes are flat (-0.004).

## Pilot (seed 0; the gate that released the full run)

| read-out | OTP-F-s0 | reference | difference [95% CI] |
|:--|--:|--:|:--|
| navhard combined vs the registered reference mean | 33.52 | 32.13 | **+1.39 [+0.06, +2.88]** (gate: point estimate >= +1.0, passed) |
| navhard combined vs SHP-F-s0 only | 33.52 | 32.36 | +1.17 [-0.27, +2.76] |
| navhard stage 1 / stage 2 | 74.83 / 43.99 | 74.04 / 43.44 | +0.79 [-0.80, +2.55] / +0.55 [-0.90, +1.98] |
| navtest EPDMS | 88.60 | 88.47 | +0.13 [-0.03, +0.28] (DAC +0.25, EP -0.15 [-0.19, -0.11]) |
| HUGSIM 64 HD vs SHP-F-s0 | 0.396 | 0.397 | -0.001 [-0.038, +0.035]; launch stalls 3 vs 1 |

`OTC-F-s0` (this lane's loop with 0 % off-track rows) reproduces `SHP-F-s0` exactly (navhard 32.356 both, identical probe rows): the separate training loop is equivalent to pp_train. It also means the registered three-checkpoint reference mean counts seed 0 twice; with the two distinct checkpoints (32.36, 31.68) the pilot difference is about +1.5, so the gate outcome does not depend on it.
The pilot's launch-stall count (3 vs 1) did not repeat at full scale (1.0 vs 1.0). [pilot/](ot_rows/pilot/).

## AlpaSim-standard offline read (navtest, `ap2_offline.py`, cold-start rule backwarp; EPDMS on a 2 035-token subset, no EC, non-reactive, seed 0)

| inputs | SH30-F-s0 ADE (m) | OT30-F-s0 ADE - SH30 [95% CI] | OT30-F-s1 ADE - SH30 | SH30 EPDMS | OT30-F-s0 EPDMS - SH30 [95% CI] |
|:--|--:|:--|:--|--:|:--|
| m = 4 | 0.585 | +0.005 [+0.002, +0.008] | +0.003 [+0.000, +0.006] | 90.14 | +0.44 [-0.04, +0.90] |
| m = 3 | 0.601 | +0.004 [+0.001, +0.007] | +0.003 [+0.000, +0.006] | 89.59 | +0.19 [-0.29, +0.65] |
| m = 2 | 0.677 | -0.000 [-0.004, +0.004] | +0.005 [+0.001, +0.008] | 88.38 | -0.06 [-0.56, +0.49] |
| m = 1 | 0.992 | +0.002 [-0.003, +0.008] | -0.003 [-0.008, +0.002] | 85.14 | **-1.49 [-2.69, -0.46]** |

ADE to the log is unchanged to within 5 mm at every m. The one thing to look at before an AP2 variant is trained: at m = 1 (the first decision of an AlpaSim rollout, fabricated history) the subset EPDMS of seed 0 drops by 1.49; the pilot shows the same sign (-0.47 [-1.11, +0.12]). Seed 1 was not scored. [ap2_offline_ot-full.md](ot_rows/ap2_offline_ot-full.md), [ap2_offline_ot-pilot.md](ot_rows/ap2_offline_ot-pilot.md).

## What the rows teach (held-out off-track rows, 256 dev tokens of shards 2-4; no line)

| model | ADE off-track (m) | ADE same tokens, logged pose (m) | response to the lateral offset, 1 / 2 / 4 s | response to the yaw offset, 1 / 2 / 4 s |
|:--|--:|--:|:--|:--|
| SH30-F-s0 / s1 | 0.789 / 0.790 | 0.626 / 0.619 | 0.04 / 0.09 / 0.21 | -0.07 / 0.00 / 0.28 |
| OT30-F-s0 / s1 | 0.722 / 0.723 | 0.622 / 0.621 | 0.18 / 0.37 / 0.64 | 0.33 / 0.40 / 0.60 |

Response = least-squares coefficient of the plan's lateral shift between the perturbed and the logged view on the shift the re-expressed target asks for (1 = full return to the logged path, 0 = the offset is ignored).
SH30 corrects about a fifth of a 0.5 m offset within 4 s; with 10 % off-track rows it corrects about two thirds, and the logged-pose ADE does not move. [probe_full.md](ot_rows/full/probe_full.md).

## Setup, verified, not verified, deviations

- Rows: navtrain tokens with speed > 3 m/s and a logged future (68 423 of the 101 499 training rows, 67 %; one perturbation each): dy ~ U(-0.5, 0.5) m, dpsi ~ U(-2, 2) deg, history offsets by `op_adapt_h.drift`, every frame one plane re-projection (`op_interp.warp_frame`) of the nearest real key; target = the logged future in the perturbed frame; hinge 30 / 0.5 m on the plan mapped back to the logged frame; never anchor rows. No rollout, no longitudinal offset.
- Verified: zero offsets reproduce the stored W tokens bit for bit (64 rows); sign of both offsets (warp map checked numerically; positive, growing responses of P0 and SHP-F-s0); OTC-F-s0 = SHP-F-s0; hinge labels cover 100 % of the off-track rows; all scores through `jevdrive.bench`; trainings pass `pp_full_check.py train`.
- Not verified / limits: two seeds, one perturbation setting, one share (10 %); navhard CI over 76 logs; HUGSIM one run per scenario; the offline EPDMS is seed 0 on a 2 035-token subset; the plane engine distorts objects above the road (decision 141), not separated from the effect of the off-track state itself; the pilot reference mean double-counts seed 0 (above).
- Deviations: amendment 1 (pre-training sign gate: yaw and lateral response at 4 s >= 0.1 for both models, was yaw >= 0.3; made before any OT score, verdict lines untouched). Per-row metadata comes from the shard tabs and the NAVSIM logs because the navsim_zs navtrain index no longer exists on the box (row selection unchanged). Process: the prep jobs declared 12 GB of VRAM and use 29-42 GB; one was killed by an out-of-memory error on an over-packed card and the chain stood still for about 90 minutes (declarations fixed, 44 / 48 GB).
- Cost: about 4.5 card-hours (12 prep jobs of 5-7 min, pilot trainings 2 x ~6 min, full trainings 2 x ~30 min at 5.7 it/s and 25.8 GB, HUGSIM 64 x 5 checkpoints, plans and offline read).
- Checkpoints (box): `$DATA_DIR/runs/op_parity/runs/{OT30-F-s0, OT30-F-s1, OTP-F-s0, OTC-F-s0}/ckpt-final.pt`; off-track token caches `$DATA_DIR/runs/op_parity/cache/ot1_navtrain_full.s*of12{,@warp}/`.
