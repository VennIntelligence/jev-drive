# EYES-vs-SENSE: the 2 x 2 of frozen side views and the body lesson in one joint training (2026-10-11, decision 255)

Pre-registration and amendment 1 [plans/2026-10-11-eyes-sense-prereg.md](../plans/2026-10-11-eyes-sense-prereg.md) (pushed before any read). Generated
tables [tables/contrast_tables.md](tables/contrast_tables.md) (every number below comes from it; long form `tables/cells.csv`, flips `tables/flips.csv`,
verdicts `tables/stats.json`), identity [identity_*.json](identity.json), pilots `pilot_b.json` / `pilot_d.json`. Code: trainer
`experiments/alpasim/scripts/joint_train.py`, chain `scripts/es_chain.sh`, bank `scripts/es_bank.py`, reader `scripts/es_read.py` (imports
experiments/vis_train/scripts/vt_read.py's replay, token classes and bootstrap).

## Result in short

- **Side views do not reopen.** (d) - (c), side views on top of the body lesson: > 45 deg cut-inside **+1.02 [+0.19, +1.91]** pp and > 45 deg
  off-road **+1.32 [+0.32, +2.34]** pp (both more failures), EPDMS in the < 5 deg bucket -0.20 [-0.34, -0.07] and EC -0.51 [-0.82, -0.23] down; the
  channel is read (arm - `:mshuf` +0.36): **stays closed**. (b) - (a), side views on the plain base: cut-inside +0.92 [+0.19, +1.70], off-road
  +1.19 [+0.19, +2.22] (also more failures), no guard cell down, but the channel-read check gives +0.18, under the 0.2 line: **not measured** by
  the registered rule. In both comparisons the side views move the turn failures the wrong way.
- **The body lesson moves the boards with or without the extra eyes.** (c) - (a): navtest EPDMS +0.38 [+0.20, +0.57], > 45 deg cut-inside -0.73
  [-1.39, -0.19], > 45 deg off-road -0.86 [-1.66, -0.07], navhard combined +2.22 [+0.25, +4.36] (stage 2 +3.33 [+1.27, +5.48]), 17 navtest cells up and 1
  down. (d) - (b): navhard combined +1.98 [+0.60, +3.43] (stage 2 +2.52 [+0.77, +4.34]), cut-inside -0.63 [-1.42, +0.14], off-road -0.73 [-1.64, +0.16],
  EPDMS +0.14 [-0.08, +0.38], 3 cells up and 1 down.
- **nuPlan closed loop (700 scenes, (a) and (c))**: mean +0.0003 [-0.0060, +0.0078]; at-fault collision + offroad zeros -2.0 [-6.5, +2.5], corridor
  zeros +2.0 [-3.0, +7.5]: the decision 233 composition shift in direction, inside the CIs at two seeds (on all 1491 scenes, decision 254:
  collision + offroad -8.5 [-18.0, +0.5], corridor +7.0 [-0.5, +15.0]).

## Arms (one trainer, lambda 10, 10 000 steps x 128, from the shipped weights, seeds 0 / 1)

| Arm | Checkpoints | What it adds |
|:--|:--|:--|
| (a) | `P2H10-F-s{0,1}` | the base (the joint trainer's `ESa-s0 / s1` equal them bit for bit) |
| (b) | `ESb-s{0,1}` | EYES: `--mem vtr_W0`, the frozen shipped encoder's tokens of the t0 views CAM_F0 / CAM_L0 / CAM_R0 through the memory channel (masked on 25 % of rows) |
| (c) | `P2H10S-F-s{0,1}` | SENSE: the body lesson S (the joint trainer's `ESc-s0 / s1` equal them bit for bit) |
| (d) | `ESd-s{0,1}` | both; the 13 hinge-only rows per batch read the memory masked |

Identity (`identity_{a,c}_s{0,1}.json`): every one of the 109 net and 46 adapter tensors of ESa / ESc equals the stored checkpoint, max |difference|
0.0, so (a) / (c) are the stored checkpoints and all their stored reads. The `vtr_W0` bank was rebuilt after the disk cleanup of 13:00 CST removed
it (amendment 1): CAM_F0 from the W token cache's t0 slot (the deleted bank's view differed from it by 3e-4 mean |d|), CAM_L0 / R0 re-rendered with
vis_train's unchanged `vt_side.py` and encoded with the frozen encoder; token RMS per view 1.70 / 1.63 / 1.63 on navtest. Final dev ADE: (b) 0.540 /
0.535 (memory masked 0.551 / 0.548, another log's memory 0.596 / 0.582), (d) 0.536 / 0.542 (0.565 / 0.561, 0.614 / 0.615).

## The 2 x 2 (two-seed means; navtest 12 146 tokens, rates in % of the bucket; navhard 225 groups, G frames)

| Arm | navtest EPDMS | < 5 deg | > 45 deg | NC | EC | DAC | > 45 cut-inside | > 45 off-road | > 45 under-turn | > 20 off-road | navhard combined | stage 1 | stage 2 | nuPlan 700 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| (a) base | 88.67 | 92.94 | 78.61 | 98.58 | 88.62 | 96.17 | 4.78 | 10.18 | 2.60 | 8.80 | 31.84 | 74.73 | 43.04 | 0.9481; zeros 7 / 10 / 7.5 |
| (b) + EYES | 88.54 | 92.82 | 77.49 | 98.60 | 88.38 | 95.99 | 5.70 | 11.37 | 2.93 | 9.32 | 31.63 | 73.14 | 43.99 | not run |
| (c) + SENSE | 89.05 | 93.14 | 79.42 | 98.64 | 88.77 | 96.52 | 4.05 | 9.33 | 2.77 | 7.88 | 34.06 | 74.16 | 46.38 | 0.9484; zeros 4.5 / 10.5 / 9.5 |
| (d) both | 88.68 | 92.94 | 78.18 | 98.60 | 88.26 | 96.16 | 5.08 | 10.65 | 3.00 | 9.02 | 33.61 | 73.32 | 46.51 | not run |

nuPlan zeros: at-fault collision / offroad / corridor, two-seed means (the closed-loop reads are phase 1's, c0b chunk0-2, the box's default path).

## Differences and the registered verdicts (seed mean, by-log cluster bootstrap, B 10 000)

| Comparison | EPDMS | > 45 cut-inside (pp) | > 45 off-road (pp) | EPDMS < 5 | NC | EC | navhard combined | navhard stage 2 | channel (arm - `:mshuf`) | Verdict |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|--:|:--|
| (b) - (a) | -0.13 [-0.36, +0.09] | +0.92 [+0.19, +1.70] | +1.19 [+0.19, +2.22] | -0.12 [-0.25, +0.01] | +0.02 [-0.06, +0.10] | -0.23 [-0.48, +0.01] | -0.21 [-1.77, +1.33] | +0.94 [-0.59, +2.59] | +0.18 | **not measured** |
| (d) - (c) | -0.37 [-0.64, -0.10] | +1.02 [+0.19, +1.91] | +1.32 [+0.32, +2.34] | **-0.20 [-0.34, -0.07]** | -0.04 [-0.14, +0.06] | **-0.51 [-0.82, -0.23]** | -0.45 [-2.05, +0.98] | +0.13 [-1.40, +1.65] | +0.36 | **stays closed** |
| (c) - (a) | +0.38 [+0.20, +0.57] | -0.73 [-1.39, -0.19] | -0.86 [-1.66, -0.07] | +0.20 [+0.01, +0.39] | +0.06 [-0.02, +0.14] | +0.16 [+0.01, +0.32] | +2.22 [+0.25, +4.36] | +3.33 [+1.27, +5.48] | | headline |
| (d) - (b) | +0.14 [-0.08, +0.38] | -0.63 [-1.42, +0.14] | -0.73 [-1.64, +0.16] | +0.11 [-0.05, +0.29] | +0.00 [-0.08, +0.08] | -0.12 [-0.38, +0.12] | +1.98 [+0.60, +3.43] | +2.52 [+0.77, +4.34] | | headline |
| (d) - (a) | +0.01 [-0.23, +0.26] | +0.30 [-0.47, +1.06] | +0.46 [-0.50, +1.41] | -0.00 [-0.16, +0.16] | +0.02 [-0.07, +0.11] | -0.35 [-0.63, -0.09] | +1.77 [-0.00, +3.55] | +3.46 [+1.31, +5.77] | | descriptive |

Registered rule (side views reopen only if both > 45 deg rates have CIs below 0, none of EPDMS < 5 deg / NC / EC is down, and the channel is read):
both rates have CIs above 0 in both comparisons. Per seed, the rate differences point the same way (cut-inside (b) - (a) +1.32 / +0.53, (d) - (c)
+1.52 / +0.53).

**Flips** (navtest tokens, both seeds summed; saved = the comparison arm fails and the arm passes / hurt = the reverse):

| Comparison | > 45 cut-inside | > 45 off-road | > 20 off-road | > 45 under-turn | NC failures |
|:--|:--|:--|:--|:--|:--|
| (b) - (a) | saved 27 / hurt 55 (145 -> 173) | 50 / 86 (309 -> 345) | 111 / 144 | 24 / 34 | 44 / 39 |
| (d) - (c) | 32 / 63 (123 -> 154) | 57 / 97 (283 -> 323) | 113 / 185 | 28 / 35 | 42 / 51 |
| (c) - (a) | **29 / 7** (145 -> 123) | 54 / 28 (309 -> 283) | 110 / 52 | 15 / 20 | 37 / 21 |
| (d) - (b) | 61 / 42 (173 -> 154) | 95 / 73 (345 -> 323) | 166 / 147 | 33 / 35 | 51 / 49 |

**No-regression table** (decision 248 amendment 2; 60 navtest score cells, sub-score x bucket): (b) - (a) up 6 / down 8 (down: EPDMS, DAC and EC
> 45 deg, EC > 20 deg, HC all / > 45 / > 20, TTC < 5 deg); (d) - (c) up 3 / down 12 (EPDMS, DAC, EC all; EPDMS, NC, DAC < 5 deg; EPDMS, DAC, EC
> 45 and > 20 deg); (c) - (a) up 17 / down 1 (DDC 20-45 deg -0.08); (d) - (b) up 3 / down 1 (EP 20-45 deg -0.11). Full lists in the generated tables.

## Channel-read checks (arm minus option, navtest EPDMS; positive = the option costs points)

| Arm | `:noside` (bank masked) | `:mshuf` (another log's bank) | `:sideoff` (side views masked, CAM_F0 kept; out of the training distribution) |
|:--|:--|:--|:--|
| (b) | +0.01 [-0.24, +0.25] | +0.18 [-0.06, +0.41] (seeds +0.15 / +0.21) | +0.24 [+0.04, +0.46] (seeds -0.06 / +0.54) |
| (d) | **-0.50 [-0.81, -0.21]** | +0.36 [+0.09, +0.62] (seeds +0.21 / +0.51) | +0.03 [-0.17, +0.23] |

(b) reads its bank lightly (the same bank in decision 248's RW0, SH30 recipe, lost 0.71 under `:mshuf`): the check misses the registered 0.2 by 0.02.
(d) is better with its bank masked (89.18, the level of (c)'s 89.05): in this arm the hinge-only rows train with the memory masked, so the body
lesson is learned in the memory-off condition and the bank, when present, pulls the plan the other way; this is a consequence of the registered
design choice (no rendered side views exist for perturbed states), not tested further.

## What was cut and why

- (b) / (d) closed loop: AlpaSim delivers CAM_L0 / R0 JPEGs, but the driver has no path that renders them as vis_train's side-view pairs and encodes
  them into the bank; a new serving path with its own equivalence check was not a cheap addition (registered as cut).
- WOD-E2E: not part of this phase.

## Limits

- Two seeds; navtest has been read for (a) and (c) before (the (c) - (a) row repeats decision 244's S - base exactly, as it must).
- The bank is the rebuilt `vtr_W0` (equal to decision 248's up to fp16 noise); `:sideoff` is out of distribution.
- (b) / (d) differ from (a) / (c) by the channel and the side views together; decision 248's RW0 - R0 (the same bank against Cinque's own t0 token in the
  same channel) was a null on every cell.
- About 5 % of the navtest cells cross 0 under no effect; the registered line uses four named quantities.

## Cost

Phase 2's GPU work: four identity runs and four bank-arm runs (10 000 steps each, 25-40 min next to other jobs), two pilots, the bank rebuild (14 CPU
render jobs, one 10-minute encode), bench navtest for 4 arms plus 12 option reads and navhard for 4 (mostly CPU scoring), the read-out (CPU).
Both phases together: 11.5 card-hours by booked VRAM share ([../../alpasim/results/joint1.md](../../alpasim/results/joint1.md#cost-pool-event-log-scriptsjoint_usagepy-both-phases-together)).
