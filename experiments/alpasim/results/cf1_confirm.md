# CF1: confirmation read of P2H10 / YR10m10 / APY10m10 on scenes no recipe choice touched (2026-10-09, decision 216)

Pre-registration: [plans/2026-10-09-cf1-confirm-prereg.md](../plans/2026-10-09-cf1-confirm-prereg.md) (pushed, commit 4a55ed02, before any score was read). Local closed loop only; nothing was
submitted to AlpaSim; no WA-JEPA weights or features. Full generated tables: [cf1/cf1_report.md](cf1/cf1_report.md), per-scene [cf1/cf1_per_scene.csv](cf1/cf1_per_scene.csv),
numbers [cf1/cf1_stats.json](cf1/cf1_stats.json). Code: `scripts/cf1_chain.sh` (-> `ot2_loop.py`, `OT_LANE=cf1`), `scripts/cf1_report.py`.

## Result in short

- **Primary: refuted.** YR10m10 minus P2H10 on the 391 scenes of part012-015 (6 logs): **-0.0069**, log-clustered CI [-0.0110, -0.0016], scene CI [-0.0244, +0.0106]. Both seeds agree
  (-0.0070 / -0.0068). Zeros 32.5 against 29.5, at-fault events 23.5 against 23. YR10m10 is below P2H10 on 4 of the 6 logs.
- YR10m10 minus P2H10 pooled: 791 fresh scenes (22 logs) +0.0043 [-0.0031, +0.0148]; all 1491 (44 logs) +0.0052 [-0.0006, +0.0120]. The 400-scene read of decision 213
  (+0.0153) was the high draw of an effect that is between 0 and +0.005; it does not clear the +0.005 line on fresh scenes.
- **Secondary (exploratory): APY10m10 meets the registered candidate line, narrowly.** APY10m10 minus YR10m10 on the 791 fresh scenes +0.0146, log-clustered CI [+0.0006, +0.0266]
  (scene CI [+0.0003, +0.0287]); APY10m10 minus P2H10 +0.0189 [+0.0076, +0.0304]. The gain is all in part012-015 (+0.0296 [+0.0148, +0.0472] over YR10m10, +0.0228 [+0.0052, +0.0431]
  over P2H10, both seeds alike); on OT3's 400 scenes APY10m10 equals YR10m10 (-0.0001) and its seeds disagree (+0.0305 / -0.0002 over P2H10). Slow scenes are halved everywhere
  (78.5 against 155.5 on the 791; 136 against 262 on all 1491).
- All 1491 scenes: P2H10 0.9196, YR10m10 0.9248, APY10m10 0.9335.

## Per recipe (two-seed per-scene means; zeros = seed means; CI over logs)

| set | recipe | seeds | mean [CI logs] | zeros (collision / offroad / corridor) | slow | at-fault events |
|:--|:--|:--|:--|:--|--:|--:|
| Fresh-A, 391 / 6 logs | P2H10 | 0.8976 / 0.8942 | 0.8959 [0.8792, 0.9261] | 29.5 (2 / 21 / 4.5) | 85.5 | 23 |
| | YR10m10 | 0.8906 / 0.8874 | 0.8890 [0.8709, 0.9213] | 32.5 (1 / 22.5 / 7) | 82 | 23.5 |
| | APY10m10 | 0.9183 / 0.9191 | 0.9187 [0.8978, 0.9423] | 28 (0.5 / 18 / 7.5) | 39.5 | 18.5 |
| Fresh-B, 400 / 17 | P2H10 | 0.8892 / 0.8960 | 0.8926 | 33 (2.5 / 13 / 7.5) | 70 | 15.5 |
| | YR10m10 | 0.9080 / 0.9078 | 0.9079 | 27.5 (2 / 10.5 / 4.5) | 71 | 12.5 |
| | APY10m10 | 0.9197 / 0.8958 | 0.9078 | 33 (5 / 11 / 7) | 39 | 16 |
| Fresh, 791 / 22 | P2H10 | 0.8933 / 0.8951 | 0.8942 [0.8746, 0.9169] | 62.5 (4.5 / 34 / 12) | 155.5 | 38.5 |
| | YR10m10 | 0.8994 / 0.8977 | 0.8986 [0.8808, 0.9230] | 60 (3 / 33 / 11.5) | 153 | 36 |
| | APY10m10 | 0.9190 / 0.9073 | 0.9131 [0.8913, 0.9366] | 61 (5.5 / 29 / 14.5) | 78.5 | 34.5 |
| All, 1491 / 44 | P2H10 | 0.9192 / 0.9200 | 0.9196 [0.9024, 0.9387] | 87 (11 / 44.5 / 19.5) | 262 | 55.5 |
| | YR10m10 | 0.9248 / 0.9249 | 0.9248 [0.9079, 0.9441] | 78.5 (7 / 43 / 16) | 263 | 50 |
| | APY10m10 | 0.9351 / 0.9320 | 0.9335 [0.9172, 0.9505] | 86 (12 / 39.5 / 22.5) | 136 | 51.5 |

Scene-resampled CIs for every row and the paired differences with both CIs for Fresh-A, Fresh-B, Fresh, the 700 and All are in `cf1/cf1_report.md`. The 700-scene rows reproduce OT3 exactly.
Fresh-A per log (P2H10 / YR10m10 / APY10m10, scenes): veh-28 14.44 (10) 0.946 / 0.959 / 0.989; veh-28 15.23 (25) 0.918 / 0.921 / 0.946; veh-28 18.19 (49) 0.954 / 0.943 / 0.946;
veh-28 19.02 (87) 0.872 / 0.868 / 0.927; veh-52 07.26 (106) 0.899 / 0.894 / 0.924; veh-52 08.16 (114) 0.877 / 0.864 / 0.884.

## Route-fold scenes (in every denominator)

12 scenes score 0 with no decision for every driver (10 in the 700 + Fresh-B scenes, 2 in Fresh-A): `2021.08.30.13.45.25_veh-40_00878_01104-257d737fc3865fd1`,
`2021.08.30.14.54.34_veh-40_00439_00835-{1ca75f05f31d51cc,673a88a4037f5b6b}`, `2021.09.16.17.40.09_veh-45_02539_02745-a96abad3a09753c5`, `2021.09.16.19.12.04_veh-42_01438_01677-3d37ed78124057a1`,
`2021.09.16.19.27.01_veh-45_00472_00711-e5599a8884235d93`, `2021.09.16.19.49.00_veh-42_00990_01609-{4302a0a4b9f05b61,458e833803315b4f,a067e1b873c8534d,f9349f5d723b5421}`,
`2021.09.29.19.02.14_veh-28_03198_03360-f1ceb70bd72a5048` and `2021.10.06.08.16.17_veh-52_00922_01296-da751fd130625cce` (the last two are in Fresh-A). They do not move any difference.

## Driver code version

All CF1 runs used the box checkout `c9d2e07b` (launched 19:55 box time, no worker died, no scene missing, 8 pool jobs, one try each): LAT1's driver, i.e. slot warp on the card plus compiled
model passes (`SH30_COMPILE=1` default), nvJPEG off. The OT3 runs it is paired with (the 700 selection scenes; P2H10 / YR10m10 on Fresh-B, 18:12-18:55) were launched on the box checkout of their time, not recorded per run; the 700 predate the compiled
default. Within CF1 the Fresh-A comparison (all three recipes) and APY10m10 on Fresh-B are current code; P2H10 / YR10m10 on Fresh-B are OT3's. The bit-identity claim is documented, and only for part of the change:
the GPU slot warp gives the CPU frames pixel for pixel (0 differing pixels in 12.58 G, [lat1_frame_synthesis.md](lat1_frame_synthesis.md)); the compiled passes are tolerance-level,
not bit-identical (plans move 1 cm median), and that document's closed-loop check reran `P2H10-F-s0` on all 700 scenes: 0.9481 against 0.9484, same 25 zeros, no scene moves by more than 0.05
(largest 0.020, mean absolute change 0.0004). That covers the question for the `sh30` driver, so no chunk was rerun here; it was not run for the `ap2` driver (APY10m10) or the YR10m10 checkpoints.

## Limits

- Fresh-A has 6 logs: the log-clustered CI is coarse, and the registered verdict used it; the scene CI of the primary includes 0 (the point estimate is negative on both seeds).
- Two seeds, one simulation per checkpoint; local render, not the official environment.
- The APY10m10 candidate verdict has a lower bound of +0.0006 on a difference that is heterogeneous across the two fresh sets (+0.0296 against -0.0001); it also costs -0.21 navtest / -1.31 navhard
  open loop (decision 212) and its two seeds differ by 0.024 on Fresh-B. Shard labels are inferred.

## Amendment 1 (follow-up read: AP2H10-AB on the 791 fresh scenes; pre-registered in the CF1 prereg, pushed before any AP2H10 fresh-scene score was read)

Question: is APY10m10's gain the AlpaSim input standard alone, or do the yaw-rate rows add anything on top? Runs: AP2H10-AB-s0 / s1 (lambda 10, AlpaSim input standard, no rows) on `cf1fresh` (391) and
`ot3new` (400), same lists and chunking as the other recipes there; OT3 has them on the first 700. All four runs complete (391 / 400 rollouts each, driver errors 0, no worker death). **This was the last
read: no untouched scenes remain.** Every one of the 1491 public scenes is now scored for all four recipes; any further read of these recipes on these scenes is a repeat, not a confirmation.

Registered lines and outcomes (2-seed means, log-clustered 95 % CI):
- (a) APY10m10 - AP2H10 on the 791: **+0.0081 [+0.0001, +0.0156]** (scene CI [-0.0041, +0.0210]); by the line (>= +0.005 and lower bound > 0) the rows add on top of the input standard, with a lower bound of +0.0001. Seeds: +0.0188 / -0.0026.
- (b) AP2H10 - P2H10: 791 **+0.0108 [-0.0042, +0.0267]**, all 1491 **+0.0068 [-0.0048, +0.0185]**: the input standard alone does not clear the line (point estimate above +0.005, lower bound below 0).

The two sets disagree: on part012-015 AP2H10 - P2H10 is +0.0208 [-0.0002, +0.0499] and APY10m10 - AP2H10 +0.0020 [-0.0120, +0.0117] (the input standard carries it, rows add nothing); on OT3's 400 AP2H10 - P2H10 is +0.0011
and APY10m10 - AP2H10 +0.0141 [+0.0031, +0.0237] (the rows carry it, the standard adds nothing; seed 0 +0.0282, seed 1 +0.0000).

Per recipe on the 791 (two-seed mean per-scene scores; zeros = seed means):

| recipe | seeds | mean [CI logs] | CI scenes | zeros (collision / offroad / corridor) | slow | at-fault events |
|:--|:--|:--|:--|:--|--:|--:|
| P2H10 | 0.8933 / 0.8951 | 0.8942 [0.8746, 0.9169] | [0.8753, 0.9120] | 62.5 (4.5 / 34 / 12) | 155.5 | 38.5 |
| YR10m10 | 0.8994 / 0.8977 | 0.8986 [0.8808, 0.9230] | [0.8805, 0.9158] | 60 (3 / 33 / 11.5) | 153 | 36 |
| AP2H10 | 0.9001 / 0.9099 | 0.9050 [0.8799, 0.9308] | [0.8855, 0.9233] | 68 (6 / 28.5 / 21.5) | 71.5 | 34.5 |
| APY10m10 | 0.9190 / 0.9073 | 0.9131 [0.8913, 0.9366] | [0.8952, 0.9302] | 61 (5.5 / 29 / 14.5) | 78.5 | 34.5 |

All four recipes on all 1491 scenes (44 logs):

| recipe | seeds | mean [CI logs] | CI scenes | zeros (collision / offroad / corridor) | slow | at-fault events |
|:--|:--|:--|:--|:--|--:|--:|
| P2H10 | 0.9192 / 0.9200 | 0.9196 [0.9024, 0.9387] | [0.9077, 0.9311] | 87 (11 / 44.5 / 19.5) | 262 | 55.5 |
| YR10m10 | 0.9248 / 0.9249 | 0.9248 [0.9079, 0.9441] | [0.9135, 0.9354] | 78.5 (7 / 43 / 16) | 263 | 50 |
| AP2H10 | 0.9246 / 0.9282 | 0.9264 [0.9093, 0.9431] | [0.9140, 0.9384] | 97.5 (14 / 37.5 / 34) | 126 | 51.5 |
| APY10m10 | 0.9351 / 0.9320 | 0.9335 [0.9172, 0.9505] | [0.9226, 0.9448] | 86 (12 / 39.5 / 22.5) | 136 | 51.5 |

Pairwise differences on all 1491 (row minus column; point, CI logs, CI scenes, seed 0 / seed 1):

| difference | point | CI logs | CI scenes | seeds |
|:--|--:|:--|:--|:--|
| YR10m10 - P2H10 | +0.0052 | [-0.0006, +0.0120] | [-0.0028, +0.0129] | +0.0056 / +0.0048 |
| AP2H10 - P2H10 | +0.0068 | [-0.0048, +0.0185] | [-0.0025, +0.0160] | +0.0055 / +0.0082 |
| APY10m10 - P2H10 | +0.0139 | [+0.0048, +0.0227] | [+0.0040, +0.0238] | +0.0159 / +0.0120 |
| AP2H10 - YR10m10 | +0.0016 | [-0.0098, +0.0132] | [-0.0088, +0.0119] | -0.0001 / +0.0033 |
| APY10m10 - YR10m10 | +0.0087 | [-0.0004, +0.0173] | [+0.0000, +0.0177] | +0.0103 / +0.0071 |
| APY10m10 - AP2H10 | +0.0071 | [+0.0010, +0.0128] | [-0.0008, +0.0154] | +0.0105 / +0.0038 |

The AlpaSim input standard halves slow scenes (126 / 136 against 262 / 263) and pays for it in left-corridor zeros (34 / 22.5 against 16 / 19.5); the rows lower zeros relative to AP2H10 (86 against 97.5) and keep the slow-scene gain.
APY10m10 is the best of the four on every pooled read and the only recipe whose gain over P2H10 has both CIs above 0 on the 791 and on all 1491.

Checkout per run (`checkout.txt` in each run dir): AP2H10-AB-s0 (both lists) and s1 on ot3new c08f414a; s1 on cf1fresh d3441732 (relaunch after the 21:01 box restart killed the chain; the three finished runs
were adopted into the manifest after checking them). `git diff c9d2e07b HEAD` over `experiments/alpasim/lib`, `jevdrive` and the driver scripts is empty for all of them, so the driver is the CF1 driver. The CF1 runs (a-stage) carry c9d2e07b.
