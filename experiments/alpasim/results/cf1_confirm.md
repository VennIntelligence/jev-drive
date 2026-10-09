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
