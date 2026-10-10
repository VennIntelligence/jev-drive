# FOV check: was the road the driver needed inside the model's view at decision time?

## Reading (numbers from the tables below; unit = token x seed for SH30, token for WA-JEPA; 1 517 navtest tokens with a logged 4 s heading change > 45 deg in 101 logs)

- **View.** SH30's horizontal field is **58.7 deg (+-29.4 deg about the ego heading)** (wide frame, focal 455 in 512 px, rendered from CAM_F0; the road frame is 31.4 deg). Ground nearer than 8.2 m ahead is below the frame. Reading as 'visible' means geometrically inside the frame; occlusion is not measurable from the map.
- **Premise, mechanically true for one failure class.** SH30 fails DAC on 274 of 3 034 token-seeds. The point where the replayed plan leaves the drivable area is outside the horizontal field at t0 in **59.9% [47.5, 72.3]** of them and was in no keyframe's field in 51.8% [40.5, 63.7]. The split is by class: cut-inside (130 of 274) **93.8% [86.3, 98.7]** outside at t0 (83.1% never in view; median bearing 43 deg, turn side), cannot-make-turn (75) 28.0% [13.0, 45.9], other (69) 30.4% [13.0, 54.8]. The logged-path point at the plan's arc length (the road the plan needed to follow) is outside the field at t0 in 56.6% [44.3, 68.2] of the failures.
- **But it does not discriminate failures from passes.** The inside edge of a > 45 deg turn is outside the field for nearly every token (inside edge in field at t0: passes 6.8% [5.3, 8.6], failures 10.8% [6.0, 16.7]); the 4 s logged path is beyond +-29.4 deg at its end for 71.9% of passes and 82.1% of failures; the share of the far path inside the frame is 54.0% for passes and 45.4% for failures (difference -8.5 pp [-17.9, +0.9]), and WA-JEPA's failures show the same direction (-2.8 pp [-17.2, +12.7]). DAC failure rate by the largest path bearing (< 20 / 20-29.4 / 29.4-45 / > 45 deg): SH30 10.7 / 5.6 / 9.7 / 10.3%, WA-JEPA 6.6 / 2.2 / 3.0 / 4.0%, so the SH30:WA-JEPA ratio (1.6 / 2.5 / 3.2 / 2.6) does not climb with bearing. SH30 fails more when under 80% of the path is in frame (10.9% against 6.1%; difference of differences against WA-JEPA +4.1 pp [+0.6, +7.6]), but within heading-change strata it is +1.6 pp [-3.3, +7.1]: the raw contrast is turn sharpness.
- **WA-JEPA sees it.** Its four native cameras contain 100% of the departure points of both models' failures (98 - 99% inside the image rows); it still fails 49 of 1 517 tokens (3.2%) against SH30's 9.0%.
- **Net.** The readings do not support 'the needed road is out of view on failures' as the explanation: where the road is out of view (the inside curb of a cut-inside failure) it is out of view on the 90% of turns that pass too, and no view-share or bearing measure separates SH30's failures from its passes once turn angle is held fixed. They are compatible with a side-view benefit confined to cut-inside failures, but nothing here shows it; the pilot side / rear camera arm (P3, frozen encoder, whole board) added nothing. A direct test needs a model that actually sees the curb.

Written 2026-10-10 (corridor lane). Measurement only: stored SH30 / WA-JEPA plans and DAC scores (decision 207's `score_all.csv`), the logged future, the nuPlan drivable area of the v2 metric cache; CPU. Code `scripts/fov_build.py` (geometry, departure replay) and `scripts/fov_report.py`. Raw rows: `$DATA_DIR/runs/corridor/fov/{tok,unit}.parquet`. 95% CIs are log-cluster bootstraps (B 10 000). No decision entry.

The map, the logged future and the replayed departure points are privileged: this is an analysis of where the road lies relative to the camera, not a method input. **Occlusion by buildings, vehicles or hedges is not measurable from the map**; every 'visible' below means 'inside the frame geometrically', an upper bound on what the model can actually see.

## Views used (from code)

SH30 is a front-camera model (P2 family: Cinque + ego / pose / command; the side / rear camera arm P3 was tested and added nothing, navtest EPDMS 86.10 vs 86.57, `experiments/op_parity` pilot). Its image input is openpilot's two frames rendered from NAVSIM `CAM_F0` (`jevdrive/navsim_zs.py` `OpenpilotMaps`, `jevdrive/openpilot/frames.py`), optical axis along the ego x axis, 512 x 256:

| frame | focal px | horizontal FOV | vertical FOV | horizon row | ground nearer than this is below the frame (camera height 1.87 m) |
|:--|--:|--:|--:|--:|--:|
| wide | 455 | 58.7 deg (+-29.4) | 31.3 deg (18.5 up / 12.9 down) | 151.8 | 8.2 m ahead |
| road (narrow) | 910 | 31.4 deg | 15.9 deg | 47.6 | 8.2 m ahead |

The wide frame contains the road frame horizontally, so the model's horizontal field is **58.7 deg, +-29.4 deg about the ego heading**. The source camera bounds it: the native CAM_F0 image is 1920 x 1080; the sampling coverage of both frames is 100% on the checked calibration. Native cameras of the log (heading in the ego frame, left +; WA-JEPA reads L0 / F0 / R0 / B0):

| camera | mount yaw deg | native HFOV deg | native VFOV deg | position x, y, z (m, ego) |
|:--|--:|--:|--:|:--|
| CAM_F0 | +0.2 | 63.7 | 38.5 | 1.67, -0.03, 1.52 |
| CAM_L0 | +55.2 | 63.7 | 38.5 | 1.65, +0.14, 1.52 |
| CAM_R0 | -55.5 | 63.7 | 38.5 | 1.63, -0.16, 1.53 |
| CAM_B0 | +179.7 | 63.7 | 38.5 | -0.49, -0.00, 1.49 |

So WA-JEPA's horizontal union is about 4 x 64 deg minus overlaps (nearly all-round); ours is 58.7 deg. The frame is built from one camera; the three-front-camera wide input the rig note asked for was never used by SH30. Whether the model uses its wide frame at all is a separate question (`experiments/op_fov`: widening it did not help).

Definitions. Positions are in the t0 rear-axle frame; the camera sits at the CAM_F0 mount (x +1.67 m); bearing = angle of the point seen from the camera against the ego heading, left +. 'in FOV' = |bearing| <= 29.4 deg and ahead; 'in image' also needs the pinhole row inside the 256-row frame for a ground point (z = -0.35 m). 'History' = the 4 keyframes the model reads (t0 - 1.5, -1.0, -0.5, 0 s), the point being in the frame of at least one of them. Logged path = the 4 s logged future, sampled every 0.25 m. Edges = nearest drivable-area boundary on each side of the logged path (lateral search 12 m, every 1 m; inside = the turn side); a hit is absent where the road is wider than 12 m (intersections). 'Departure' = first footprint corner outside the scorer's drivable polygons along the LQR replay (decision 153's replay, the same replay that scores DAC). 'Needed road point' = the logged-path point at the plan's own arc length up to the departure. 'Relevant edge' = the drivable-area boundary within 10 m of the departure point.

Population: navtest tokens with |logged heading change at 4 s| > 45 deg: 1517 tokens in 101 logs; 274 SH30 failing token-seeds (of 3034), 49 WA-JEPA failing tokens (of 1517; 119 navtest tokens have no WA-JEPA plan). Failure classes are decision 240's: cut-inside = the first corner outside is on the turn side; cannot-make-turn = otherwise, plan heading gain < 0.9.

## 1. Bearing of the road from the t0 camera

Median bearing (deg, signed so that the turn direction is positive) of the logged path at 1 - 4 s and at fixed arc lengths, and the share of path points beyond the wide frame's +-29.4 deg. Points nearer than 8 m ahead of the camera are below the frame and are listed in the 1 s column only as a bearing.

| set | n | med 1s | med 2s | med 3s | med 4s | med s10 | med s20 | med s30 | outside FOV 4s % | outside FOV s10 % | outside FOV s20 % | outside FOV s30 % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30 pass | 2760 | 12.0 | 17.7 | 26.0 | 34.8 | 22.1 | 29.1 | 27.7 | 71.9 [68.6, 75.1] | 19.9 [15.3, 24.8] | 49.8 [40.6, 58.1] | 37.2 [23.0, 54.3] |
| SH30 DAC fail | 274 | 12.5 | 19.6 | 28.5 | 37.0 | 23.3 | 30.0 | 39.5 | 82.1 [74.1, 89.0] | 26.7 [17.7, 36.0] | 57.3 [39.2, 72.1] | 75.0 [0.0, 100.0] |
| - cut-inside | 130 | 12.7 | 19.5 | 28.1 | 35.8 | 23.4 | 29.4 |  | 86.9 [75.9, 95.2] | 15.5 [7.0, 25.0] | 55.8 [35.0, 77.8] | nan [nan, nan] |
| - cannot-make-turn | 75 | 11.6 | 19.8 | 30.9 | 42.2 | 21.0 | 31.0 | 34.4 | 74.7 [58.7, 88.2] | 39.4 [19.0, 60.5] | 58.3 [20.0, 84.6] | 50.0 [0.0, 100.0] |
| - other fail | 69 | 12.3 | 20.5 | 31.6 | 40.9 | 22.6 | 31.6 | 40.1 | 81.2 [68.0, 94.3] | 34.8 [18.6, 55.3] | 59.3 [36.4, 81.8] | 100.0 [100.0, 100.0] |
| SH30 fail, WA-JEPA pass | 216 | 12.6 | 19.8 | 28.8 | 36.7 | 23.4 | 29.9 | 40.9 | 82.9 [74.2, 90.5] | 25.4 [15.3, 36.1] | 57.9 [40.8, 72.0] | 66.7 [0.0, 100.0] |
| SH30 fail, WA-JEPA fail | 58 | 11.3 | 19.1 | 28.0 | 38.2 | 19.9 | 31.6 | 38.2 | 79.3 [60.0, 94.7] | 32.1 [12.2, 53.5] | 55.6 [22.2, 85.7] | 100.0 [100.0, 100.0] |
| WA-JEPA pass | 1468 | 12.1 | 17.9 | 26.2 | 35.1 | 22.3 | 29.1 | 27.8 | 72.7 [69.6, 75.7] | 20.3 [15.7, 25.0] | 49.9 [40.5, 57.9] | 38.1 [23.4, 55.1] |
| WA-JEPA DAC fail | 49 | 11.2 | 18.8 | 27.8 | 37.9 | 20.3 | 33.1 | 38.2 | 77.6 [60.8, 92.3] | 28.3 [11.5, 45.8] | 66.7 [43.8, 88.2] | 100.0 [100.0, 100.0] |

`med`: median of sgn * bearing, deg. `outside FOV`: share of members whose path point (at that time / arc length) has |bearing| > 29.4 deg, among members whose path reaches it (arc-length columns drop slow tokens; 30 m is reached by few).

| set | centreline med s10 | centreline med s20 | centreline med s30 | centreline med s40 | inside edge med s10 | n in s10 | inside edge med s20 | n in s20 | outside edge med s10 | n out s10 | outside edge med s20 | n out s20 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30 pass | 18.1 | 39.1 | 52.2 | 58.6 | 65.1 | 868 | 52.5 | 907 | -7.4 | 1172 | 21.0 | 869 |
| SH30 DAC fail | 19.4 | 39.1 | 49.7 | 54.8 | 57.7 | 92 | 46.4 | 129 | -5.6 | 130 | 20.2 | 113 |
| - cut-inside | 19.2 | 36.8 | 44.9 | 49.8 | 56.8 | 23 | 44.8 | 66 | -8.9 | 52 | 14.5 | 64 |
| - cannot-make-turn | 25.8 | 50.8 | 63.6 | 69.9 | 66.3 | 30 | 37.0 | 25 | 9.0 | 36 | 15.1 | 17 |
| - other fail | 18.0 | 41.9 | 54.7 | 59.6 | 59.5 | 39 | 51.3 | 38 | -2.1 | 42 | 24.3 | 32 |
| SH30 fail, WA-JEPA pass | 19.5 | 38.8 | 49.9 | 54.8 | 65.3 | 66 | 46.4 | 94 | -5.4 | 101 | 14.7 | 82 |
| SH30 fail, WA-JEPA fail | 14.7 | 40.3 | 49.5 | 52.7 | 52.9 | 26 | 48.8 | 35 | -5.8 | 29 | 23.3 | 31 |
| WA-JEPA pass | 18.2 | 39.1 | 52.0 | 58.4 | 65.3 | 464 | 51.5 | 493 | -7.3 | 628 | 20.6 | 468 |
| WA-JEPA DAC fail | 20.1 | 39.3 | 48.6 | 55.0 | 50.0 | 16 | 46.3 | 25 | -4.4 | 23 | 23.7 | 23 |

Exit-lane centreline arc length counts from abeam of the ego along the driven lane sequence (privileged map; matched tokens only); edge columns are the hit nearest to the path point at that arc length, `n` = members with a hit.

## 2. Share of the road inside the model's view at t0, and over the history keyframes

Mean over members of the share of the object that is inside the wide frame (SH30) at t0 / in at least one of the 4 keyframes. `FOV` = horizontal test, `image` = also inside the 256 rows.

**logged path 4 s** (% of arc length / edge samples)

| set | n | fov t0 | fov any | img t0 | img any |
|:--|--:|--:|--:|--:|--:|
| SH30 pass | 2760 | 54.0 [50.0, 58.2] | 59.3 [55.5, 63.2] | 45.7 [41.6, 50.3] | 57.3 [53.2, 61.5] |
| SH30 DAC fail | 274 | 45.4 [37.4, 53.9] | 54.4 [46.0, 63.2] | 39.3 [31.5, 47.7] | 52.1 [44.0, 60.9] |
| - cut-inside | 130 | 48.9 [38.9, 59.1] | 55.3 [44.5, 66.6] | 39.9 [29.6, 51.3] | 51.5 [40.3, 63.5] |
| - cannot-make-turn | 75 | 44.0 [26.3, 63.1] | 53.2 [32.0, 75.7] | 40.3 [23.2, 58.7] | 51.7 [30.6, 74.1] |
| - other fail | 69 | 40.2 [21.2, 55.1] | 53.9 [34.8, 68.6] | 36.9 [17.5, 51.7] | 53.7 [34.6, 68.4] |
| SH30 fail, WA-JEPA pass | 216 | 45.1 [36.0, 54.2] | 54.7 [45.6, 63.9] | 37.8 [29.2, 47.0] | 51.7 [43.1, 61.0] |
| SH30 fail, WA-JEPA fail | 58 | 46.9 [30.2, 65.2] | 53.5 [37.2, 72.6] | 45.0 [28.5, 63.6] | 53.5 [37.2, 72.6] |
| WA-JEPA pass | 1468 | 53.3 [49.6, 57.3] | 59.0 [55.5, 62.6] | 45.0 [41.1, 49.4] | 56.9 [53.1, 60.9] |
| WA-JEPA DAC fail | 49 | 50.5 [36.6, 65.9] | 55.7 [42.1, 71.5] | 47.7 [33.8, 63.2] | 55.7 [42.1, 71.5] |

**centreline 40 m** (% of arc length / edge samples)

| set | n | fov t0 | fov any | img t0 | img any |
|:--|--:|--:|--:|--:|--:|
| SH30 pass | 2760 | 22.9 [19.8, 26.2] | 26.6 [23.5, 29.9] | 21.8 [18.7, 25.2] | 26.4 [23.4, 29.7] |
| SH30 DAC fail | 274 | 21.0 [16.4, 26.2] | 27.4 [21.4, 34.4] | 20.2 [15.5, 25.4] | 27.3 [21.2, 34.3] |
| - cut-inside | 130 | 22.5 [16.0, 29.4] | 27.1 [18.9, 36.0] | 21.2 [14.7, 28.4] | 26.8 [18.6, 35.9] |
| - cannot-make-turn | 75 | 20.0 [11.2, 30.6] | 29.8 [17.0, 44.3] | 19.4 [10.6, 30.1] | 29.6 [16.9, 44.2] |
| - other fail | 69 | 19.5 [8.7, 28.0] | 25.6 [12.1, 36.2] | 19.0 [8.2, 27.5] | 25.6 [12.0, 36.2] |
| SH30 fail, WA-JEPA pass | 216 | 20.0 [14.9, 25.8] | 26.0 [19.5, 33.2] | 19.1 [13.9, 24.9] | 25.9 [19.4, 33.1] |
| SH30 fail, WA-JEPA fail | 58 | 24.7 [15.2, 35.1] | 32.6 [19.7, 47.4] | 24.2 [14.5, 34.7] | 32.4 [19.4, 47.3] |
| WA-JEPA pass | 1468 | 22.7 [19.7, 26.0] | 26.5 [23.5, 29.8] | 21.6 [18.6, 24.9] | 26.4 [23.4, 29.6] |
| WA-JEPA DAC fail | 49 | 23.7 [16.4, 31.9] | 30.9 [20.9, 42.6] | 23.0 [15.6, 31.4] | 30.8 [20.8, 42.5] |

**inside edge** (% of arc length / edge samples)

| set | n | fov t0 | fov any | img t0 | img any |
|:--|--:|--:|--:|--:|--:|
| SH30 pass | 2760 | 6.8 [5.3, 8.6] | 17.0 [14.3, 19.8] | 5.6 [4.3, 7.1] | 16.9 [14.2, 19.6] |
| SH30 DAC fail | 274 | 10.8 [6.0, 16.7] | 25.7 [16.6, 35.6] | 9.4 [5.1, 14.7] | 25.6 [16.5, 35.5] |
| - cut-inside | 130 | 8.3 [3.2, 14.2] | 17.8 [7.8, 29.4] | 7.2 [2.6, 12.6] | 17.6 [7.5, 29.2] |
| - cannot-make-turn | 75 | 20.6 [8.1, 34.7] | 39.7 [16.9, 63.8] | 17.5 [6.8, 30.9] | 39.7 [16.9, 63.8] |
| - other fail | 69 | 5.4 [0.0, 17.5] | 26.4 [8.9, 42.6] | 5.0 [0.0, 15.9] | 26.4 [8.9, 42.6] |
| SH30 fail, WA-JEPA pass | 216 | 9.6 [4.9, 15.2] | 24.3 [15.2, 34.4] | 8.3 [4.1, 13.3] | 24.1 [15.0, 34.2] |
| SH30 fail, WA-JEPA fail | 58 | 15.6 [5.2, 30.0] | 31.1 [11.2, 53.5] | 13.4 [3.5, 27.3] | 31.1 [11.2, 53.5] |
| WA-JEPA pass | 1468 | 6.9 [5.3, 8.7] | 17.5 [14.7, 20.3] | 5.7 [4.3, 7.3] | 17.3 [14.6, 20.2] |
| WA-JEPA DAC fail | 49 | 16.2 [5.1, 29.3] | 29.4 [12.4, 48.4] | 14.1 [3.9, 26.0] | 29.4 [12.4, 48.4] |

**outside edge** (% of arc length / edge samples)

| set | n | fov t0 | fov any | img t0 | img any |
|:--|--:|--:|--:|--:|--:|
| SH30 pass | 2760 | 73.9 [70.5, 77.5] | 81.3 [77.7, 85.0] | 73.0 [69.8, 76.6] | 81.1 [77.5, 84.9] |
| SH30 DAC fail | 274 | 74.8 [69.3, 81.2] | 83.9 [79.0, 89.2] | 74.3 [68.7, 80.8] | 83.8 [78.8, 89.2] |
| - cut-inside | 130 | 81.4 [73.9, 89.0] | 87.8 [80.9, 94.0] | 81.0 [73.7, 88.6] | 87.8 [80.9, 94.0] |
| - cannot-make-turn | 75 | 67.2 [53.5, 80.4] | 82.4 [70.1, 93.4] | 66.8 [52.7, 80.4] | 82.1 [69.3, 93.3] |
| - other fail | 69 | 70.9 [58.5, 80.4] | 78.2 [65.4, 87.1] | 70.1 [57.6, 79.9] | 78.2 [65.4, 87.1] |
| SH30 fail, WA-JEPA pass | 216 | 74.5 [69.1, 81.0] | 84.8 [79.6, 90.2] | 74.0 [68.5, 80.6] | 84.7 [79.4, 90.2] |
| SH30 fail, WA-JEPA fail | 58 | 75.7 [63.3, 87.7] | 81.0 [69.5, 91.3] | 75.5 [63.1, 87.6] | 81.0 [69.5, 91.3] |
| WA-JEPA pass | 1468 | 73.8 [70.6, 77.4] | 81.5 [78.0, 85.1] | 73.0 [69.8, 76.5] | 81.3 [77.8, 85.0] |
| WA-JEPA DAC fail | 49 | 78.0 [66.0, 88.4] | 83.4 [72.4, 92.7] | 77.8 [65.8, 88.3] | 83.4 [72.4, 92.7] |

All fractions in this section are over points at least 8.2 m ahead of the camera: ground nearer than that falls below the bottom row of the wide frame at any heading (near-field blind spot, not a side-view question). The share of the 4 s path that is nearer than that:

| set | path arc nearer than 8.2 m (mean share) % |
|:--|--:|
| SH30 pass | 59.4 [56.1, 63.0] |
| SH30 DAC fail | 58.4 [53.0, 64.0] |
| - cut-inside | 55.1 [49.8, 60.2] |
| - cannot-make-turn | 64.4 [56.0, 72.6] |
| - other fail | 58.2 [49.6, 71.0] |
| SH30 fail, WA-JEPA pass | 59.0 [53.3, 64.8] |
| SH30 fail, WA-JEPA fail | 56.4 [49.0, 64.7] |
| WA-JEPA pass | 59.5 [56.1, 63.0] |
| WA-JEPA DAC fail | 55.2 [50.0, 60.6] |

| set | whole 4 s path in FOV at t0 % | at least half % | whole path in FOV over history % |
|:--|--:|--:|--:|
| SH30 pass | 26.3 [23.0, 29.8] | 50.8 [46.0, 56.2] | 33.3 [29.5, 36.9] |
| SH30 DAC fail | 18.6 [11.5, 26.8] | 43.1 [33.3, 53.6] | 28.5 [19.5, 38.6] |
| - cut-inside | 14.6 [5.6, 26.4] | 48.5 [33.3, 63.6] | 20.8 [9.6, 34.6] |
| - cannot-make-turn | 25.3 [11.8, 41.3] | 37.3 [19.5, 57.4] | 40.0 [19.7, 62.5] |
| - other fail | 18.8 [5.7, 32.0] | 39.1 [19.5, 56.9] | 30.4 [16.1, 43.1] |
| SH30 fail, WA-JEPA pass | 18.1 [10.2, 27.0] | 43.5 [31.9, 56.2] | 27.8 [18.4, 38.1] |
| SH30 fail, WA-JEPA fail | 20.7 [5.3, 40.0] | 41.4 [21.7, 60.9] | 31.0 [11.5, 54.1] |
| WA-JEPA pass | 25.7 [22.6, 28.9] | 50.2 [45.7, 55.5] | 32.8 [29.4, 36.3] |
| WA-JEPA DAC fail | 22.4 [7.7, 39.2] | 46.9 [30.2, 64.4] | 32.7 [16.2, 51.9] |

## 3. Where the plan leaves the drivable area, against the view

Failures only; the replayed departure point (first footprint corner outside along the LQR replay). Members whose replay shows no departure (DAC < 1 from another cause) are excluded: sh0 0, sh1 0, wa 0.

| set | n | departure in FOV t0 % | departure in image t0 % | departure in FOV any history % | departure in image any history % |
|:--|--:|--:|--:|--:|--:|
| SH30 DAC fail | 274 | 40.1 [27.7, 52.5] | 33.2 [21.5, 44.8] | 48.2 [36.3, 59.5] | 45.3 [33.6, 56.5] |
| - cut-inside | 130 | 6.2 [1.3, 13.7] | 4.6 [0.0, 11.3] | 16.9 [7.0, 28.9] | 15.4 [5.9, 27.0] |
| - cannot-make-turn | 75 | 72.0 [54.1, 87.0] | 57.3 [37.9, 76.7] | 77.3 [60.2, 91.4] | 72.0 [51.7, 90.1] |
| - other fail | 69 | 69.6 [45.2, 87.0] | 60.9 [32.6, 80.6] | 75.4 [56.9, 88.5] | 72.5 [51.9, 86.3] |
| SH30 fail, WA-JEPA pass | 216 | 32.4 [21.0, 44.3] | 26.4 [16.3, 36.9] | 42.1 [31.0, 53.6] | 39.4 [28.4, 50.5] |
| SH30 fail, WA-JEPA fail | 58 | 69.0 [44.4, 90.5] | 58.6 [32.7, 81.2] | 70.7 [46.7, 91.4] | 67.2 [42.6, 88.7] |
| WA-JEPA DAC fail | 49 | 53.1 [32.4, 74.0] | 46.9 [26.5, 67.4] | 55.1 [34.5, 76.0] | 53.1 [32.6, 74.1] |

| set | n | needed-road point in FOV t0 % | needed-road point in image t0 % | needed-road point in FOV any history % | relevant edge in FOV t0 (mean share) % | relevant edge in FOV any history % | median departure bearing (turn side +) deg | median departure range m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30 DAC fail | 274 | 43.4 [31.8, 55.7] | 31.4 [21.5, 42.4] | 54.1 [42.2, 66.5] | 36.5 [32.2, 41.2] | 47.7 [41.8, 54.2] | 34.9 | 15.2 |
| - cut-inside | 130 | 35.4 [19.5, 52.6] | 30.1 [15.9, 45.8] | 43.4 [28.0, 59.3] | 24.7 [19.7, 29.1] | 33.5 [25.9, 41.0] | 43.0 | 15.8 |
| - cannot-make-turn | 75 | 49.3 [26.5, 72.7] | 37.3 [17.4, 60.7] | 55.2 [30.9, 79.2] | 51.2 [42.1, 61.3] | 63.4 [51.1, 75.1] | 23.1 | 14.8 |
| - other fail | 69 | 51.6 [28.3, 68.8] | 27.4 [7.7, 41.8] | 72.6 [50.0, 88.0] | 42.6 [34.3, 48.0] | 57.4 [46.7, 65.7] | 25.3 | 14.2 |
| SH30 fail, WA-JEPA pass | 216 | 37.0 [25.1, 50.5] | 27.0 [16.7, 38.7] | 48.7 [36.9, 61.8] | 33.4 [28.6, 38.6] | 44.6 [38.5, 51.4] | 36.5 | 15.3 |
| SH30 fail, WA-JEPA fail | 58 | 66.0 [41.0, 87.5] | 47.2 [27.8, 69.2] | 73.6 [50.0, 93.9] | 48.0 [37.9, 58.1] | 59.4 [46.3, 71.9] | 20.5 | 14.5 |
| WA-JEPA DAC fail | 49 | 53.5 [32.5, 75.0] | 41.9 [23.9, 61.7] | 60.5 [39.1, 82.1] | 43.2 [32.6, 54.7] | 51.9 [39.2, 65.5] | 27.9 | 17.0 |

The 'in FOV' columns are the headline: the share of failures whose departure point (or needed-road point) lies inside the horizontal field at t0 / in some keyframe. Its complement is the share out of view. WA-JEPA's row evaluates WA-JEPA's own departure points against SH30's geometry (our frame), not against its own cameras; see section 4.

Where the departure lies relative to the turn (share on the inside of the turn / forward range):

| set | n | departure on the turn side % | departure beyond 29.4 deg bearing % |
|:--|--:|--:|--:|
| SH30 DAC fail | 274 | 98.5 [95.0, 100.0] | 59.9 [47.5, 72.3] |
| - cut-inside | 130 | 100.0 [100.0, 100.0] | 93.8 [86.3, 98.7] |
| - cannot-make-turn | 75 | 96.0 [87.8, 100.0] | 28.0 [13.0, 45.9] |
| - other fail | 69 | 98.6 [94.1, 100.0] | 30.4 [13.0, 54.8] |
| SH30 fail, WA-JEPA pass | 216 | 99.1 [96.8, 100.0] | 67.6 [55.7, 79.0] |
| SH30 fail, WA-JEPA fail | 58 | 96.6 [87.5, 100.0] | 31.0 [9.5, 55.6] |
| WA-JEPA DAC fail | 49 | 98.0 [93.2, 100.0] | 46.9 [26.0, 67.6] |

## 4. Does SH30 fail where the road leaves its view and WA-JEPA does not?

Difference in the share of the 4 s logged path inside the wide frame at t0 (fail minus pass), SH30 and WA-JEPA, and in WA-JEPA's own four cameras. The path geometry is the same logged path for both models; a view-limited failure would show a more negative difference for SH30 than for WA-JEPA in the SH30 view and a flat one in WA-JEPA's view.

| model | view | n fail | n pass | path FOV t0 fail - pass (pp) | path image t0 fail - pass (pp) | path FOV any history fail - pass (pp) |
|:--|--:|--:|--:|--:|--:|--:|
| SH30 | SH30 wide frame | 274 | 2760 | -8.5 [-17.9, +0.9] | -6.4 [-15.6, +2.7] | -4.9 [-14.8, +5.0] |
| SH30 | WA-JEPA 4 cameras | 274 | 2760 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| WA-JEPA | SH30 wide frame | 49 | 1468 | -2.8 [-17.2, +12.7] | +2.7 [-11.8, +18.2] | -3.2 [-17.6, +13.1] |
| WA-JEPA | WA-JEPA 4 cameras | 49 | 1468 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |

Failure rate (DAC) by the largest path bearing within 4 s (turn side +), per model; columns are bearing bins in the SH30 view:

| model | < 20 deg: n | < 20 deg: DAC fail % | 20 - 29.4 deg: n | 20 - 29.4 deg: DAC fail % | 29.4 - 45 deg: n | 29.4 - 45 deg: DAC fail % | > 45 deg: n | > 45 deg: DAC fail % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30 | 122 | 10.7 [3.8, 19.6] | 628 | 5.6 [3.0, 8.8] | 1582 | 9.7 [7.0, 12.8] | 702 | 10.3 [6.3, 14.3] |
| WA-JEPA | 61 | 6.6 [1.4, 14.5] | 314 | 2.2 [0.6, 4.4] | 791 | 3.0 [1.7, 4.6] | 351 | 4.0 [1.6, 7.1] |

Same by the share of the logged path inside the SH30 frame at t0:

| model | < 50%: n | < 50%: DAC fail % | 50 - 80%: n | 50 - 80%: DAC fail % | 80 - 99.9%: n | 80 - 99.9%: DAC fail % | 100%: n | 100%: DAC fail % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30 | 1366 | 10.5 [7.6, 13.8] | 492 | 11.6 [7.2, 16.7] | 212 | 4.7 [1.4, 9.2] | 776 | 6.6 [3.9, 9.7] |
| WA-JEPA | 683 | 3.4 [1.7, 5.4] | 246 | 3.7 [1.3, 6.5] | 106 | 2.8 [0.0, 6.1] | 388 | 2.8 [0.9, 5.5] |

Failure rate of the part of the population whose logged path is mostly out of the SH30 frame (< 80% of its far path inside at t0) against the rest, per model, and the difference of the two differences (positive = SH30's failures lean more on the out-of-view part than WA-JEPA's, the view-limit signature). A joint log-cluster bootstrap.

| model | fail % < 80% in frame | fail % >= 80% in frame | difference (pp) |
|:--|--:|--:|--:|
| SH30 | 10.9 | 6.1 | +4.8 [+1.1, +8.5] |
| WA-JEPA | 3.5 | 2.8 | +0.7 [-1.8, +3.0] |
| SH30 - WA-JEPA |  |  | +4.1 [+0.6, +7.6] |

The in-frame share is not independent of the turn: tighter turns put more of the path beyond the frame and are harder for any model. The same comparison stratified by the logged 4 s heading change (45 - 55, 55 - 65, > 65 deg; strata weighted by their size), so the contrast is within similar turn angles:

| model | stratified difference (pp) |
|:--|--:|
| SH30 | +1.2 [-4.0, +6.5] |
| WA-JEPA | -0.4 [-4.4, +2.8] |
| SH30 - WA-JEPA | +1.6 [-3.3, +7.1] |

WA-JEPA failure departure points in WA-JEPA's own cameras (share inside the union of its four native cameras, pinhole test):

| set | n | departure in its 4 cameras (FOV) t0 % | departure in its 4 cameras (image) t0 % | needed-road point in its 4 cameras (image) t0 % |
|:--|--:|--:|--:|--:|
| WA-JEPA DAC fail | 49 | 100.0 [100.0, 100.0] | 98.0 [92.7, 100.0] | 88.4 [78.2, 97.1] |
| SH30 DAC fail | 274 | 100.0 [100.0, 100.0] | 98.9 [97.2, 100.0] | 90.9 [83.5, 97.0] |

