# vred3: vred2 + junction slow-down on the approach + a yellow / commit rule (zero-shot Qwen3-VL-4B reads the light), closed loop

Diagnostic batch, 19 routes x 2 traffic seeds. Every number is a read, not a confirmation; no registered line applies to this arm and the confirmation lines are listed as not evaluated. `drive`, `pred`, `jslow`, `vred`, `vred2` are existing runs (not rerun in this arm's batch). `vred3` = `vred2` (R2 stops short of the stop line) + R1 on the approach only (cap 4.5 m/s from 25 m before the junction until the front bumper passes the stop line; lifted entirely afterwards and during a committed go) + the yellow decision at the first non-green answer after green (comfortable stop, else go if the whole car clears the scorer's line before the red, else hard stop) + the commit rule (no R2 start once the front bumper is past the stop line). Parameters, rule and offline basis: [plans/2026-10-02-pbyp2-vred2.md](../plans/2026-10-02-pbyp2-vred2.md) section 11, [vred3_yellow_offline.md](vred3_yellow_offline.md); inputs by source: section 12 of the plan.

## Arms

| arm   |   runs |   crashes |    DS |    RC |   red_light |   stop_sign |   collisions |   blocked |   timeouts |   v_mean |
|:------|-------:|----------:|------:|------:|------------:|------------:|-------------:|----------:|-----------:|---------:|
| drive |     38 |         0 | 65.9  | 83.27 |          13 |           2 |           11 |         2 |          0 |     2.2  |
| pred  |     38 |         0 | 70.23 | 83.99 |           5 |           2 |           13 |         0 |          0 |     1.87 |
| vred  |     38 |         0 | 70.92 | 83.86 |           6 |           2 |           11 |         2 |          0 |     2.01 |
| vred2 |     38 |         0 | 66.52 | 81.37 |           7 |           2 |           14 |         2 |          0 |     1.76 |
| vred3 |     38 |         0 | 67.58 | 80.73 |           7 |           2 |           12 |         1 |          0 |     1.71 |
| jslow |     38 |         0 | 68.02 | 83.97 |          11 |           2 |           10 |         0 |          0 |     2.02 |

Official infraction counts summed over the runs; DS, RC and mean speed averaged over runs; 38 runs expected per arm.

## Paired differences (mean over routes of the per-route mean over the two seeds; [95% route-cluster CI])

`logged-light routes` = routes on which an ego light was seen within 50 m in the vred3 logs (15102, 15483, 15612, 16390, 16508, 16529, 24944, 27043, 27297, 27870, 28147, 9196); `no logged light` = the other 7. Counts are per run.

| contrast | routes | n routes | DS | red light | blocked | collisions | mean speed m/s |
|:--|:--|--:|:--|:--|:--|:--|:--|
| vred3 - drive | all routes | 19 | +1.68 [-6.81, +9.27] | -0.16 [-0.37, +0.03] | -0.03 [-0.08, +0.00] | +0.03 [+0.00, +0.08] | -0.49 [-0.88, -0.17] |
| vred3 - drive | logged-light routes | 12 | +6.72 [-2.67, +15.47] | -0.25 [-0.54, +0.08] | -0.04 [-0.12, +0.00] | +0.04 [+0.00, +0.12] | -0.51 [-0.81, -0.21] |
| vred3 - drive | no logged light | 7 | -6.96 [-20.87, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.46 [-1.32, +0.02] |
| vred3 - vred2 | all routes | 19 | +1.05 [-3.54, +6.52] | +0.00 [+0.00, +0.00] | -0.03 [-0.08, +0.00] | -0.05 [-0.16, +0.00] | -0.05 [-0.29, +0.16] |
| vred3 - vred2 | logged-light routes | 12 | +3.40 [-0.12, +10.32] | +0.00 [+0.00, +0.00] | -0.04 [-0.12, +0.00] | -0.08 [-0.25, +0.00] | +0.02 [-0.17, +0.24] |
| vred3 - vred2 | no logged light | 7 | -2.97 [-9.98, +1.23] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.16 [-0.72, +0.22] |
| vred3 - jslow | all routes | 19 | -0.45 [-8.01, +7.08] | -0.11 [-0.34, +0.13] | +0.03 [+0.00, +0.08] | +0.05 [+0.00, +0.16] | -0.31 [-0.62, +0.02] |
| vred3 - jslow | logged-light routes | 12 | +1.20 [-10.00, +12.45] | -0.17 [-0.54, +0.21] | +0.04 [+0.00, +0.12] | +0.08 [+0.00, +0.25] | -0.39 [-0.83, +0.08] |
| vred3 - jslow | no logged light | 7 | -3.27 [-9.96, +0.25] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.17 [-0.46, +0.01] |
| vred3 - vred | all routes | 19 | -3.35 [-10.08, +2.32] | +0.03 [-0.11, +0.16] | -0.03 [-0.08, +0.00] | +0.03 [+0.00, +0.08] | -0.30 [-0.66, -0.05] |
| vred3 - vred | logged-light routes | 12 | -1.33 [-7.77, +4.84] | +0.04 [-0.17, +0.25] | -0.04 [-0.12, +0.00] | +0.04 [+0.00, +0.12] | -0.17 [-0.33, -0.02] |
| vred3 - vred | no logged light | 7 | -6.81 [-20.43, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.52 [-1.42, +0.00] |
| vred3 - pred | all routes | 19 | -2.65 [-5.17, -0.47] | +0.05 [+0.00, +0.13] | +0.03 [+0.00, +0.08] | -0.03 [-0.08, +0.00] | -0.16 [-0.39, +0.03] |
| vred3 - pred | logged-light routes | 12 | -3.13 [-6.88, +0.00] | +0.08 [+0.00, +0.21] | +0.04 [+0.00, +0.12] | +0.00 [+0.00, +0.00] | -0.11 [-0.30, +0.10] |
| vred3 - pred | no logged light | 7 | -1.83 [-5.36, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.07 [-0.21, +0.00] | -0.26 [-0.76, -0.00] |

Harm on routes without a logged light (14 runs): runs with more blocked events than their `drive` pair 0; with more collisions 0.

Repeat noise: 13 routes with 2-4 identical `drive` runs: per-route DS standard deviation mean 4.7, median 0.0, max 30.5; 5 of 13 routes changed DS between repeats (results/report.md). A difference whose CI includes 0 is within that noise at this size.

## In-batch latency of the light answers

L = 0.35 s: an answer is used at t_q + max(L, latency); TTL = 2.5 s.

| requests | answered ok | p50 ms | p95 ms | p99 ms | max ms | share > L | share > TTL | server queue p95 ms | server service p50 ms |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 6772 | 100.00% | 220 | 350 | 404 | 558 | 5.0% | 0.00% | 129 | 160 |

Calibration before the batch (shadow run, 3 shadow workers per card on 3 cards, nothing else on the box (the `vred` batch's load)): p50 216 ms, p95 342 ms, p99 391 ms (3203 requests); registered L = 0.35 s held (`v2_calibration_cal3.md`).

A first calibration at 6 workers per card (3 shadow + 3 pbyp2 workers, `v2_calibration_cal2.md`) gave p50 264 ms, p95 446 ms, p99 566 ms (2731 requests) and failed the registered line (p95 <= 350 ms); concurrency was reduced to the load above before any vred2 unit ran (vred3 ran at that reduced load as well).

## In-loop reading quality (answers logged during the runs against the simulator's light state)

Ego-green recall is shown for all requests and split by whether R2 was holding the car when the request was made (`while holding`): vred holds the car at the junction entrance, beyond the stop line; vred2 holds it short of the line, with the light in view. Estimate [95% route-cluster CI]; requests and routes behind each.

| readout | arm | all requests | while R2 holds | not holding |
|:--|:--|:--|:--|:--|
| ego red / yellow answered red, 0-50 m | vred | 91.9% [84.0, 98.7] (n=959, 11 routes) | 97.6% [94.8, 99.2] (n=801, 10 routes) | 62.7% [53.9, 94.3] (n=158, 11 routes) |
| ego green answered green, 0-50 m | vred | 74.1% [47.1, 95.4] (n=406, 12 routes) | 28.6% [16.0, 91.7] (n=126, 9 routes) | 94.6% [88.7, 97.2] (n=280, 12 routes) |
| ego red / yellow answered red, 0-50 m | vred2 | 91.9% [82.3, 99.7] (n=998, 11 routes) | 99.0% [97.0, 99.9] (n=832, 10 routes) | 56.0% [42.4, 100.0] (n=166, 11 routes) |
| ego green answered green, 0-50 m | vred2 | 95.3% [92.0, 97.7] (n=361, 12 routes) | 94.9% [85.1, 100.0] (n=39, 9 routes) | 95.3% [91.3, 97.8] (n=322, 12 routes) |
| ego red / yellow answered red, 0-50 m | vred3 | 92.7% [85.1, 99.4] (n=1054, 11 routes) | 98.7% [96.9, 99.8] (n=869, 10 routes) | 64.3% [54.5, 100.0] (n=185, 11 routes) |
| ego green answered green, 0-50 m | vred3 | 93.5% [89.4, 97.3] (n=382, 12 routes) | 93.2% [85.2, 100.0] (n=44, 10 routes) | 93.5% [88.2, 97.6] (n=338, 12 routes) |

## Stop position at every red-light stop

Front bumper to the stop line of the governing light (ctx `tl_dist`, simulator truth, evaluation only) at the last plan step of standstill under the arm's hold (where the car finally stood; a car that starts the route standing under a hold is counted where it stood at the end) (pred: privileged red stop; vred / vred2: R2). Positive = short of the line. `min` = the smallest distance during the hold (negative = the car crept across the line while holding).

| arm | stops | median d_stop | min | max | stopped beyond the line | crept across while holding (min < 0) |
|:--|--:|--:|--:|--:|--:|--:|
| pred | 27 | 0.24 | -0.84 | 46.97 | 2 | 2 |
| vred | 23 | -0.84 | -3.84 | 47.09 | 13 | 15 |
| vred2 | 20 | 0.16 | -0.78 | 33.77 | 1 | 1 |
| vred3 | 22 | 0.70 | 0.16 | 35.00 | 0 | 0 |

| arm   |   route |   seed |    t |   d_stop |   d_min |   hold_s |
|:------|--------:|-------:|-----:|---------:|--------:|---------:|
| vred3 |   27043 |      0 | 13.9 |     4.16 |    4.16 |      8.8 |
| vred3 |   15102 |      0 | 45.4 |     0.16 |    0.16 |     40.3 |
| vred3 |   24944 |      0 | 30.9 |    35    |   35    |      2   |
| vred3 |   24944 |      0 | 75.9 |     0.16 |    0.16 |     34.5 |
| vred3 |   27870 |      0 | 15.3 |     1.16 |    1.16 |     10.3 |
| vred3 |   27297 |      0 | 30.4 |     0.24 |    0.24 |     20.5 |
| vred3 |    9196 |      0 | 30.4 |     2.16 |    2.16 |     25.3 |
| vred3 |   28147 |      0 |  5.7 |    16.32 |   16.32 |      1.8 |
| vred3 |   15612 |      0 | 15   |     2.16 |    1.16 |     10.3 |
| vred3 |   15483 |      0 | 30.4 |     0.16 |    0.16 |     25.3 |
| vred3 |   16529 |      0 | 30.4 |     0.16 |    0.16 |     25.3 |
| vred3 |   27043 |      1 | 13.9 |     4.16 |    4.16 |      8.8 |
| vred3 |   15102 |      1 | 45.4 |     0.16 |    0.16 |     40.3 |
| vred3 |   24944 |      1 | 75.9 |     0.16 |    0.16 |     36   |
| vred3 |   27870 |      1 | 15.4 |     1.16 |    1.16 |     10.4 |
| vred3 |   27297 |      1 | 15.2 |     0.24 |    0.24 |      5.5 |
| vred3 |    9196 |      1 | 30.4 |     0.16 |    0.16 |     25.3 |
| vred3 |   28147 |      1 |  5.7 |    16.32 |   16.32 |      1.9 |
| vred3 |   15612 |      1 | 15.4 |     1.16 |    1.16 |     10.3 |
| vred3 |   15483 |      1 | 30.4 |     0.16 |    0.16 |     25.3 |
| vred3 |   16529 |      1 |  7.1 |     2.16 |    2.16 |      2.3 |
| vred3 |   16529 |      1 | 30.4 |     0.16 |    0.16 |     22.5 |

## Green after red

20 red-to-green changes of the ego light while R2 held the car. Seconds after the light turned green (median [p5, p95]): second consecutive green answer in force 0.9 [0.9, 1.4] (0 never reached); R2 released 1.0 [1.0, 1.5]; car rolling (> 1 m/s) 2.0 [1.5, 2.0] (0 never).

|   route |   seed |   t_green |   answer |   r2_end |   roll |
|--------:|-------:|----------:|---------:|---------:|-------:|
|   27043 |      0 |      13.1 |      0.9 |      1   |    2   |
|   27043 |      1 |      13.1 |      0.9 |      1   |    2   |
|   15102 |      0 |      44.6 |      0.9 |      1   |    2   |
|   15102 |      1 |      44.6 |      0.9 |      1   |    2   |
|   24944 |      0 |      29.6 |      1.4 |      1.5 |    2   |
|   24944 |      0 |      75.1 |      0.9 |      1   |    2   |
|   24944 |      1 |      75.1 |      0.9 |      1   |    2   |
|   27870 |      0 |      14.6 |      0.9 |      1   |    1.5 |
|   27870 |      1 |      14.6 |      0.9 |      1   |    1.5 |
|   27297 |      0 |      29.6 |      0.9 |      1   |    2   |
|    9196 |      0 |      29.6 |      0.9 |      1   |    2   |
|    9196 |      1 |      29.6 |      0.9 |      1   |    2   |
|   28147 |      0 |       5.6 |      1.4 |      1.5 |    1.5 |
|   28147 |      1 |       5.6 |      1.4 |      1.5 |    1.5 |
|   15612 |      0 |      14.6 |      0.9 |      1   |    1.5 |
|   15612 |      1 |      14.6 |      0.9 |      1   |    1.5 |
|   15483 |      0 |      29.6 |      0.9 |      1   |    2   |
|   15483 |      1 |      29.6 |      0.9 |      1   |    2   |
|   16529 |      0 |      29.6 |      0.9 |      1   |    2   |
|   16529 |      1 |      29.6 |      0.9 |      1   |    2   |

## R5 fallback

0 episodes in the 38 vred3 runs.

## Remaining red-light infractions in vred3 (7)

Located with the scorer's rule (`RunningRedLightTest`: the light of the official message is red while the car's tail segment crosses the junction entrance line, i.e. the front bumper is 4.4-5.9 m beyond it): the time the front bumper passed that light's stop line (ctx `tl_dist` <= 0), the light's logged state there, its red onset, and a cause label from the logs (red began after the front bumper had passed the stop line, otherwise the `vred.md` rules: released early / not answered red / answered late / stale / R5). The timelines are the evidence.

|   route |   seed |   light |   t_front_stop_line | state_stop_line   |   v_stop_line |   t_front_entrance |   t_red_onset |   t_tail_crossing |   v_at_red |   front_vs_entrance_at_red | cause                                                                                                                                                                                              |
|--------:|-------:|--------:|--------------------:|:------------------|--------------:|-------------------:|--------------:|------------------:|-----------:|---------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   27297 |      1 |    4174 |                16.3 | red               |           1.7 |               17.6 |          12.6 |              18.6 |        0.9 |                       -5.6 | released early (R2 dropped while the light was red)                                                                                                                                                |
|   16390 |      0 |    6923 |                 7.6 | green             |           4.3 |                8.6 |           8.1 |               9.1 |        5   |                       -0.1 | the light turned red while the car was crossing (front 0.1 m before the entrance, 5.0 m/s, 0.5 s after its front passed the stop line); the car kept going                                         |
|   16390 |      1 |    6912 |                 7.6 | green             |           4.3 |                8.6 |           8.1 |               9.1 |        5   |                       -0.1 | the light turned red while the car was crossing (front 0.1 m before the entrance, 5.0 m/s, 0.5 s after its front passed the stop line); the car kept going                                         |
|   15612 |      0 |     881 |                16.7 | green             |           2.2 |               21.1 |          17.6 |              22.6 |        2.5 |                       -1.6 | the light turned red 0.9 s after the front bumper had passed the stop line; the car stopped at t = 18.6 s past the line (R2 never starts there) and moved on at t = 20.6 s while the light was red |
|   15612 |      1 |     881 |                16.6 | green             |           2   |               18.6 |          17.6 |              22.6 |        2.7 |                       -1.4 | the light turned red 1.0 s after the front bumper had passed the stop line; the car stopped at t = 19.1 s past the line (R2 never starts there) and moved on at t = 21.1 s while the light was red |
|   15483 |      0 |    1462 |                32.1 | green             |           2.4 |               33.1 |          32.6 |              34.6 |        3   |                       -1.4 | the light turned red while the car was crossing (front 1.4 m before the entrance, 3.0 m/s, 0.5 s after its front passed the stop line); the car kept going                                         |
|   15483 |      1 |    1837 |                32.1 | green             |           2.3 |               33.1 |          32.6 |              34.6 |        2.9 |                       -1.4 | the light turned red while the car was crossing (front 1.4 m before the entrance, 2.9 m/s, 0.5 s after its front passed the stop line); the car kept going                                         |


Route 27297 seed 1, light 4174, front bumper at the stop line at t = 16.3 s (light red): released early (R2 dropped while the light was red)

|    t |   v |   line_m | rules   | fresh   | last_K   | answer_in_force          | truth          |
|-----:|----:|---------:|:--------|:--------|:---------|:-------------------------|:---------------|
|  7.6 | 4.2 |    15.6  | R1      | True    | gre,gre  | gre (q 7.1, lat 0.20 s)  | tl 0 @ 17.57 m |
|  8.1 | 4.8 |    13.37 | R1      | True    | gre,gre  | gre (q 7.6, lat 0.31 s)  | tl 0 @ 15.53 m |
|  8.6 | 4.6 |    11.07 | R1      | True    | gre,gre  | gre (q 8.1, lat 0.19 s)  | tl 0 @ 13.49 m |
|  9.1 | 4.2 |     8.85 | R1      | True    | gre,gre  | gre (q 8.6, lat 0.34 s)  | tl 0 @ 11.45 m |
|  9.6 | 3.7 |     6.7  | R1      | True    | gre,gre  | gre (q 9.1, lat 0.20 s)  | tl 0 @ 8.4 m   |
| 10.1 | 3.4 |     4.91 | R1+R2   | True    | gre,red  | red (q 9.6, lat 0.33 s)  | tl 1 @ 6.36 m  |
| 10.6 | 2.8 |     3.32 | R1+R2   | True    | red,red  | red (q 10.1, lat 0.19 s) | tl 1 @ 5.34 m  |
| 11.1 | 2.1 |     2.19 | R1+R2   | True    | red,red  | red (q 10.6, lat 0.19 s) | tl 1 @ 3.3 m   |
| 11.6 | 0   |     1.42 | R1+R2   | True    | red,gre  | gre (q 11.1, lat 0.20 s) | tl 1 @ 2.28 m  |
| 12.1 | 0.1 |     1.47 | R1+R2   | True    | gre,red  | red (q 11.6, lat 0.19 s) | tl 1 @ 1.26 m  |
| 12.6 | 0.9 |     1.2  | R1+R2   | True    | red,red  | red (q 12.1, lat 0.20 s) | tl 1 @ 1.26 m  |
| 13.1 | 0   |     1.1  | R1+R2   | True    | red,red  | red (q 12.6, lat 0.20 s) | tl 2 @ 1.26 m  |
| 13.6 | 0.6 |     1.11 | R1+R2   | True    | red,red  | red (q 13.1, lat 0.20 s) | tl 2 @ 1.26 m  |
| 14.1 | 0.3 |     0.97 | R1+R2   | True    | red,red  | red (q 13.6, lat 0.20 s) | tl 2 @ 1.26 m  |
| 14.6 | 0.4 |     0.71 | R1+R2   | True    | red,red  | red (q 14.1, lat 0.20 s) | tl 2 @ 1.26 m  |
| 15.1 | 0.5 |     0.45 | R1+R2   | True    | red,gre  | gre (q 14.6, lat 0.21 s) | tl 2 @ 1.26 m  |
| 15.6 | 0.6 |     0.33 | R1      | True    | gre,gre  | gre (q 15.1, lat 0.20 s) | tl 2 @ 0.24 m  |
| 16.1 | 1.4 |    -0.17 | R1      | True    | gre,gre  | gre (q 15.6, lat 0.20 s) | tl 2 @ 0.24 m  |
| 16.6 | 2.3 |    -1.08 | -       | True    | gre,gre  | gre (q 16.1, lat 0.21 s) | tl 2 @ 0.24 m  |

Route 16390 seed 0, light 6923, front bumper at the stop line at t = 7.6 s (light green): the light turned red while the car was crossing (front 0.1 m before the entrance, 5.0 m/s, 0.5 s after its front passed the stop line); the car kept going

|   t |   v |   line_m | rules   | fresh   | last_K   | answer_in_force         | truth          |
|----:|----:|---------:|:--------|:--------|:---------|:------------------------|:---------------|
| 0.1 | 0   |     1.3  | R1      | False   |          | -                       | nan            |
| 0.6 | 0   |     1.3  | R1      | True    | gre      | gre (q 0.1, lat 0.21 s) | tl 0 @ 2.16 m  |
| 1.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 0.6, lat 0.20 s) | tl 0 @ 2.16 m  |
| 1.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 1.1, lat 0.21 s) | tl 0 @ 2.16 m  |
| 2.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 1.6, lat 0.20 s) | tl 0 @ 2.16 m  |
| 2.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 2.1, lat 0.20 s) | tl 0 @ 2.16 m  |
| 3.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 2.6, lat 0.20 s) | tl 0 @ 2.16 m  |
| 3.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 3.1, lat 0.20 s) | tl 0 @ 2.16 m  |
| 4.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 3.6, lat 0.19 s) | tl 0 @ 2.16 m  |
| 4.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 4.1, lat 0.20 s) | tl 0 @ 2.16 m  |
| 5.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 4.6, lat 0.21 s) | tl 0 @ 2.16 m  |
| 5.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 5.1, lat 0.20 s) | tl 0 @ 2.16 m  |
| 6.1 | 1.2 |     1.3  | R1      | True    | gre,gre  | gre (q 5.6, lat 0.19 s) | tl 0 @ 2.16 m  |
| 6.6 | 2.2 |     1.3  | R1      | True    | gre,gre  | gre (q 6.1, lat 0.21 s) | tl 0 @ 2.16 m  |
| 7.1 | 3.3 |     0.15 | R1      | True    | gre,gre  | gre (q 6.6, lat 0.21 s) | tl 0 @ 2.16 m  |
| 7.6 | 4.3 |    -1.79 | -       | True    | gre,gre  | gre (q 7.1, lat 0.23 s) | tl 0 @ 0.16 m  |
| 8.1 | 5   |    -4.27 | -       | True    | gre,red  | red (q 7.6, lat 0.24 s) | tl 0 @ -1.84 m |

Route 16390 seed 1, light 6912, front bumper at the stop line at t = 7.6 s (light green): the light turned red while the car was crossing (front 0.1 m before the entrance, 5.0 m/s, 0.5 s after its front passed the stop line); the car kept going

|   t |   v |   line_m | rules   | fresh   | last_K   | answer_in_force         | truth          |
|----:|----:|---------:|:--------|:--------|:---------|:------------------------|:---------------|
| 0.1 | 0   |     1.3  | R1      | False   |          | -                       | nan            |
| 0.6 | 0   |     1.3  | R1      | True    | gre      | gre (q 0.1, lat 0.21 s) | tl 0 @ 2.16 m  |
| 1.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 0.6, lat 0.19 s) | tl 0 @ 2.16 m  |
| 1.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 1.1, lat 0.20 s) | tl 0 @ 2.16 m  |
| 2.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 1.6, lat 0.20 s) | tl 0 @ 2.16 m  |
| 2.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 2.1, lat 0.20 s) | tl 0 @ 2.16 m  |
| 3.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 2.6, lat 0.19 s) | tl 0 @ 2.16 m  |
| 3.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 3.1, lat 0.23 s) | tl 0 @ 2.16 m  |
| 4.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 3.6, lat 0.21 s) | tl 0 @ 2.16 m  |
| 4.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 4.1, lat 0.20 s) | tl 0 @ 2.16 m  |
| 5.1 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 4.6, lat 0.21 s) | tl 0 @ 2.16 m  |
| 5.6 | 0   |     1.3  | R1      | True    | gre,gre  | gre (q 5.1, lat 0.20 s) | tl 0 @ 2.16 m  |
| 6.1 | 1.2 |     1.3  | R1      | True    | gre,gre  | gre (q 5.6, lat 0.22 s) | tl 0 @ 2.16 m  |
| 6.6 | 2.2 |     1.3  | R1      | True    | gre,gre  | gre (q 6.1, lat 0.23 s) | tl 0 @ 2.16 m  |
| 7.1 | 3.3 |     0.15 | R1      | True    | gre,gre  | gre (q 6.6, lat 0.19 s) | tl 0 @ 2.16 m  |
| 7.6 | 4.3 |    -1.78 | -       | True    | gre,gre  | gre (q 7.1, lat 0.21 s) | tl 0 @ 0.16 m  |
| 8.1 | 5   |    -4.26 | -       | True    | gre,red  | red (q 7.6, lat 0.21 s) | tl 0 @ -1.84 m |

Route 15612 seed 0, light 881, front bumper at the stop line at t = 16.7 s (light green): the light turned red 0.9 s after the front bumper had passed the stop line; the car stopped at t = 18.6 s past the line (R2 never starts there) and moved on at t = 20.6 s while the light was red

|    t |    v |   line_m | rules   | fresh   | last_K   | answer_in_force          | truth         |
|-----:|-----:|---------:|:--------|:--------|:---------|:-------------------------|:--------------|
|  8.1 |  0   |     1.95 | R1+R2   | True    | red,red  | red (q 7.6, lat 0.24 s)  | tl 2 @ 2.16 m |
|  8.6 |  0   |     1.95 | R1+R2   | True    | red,red  | red (q 8.1, lat 0.23 s)  | tl 2 @ 2.16 m |
|  9.1 |  0   |     1.95 | R1+R2   | True    | red,red  | red (q 8.6, lat 0.22 s)  | tl 2 @ 2.16 m |
|  9.6 |  0   |     1.95 | R1+R2   | True    | red,red  | red (q 9.1, lat 0.23 s)  | tl 2 @ 2.16 m |
| 10.1 |  0   |     1.95 | R1+R2   | True    | red,red  | red (q 9.6, lat 0.23 s)  | tl 2 @ 2.16 m |
| 10.6 |  0   |     1.95 | R1+R2   | True    | red,red  | red (q 10.1, lat 0.22 s) | tl 2 @ 2.16 m |
| 11.1 |  0   |     1.95 | R1+R2   | True    | red,red  | red (q 10.6, lat 0.22 s) | tl 2 @ 2.16 m |
| 11.6 |  0   |     1.95 | R1+R2   | True    | red,red  | red (q 11.1, lat 0.22 s) | tl 2 @ 2.16 m |
| 12.1 |  0.3 |     1.95 | R1+R2   | True    | red,red  | red (q 11.6, lat 0.21 s) | tl 2 @ 2.16 m |
| 12.6 |  0.9 |     1.94 | R1+R2   | True    | red,red  | red (q 12.1, lat 0.31 s) | tl 2 @ 2.16 m |
| 13.1 |  0.2 |     1.57 | R1+R2   | True    | red,red  | red (q 12.6, lat 0.22 s) | tl 2 @ 2.16 m |
| 13.6 |  0   |     1.63 | R1+R2   | True    | red,red  | red (q 13.1, lat 0.22 s) | tl 2 @ 2.16 m |
| 14.1 |  0   |     1.69 | R1+R2   | True    | red,red  | red (q 13.6, lat 0.31 s) | tl 2 @ 2.16 m |
| 14.6 | -0   |     1.59 | R1+R2   | True    | red,red  | red (q 14.1, lat 0.25 s) | tl 2 @ 2.16 m |
| 15.1 |  0.4 |     1.47 | R1+R2   | True    | red,gre  | gre (q 14.6, lat 0.37 s) | tl 0 @ 2.16 m |
| 15.6 |  0.9 |     1.19 | R1      | True    | gre,gre  | gre (q 15.1, lat 0.22 s) | tl 0 @ 2.16 m |
| 16.1 |  1.5 |     0.7  | R1      | True    | gre,gre  | gre (q 15.6, lat 0.45 s) | tl 0 @ 1.16 m |
| 16.6 |  2.1 |    -0.29 | R1      | True    | gre,gre  | gre (q 16.1, lat 0.22 s) | tl 0 @ 1.16 m |
| 17.1 |  2.5 |    -1.44 | -       | True    | gre,gre  | gre (q 16.6, lat 0.23 s) | tl 0 @ 0.16 m |

Route 15612 seed 1, light 881, front bumper at the stop line at t = 16.6 s (light green): the light turned red 1.0 s after the front bumper had passed the stop line; the car stopped at t = 19.1 s past the line (R2 never starts there) and moved on at t = 21.1 s while the light was red

|    t |   v |   line_m | rules   | fresh   | last_K   | answer_in_force          | truth         |
|-----:|----:|---------:|:--------|:--------|:---------|:-------------------------|:--------------|
|  8.1 | 0   |     1.95 | R1+R2   | True    | red,red  | red (q 7.6, lat 0.27 s)  | tl 2 @ 2.16 m |
|  8.6 | 0   |     1.95 | R1+R2   | True    | red,red  | red (q 8.1, lat 0.22 s)  | tl 2 @ 2.16 m |
|  9.1 | 0   |     1.95 | R1+R2   | True    | red,red  | red (q 8.6, lat 0.32 s)  | tl 2 @ 2.16 m |
|  9.6 | 0   |     1.95 | R1+R2   | True    | red,red  | red (q 9.1, lat 0.22 s)  | tl 2 @ 2.16 m |
| 10.1 | 0   |     1.95 | R1+R2   | True    | red,red  | red (q 9.6, lat 0.24 s)  | tl 2 @ 2.16 m |
| 10.6 | 0   |     1.95 | R1+R2   | True    | red,red  | red (q 10.1, lat 0.23 s) | tl 2 @ 2.16 m |
| 11.1 | 0.4 |     1.95 | R1+R2   | True    | red,red  | red (q 10.6, lat 0.25 s) | tl 2 @ 2.16 m |
| 11.6 | 1   |     1.95 | R1+R2   | True    | red,red  | red (q 11.1, lat 0.24 s) | tl 2 @ 2.16 m |
| 12.1 | 0.2 |     1.5  | R1+R2   | True    | red,red  | red (q 11.6, lat 0.23 s) | tl 2 @ 2.16 m |
| 12.6 | 0   |     1.34 | R1+R2   | True    | red,red  | red (q 12.1, lat 0.23 s) | tl 2 @ 1.16 m |
| 13.1 | 0   |     1.38 | R1+R2   | True    | red,red  | red (q 12.6, lat 0.22 s) | tl 2 @ 1.16 m |
| 13.6 | 0   |     1.45 | R1+R2   | True    | red,red  | red (q 13.1, lat 0.35 s) | tl 2 @ 1.16 m |
| 14.1 | 0.6 |     1.38 | R1+R2   | True    | red,red  | red (q 13.6, lat 0.31 s) | tl 2 @ 1.16 m |
| 14.6 | 0.6 |     1.04 | R1+R2   | True    | red,red  | red (q 14.1, lat 0.23 s) | tl 2 @ 1.16 m |
| 15.1 | 0.2 |     0.78 | R1+R2   | True    | red,gre  | gre (q 14.6, lat 0.23 s) | tl 0 @ 1.16 m |
| 15.6 | 0.5 |     0.67 | R1      | True    | gre,gre  | gre (q 15.1, lat 0.22 s) | tl 0 @ 1.16 m |
| 16.1 | 1.2 |     0.4  | R1      | True    | gre,gre  | gre (q 15.6, lat 0.23 s) | tl 0 @ 1.16 m |
| 16.6 | 1.9 |    -0.48 | R1      | True    | gre,gre  | gre (q 16.1, lat 0.21 s) | tl 0 @ 0.16 m |
| 17.1 | 2.5 |    -1.59 | -       | True    | gre,gre  | gre (q 16.6, lat 0.26 s) | tl 0 @ 0.16 m |

Route 15483 seed 0, light 1462, front bumper at the stop line at t = 32.1 s (light green): the light turned red while the car was crossing (front 1.4 m before the entrance, 3.0 m/s, 0.5 s after its front passed the stop line); the car kept going

|    t |    v |   line_m | rules   | fresh   | last_K   | answer_in_force          | truth         |
|-----:|-----:|---------:|:--------|:--------|:---------|:-------------------------|:--------------|
| 23.6 |  0   |     0.14 | R1+R2   | True    | red,red  | red (q 23.1, lat 0.24 s) | tl 2 @ 0.16 m |
| 24.1 |  0   |     0.14 | R1+R2   | True    | red,red  | red (q 23.6, lat 0.24 s) | tl 2 @ 0.16 m |
| 24.6 |  0   |     0.11 | R1+R2   | True    | red,red  | red (q 24.1, lat 0.24 s) | tl 2 @ 0.16 m |
| 25.1 |  0   |     0.12 | R1+R2   | True    | red,red  | red (q 24.6, lat 0.22 s) | tl 2 @ 0.16 m |
| 25.6 |  0   |     0.12 | R1+R2   | True    | red,red  | red (q 25.1, lat 0.22 s) | tl 2 @ 0.16 m |
| 26.1 |  0   |     0.06 | R1+R2   | True    | red,red  | red (q 25.6, lat 0.23 s) | tl 2 @ 0.16 m |
| 26.6 |  0   |     0.2  | R1+R2   | True    | red,red  | red (q 26.1, lat 0.24 s) | tl 2 @ 0.16 m |
| 27.1 |  0   |     0.28 | R1+R2   | True    | red,red  | red (q 26.6, lat 0.22 s) | tl 2 @ 0.16 m |
| 27.6 |  0.1 |     0.17 | R1+R2   | True    | red,red  | red (q 27.1, lat 0.20 s) | tl 2 @ 0.16 m |
| 28.1 |  0   |     0.09 | R1+R2   | True    | red,red  | red (q 27.6, lat 0.22 s) | tl 2 @ 0.16 m |
| 28.6 |  0   |     0.16 | R1+R2   | True    | red,red  | red (q 28.1, lat 0.23 s) | tl 2 @ 0.16 m |
| 29.1 |  0   |     0.21 | R1+R2   | True    | red,red  | red (q 28.6, lat 0.23 s) | tl 2 @ 0.16 m |
| 29.6 |  0   |     0.18 | R1+R2   | True    | red,red  | red (q 29.1, lat 0.24 s) | tl 2 @ 0.16 m |
| 30.1 |  0   |     0.08 | R1+R2   | True    | red,gre  | gre (q 29.6, lat 0.23 s) | tl 0 @ 0.16 m |
| 30.6 | -0   |     0.01 | R1      | True    | gre,gre  | gre (q 30.1, lat 0.19 s) | tl 0 @ 0.16 m |
| 31.1 |  0.6 |    -0.06 | R1      | True    | gre,gre  | gre (q 30.6, lat 0.21 s) | tl 0 @ 0.16 m |
| 31.6 |  1.5 |    -0.6  | -       | True    | gre,gre  | gre (q 31.1, lat 0.21 s) | tl 0 @ 0.16 m |
| 32.1 |  2.3 |    -1.47 | -       | True    | gre,gre  | gre (q 31.6, lat 0.21 s) | tl 0 @ 0.16 m |
| 32.6 |  3   |    -2.9  | -       | True    | gre,gre  | gre (q 32.1, lat 0.22 s) | tl 0 @ 0.16 m |

Route 15483 seed 1, light 1837, front bumper at the stop line at t = 32.1 s (light green): the light turned red while the car was crossing (front 1.4 m before the entrance, 2.9 m/s, 0.5 s after its front passed the stop line); the car kept going

|    t |   v |   line_m | rules   | fresh   | last_K   | answer_in_force          | truth          |
|-----:|----:|---------:|:--------|:--------|:---------|:-------------------------|:---------------|
| 23.6 | 0   |     0.15 | R1+R2   | True    | red,red  | red (q 23.1, lat 0.38 s) | tl 2 @ 0.16 m  |
| 24.1 | 0   |     0.15 | R1+R2   | True    | red,red  | red (q 23.6, lat 0.25 s) | tl 2 @ 0.16 m  |
| 24.6 | 0   |     0.12 | R1+R2   | True    | red,red  | red (q 24.1, lat 0.22 s) | tl 2 @ 0.16 m  |
| 25.1 | 0   |     0.13 | R1+R2   | True    | red,red  | red (q 24.6, lat 0.24 s) | tl 2 @ 0.16 m  |
| 25.6 | 0   |     0.13 | R1+R2   | True    | red,red  | red (q 25.1, lat 0.21 s) | tl 2 @ 0.16 m  |
| 26.1 | 0   |     0.07 | R1+R2   | True    | red,red  | red (q 25.6, lat 0.26 s) | tl 2 @ 0.16 m  |
| 26.6 | 0   |     0.21 | R1+R2   | True    | red,red  | red (q 26.1, lat 0.21 s) | tl 2 @ 0.16 m  |
| 27.1 | 0   |     0.29 | R1+R2   | True    | red,red  | red (q 26.6, lat 0.21 s) | tl 2 @ 0.16 m  |
| 27.6 | 0   |     0.16 | R1+R2   | True    | red,red  | red (q 27.1, lat 0.21 s) | tl 2 @ 0.16 m  |
| 28.1 | 0   |     0.06 | R1+R2   | True    | red,red  | red (q 27.6, lat 0.22 s) | tl 2 @ 0.16 m  |
| 28.6 | 0   |     0.14 | R1+R2   | True    | red,red  | red (q 28.1, lat 0.34 s) | tl 2 @ 0.16 m  |
| 29.1 | 0   |     0.18 | R1+R2   | True    | red,red  | red (q 28.6, lat 0.22 s) | tl 2 @ 0.16 m  |
| 29.6 | 0   |     0.16 | R1+R2   | True    | red,red  | red (q 29.1, lat 0.21 s) | tl 2 @ 0.16 m  |
| 30.1 | 0   |     0.06 | R1+R2   | True    | red,gre  | gre (q 29.6, lat 0.22 s) | tl 0 @ 0.16 m  |
| 30.6 | 0   |    -0.01 | R1      | True    | gre,gre  | gre (q 30.1, lat 0.20 s) | tl 0 @ 0.16 m  |
| 31.1 | 0.6 |    -0.11 | R1      | True    | gre,gre  | gre (q 30.6, lat 0.21 s) | tl 0 @ 0.16 m  |
| 31.6 | 1.5 |    -0.66 | -       | True    | gre,gre  | gre (q 31.1, lat 0.21 s) | tl 0 @ 0.16 m  |
| 32.1 | 2.3 |    -1.55 | -       | True    | gre,gre  | gre (q 31.6, lat 0.21 s) | tl 0 @ 0.16 m  |
| 32.6 | 2.9 |    -2.96 | -       | True    | gre,gre  | gre (q 32.1, lat 0.22 s) | tl 0 @ -1.84 m |

## Per route (DS and red-light counts: means / sums over the two seeds; light: yes = scenario set, * = ego light seen in the vred3 logs; `_A` columns are vred3)

|   route | light   |   DS_drive |   DS_pred |   DS_vred |   DS_vred2 |   DS_vred3 |   DS_jslow |   dDS_vs_drive |   red_drive |   red_pred |   red_vred |   red_vred2 |   red_vred3 |   red_jslow |   R2_stops_s0 |   R2_stops_s1 |   blocked_A |   coll_drive |   coll_A |   v_A |
|--------:|:--------|-----------:|----------:|----------:|-----------:|-----------:|-----------:|---------------:|------------:|-----------:|-----------:|------------:|------------:|------------:|--------------:|--------------:|------------:|-------------:|---------:|------:|
|   27043 | yes*    |       60   |      60   |      60   |       60   |       60   |       60   |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             1 |             1 |           0 |            2 |        2 |  3.16 |
|   15102 | yes*    |      100   |     100   |     100   |      100   |      100   |      100   |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             1 |             1 |           0 |            0 |        0 |  1.07 |
|   24944 | *       |       70   |     100   |     100   |      100   |      100   |       70   |           30   |           2 |          0 |          0 |           0 |           0 |           2 |             2 |             1 |           0 |            0 |        0 |  1.26 |
|   27870 | *       |       85   |     100   |     100   |      100   |      100   |       85   |           15   |           1 |          0 |          0 |           0 |           0 |           1 |             1 |             1 |           0 |            0 |        0 |  2.41 |
|   22535 |         |      100   |     100   |     100   |      100   |      100   |      100   |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             0 |             0 |           0 |            0 |        0 |  3.08 |
|   37969 |         |       60   |      23.6 |      59   |       34.6 |       11.3 |       34.5 |          -48.7 |           0 |          0 |          0 |           0 |           0 |           0 |             0 |             0 |           0 |            2 |        2 |  0.2  |
|   24497 |         |       21.2 |      21.7 |      21.2 |       21.7 |       21.2 |       21.7 |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             0 |             0 |           0 |            2 |        2 |  0.23 |
|   27297 | *       |       70   |     100   |      70   |       43.7 |       85   |       70   |           15   |           2 |          0 |          2 |           1 |           1 |           2 |             1 |             1 |           0 |            0 |        0 |  2.45 |
|    9196 | *       |       28.4 |      34   |      27.4 |       26.9 |       26.4 |       42   |           -2   |           2 |          0 |          0 |           0 |           0 |           2 |             1 |             1 |           1 |            3 |        4 |  0.57 |
|   28147 | yes*    |      100   |     100   |     100   |      100   |      100   |      100   |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             1 |             2 |           0 |            0 |        0 |  1.58 |
|   16390 | yes*    |       70   |      70   |      70   |       70   |       70   |       70   |            0   |           2 |          2 |          2 |           2 |           2 |           2 |             0 |             0 |           0 |            0 |        0 |  4.15 |
|   15612 | yes*    |       47.4 |      70   |      85   |       70   |       70   |      100   |           22.6 |           2 |          2 |          1 |           2 |           2 |           0 |             1 |             1 |           0 |            0 |        0 |  1.42 |
|   15483 | yes*    |      100   |      85   |     100   |       70   |       70   |      100   |          -30   |           0 |          1 |          0 |           2 |           2 |           0 |             1 |             1 |           0 |            0 |        0 |  0.99 |
|   17280 |         |       80   |      80   |      80   |       80   |       80   |       80   |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             0 |             0 |           0 |            0 |        0 |  4.96 |
|   16529 | *       |       70   |     100   |      85   |      100   |      100   |       70   |           30   |           2 |          0 |          1 |           0 |           0 |           2 |             1 |             2 |           0 |            0 |        0 |  1.07 |
|   16508 | *       |      100   |     100   |     100   |      100   |      100   |      100   |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             0 |             0 |           0 |            0 |        0 |  3.2  |
|   19324 |         |       33.4 |      33.4 |      33.4 |       30.3 |       33.4 |       32.6 |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             0 |             0 |           0 |            0 |        0 |  0.23 |
|    2520 |         |       23.5 |      23.5 |      23.5 |       23.5 |       23.5 |       23.5 |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             0 |             0 |           0 |            2 |        2 |  0.25 |
|   19832 |         |       33.1 |      33.1 |      33.1 |       33.1 |       33.1 |       33.1 |            0   |           0 |          0 |          0 |           0 |           0 |           0 |             0 |             0 |           0 |            0 |        0 |  0.23 |

## Yellow encounters (every decision of the rule, `k = y` lines)

`d_stop_front` = front bumper to the stop line, `d_line` = front bumper to the scorer's line (junction entrance), `age` = time since the frame of the last green answer, `remaining` = yellow time left after the age and the lag margin, `stop_need` = reaction + comfortable braking distance at the speed; `truth` = the simulator's ego light at the decision (evaluation only). Outcome: infraction = an official red-light event within 15 s; cleared = the front bumper 5.9 m past the junction entrance (tail clear) and the light then; stopped = the last standstill of the R2 hold against the stop line (+ = short of it).

|   route |   seed |    t |    v |   d_stop_front |   d_line |   age |   remaining |   stop_need | truth            | decision   | second_answer   | outcome                                              |
|--------:|-------:|-----:|-----:|---------------:|---------:|------:|------------:|------------:|:-----------------|:-----------|:----------------|:-----------------------------------------------------|
|   24944 |      1 | 39.9 | 0    |          17.22 |    27.35 |  0.85 |        1.65 |        0    | tl 1 @ 17.16 m   | stop       | confirm         | stopped before the stop line (0.2 m)                 |
|   27297 |      0 |  9.9 | 4.43 |           5.25 |     9.17 |  0.85 |        1.65 |        5.48 | tl 1 @ 5.34 m    | stop_hard  | confirm         | stopped before the stop line (0.2 m)                 |
|   27297 |      1 |  9.9 | 3.5  |           5.88 |     9.79 |  0.85 |        1.65 |        3.79 | tl 1 @ 5.34 m    | stop       | confirm         | infraction (red light at t = 16.3)                   |
|   28147 |      0 | 33.9 | 4.44 |          -4.16 |    -4.16 |  0.85 |        1.65 |        5.5  | tl None @ None m | go         | -               | cleared before red (tail clear at t = 34.6, light ?) |
|   28147 |      1 |  8.9 | 3.49 |           7.56 |    13.4  |  0.85 |        1.65 |        3.78 | tl 0 @ 8.32 m    | stop       | cancel          | tentative stop cancelled by the second answer        |
|   28147 |      1 | 30.9 | 3.24 |          -4.66 |    -4.66 |  0.85 |        1.65 |        3.38 | tl None @ None m | go         | -               | cleared before red (tail clear at t = 32.1, light ?) |
|   16529 |      1 |  7.9 | 1.21 |           1.68 |     5.83 |  0.85 |        1.65 |        0.85 | tl 2 @ 2.16 m    | stop       | confirm         | stopped before the stop line (0.2 m)                 |

Decisions: {'stop': 4, 'go': 2, 'stop_hard': 1}. Outcomes: {'stopped before the stop line': 3, 'cleared before red': 2, 'infraction': 1, 'tentative stop cancelled by the second answer': 1}.

## Tentative starts, false stops and collisions

Tentative R2 starts at a first non-green answer: confirmed by the second answer 4, cancelled 1. R2 stops whose hold saw no red or yellow truth state (false stops): 1:

|   route |   seed |   t0 |   hold_s | truth_states   |
|--------:|-------:|-----:|---------:|:---------------|
|   28147 |      1 |  8.9 |      0.5 | ['0']          |.

Blocked events / collisions summed over the 38 runs: vred3 1 / 12; vred2 2 / 14; jslow 0 / 10; drive 2 / 11.

Collisions of vred3 with a vehicle (`along` < 0: the other vehicle was behind the ego at first contact = a rear-end hit on the ego; ego speed in m/s):

|   route |   seed |    t | actor                         |   along |   ego_v |
|--------:|-------:|-----:|:------------------------------|--------:|--------:|
|   27043 |      0 | 19.3 | vehicle.mini.cooper_s_2021    |    -1.8 |     4   |
|   27043 |      1 | 19.3 | vehicle.mini.cooper_s_2021    |    -2.1 |     4   |
|    9196 |      0 | 35.7 | vehicle.carlamotors.firetruck |    -1.5 |     4.4 |
|    9196 |      0 | 36.3 | vehicle.lincoln.mkz_2017      |     5.5 |     8   |
|    9196 |      1 | 35.2 | vehicle.carlamotors.firetruck |    -1.9 |    16.1 |
|    9196 |      1 | 35.7 | vehicle.lincoln.mkz_2017      |     5.5 |    11.6 |

## Registered lines (plan 4.3), vred3

| line | read | status |
|:--|:--|:--|
| harmless on routes without a light: DS CI lower bound >= -5, no new blocked, no new collision | DS -6.96 [-20.87, +0.00]; blocked +0, collisions +0 | not evaluated at this size (7 routes < 30) |
| useful on routes with a light: red-light infractions fall, DS CI lower bound > 0 | DS +6.72 [-2.67, +15.47]; red light -6 runs | not evaluated at this size (12 routes < 30) |

![paired differences](vred3_paired.png)

Figure: paired differences for vred3 against drive, vred2, jslow, vred, pred per route set (dot: mean over routes, bar: 95% route-cluster CI), DS on the left and red-light infractions per run on the right. Look at whether vred3 moved against the previous arms and whether any bar clears zero.

![stop position](vred3_stop_position.png)

Figure: distance of the front bumper to the stop line at standstill for every red-light stop (dots; ticks below the dots: the smallest distance during the hold). Look at which side of the zero line the dots sit: vred sat beyond it, the stop-line arms should sit just short of it, like pred.

![in-batch latency](vred3_latency.png)

Figure: cumulative distribution of the in-batch answer latency with L = 0.35 s and the TTL. Look at how much of the curve lies right of L.
