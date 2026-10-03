# vmerge3: the bypass without privileged input + a cross-traffic release check (19 routes, diagnostic)

Plan: [../plans/2026-10-04-vmerge3.md](../plans/2026-10-04-vmerge3.md). `drive` and `vmerge2` are the logged runs of the earlier batches (drive seeds 2, 3 = the reruns of the vmerge2 batch). Arms: `vmerge3` = vmerge2 + perceived bypass + release check; `vm3priv` = vmerge2 + release check (privileged bypass); `vm3norel` = vmerge2 + perceived bypass.

## Missing runs

none: every registered run finished

## Registered lines (seeds 0 and 1)

- Bypass: vmerge3 - drive on the 4 obstacle routes = +12.88 [+3.56, +24.34] DS; line >= +20.7: **no**
- Collisions: vehicle collisions per run vmerge3 0.158 (6 / 38) vs drive 0.158 (6 / 38): **yes**

Vehicle collisions per run, seeds 0 and 1: drive 0.158, vmerge2 0.289, vmerge3 0.158, vm3priv 0.263, vm3norel 0.237

## Arms, seeds 0 and 1

| arm      |   runs |   crashes |    DS |    RC |   red_light |   stop_sign |   collisions |   blocked |   timeouts |   v_mean |
|:---------|-------:|----------:|------:|------:|------------:|------------:|-------------:|----------:|-----------:|---------:|
| drive    |     38 |         0 | 65.9  | 83.27 |          13 |           2 |           11 |         2 |          0 |     2.2  |
| vmerge2  |     38 |         0 | 76.67 | 92.04 |           5 |           0 |           16 |         4 |          0 |     2.17 |
| vmerge3  |     38 |         0 | 68.3  | 82.91 |           8 |           0 |           14 |         3 |          0 |     1.62 |
| vm3priv  |     38 |         0 | 75.38 | 93.83 |           9 |           0 |           15 |         3 |          0 |     1.99 |
| vm3norel |     38 |         0 | 70.61 | 84.02 |           5 |           0 |           15 |         2 |          0 |     1.73 |

## Arms, every finished seed

| arm      |   runs |   crashes |    DS |    RC |   red_light |   stop_sign |   collisions |   blocked |   timeouts |   v_mean |
|:---------|-------:|----------:|------:|------:|------------:|------------:|-------------:|----------:|-----------:|---------:|
| drive    |     76 |         0 | 66.86 | 83.99 |          25 |           4 |           21 |         3 |          0 |     2.21 |
| vmerge2  |     76 |         0 | 76.08 | 90.56 |          10 |           0 |           29 |         8 |          0 |     2.09 |
| vmerge3  |     38 |         0 | 68.3  | 82.91 |           8 |           0 |           14 |         3 |          0 |     1.62 |
| vm3priv  |     38 |         0 | 75.38 | 93.83 |           9 |           0 |           15 |         3 |          0 |     1.99 |
| vm3norel |     38 |         0 | 70.61 | 84.02 |           5 |           0 |           15 |         2 |          0 |     1.73 |

## Infractions by type (official, summed over runs, every finished seed)

| arm      |   red_light |   stop_infraction |   collisions_vehicle |   collisions_layout |   collisions_pedestrian |   outside_route_lanes |   vehicle_blocked |   route_timeout |   scenario_timeouts |
|:---------|------------:|------------------:|---------------------:|--------------------:|------------------------:|----------------------:|------------------:|----------------:|--------------------:|
| drive    |          25 |                 4 |                   12 |                   9 |                       0 |                     4 |                 3 |               0 |                   0 |
| vmerge2  |          10 |                 0 |                   21 |                   8 |                       0 |                    10 |                 8 |               0 |                   0 |
| vmerge3  |           8 |                 0 |                    6 |                   8 |                       0 |                     7 |                 3 |               0 |                   0 |
| vm3priv  |           9 |                 0 |                   10 |                   5 |                       0 |                     5 |                 3 |               0 |                   0 |
| vm3norel |           5 |                 0 |                    9 |                   6 |                       0 |                     6 |                 2 |               0 |                   0 |

## Paired differences, seeds 0 and 1 (registered primary; mean over routes [95% route-cluster CI])

| contrast | routes | n routes | DS | RC | red_light | stop_infraction | collisions_vehicle | collisions | vehicle_blocked |
|:--|:--|--:|:--|:--|:--|:--|:--|:--|:--|
| vmerge3 - drive | all | 19 | +2.40 [-6.29, +9.87] | -0.36 [-12.05, +9.12] | -0.13 [-0.32, +0.05] | -0.05 [-0.16, +0.00] | +0.00 [-0.13, +0.11] | +0.08 [+0.00, +0.16] | +0.03 [+0.00, +0.08] |
| vmerge3 - drive | obstacle | 4 | +12.88 [+3.56, +24.34] | +24.29 [+8.55, +32.84] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.12 [+0.00, +0.38] | +0.25 [+0.00, +0.50] | +0.00 [+0.00, +0.00] |
| vmerge3 - drive | light | 6 | -2.77 [-15.00, +6.68] | -0.39 [-1.17, +0.00] | +0.08 [-0.25, +0.50] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| vmerge3 - drive | stop sign | 3 | +5.00 [+0.00, +15.00] | +0.00 [+0.00, +0.00] | -0.17 [-0.50, +0.00] | -0.33 [-1.00, +0.00] | +0.17 [+0.00, +0.50] | +0.17 [+0.00, +0.50] | +0.00 [+0.00, +0.00] |
| vmerge3 - drive | other | 6 | -0.71 [-22.00, +15.00] | -16.94 [-43.82, +0.00] | -0.42 [-0.75, -0.08] | +0.00 [+0.00, +0.00] | -0.17 [-0.50, +0.00] | +0.00 [+0.00, +0.00] | +0.08 [+0.00, +0.25] |
| vmerge3 - vmerge2 | all | 19 | -8.37 [-17.91, -0.71] | -9.13 [-19.76, +0.04] | +0.08 [+0.00, +0.21] | +0.00 [+0.00, +0.00] | -0.13 [-0.26, -0.03] | -0.05 [-0.24, +0.13] | -0.03 [-0.08, +0.00] |
| vmerge3 - vmerge2 | obstacle | 4 | -31.94 [-54.28, -9.61] | -41.89 [-58.31, -32.68] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.50 [-0.88, -0.12] | +0.00 [-0.75, +0.75] | +0.00 [+0.00, +0.00] |
| vmerge3 - vmerge2 | light | 6 | -5.18 [-15.18, +0.00] | -0.26 [-0.78, +0.00] | +0.17 [+0.00, +0.50] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| vmerge3 - vmerge2 | stop sign | 3 | +2.20 [-15.00, +21.60] | +12.00 [+0.00, +35.99] | +0.17 [+0.00, +0.50] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.17 [-0.50, +0.00] |
| vmerge3 - vmerge2 | other | 6 | -1.13 [-6.57, +3.19] | -6.72 [-20.16, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.08 [-0.25, +0.00] | -0.17 [-0.33, +0.00] | +0.00 [+0.00, +0.00] |
| vmerge3 - vm3priv | all | 19 | -7.08 [-17.99, +1.12] | -10.92 [-21.52, -1.61] | -0.03 [-0.08, +0.00] | +0.00 [+0.00, +0.00] | -0.11 [-0.24, +0.00] | -0.03 [-0.18, +0.16] | +0.00 [-0.08, +0.08] |
| vmerge3 - vm3priv | obstacle | 4 | -32.24 [-63.59, -6.36] | -41.89 [-58.31, -32.68] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.38 [-0.75, +0.00] | +0.00 [-0.75, +0.75] | +0.00 [+0.00, +0.00] |
| vmerge3 - vm3priv | light | 6 | -4.04 [-12.12, +0.00] | -5.77 [-17.31, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.08 [+0.00, +0.25] |
| vmerge3 - vm3priv | stop sign | 3 | +12.01 [+0.00, +21.02] | +11.67 [+0.00, +35.02] | -0.17 [-0.50, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.17 [-0.50, +0.00] |
| vmerge3 - vm3priv | other | 6 | -2.89 [-11.66, +2.98] | -6.72 [-20.16, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.08 [-0.25, +0.00] | -0.08 [-0.25, +0.00] | +0.00 [+0.00, +0.00] |
| vmerge3 - vm3norel | all | 19 | -2.31 [-10.13, +4.53] | -1.11 [-12.91, +8.78] | +0.08 [+0.00, +0.21] | +0.00 [+0.00, +0.00] | -0.08 [-0.21, +0.05] | -0.03 [-0.13, +0.08] | +0.03 [-0.05, +0.11] |
| vmerge3 - vm3norel | obstacle | 4 | +6.24 [+1.75, +10.74] | +15.01 [-2.65, +32.68] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [-0.38, +0.38] | +0.12 [-0.25, +0.50] | +0.00 [+0.00, +0.00] |
| vmerge3 - vm3norel | light | 6 | -9.04 [-19.04, +0.00] | -5.77 [-17.31, +0.00] | +0.17 [+0.00, +0.50] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.08 [+0.00, +0.25] |
| vmerge3 - vm3norel | stop sign | 3 | +1.81 [-15.00, +20.43] | +11.35 [+0.00, +34.05] | +0.17 [+0.00, +0.50] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.17 [-0.50, +0.00] |
| vmerge3 - vm3norel | other | 6 | -3.35 [-22.17, +10.00] | -13.44 [-40.33, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.25 [-0.58, +0.00] | -0.17 [-0.33, +0.00] | +0.08 [+0.00, +0.25] |
| vm3priv - drive | all | 19 | +9.48 [-1.56, +21.50] | +10.56 [-3.05, +25.91] | -0.11 [-0.29, +0.08] | -0.05 [-0.16, +0.00] | +0.11 [-0.03, +0.26] | +0.11 [-0.08, +0.26] | +0.03 [-0.05, +0.11] |
| vm3priv - drive | obstacle | 4 | +45.12 [+27.42, +66.14] | +66.18 [+64.60, +67.18] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.50 [+0.12, +0.88] | +0.25 [-0.62, +0.88] | +0.00 [+0.00, +0.00] |
| vm3priv - drive | light | 6 | +1.26 [-15.00, +18.79] | +5.38 [+0.00, +16.13] | +0.08 [-0.25, +0.50] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.08 [-0.25, +0.00] |
| vm3priv - drive | stop sign | 3 | -7.00 [-21.02, +0.00] | -11.67 [-35.02, +0.00] | +0.00 [+0.00, +0.00] | -0.33 [-1.00, +0.00] | +0.17 [+0.00, +0.50] | +0.17 [+0.00, +0.50] | +0.17 [+0.00, +0.50] |
| vm3priv - drive | other | 6 | +2.19 [-11.33, +15.39] | -10.22 [-23.66, +0.00] | -0.42 [-0.75, -0.08] | +0.00 [+0.00, +0.00] | -0.08 [-0.25, +0.00] | +0.08 [+0.00, +0.25] | +0.08 [+0.00, +0.25] |
| vm3norel - drive | all | 19 | +4.71 [-2.02, +11.93] | +0.75 [-5.71, +7.22] | -0.21 [-0.39, -0.05] | -0.05 [-0.16, +0.00] | +0.08 [+0.00, +0.16] | +0.11 [+0.03, +0.21] | +0.00 [-0.08, +0.08] |
| vm3norel - drive | obstacle | 4 | +6.63 [+0.00, +19.53] | +9.28 [+0.00, +27.26] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.12 [+0.00, +0.38] | +0.12 [+0.00, +0.38] | +0.00 [+0.00, +0.00] |
| vm3norel - drive | light | 6 | +6.26 [+0.00, +18.79] | +5.38 [+0.00, +16.13] | -0.08 [-0.25, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.08 [-0.25, +0.00] |
| vm3norel - drive | stop sign | 3 | +3.19 [-20.43, +30.00] | -11.35 [-34.05, +0.00] | -0.33 [-1.00, +0.00] | -0.33 [-1.00, +0.00] | +0.17 [+0.00, +0.50] | +0.17 [+0.00, +0.50] | +0.17 [+0.00, +0.50] |
| vm3norel - drive | other | 6 | +2.64 [-9.32, +15.98] | -3.50 [-10.49, +0.00] | -0.42 [-0.75, -0.08] | +0.00 [+0.00, +0.00] | +0.08 [+0.00, +0.25] | +0.17 [+0.00, +0.33] | +0.00 [+0.00, +0.00] |
| vmerge2 - drive | all | 19 | +10.77 [+0.29, +22.18] | +8.77 [-4.76, +24.37] | -0.21 [-0.39, -0.05] | -0.05 [-0.16, +0.00] | +0.13 [-0.03, +0.32] | +0.13 [-0.05, +0.34] | +0.05 [+0.00, +0.13] |
| vmerge2 - drive | obstacle | 4 | +44.82 [+25.27, +58.89] | +66.18 [+64.60, +67.18] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.62 [+0.12, +1.25] | +0.25 [-0.50, +1.00] | +0.00 [+0.00, +0.00] |
| vmerge2 - drive | light | 6 | +2.41 [+0.00, +7.23] | -0.13 [-0.39, +0.00] | -0.08 [-0.25, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| vmerge2 - drive | stop sign | 3 | +2.80 [-21.60, +30.00] | -12.00 [-35.99, +0.00] | -0.33 [-1.00, +0.00] | -0.33 [-1.00, +0.00] | +0.17 [+0.00, +0.50] | +0.17 [+0.00, +0.50] | +0.17 [+0.00, +0.50] |
| vmerge2 - drive | other | 6 | +0.42 [-16.49, +15.00] | -10.22 [-23.66, +0.00] | -0.42 [-0.75, -0.08] | +0.00 [+0.00, +0.00] | -0.08 [-0.25, +0.00] | +0.17 [+0.00, +0.33] | +0.08 [+0.00, +0.25] |

13 routes with 2-4 identical `drive` runs: per-route DS standard deviation mean 4.7, median 0.0, max 30.5; 5 of 13 routes changed DS between repeats (results/report.md)

## Latency (vmerge3, every answered request)

n 4169, answered 100.0%, p50 / p95 / p99 384 / 1088 / 2004 ms (cross 0 requests, p95 nan ms)

## Perceived bypass against the logged truth (vmerge3 and vm3norel; truth = vmerge2's privileged pbyp3 extent / offset on the same route)

|   route |   seed | obstacle   | activated   |   t_act | side   |   offset |   t_offset |   start_s |   end_s |   t_start_s |   t_end_s |   gap_hold_snaps |   t_pullout |   path_on_snaps | gap_reasons                 | arm      |
|--------:|-------:|:-----------|:------------|--------:|:-------|---------:|-----------:|----------:|--------:|------------:|----------:|-----------------:|------------:|----------------:|:----------------------------|:---------|
|   19324 |      0 | True       | True        |   17.05 | left   |    -3.38 |      -3.25 |     43.97 |   69.75 |       49.44 |     54.68 |              617 |       24.6  |            2717 | behind;side_memory          | vmerge3  |
|    2520 |      0 | True       | True        |   17.25 | left   |    -3.4  |      -3.25 |     44.39 |   87.18 |       50.43 |     65.23 |              631 |       24.35 |            2770 | behind;side_memory          | vmerge3  |
|   27043 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   27870 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   17280 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   19832 |      0 | True       | True        |   31.65 | left   |    -3.41 |      -3.25 |     48.42 |   65.08 |       49.62 |     54.29 |              836 |       44    |               0 | behind;side_memory          | vmerge3  |
|   24944 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   15612 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   22535 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   15102 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   27297 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   16390 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   24497 |      0 | True       | True        |   18.25 | left   |    -3.5  |      -3.5  |     42.46 |   80.6  |       46.46 |     61.06 |              693 |      134.35 |            1839 | behind;side_memory          | vmerge3  |
|   37969 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|    9196 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   28147 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   15483 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   16529 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   16508 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   19324 |      1 | True       | True        |   35.65 | left   |    -3.61 |      -3.25 |     48.83 |   73.41 |       49.44 |     54.68 |               67 |       46.8  |             293 | behind;oncoming;side_memory | vmerge3  |
|    2520 |      1 | True       | True        |   17.25 | left   |    -3.36 |      -3.25 |     43.3  |   72.55 |       50.43 |     65.23 |              106 |       24.35 |             358 | behind;oncoming;side_memory | vmerge3  |
|   27043 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   27870 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   17280 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   19832 |      1 | True       | True        |   33.05 | left   |    -3.43 |      -3.25 |     49.34 |   67.95 |       49.62 |     54.29 |              607 |       44    |             254 | behind;oncoming;side_memory | vmerge3  |
|   24944 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   15612 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   22535 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   15102 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   27297 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   16390 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   24497 |      1 | True       | True        |   18.25 | left   |    -3.52 |      -3.5  |     43.26 |   78.24 |       46.46 |     61.06 |              905 |      nan    |               0 | behind;side_memory          | vmerge3  |
|   37969 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|    9196 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   28147 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   15483 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   16529 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   16508 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vmerge3  |
|   19324 |      0 | True       | True        |   16.85 | left   |    -3.41 |      -3.25 |     43.13 |   69.97 |       49.44 |     54.68 |              163 |       24.35 |            3120 | behind;side_memory          | vm3norel |
|    2520 |      0 | True       | True        |   17.05 | left   |    -3.35 |      -3.25 |     42.01 |   87.97 |       50.43 |     65.23 |              897 |       24.35 |               0 | behind;side_memory          | vm3norel |
|   27043 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   27870 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   17280 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   19832 |      0 | True       | True        |   36.05 | left   |    -3.46 |      -3.25 |     48.86 |   65.9  |       49.62 |     54.29 |              816 |       69.6  |               0 | behind;side_memory          | vm3norel |
|   24944 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   15612 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   22535 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   15102 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   27297 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   16390 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   24497 |      0 | True       | True        |   18.45 | left   |    -3.53 |      -3.5  |     42.57 |   76.68 |       46.46 |     61.06 |              732 |      136.55 |            1565 | behind;side_memory          | vm3norel |
|   37969 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|    9196 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   28147 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   15483 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   16529 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   16508 |      0 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   19324 |      1 | True       | True        |   17.05 | left   |    -3.4  |      -3.25 |     42.03 |   69.44 |       49.44 |     54.68 |              156 |       24.6  |             273 | behind;oncoming;side_memory | vm3norel |
|    2520 |      1 | True       | True        |   17.25 | left   |    -3.36 |      -3.25 |     43.81 |   85.99 |       50.43 |     65.23 |              901 |       24.35 |               0 | behind;side_memory          | vm3norel |
|   27043 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   27870 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   17280 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   19832 |      1 | True       | True        |   34.05 | left   |    -3.42 |      -3.25 |     48.85 |   65.65 |       49.62 |     54.29 |              818 |      136.55 |               0 | behind;side_memory          | vm3norel |
|   24944 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   15612 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   22535 |      1 | False      | True        |   18.65 | left   |    -3.59 |     nan    |     57.52 |   81.31 |      nan    |    nan    |                9 |       18.5  |             291 | behind;side_memory          | vm3norel |
|   15102 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   27297 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   16390 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   24497 |      1 | True       | True        |   33.25 | left   |    -3.54 |      -3.5  |     46.51 |   79.63 |       46.46 |     61.06 |              657 |      136.55 |            1527 | behind;side_memory          | vm3norel |
|   37969 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|    9196 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   28147 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   15483 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   16529 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |
|   16508 |      1 | False      | False       |  nan    | nan    |   nan    |     nan    |    nan    |  nan    |      nan    |    nan    |                0 |      nan    |               0 |                             | vm3norel |

## Release check (vmerge3 and vm3priv): cross questions, waits, timeouts, R6 re-holds

|   route |   seed |   n_cross |   n_cross_pos |   cusum_releases |   waits |   timeouts |   reholds |   s_waiting |   r6_snaps | arm     |
|--------:|-------:|----------:|--------------:|-----------------:|--------:|-----------:|----------:|------------:|-----------:|:--------|
|   19324 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|    2520 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   27043 |      0 |         0 |             0 |                1 |       2 |          2 |         1 |        16   |         16 | vmerge3 |
|   27870 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   17280 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   19832 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   24944 |      0 |         0 |             0 |                2 |       2 |          0 |         1 |         1.5 |          2 | vmerge3 |
|   15612 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   22535 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   15102 |      0 |         0 |             0 |                1 |       1 |          0 |         0 |         2.5 |          0 | vmerge3 |
|   27297 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   16390 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   24497 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   37969 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|    9196 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   28147 |      0 |         0 |             0 |                1 |       1 |          0 |         1 |         1   |          2 | vmerge3 |
|   15483 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   16529 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   16508 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   19324 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|    2520 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   27043 |      1 |         0 |             0 |                1 |       2 |          2 |         1 |        16   |         16 | vmerge3 |
|   27870 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   17280 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   19832 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   24944 |      1 |         0 |             0 |                2 |       2 |          0 |         1 |         3   |          4 | vmerge3 |
|   15612 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   22535 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   15102 |      1 |         0 |             0 |                1 |       1 |          0 |         0 |         2.5 |          0 | vmerge3 |
|   27297 |      1 |         0 |             0 |                1 |       1 |          0 |         1 |         2   |          4 | vmerge3 |
|   16390 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   24497 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   37969 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|    9196 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   28147 |      1 |         0 |             0 |                1 |       1 |          0 |         1 |         1   |          2 | vmerge3 |
|   15483 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   16529 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   16508 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vmerge3 |
|   19324 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|    2520 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   27043 |      0 |         0 |             0 |                1 |       2 |          2 |         1 |        16   |         16 | vm3priv |
|   27870 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   17280 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   19832 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   24944 |      0 |         0 |             0 |                2 |       2 |          0 |         1 |         3   |          4 | vm3priv |
|   15612 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   22535 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   15102 |      0 |         0 |             0 |                1 |       1 |          0 |         0 |         2.5 |          0 | vm3priv |
|   27297 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   16390 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   24497 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   37969 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|    9196 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   28147 |      0 |         0 |             0 |                1 |       1 |          0 |         1 |         0.5 |          1 | vm3priv |
|   15483 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   16529 |      0 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   16508 |      0 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   19324 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|    2520 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   27043 |      1 |         0 |             0 |                1 |       2 |          2 |         1 |        16   |         16 | vm3priv |
|   27870 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   17280 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   19832 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   24944 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   15612 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   22535 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   15102 |      1 |         0 |             0 |                1 |       1 |          0 |         0 |         2.5 |          0 | vm3priv |
|   27297 |      1 |         0 |             0 |                1 |       1 |          0 |         1 |         2   |          4 | vm3priv |
|   16390 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   24497 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   37969 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|    9196 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   28147 |      1 |         0 |             0 |                1 |       1 |          0 |         1 |         1   |          2 | vm3priv |
|   15483 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   16529 |      1 |         0 |             0 |                1 |       0 |          0 |         0 |         0   |          0 | vm3priv |
|   16508 |      1 |         0 |             0 |                0 |       0 |          0 |         0 |         0   |          0 | vm3priv |

## DS per route and seed

|   route |   ('drive', 0) |   ('drive', 1) |   ('drive', 2) |   ('drive', 3) |   ('vm3norel', 0) |   ('vm3norel', 1) |   ('vm3priv', 0) |   ('vm3priv', 1) |   ('vmerge2', 0) |   ('vmerge2', 1) |   ('vmerge2', 2) |   ('vmerge2', 3) |   ('vmerge3', 0) |   ('vmerge3', 1) |
|--------:|---------------:|---------------:|---------------:|---------------:|------------------:|------------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|   15102 |          100   |          100   |          100   |          100   |             100   |             100   |             70   |             70   |            100   |            100   |            100   |            100   |             70   |             70   |
|   15483 |          100   |          100   |          100   |          100   |             100   |             100   |            100   |            100   |            100   |            100   |            100   |             34.2 |            100   |            100   |
|   15612 |           70   |           24.8 |           70   |          100   |              70   |             100   |            100   |             70   |            100   |             23.7 |             21.5 |            100   |             21.5 |            100   |
|   16390 |           70   |           70   |           70   |           70   |              70   |              70   |             70   |             70   |             70   |             70   |             70   |             70   |             70   |             70   |
|   16508 |          100   |          100   |          100   |          100   |             100   |             100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |
|   16529 |           70   |           70   |           70   |           70   |             100   |             100   |             70   |             70   |            100   |            100   |            100   |            100   |            100   |             70   |
|   17280 |           80   |           80   |           80   |           80   |             100   |              19.1 |            100   |             18   |            100   |             16.8 |             60   |             60   |            100   |             60   |
|   19324 |           33.4 |           33.4 |           33.4 |           28.8 |              21.8 |              97   |             60   |             60   |             60   |            100   |             97   |            100   |             28.8 |             97   |
|   19832 |           33.1 |           33.1 |           33.1 |           33.1 |              33.1 |              33.1 |             60   |            100   |             36   |             60   |            100   |             60   |             33.1 |             58.6 |
|   22535 |          100   |          100   |          100   |          100   |             100   |              60   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |
|   24497 |           20.7 |           21.7 |           20.7 |           20.7 |              21.7 |              21.7 |            100   |            100   |             65   |            100   |             21.7 |             21.7 |             21.7 |             21.7 |
|   24944 |           70   |           70   |           70   |           70   |             100   |             100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |
|    2520 |           23.5 |           23.5 |           23.5 |           23.5 |              23.5 |              23.5 |             38.4 |             65   |             60   |            100   |             20.8 |            100   |             23.5 |             41   |
|   27043 |           60   |           60   |           60   |           60   |              60   |              60   |             60   |             60   |             60   |             60   |             60   |             60   |             60   |             60   |
|   27297 |           70   |           70   |           70   |           70   |              70   |              70   |             70   |             70   |             70   |             70   |             70   |             70   |             70   |             70   |
|   27870 |           70   |          100   |           70   |          100   |             100   |             100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |
|   28147 |          100   |          100   |          100   |          100   |             100   |             100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |            100   |
|   37969 |           60   |           60   |           58.2 |           58.8 |              57.6 |              57.6 |             11.3 |             57.6 |             11.3 |             37.3 |             58.5 |             58.2 |             11.3 |             11   |
|    9196 |           42   |           14.9 |           22.9 |           40.5 |              22.1 |              21.2 |             22.1 |             22.1 |             22.1 |             21.2 |             21.2 |             34   |             34   |             22.1 |

## Reading (added by hand after the batch; the tables above come from vmerge3_report.py)

Registered lines: bypass (obstacle routes vmerge3 - drive >= +20.7 DS): +12.88 [+3.56, +24.34], **no**. Collisions (vehicle collisions per run <= drive 0.158): 0.158 (6 / 38), **yes**.
Vehicle collisions on 17280 / 27297 (seeds 0, 1): drive 0 / 0 and 0 / 0; vmerge2, vm3priv, vm3norel, vmerge3 each one on 17280 seed 1, none on 27297.
D8 check: every answer row of the vm3 arms has t_eff - t_q = 0.35 s (7139 rows); measured p95 latency 1088 ms (shared card) therefore does not enter the simulation.
Release check: 0 cross questions (the check reads the front radar), see the release table.
