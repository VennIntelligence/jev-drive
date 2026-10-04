# Low-speed lateral transfer limit on B2D: `drive` arm, 19 routes, seeds 2, 3

Plan: [../../plans/2026-10-04-lowspeed-ctrl-prereg.md](../../plans/2026-10-04-lowspeed-ctrl-prereg.md), section B. `drive` = vmerge2 reruns of the unchanged shipped arm; `lsc` = + LOWSPEED_CTRL.

## Missing runs

none

## Arms

| arm   |   runs |   crashes |    DS |    RC |   red_light |   stop_sign |   collisions |   blocked |   timeouts |   v_mean |
|:------|-------:|----------:|------:|------:|------------:|------------:|-------------:|----------:|-----------:|---------:|
| drive |     38 |         0 | 67.82 | 84.71 |          12 |           2 |           10 |         1 |          0 |     2.22 |
| lsc   |     38 |         0 | 63.72 | 80.57 |          13 |           2 |           12 |         1 |          0 |     2    |

## Infractions by type (summed over runs)

| arm   |   red_light |   stop_infraction |   collisions_vehicle |   collisions_layout |   collisions_pedestrian |   outside_route_lanes |   vehicle_blocked |   route_timeout |   scenario_timeouts |
|:------|------------:|------------------:|---------------------:|--------------------:|------------------------:|----------------------:|------------------:|----------------:|--------------------:|
| drive |          12 |                 2 |                    6 |                   4 |                       0 |                     3 |                 1 |               0 |                   0 |
| lsc   |          13 |                 2 |                    5 |                   7 |                       0 |                     2 |                 1 |               0 |                   0 |

## Paired differences (mean over routes [95% route-cluster CI])

| contrast | routes | n routes | DS | RC | red_light | stop_infraction | collisions | vehicle_blocked |
|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| lsc - drive | all | 19 | -4.11 [-10.69, +1.05] | -4.14 [-13.10, +0.48] | +0.03 [-0.05, +0.13] | +0.00 [+0.00, +0.00] | +0.05 [+0.00, +0.13] | +0.00 [+0.00, +0.00] |
| lsc - drive | obstacle | 4 | +0.62 [+0.00, +1.25] | +0.76 [+0.00, +1.51] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| lsc - drive | light | 6 | -7.42 [-17.27, +0.00] | +0.00 [+0.00, +0.00] | +0.17 [+0.00, +0.33] | +0.00 [+0.00, +0.00] | +0.08 [+0.00, +0.25] | +0.00 [+0.00, +0.00] |
| lsc - drive | stop sign | 3 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| lsc - drive | other | 6 | -6.00 [-23.88, +6.96] | -13.61 [-41.73, +0.89] | -0.08 [-0.25, +0.00] | +0.00 [+0.00, +0.00] | +0.08 [+0.00, +0.25] | +0.00 [+0.00, +0.00] |

Repeat noise: drive: per-route DS sd over seeds mean 3.1, median 0.0, max 21.2; lsc: per-route DS sd over seeds mean 3.3, median 0.0, max 21.2

## Selector

no selector (the rule is a steering filter)



## DS per route and seed

|   route |   ('drive', 2) |   ('drive', 3) |   ('lsc', 2) |   ('lsc', 3) |
|--------:|---------------:|---------------:|-------------:|-------------:|
|   15102 |          100   |          100   |        100   |        100   |
|   15483 |          100   |          100   |         70   |        100   |
|   15612 |           70   |          100   |         40.9 |         70   |
|   16390 |           70   |           70   |         70   |         70   |
|   16508 |          100   |          100   |        100   |        100   |
|   16529 |           70   |           70   |         70   |         70   |
|   17280 |           80   |           80   |         80   |         80   |
|   19324 |           33.4 |           28.8 |         33.4 |         31.8 |
|   19832 |           33.1 |           33.1 |         33.1 |         33.1 |
|   22535 |          100   |          100   |        100   |        100   |
|   24497 |           20.7 |           20.7 |         21.7 |         21.7 |
|   24944 |           70   |           70   |         70   |         70   |
|    2520 |           23.5 |           23.5 |         23.5 |         23.5 |
|   27043 |           60   |           60   |         60   |         60   |
|   27297 |           70   |           70   |         70   |         70   |
|   27870 |           70   |          100   |        100   |        100   |
|   28147 |          100   |          100   |        100   |        100   |
|   37969 |           58.2 |           58.8 |         11   |         10.5 |
|    9196 |           22.9 |           40.5 |         14.9 |         42   |
