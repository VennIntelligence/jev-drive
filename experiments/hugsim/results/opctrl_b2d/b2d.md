# openpilot lateral path on B2D: `drive` arm, 19 routes, seeds 2, 3

Plan: [../../plans/2026-10-04-op-control-stack-b2d-prereg.md](../../plans/2026-10-04-op-control-stack-b2d-prereg.md). `drive` = vmerge2 reruns of the unchanged shipped arm; `opc` = + OP_CTRL.

## Missing runs

none

## Arms

| arm   |   runs |   crashes |    DS |    RC |   red_light |   stop_sign |   collisions |   blocked |   timeouts |   v_mean |
|:------|-------:|----------:|------:|------:|------------:|------------:|-------------:|----------:|-----------:|---------:|
| drive |     38 |         0 | 67.82 | 84.71 |          12 |           2 |           10 |         1 |          0 |     2.22 |
| opc   |     38 |         0 | 64.16 | 82.16 |          13 |           2 |           11 |         1 |          0 |     2.16 |

## Infractions by type (summed over runs)

| arm   |   red_light |   stop_infraction |   collisions_vehicle |   collisions_layout |   collisions_pedestrian |   outside_route_lanes |   vehicle_blocked |   route_timeout |   scenario_timeouts |
|:------|------------:|------------------:|---------------------:|--------------------:|------------------------:|----------------------:|------------------:|----------------:|--------------------:|
| drive |          12 |                 2 |                    6 |                   4 |                       0 |                     3 |                 1 |               0 |                   0 |
| opc   |          13 |                 2 |                    6 |                   5 |                       0 |                     3 |                 1 |               0 |                   0 |

## Paired differences (mean over routes [95% route-cluster CI])

| contrast | routes | n routes | DS | RC | red_light | stop_infraction | collisions | vehicle_blocked |
|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| opc - drive | all | 19 | -3.66 [-9.02, +0.34] | -2.55 [-8.73, +2.63] | +0.03 [-0.05, +0.11] | +0.00 [+0.00, +0.00] | +0.03 [+0.00, +0.08] | +0.00 [-0.08, +0.08] |
| opc - drive | obstacle | 4 | +0.69 [+0.00, +1.70] | +0.76 [+0.00, +1.70] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| opc - drive | light | 6 | -6.45 [-19.34, +0.00] | -5.64 [-16.92, +0.00] | +0.08 [+0.00, +0.25] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.08 [+0.00, +0.25] |
| opc - drive | stop sign | 3 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| opc - drive | other | 6 | -5.60 [-14.53, +1.82] | -2.93 [-20.16, +11.38] | +0.00 [-0.25, +0.25] | +0.00 [+0.00, +0.00] | +0.08 [+0.00, +0.25] | -0.08 [-0.25, +0.00] |

Repeat noise: drive: per-route DS sd over seeds mean 3.1, median 0.0, max 21.2; opc: per-route DS sd over seeds mean 3.9, median 0.0, max 33.5

## Selector

no selector (the rule is a steering filter)



## DS per route and seed

|   route |   ('drive', 2) |   ('drive', 3) |   ('opc', 2) |   ('opc', 3) |
|--------:|---------------:|---------------:|-------------:|-------------:|
|   15102 |          100   |          100   |        100   |        100   |
|   15483 |          100   |          100   |        100   |        100   |
|   15612 |           70   |          100   |         70   |         22.6 |
|   16390 |           70   |           70   |         70   |         70   |
|   16508 |          100   |          100   |        100   |        100   |
|   16529 |           70   |           70   |         70   |         70   |
|   17280 |           80   |           80   |         80   |         80   |
|   19324 |           33.4 |           28.8 |         33.4 |         33.4 |
|   19832 |           33.1 |           33.1 |         33.1 |         33.1 |
|   22535 |          100   |          100   |        100   |        100   |
|   24497 |           20.7 |           20.7 |         20.7 |         21.7 |
|   24944 |           70   |           70   |         70   |         70   |
|    2520 |           23.5 |           23.5 |         23.5 |         23.5 |
|   27043 |           60   |           60   |         60   |         60   |
|   27297 |           70   |           70   |         70   |         70   |
|   27870 |           70   |          100   |         70   |         70   |
|   28147 |          100   |          100   |        100   |        100   |
|   37969 |           58.2 |           58.8 |         11.3 |         57.6 |
|    9196 |           22.9 |           40.5 |         32.3 |         42   |
