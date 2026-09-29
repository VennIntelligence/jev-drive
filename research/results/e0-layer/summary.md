# E0 per-layer probe (CARLA P5 v1 BA pedestrian pairs vs nuScenes val corridor pedestrian)

| layer | dim | CARLA AUC [95% CI] | CARLA Δ vs vision [CI] | CARLA MLP | real AUC [95% CI] | real Δ vs vision [CI] | real MLP | real wide-corridor | domain AUC |
|:--|--:|:--|:--|--:|:--|:--|--:|--:|--:|
| stage1 | 512 | 0.526 [0.508, 0.556] | +0.018 [0.000, 0.040] | 0.533 | 0.715 [0.628, 0.799] | +0.014 [-0.093, 0.120] | 0.727 | 0.703 | 1.0000 |
| stage2 | 1024 | 0.523 [0.504, 0.548] | +0.015 [-0.003, 0.033] | 0.538 | 0.740 [0.644, 0.821] | +0.039 [-0.051, 0.136] | 0.787 | 0.742 | 1.0000 |
| stage3 | 2048 | 0.523 [0.507, 0.544] | +0.014 [0.002, 0.028] | 0.540 | 0.831 [0.758, 0.895] | +0.130 [0.049, 0.230] | 0.805 | 0.721 | 0.9998 |
| stage4 | 4096 | 0.522 [0.502, 0.546] | +0.013 [-0.002, 0.028] | 0.518 | 0.748 [0.666, 0.813] | +0.047 [-0.004, 0.121] | 0.769 | 0.717 | 0.9994 |
| vision | 512 | 0.509 [0.499, 0.520] | - | 0.524 | 0.702 [0.599, 0.785] | - | 0.765 | 0.713 | 0.9992 |
| temporal | 512 | 0.506 [0.499, 0.515] | -0.003 [-0.008, 0.003] | 0.508 | 0.709 [0.598, 0.799] | +0.008 [-0.140, 0.137] | 0.784 | 0.662 | 0.9944 |

CARLA: 4 414 pedestrian pairs, 42 routes, route-cluster bootstrap 500. Real: 6 019 keyframes (203 corridor positives), 150 scenes, scene-cluster bootstrap 500.

## CARLA per family (descriptive), linear AUC

| scope | pairs | stage1 | stage2 | stage3 | stage4 | vision | temporal |
|:--|--:|--:|--:|--:|--:|--:|--:|
| pedestrian | 4414 | 0.526 | 0.523 | 0.523 | 0.522 | 0.509 | 0.506 |
| hazard pooled | 7639 | 0.522 | 0.517 | 0.544 | 0.542 | 0.533 | 0.547 |
| DynamicObjectCrossing | 609 | 0.549 | 0.553 | 0.613 | 0.586 | 0.566 | 0.545 |
| HighwayCutIn | 1319 | 0.504 | 0.475 | 0.647 | 0.667 | 0.634 | 0.762 |
| OppositeVehicleRunningRedLight | 58 | 0.422 | 0.428 | 0.508 | 0.537 | 0.534 | 0.521 |
| ParkingCrossingPedestrian | 310 | 0.494 | 0.477 | 0.502 | 0.500 | 0.488 | 0.484 |
| ParkingCutIn | 817 | 0.558 | 0.554 | 0.623 | 0.600 | 0.592 | 0.557 |
| PedestrianCrossing | 836 | 0.632 | 0.611 | 0.596 | 0.604 | 0.547 | 0.537 |
| StaticCutIn | 1031 | 0.520 | 0.513 | 0.593 | 0.595 | 0.583 | 0.584 |
| VehicleTurningRoutePedestrian | 2659 | 0.504 | 0.497 | 0.500 | 0.495 | 0.502 | 0.502 |
