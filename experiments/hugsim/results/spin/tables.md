## T1 spin-outs by agent and controller (64 scenarios each; spin = heading error to the route reaches 60 deg)

| agent | controller | complete | failed (not complete) | spin >= 60 deg | of failed | spin >= 90 deg | spin >= 45 deg | left / right | share of failures that are spins |
|---|---|---|---|---|---|---|---|---|---|
| cinque | official | 0 | 64 | 52 | 52 | 40 | 58 | 3 / 49 | 81% |
| cinque | fixed | 10 | 54 | 10 | 9 | 8 | 10 | 10 / 0 | 17% |
| lebowski | official | 0 | 64 | 50 | 50 | 40 | 59 | 1 / 49 | 78% |
| lebowski | fixed | 8 | 56 | 11 | 10 | 8 | 11 | 8 / 3 | 18% |
| ltf | official | 12 | 52 | 0 | 0 | 0 | 2 | 0 / 0 | 0% |
| ltf | fixed | 14 | 50 | 0 | 0 | 0 | 1 | 0 / 0 | 0% |
| cv | official | 0 | 64 | 55 | 55 | 48 | 60 | 0 / 55 | 86% |
| cv | fixed | 3 | 61 | 0 | 0 | 0 | 2 | 0 / 0 | 0% |

## T2 paired by scenario: spin under the official controller vs under PR #57 (same agent, same scenario)

| agent | spin official & not PR#57 | spin both | spin PR#57 only | no spin either |
|---|---|---|---|---|
| cinque | 45 | 7 | 3 | 9 |
| lebowski | 43 | 7 | 4 | 10 |
| cv | 55 | 0 | 0 | 9 |

## T3 the PR #57 openpilot spin-outs (Cinque and Lebowski)

| agent | scenario | dataset | end | steps | first direction | divergence step (|e| > 5 deg) | speed there (m/s) | plan error before divergence (deg, signed toward the turn) | share of the yaw built on steps whose plan pointed >= 10 deg off the route on the turn side |
|---|---|---|---|---|---|---|---|---|---|
| cinque | scene-0013-medium-00 | nuscenes | bg_collision | 22 | left | 8 | 2.9 | 39 | 0.93 |
| cinque | scene-0528-medium-00 | nuscenes | bg_collision | 17 | left | 7 | 1.8 | 55 | 1.00 |
| cinque | scene-0254-extreme-00 | nuscenes | complete | 76 | left | 8 | 1.6 | 46 | 0.99 |
| cinque | scene-102751446607-medium-01 | waymo | bg_collision | 66 | left | 8 | 1.5 | 42 | 0.99 |
| cinque | scene-152217047339-medium-00 | waymo | bg_collision | 27 | left | 10 | 1.8 | 65 | 1.00 |
| cinque | scene-570_770-easy-00 | kitti360 | bg_collision | 45 | left | 39 | 6.3 | 17 | 0.62 |
| cinque | scene-5980_6180-easy-00 | kitti360 | bg_collision | 118 | left | 81 | 2.9 | 37 | 0.61 |
| cinque | scene-8440_8640-easy-00 | kitti360 | bg_collision | 256 | left | 243 | 1.5 | 58 | 0.93 |
| cinque | scene-040-easy-00 | pandaset | bg_collision | 175 | left | 151 | 2.1 | 12 | 0.82 |
| cinque | scene-053-medium-02 | pandaset | max_steps | 400 | left | 5 | 2.1 | 38 | 0.99 |
| lebowski | scene-0013-medium-00 | nuscenes | off_route | 21 | right | 6 | 1.5 | 37 | 0.99 |
| lebowski | scene-0041-medium-00 | nuscenes | bg_collision | 31 | left | 17 | 5.9 | 24 | 0.91 |
| lebowski | scene-0254-hard-00 | nuscenes | fg_collision | 75 | left | 34 | 1.3 | 34 | 0.97 |
| lebowski | scene-0013-extreme-00 | nuscenes | off_route | 22 | right | 7 | 1.7 | 41 | 0.99 |
| lebowski | scene-132384196576-medium-01 | waymo | bg_collision | 122 | left | 29 | 0.7 | 21 | 0.95 |
| lebowski | scene-152217047339-medium-00 | waymo | bg_collision | 51 | left | 14 | 0.9 | 58 | 0.96 |
| lebowski | scene-144248042870-extreme-00 | waymo | off_route | 18 | right | 3 | 1.1 | 43 | 1.00 |
| lebowski | scene-5980_6180-easy-00 | kitti360 | bg_collision | 99 | left | 92 | 2.1 | 59 | 0.95 |
| lebowski | scene-3000_3200-hard-00 | kitti360 | complete | 137 | left | 131 | 2.1 | -11 | 0.03 |
| lebowski | scene-034-hard-00 | pandaset | bg_collision | 17 | left | 7 | 1.6 | 42 | 0.98 |
| lebowski | scene-053-extreme-02 | pandaset | fg_collision | 33 | left | 7 | 1.7 | 39 | 0.99 |

## T4 PR #57 spin-outs by dataset (16 scenarios per cell)

| agent | nuscenes | waymo | kitti360 | pandaset |
|---|---|---|---|---|
| cinque | 3 / 16 | 2 / 16 | 3 / 16 | 2 / 16 |
| lebowski | 4 / 16 | 3 / 16 | 2 / 16 | 2 / 16 |

## T5 controller replay on the logged states and plans (one 0.25 s step from each logged state)

| run set | moving steps | steps whose plan points < 5 deg off the ego axis | of those, executed |yaw change| > 3 deg in the step | share |
|---|---|---|---|---|
| cinque-official | 1380 | 89 | 27 | 30.3% |
| lebowski-official | 1061 | 100 | 28 | 28.0% |
| cinque-fixed | 2435 | 2011 | 5 | 0.2% |
| lebowski-fixed | 2502 | 2096 | 3 | 0.1% |

| run set | steps with plan direction at 0.5 s > 20 deg | median executed yaw change / plan direction: as run | official | PR #57 | fixed2 |
|---|---|---|---|---|---|
| cinque-fixed | 180 | 0.37 | 0.37 | 0.37 | 0.44 |
| lebowski-fixed | 163 | 0.40 | 0.42 | 0.40 | 0.48 |

## T6 plan direction relative to the route, PR #57 openpilot runs, steps with heading error < 5 deg (still aligned)

| agent | dataset | steps | mean plan error at 1 s (deg, - = left of the route direction) | median | share >= 10 deg left | share >= 10 deg right |
|---|---|---|---|---|---|---|
| cinque | kitti360 | 632 | -5.0 | -2.9 | 21% | 3% |
| cinque | nuscenes | 474 | -0.6 | -0.9 | 7% | 7% |
| cinque | pandaset | 462 | -2.5 | -1.7 | 3% | 0% |
| cinque | waymo | 373 | -6.8 | -7.0 | 34% | 2% |
| lebowski | kitti360 | 986 | -4.5 | -2.0 | 29% | 3% |
| lebowski | nuscenes | 509 | -3.1 | -1.0 | 10% | 2% |
| lebowski | pandaset | 327 | -5.5 | -5.3 | 10% | 0% |
| lebowski | waymo | 190 | -10.0 | -5.6 | 40% | 8% |

## T7 scene context at the divergence step (cinque + lebowski, PR #57): drivable ground left vs right, geometry in the lane ahead

| group | n | mean (L - R) / (L + R) of ground points 3-25 m ahead | share with scene points in the lane 3-15 m ahead |
|---|---|---|---|
| spin, turned left | 18 | 0.07 | 50% |
| spin, turned right | 3 | 0.77 | 100% |
| no spin (step 8) | 107 | 0.03 | 67% |
