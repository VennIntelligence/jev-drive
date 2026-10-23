# GARAGE020 extras

## On the 184 scenes NTV0-ag scores >= 0.98

| read | n | new zeros | new partials | zero classes |
|:--|--:|--:|--:|:--|
| GARAGE020 | 182 | 23 | 31 | AF-front 13, offroad 8, overshoot 1, side 1 |
| NTV0-ag | 184 | 0 | 0 |  |
| R6DAG-s0-ag | 184 | 12 | 9 | offroad 6, zero-other 2, side 2, AF-front 1, overshoot 1 |
| R6DAG-s1-ag | 184 | 5 | 5 | offroad 3, side 1, zero-other 1 |
| R6DAG-s2-ag | 184 | 9 | 6 | offroad 5, zero-other 1, overshoot 1, AF-front 1, side 1 |
| R8K-s0-ag | 184 | 11 | 5 | offroad 5, AF-front 3, zero-other 1, side 1, route-turn 1 |
| R8K-s1-ag | 184 | 9 | 7 | offroad 5, AF-front 2, zero-other 1, route-turn 1 |
| R8K-s2-ag | 184 | 10 | 5 | AF-front 3, offroad 3, side 2, zero-other 1, overshoot 1 |

## Corridor scorer (garage read, every rollout re-evaluated from its rollout.asl)

- old scorer re-evaluation reproduces the stored metrics.parquet: 277 / 277 rollouts
- fixed (1b024f0) - old: 277 scenes, mean 0.750 -> 0.772 (+0.022 [+0.007, +0.040]); zeros 47 -> 41; 6 scenes change
- fixed-scorer failure reasons (zeros): collision_at_fault 21, offroad 18, left_corridor_laterally 2

Paired on the fixed scorer (stored reads: scorer_v2's rescoring; their nonzero rollouts keep the stored score):

| ref | n | garage | ref | garage - ref [95 % CI] |
|:--|--:|--:|--:|--:|
| NTV0-ag | 277 | 0.772 | 0.817 | -0.046 [-0.100, +0.009] |
| R6DAG-s0-ag | 277 | 0.772 | 0.842 | -0.070 [-0.124, -0.019] |
| R6DAG-s1-ag | 277 | 0.772 | 0.879 | -0.107 [-0.158, -0.057] |
| R6DAG-s2-ag | 277 | 0.772 | 0.858 | -0.086 [-0.139, -0.035] |
| R8K-s0-ag | 277 | 0.772 | 0.856 | -0.084 [-0.137, -0.032] |
| R8K-s1-ag | 277 | 0.772 | 0.865 | -0.093 [-0.146, -0.040] |
| R8K-s2-ag | 277 | 0.772 | 0.872 | -0.100 [-0.155, -0.047] |

## The host's own per-scene results (Harmonizer ON, e121e59, their timing), on our scenes

| host seed | n (ours scored) | host mean | ours (fixed scorer) | ours - host [95 % CI] | host mean on all their scenes |
|:--|--:|--:|--:|--:|--:|
| 0 | 277 | 0.771 | 0.772 | +0.000 [-0.030, +0.032] | 0.768 (441) |
| 1 | 277 | 0.773 | 0.772 | -0.001 [-0.039, +0.037] | 0.752 (441) |
| 2 | 277 | 0.759 | 0.772 | +0.013 [-0.024, +0.051] | 0.761 (441) |

- seed 0, same scene: both zero 32, ours only 9, host only 11; score correlation 0.76
