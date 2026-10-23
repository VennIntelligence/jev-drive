# GARAGE020 extras

## On the 184 scenes NTV0-ag scores >= 0.98

| read | n | new zeros | new partials | zero classes |
|:--|--:|--:|--:|:--|
| GARAGE020 | 184 | 39 | 29 | AF-front 12, zero-other 9, route-turn 8, offroad 7, overshoot 2, side 1 |
| NTV0-ag | 184 | 0 | 0 |  |
| R6DAG-s0-ag | 184 | 12 | 9 | offroad 6, zero-other 2, side 2, AF-front 1, overshoot 1 |
| R6DAG-s1-ag | 184 | 5 | 5 | offroad 3, side 1, zero-other 1 |
| R6DAG-s2-ag | 184 | 9 | 6 | offroad 5, zero-other 1, overshoot 1, AF-front 1, side 1 |
| R8K-s0-ag | 184 | 11 | 5 | offroad 5, AF-front 3, zero-other 1, side 1, route-turn 1 |
| R8K-s1-ag | 184 | 9 | 7 | offroad 5, AF-front 2, zero-other 1, route-turn 1 |
| R8K-s2-ag | 184 | 10 | 5 | AF-front 3, offroad 3, side 2, zero-other 1, overshoot 1 |

## Corridor scorer (garage read, every rollout re-evaluated from its rollout.asl)

- old scorer re-evaluation reproduces the stored metrics.parquet: 280 / 281 rollouts
- fixed (1b024f0) - old: 281 scenes, mean 0.643 -> 0.664 (+0.021 [+0.007, +0.039]); zeros 80 -> 74; 6 scenes change
- fixed-scorer failure reasons (zeros): left_corridor_laterally 39, collision_at_fault 20, offroad 15

Paired on the fixed scorer (stored reads: scorer_v2's rescoring; their nonzero rollouts keep the stored score):

| ref | n | garage | ref | garage - ref [95 % CI] |
|:--|--:|--:|--:|--:|
| NTV0-ag | 280 | 0.667 | 0.816 | -0.149 [-0.208, -0.092] |
| R6DAG-s0-ag | 281 | 0.664 | 0.844 | -0.180 [-0.236, -0.123] |
| R6DAG-s1-ag | 281 | 0.664 | 0.881 | -0.216 [-0.271, -0.161] |
| R6DAG-s2-ag | 281 | 0.664 | 0.860 | -0.196 [-0.253, -0.137] |
| R8K-s0-ag | 281 | 0.664 | 0.858 | -0.194 [-0.248, -0.138] |
| R8K-s1-ag | 281 | 0.664 | 0.866 | -0.202 [-0.258, -0.147] |
| R8K-s2-ag | 281 | 0.664 | 0.874 | -0.209 [-0.267, -0.151] |

## The host's own per-scene results (Harmonizer ON, e121e59, their timing), on our scenes

| host seed | n (ours scored) | host mean | ours (fixed scorer) | ours - host [95 % CI] | host mean on all their scenes |
|:--|--:|--:|--:|--:|--:|
| 0 | 281 | 0.771 | 0.664 | -0.107 [-0.156, -0.059] | 0.768 (441) |
| 1 | 281 | 0.773 | 0.664 | -0.108 [-0.162, -0.055] | 0.752 (441) |
| 2 | 281 | 0.759 | 0.664 | -0.094 [-0.149, -0.040] | 0.761 (441) |

- seed 0, same scene: both zero 32, ours only 42, host only 12; score correlation 0.50
