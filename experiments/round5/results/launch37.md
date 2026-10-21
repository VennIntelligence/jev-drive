# LAUNCH-37: the 37 PAI 281 launch scenes, ACC + GAP (g10, FAST=1, CONC 4)

Scenes: lead_281.csv launch rows (NTV0's launch scenes; LEAD rule at the human's t_L: 26 lead, 11 no lead). Stored NTV0 / R2C / R2J = their PAI 281 ACC + GAP reads restricted to these scenes (not re-read); the others read LAUNCH-37 alone (the same 4 chunk lists). launched = pai_fix.py launched (dist >= 2 m and >= 0.3 x the log's); scenes with an RPC error are left out.

| read | n | launched | rear-hit | mean | zeros | launched lead / no-lead | rear-hit lead / no-lead | mean lead / no-lead |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0 | 37 | 33 | 9 | 0.774 | 3 | 23 / 10 | 8 / 1 | 0.772 / 0.779 |
| R2C | 37 | 33 | 11 | 0.734 | 4 | 23 / 10 | 10 / 1 | 0.717 / 0.774 |
| R2J | 37 | 21 | 19 | 0.528 | 1 | 15 / 6 | 14 / 5 | 0.536 / 0.508 |
| R3MV | 37 | 35 | 10 | 0.822 | 2 | 24 / 11 | 9 / 1 | 0.802 / 0.868 |
| R3MVL | 37 | 35 | 9 | 0.815 | 3 | 24 / 11 | 8 / 1 | 0.789 / 0.875 |
| R5N | 37 | 32 | 11 | 0.729 | 4 | 23 / 9 | 9 / 2 | 0.742 / 0.697 |
| R5G | 36 | 35 | 9 | 0.862 | 1 | 24 / 11 | 8 / 1 | 0.821 / 0.955 |

## Paired over LAUNCH-37 (A - B; scenes both reads scored)

| pair | n | mean (95 % CI) | launched (count, 95 % CI) | launched A only / B only | mean lead / no-lead |
|:--|--:|:--|:--|--:|:--|
| R5G - R5N | 36 | +0.113 [+0.037, +0.206] | +3 [+0, +7] | 3 / 0 | +0.049 / +0.258 |
| R5G - R3MVL | 36 | +0.024 [-0.007, +0.076] | +0 [+0, +0] | 0 / 0 | +0.000 / +0.079 |
| R5G - R3MV | 36 | +0.018 [-0.038, +0.089] | +0 [+0, +0] | 0 / 0 | -0.013 / +0.086 |
| R5N - R3MV | 37 | -0.093 [-0.174, -0.025] | -3 [-7, +0] | 0 / 3 | -0.060 / -0.172 |
| R3MVL - R3MV | 37 | -0.007 [-0.088, +0.072] | +0 [+0, +0] | 0 / 0 | -0.013 / +0.007 |
| R5G - NTV0 | 36 | +0.095 [-0.025, +0.216] | +3 [+0, +7] | 3 / 0 | +0.059 / +0.176 |
| R5N - NTV0 | 37 | -0.045 [-0.180, +0.086] | -1 [-5, +3] | 2 / 3 | -0.029 / -0.082 |
| R3MV - NTV0 | 37 | +0.048 [-0.060, +0.162] | +2 [-2, +6] | 3 / 1 | +0.030 / +0.089 |
| R5G - R2C | 36 | +0.135 [+0.033, +0.249] | +3 [+0, +6] | 3 / 0 | +0.114 / +0.181 |

## Paired over PAI-FIX TARGET (val160 STOP + STOP-L + TURN, native VC + LEAD reads; scenes both scored without RPC)

| pair | n | A / B | mean (95 % CI) | zeros A / B (new / fixed) |
|:--|--:|:--|:--|:--|
| R5G-L18-s0 - R5N-L18-s0 | 60 | 0.493 / 0.532 | -0.039 [-0.130, +0.051] | 25 / 22 (6 / 3) |
| R5G-L18-s0 - R3MVL-L18-s0 | 60 | 0.493 / 0.528 | -0.035 [-0.106, +0.030] | 25 / 22 (4 / 1) |
| R5G-L18-s0 - R3MV-L18-s0 | 60 | 0.493 / 0.574 | -0.082 [-0.166, -0.003] | 25 / 19 (7 / 1) |
| R5N-L18-s0 - R3MV-L18-s0 | 60 | 0.532 / 0.574 | -0.043 [-0.118, +0.027] | 22 / 19 (4 / 1) |

**Registered comparison R5G vs R5N (addendum 4 / 4b): separated = yes** (paired CI excludes 0 on LAUNCH-37: yes, on TARGET: no).
