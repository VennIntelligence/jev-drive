# Knows vs does: wajepa (12146 navtest tokens, 5 log folds)

| GT manoeuvre | n | knows (feat+blind) | feat only | blind | does (plan) | knows - does [95 % CI] | knows - blind [95 % CI] | verdict |
|---|--:|--:|--:|--:|--:|---|---|---|
| straight | 5267 | 0.873 | 0.870 | 0.891 | 0.914 | -0.042 [-0.060, -0.023] | -0.018 [-0.040, +0.003] | does what it sees |
| left turn | 1737 | 0.917 | 0.914 | 0.922 | 0.957 | -0.040 [-0.062, -0.022] | -0.006 [-0.028, +0.015] | does what it sees |
| curve | 1522 | 0.597 | 0.598 | 0.057 | 0.834 | -0.237 [-0.272, -0.199] | +0.540 [+0.487, +0.586] | unclear |
| launch | 1471 | 0.869 | 0.872 | 0.898 | 0.989 | -0.120 [-0.154, -0.090] | -0.029 [-0.069, +0.005] | unclear |
| right turn | 1010 | 0.852 | 0.845 | 0.853 | 0.961 | -0.109 [-0.141, -0.080] | -0.001 [-0.033, +0.033] | unclear |
| stop | 660 | 0.395 | 0.398 | 0.076 | 0.805 | -0.409 [-0.483, -0.332] | +0.320 [+0.253, +0.392] | unclear |
| lane change | 472 | 0.294 | 0.297 | 0.042 | 0.667 | -0.373 [-0.452, -0.300] | +0.252 [+0.189, +0.314] | unclear |

Balanced accuracy over these classes: blind 0.534, feat 0.685, feat+blind 0.685, plan 0.875

R^2 (out of fold): dyaw_deg: blind 0.750, feat+blind 0.865, plan 0.959; lat_4s_m: blind 0.826, feat+blind 0.898, plan 0.971

Recall per GT class; CIs = log-cluster bootstrap (B 10000, seed 0); verdict lines from plans/2026-10-23-p1-knowdo-prereg.md.
