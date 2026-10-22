# Knows vs does: recog-lmh (12146 navtest tokens, 5 log folds)

| GT manoeuvre | n | knows (feat+blind) | feat only | blind | does (plan) | knows - does [95 % CI] | knows - blind [95 % CI] | verdict |
|---|--:|--:|--:|--:|--:|---|---|---|
| straight | 5267 | 0.871 | 0.868 | 0.891 | 0.851 | +0.020 [+0.007, +0.034] | -0.020 [-0.041, -0.000] | does what it sees |
| left turn | 1737 | 0.904 | 0.902 | 0.922 | 0.952 | -0.048 [-0.068, -0.029] | -0.018 [-0.037, +0.001] | does what it sees |
| curve | 1522 | 0.648 | 0.647 | 0.057 | 0.775 | -0.126 [-0.180, -0.081] | +0.591 [+0.521, +0.649] | unclear |
| launch | 1471 | 0.857 | 0.840 | 0.898 | 0.965 | -0.109 [-0.144, -0.072] | -0.041 [-0.075, -0.009] | unclear |
| right turn | 1010 | 0.858 | 0.854 | 0.854 | 0.907 | -0.049 [-0.082, -0.017] | +0.004 [-0.025, +0.035] | does what it sees |
| stop | 660 | 0.442 | 0.438 | 0.079 | 0.152 | +0.291 [+0.222, +0.363] | +0.364 [+0.301, +0.428] | sees but does not do |
| lane change | 472 | 0.278 | 0.278 | 0.042 | 0.390 | -0.112 [-0.190, -0.043] | +0.235 [+0.173, +0.293] | unclear |

Balanced accuracy over these classes: blind 0.535, feat 0.689, feat+blind 0.694, plan 0.713

R^2 (out of fold): dyaw_deg: blind 0.750, feat+blind 0.824, plan 0.954; lat_4s_m: blind 0.826, feat+blind 0.852, plan 0.958

Recall per GT class; CIs = log-cluster bootstrap (B 10000, seed 0); verdict lines from plans/2026-10-23-p1-knowdo-prereg.md.
