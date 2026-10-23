# Knows vs does: w0-vlmkv (12146 navtest tokens, 5 log folds)

| GT manoeuvre | n | knows (feat+blind) | feat only | blind | does (plan) | knows - does [95 % CI] | knows - blind [95 % CI] | verdict |
|---|--:|--:|--:|--:|--:|---|---|---|
| straight | 5267 | 0.879 | 0.879 | 0.910 | 0.899 | -0.020 [-0.032, -0.009] | -0.031 [-0.051, -0.012] | does what it sees |
| left turn | 1737 | 0.937 | 0.937 | 0.927 | 0.949 | -0.012 [-0.030, +0.006] | +0.009 [-0.014, +0.031] | does what it sees |
| curve | 1522 | 0.685 | 0.685 | 0.022 | 0.781 | -0.095 [-0.136, -0.058] | +0.664 [+0.608, +0.713] | does what it sees |
| launch | 1471 | 0.884 | 0.886 | 0.898 | 0.985 | -0.101 [-0.125, -0.075] | -0.014 [-0.042, +0.013] | unclear |
| right turn | 1010 | 0.917 | 0.914 | 0.858 | 0.946 | -0.029 [-0.046, -0.011] | +0.058 [+0.032, +0.087] | does what it sees |
| stop | 660 | 0.589 | 0.591 | 0.000 | 0.732 | -0.142 [-0.210, -0.076] | +0.589 [+0.525, +0.653] | unclear |
| lane change | 472 | 0.358 | 0.356 | 0.013 | 0.517 | -0.159 [-0.230, -0.084] | +0.345 [+0.267, +0.422] | unclear |

Balanced accuracy over these classes: blind 0.518, feat 0.750, feat+blind 0.750, plan 0.830

R^2 (out of fold): dyaw_deg: blind 0.751, feat+blind 0.942, plan 0.960; lat_4s_m: blind 0.816, feat+blind 0.937, plan 0.958

Recall per GT class; CIs = log-cluster bootstrap (B 10000, seed 0); verdict lines from plans/2026-10-23-p1-knowdo-prereg.md.
