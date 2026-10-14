# SLOW-NOVELTY tables (rf_slownov.py analyse)

| label | rollouts | kept decisions | Y1 median (m) | Y2 median | lat median / p90 (m) | head median / p90 (deg) | junction share | held | released for a turn | junc / logdef missing |
|:--|--:|--:|--:|--:|:--|:--|--:|--:|--:|:--|
| RSG | 450 | 7217 | -0.258 | 0.924 | 0.12 / 0.90 | 0.7 / 4.7 | 0.047 | 0.000 | 0.000 | 0.000 / 0.077 |
| RSG+hold | 450 | 7214 | -0.068 | 0.983 | 0.13 / 0.92 | 0.7 / 4.5 | 0.047 | 0.327 | 0.535 | 0.000 / 0.120 |
| RSG+holdlead | 450 | 7205 | -0.069 | 0.963 | 0.13 / 0.89 | 0.7 / 4.5 | 0.047 | 0.226 | 0.537 | 0.000 / 0.098 |

Standardised OLS coefficients (outcome and covariates winsorised 1 / 99 %, z-scored within the model's rows), 95 % log-clustered bootstrap (2000 draws). Negative = more of it -> shorter plan (Y1) / slower than the log (Y2).

| model | label | Y | n (scenes / logs) | R^2 base | delta R^2 novelty [CI] | lat | head | junc |
|:--|:--|:--|:--|--:|:--|:--|:--|:--|
| 1 primary | RSG+hold | y1 | 6347 (450 / 39) | 0.784 | +0.0011 [+0.0002, +0.0038] | -0.041 [-0.082, -0.008] | +0.016 [-0.016, +0.052] | -0.004 [-0.024, +0.014] |
| 1b no ratchet controls (brief) | RSG+hold | y1 | 6347 (450 / 39) | 0.535 | +0.0060 [+0.0026, +0.0128] | -0.096 [-0.146, -0.056] | +0.065 [+0.017, +0.114] | -0.002 [-0.030, +0.030] |
| 1c within-scene | RSG+hold | y1 | 6347 (450 / 39) | 0.672 | +0.0010 [+0.0001, +0.0058] | -0.038 [-0.091, +0.010] | +0.019 [-0.017, +0.063] | -0.002 [-0.023, +0.020] |
| 1d speed ratio Y2 | RSG+hold | y2 | 6319 (450 / 39) | 0.160 | +0.0014 [+0.0003, +0.0158] | +0.039 [-0.040, +0.114] | +0.006 [-0.099, +0.083] | +0.002 [-0.040, +0.056] |
| 2 plain read, ratchet in | RSG | y1 | 6659 (450 / 39) | 0.820 | +0.0017 [+0.0007, +0.0038] | -0.048 [-0.079, -0.020] | +0.004 [-0.025, +0.038] | -0.003 [-0.018, +0.012] |
| 2b plain, no ratchet controls | RSG | y1 | 6659 (450 / 39) | 0.532 | +0.0086 [+0.0042, +0.0161] | -0.113 [-0.155, -0.073] | +0.079 [+0.022, +0.139] | +0.009 [-0.013, +0.031] |
| 2c plain within-scene | RSG | y1 | 6659 (450 / 39) | 0.733 | +0.0011 [+0.0002, +0.0050] | -0.031 [-0.069, +0.008] | +0.029 [-0.009, +0.074] | -0.009 [-0.029, +0.018] |
| 2d plain speed ratio Y2 | RSG | y2 | 6635 (450 / 39) | 0.342 | +0.0236 [+0.0068, +0.0575] | +0.130 [+0.051, +0.224] | +0.057 [-0.016, +0.117] | +0.019 [-0.027, +0.067] |
| 1r replication RSG+holdlead | RSG+holdlead | y1 | 6502 (450 / 39) | 0.790 | +0.0016 [+0.0004, +0.0045] | -0.049 [-0.089, -0.018] | +0.014 [-0.017, +0.051] | -0.003 [-0.019, +0.010] |
| 1e post-hoc: excess deficit Y1 - logdef | RSG+hold | x | 6347 (450 / 39) | 0.217 | +0.0100 [+0.0034, +0.0226] | -0.090 [-0.154, -0.032] | +0.022 [-0.039, +0.096] | -0.064 [-0.100, -0.035] |
| 2e post-hoc: excess deficit, plain | RSG | x | 6659 (450 / 39) | 0.180 | +0.0277 [+0.0144, +0.0509] | -0.168 [-0.244, -0.101] | +0.019 [-0.054, +0.095] | -0.069 [-0.107, -0.037] |

Base-block coefficients:

| model | logdef | vlog | tstop | stopped | k | vfed | dvfed | held |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 primary | +0.297* | +0.185* | -0.100 | -0.087* | +0.111 | -0.357* | +0.709* | +0.104* |
| 1b no ratchet controls (brief) | +0.631* | -0.068* | -0.368 | +0.068 | +0.399 | - | - | - |
| 1c within-scene | +0.189* | +0.047* | -0.348 | -0.013 | +0.323 | -0.419* | +0.712* | +0.201* |
| 1d speed ratio Y2 | -0.130* | -0.133* | +0.510* | +0.259* | -0.828* | - | - | +0.241* |
| 2 plain read, ratchet in | +0.155* | +0.178* | -0.066 | -0.077* | +0.068 | -0.359* | +0.859* | - |
| 2b plain, no ratchet controls | +0.588* | -0.214* | -0.459* | +0.015 | +0.566* | - | - | - |
| 2c plain within-scene | +0.134* | +0.056* | -0.068 | +0.010 | -0.040 | -0.528* | +0.784* | - |
| 2d plain speed ratio Y2 | +0.177* | +0.067 | +0.387 | +0.236* | -0.956* | - | - | - |
| 1r replication RSG+holdlead | +0.273* | +0.010 | -0.103 | -0.081* | +0.143 | -0.180* | +0.697* | +0.145* |
| 1e post-hoc: excess deficit Y1 - logdef | - | -0.500* | -0.089 | -0.216* | +0.218 | +0.387* | +0.282* | +0.227* |
| 2e post-hoc: excess deficit, plain | - | -0.212* | -0.253 | -0.306* | +0.495 | -0.025 | +0.373* | - |

(* = 95 % CI clear of 0)

**Verdict (registered rule, primary model): pure input loop (novelty a minor term)**; novelty coefficients clear of 0 in the slower direction: ['lat']; in the opposite direction: none.

Onset (RSG): excess deficit x = Y1 - logdef, threshold = 10th percentile = -1.683 m; 163 of 450 rollouts reach it (median onset k 6), 131 with >= 2 kept decisions before it; 287 never.

| variable | onset - preceding (same rollout) [CI] | onset - never-slow at the same k [CI] | never-slow mean |
|:--|:--|:--|:--|
| lat | +0.341 [+0.235, +0.438] (n 131) | +0.206 [+0.069, +0.333] (n 131) | 0.325 |
| head | +0.833 [+0.170, +1.578] (n 131) | +0.768 [-0.167, +1.791] (n 131) | 1.730 |
| junc | -0.006 [-0.020, +0.000] (n 131) | +0.064 [+0.015, +0.121] (n 131) | 0.033 |
| dvfed | -0.323 [-0.384, -0.253] (n 131) | -0.237 [-0.329, -0.121] (n 131) | -0.042 |
| vfed | -0.269 [-0.518, +0.043] (n 131) | +0.006 [-0.407, +0.455] (n 131) | 5.909 |
| x | -1.507 [-1.647, -1.375] (n 131) | -1.900 [-1.961, -1.843] (n 131) | -0.038 |

Onset supported (registered: (i) and (ii) both clear of 0 in the more-novel direction) by: ['lat'].
