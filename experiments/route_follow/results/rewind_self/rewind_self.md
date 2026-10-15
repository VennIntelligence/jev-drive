# Self rewind: does RSG-L18-s0 have a frontier on its own failed nuPlan @18 train turns?

Units: 32 failed (d316 stratum A without short: off 9, zero 23), 10 controls (completed turns with the latest entrance crossing, k_ref = that decision). Variants per (unit, Delta): curvature gain g x arc scale s on the served plan for n decisions from h = k_ref - 2 Delta, then the raw plan. Strict completion = rf_follow completed AND v(h + 3 s) >= 2 m/s AND no G-crawl (approach speed < 2.3 m/s after h). p-hat = strict completions / variants in the denominator (perturbation inside the bound at the window end: |lateral| <= 0.5 m, |heading| <= 5 deg, speed ratio 0.9-1.1; and the pre-h check within d316's tolerance). CI: bootstrap over logs, 10 000.

## p-hat per Delta

| role | Delta | units rewindable / all | units with >= 1 variant in the denominator | p-hat bins 0 / (0, 0.2) / [0.2, 0.8] / (0.8, 1) / 1 | mean p-hat [95 % CI] | mean plain p-hat | units with p-hat > 0 | identity strict (same h) | expert strict / plain at the same h | d316 expert (D3) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| failed | 3 s | 30 / 32 | 30 | 19 / 3 / 8 / 0 / 0 | 0.12 [0.05, 0.20] | 0.12 | 11 | 1 / 30 | 22 / 23 of 30 | 23 / 30 |
| failed | 5 s | 21 / 32 | 21 | 11 / 4 / 5 / 0 / 1 | 0.17 [0.06, 0.30] | 0.18 | 10 | 1 / 21 | 17 / 19 of 21 | 0 / - |
| failed | 8 s | 3 / 32 | 3 | 2 / 0 / 1 / 0 / 0 | 0.17 [0.00, 0.50] | 0.17 | 1 | 0 / 3 | 2 / 2 of 3 | 0 / - |
| control | 3 s | 10 / 10 | 10 | 3 / 0 / 3 / 1 / 3 | 0.55 [0.25, 0.89] | 0.67 | 7 | 8 / 10 | 4 / 6 of 10 | 10 / 10 |
| control | 5 s | 0 / 10 | 0 | 0 / 0 / 0 / 0 / 0 | - | nan | 0 | 0 / 0 | 0 / 0 of 0 | 0 / - |
| control | 8 s | 0 / 10 | 0 | 0 / 0 / 0 / 0 / 0 | - | nan | 0 | 0 / 0 | 0 / 0 of 0 | 0 / - |

Non-rewindable (unit, Delta) pairs (h < 1): Delta 3 s: failed 2, controls 0, Delta 5 s: failed 11, controls 10, Delta 8 s: failed 29, controls 10

## Completions by variant type (failed units, variants in the denominator)

| Delta | g < 1 | g > 1 | s < 1 | s > 1 | strict completions from s < 1 | strict + d293 gate | rows (strict; strict + gate) | units giving rows |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 3 s | 13 / 100 | 9 / 80 | 12 / 96 | 10 / 84 | 12 / 22 | 17 | 284; 224 | 11 |
| 5 s | 12 / 82 | 14 / 75 | 13 / 80 | 13 / 77 | 13 / 26 | 22 | 404; 346 | 10 |
| 8 s | 2 / 12 | 2 / 11 | 3 / 12 | 1 / 11 | 3 / 4 | 3 | 69; 53 | 1 |

## Displacement from the identity at the end of the perturbation window (all units)

| Delta | variant | n | median / max abs lateral (m) | median / max abs heading (deg) | median speed ratio [min, max] | in bound | moved | pre-h within tolerance | abs lateral 1 s later (median) |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 3 s | g0.7s0.96 | 40 | 0.052 / 0.206 | 2.54 / 7.19 | 0.925 [0.888, 0.944] | 30 | 40 | 40 | 0.263 |
| 3 s | g0.7s1.04 | 40 | 0.056 / 0.233 | 2.04 / 6.76 | 1.068 [1.035, 1.110] | 31 | 40 | 40 | 0.390 |
| 3 s | g0.85s0.96 | 40 | 0.025 / 0.100 | 1.33 / 3.72 | 0.928 [0.902, 0.937] | 40 | 40 | 40 | 0.133 |
| 3 s | g0.85s1.04 | 40 | 0.023 / 0.111 | 0.78 / 3.01 | 1.069 [1.051, 1.100] | 39 | 40 | 40 | 0.172 |
| 3 s | g1.15s0.96 | 40 | 0.032 / 0.138 | 1.40 / 4.47 | 0.932 [0.893, 0.962] | 39 | 40 | 40 | 0.228 |
| 3 s | g1.15s1.04 | 40 | 0.057 / 0.157 | 2.30 / 5.94 | 1.074 [1.061, 1.117] | 32 | 40 | 40 | 0.334 |
| 3 s | g1.3s0.96 | 40 | 0.063 / 0.257 | 3.01 / 9.00 | 0.935 [0.896, 0.984] | 27 | 40 | 40 | 0.423 |
| 3 s | g1.3s1.04 | 40 | 0.095 / 0.296 | 4.01 / 11.04 | 1.076 [1.062, 1.137] | 21 | 40 | 40 | 0.617 |
| 5 s | g0.7s0.96 | 21 | 0.068 / 0.200 | 2.72 / 5.20 | 0.923 [0.905, 0.938] | 19 | 21 | 21 | 0.235 |
| 5 s | g0.7s1.04 | 21 | 0.072 / 0.204 | 2.21 / 4.42 | 1.068 [1.050, 1.098] | 21 | 21 | 21 | 0.328 |
| 5 s | g0.85s0.96 | 21 | 0.029 / 0.107 | 1.49 / 2.79 | 0.924 [0.915, 0.944] | 21 | 21 | 21 | 0.129 |
| 5 s | g0.85s1.04 | 21 | 0.030 / 0.083 | 0.87 / 1.83 | 1.071 [1.058, 1.092] | 21 | 21 | 21 | 0.163 |
| 5 s | g1.15s0.96 | 21 | 0.043 / 0.109 | 1.17 / 2.78 | 0.934 [0.916, 0.947] | 21 | 21 | 21 | 0.165 |
| 5 s | g1.15s1.04 | 21 | 0.058 / 0.183 | 2.24 / 4.38 | 1.078 [1.067, 1.093] | 21 | 21 | 21 | 0.308 |
| 5 s | g1.3s0.96 | 21 | 0.081 / 0.221 | 2.49 / 6.17 | 0.932 [0.912, 0.951] | 19 | 21 | 21 | 0.342 |
| 5 s | g1.3s1.04 | 21 | 0.104 / 0.325 | 3.71 / 7.80 | 1.079 [1.066, 1.100] | 14 | 21 | 21 | 0.472 |
| 8 s | g0.7s0.96 | 3 | 0.077 / 0.203 | 2.17 / 4.21 | 0.931 [0.914, 0.936] | 3 | 3 | 3 | 0.244 |
| 8 s | g0.7s1.04 | 3 | 0.073 / 0.214 | 1.38 / 3.71 | 1.066 [1.064, 1.072] | 3 | 3 | 3 | 0.293 |
| 8 s | g0.85s0.96 | 3 | 0.040 / 0.096 | 1.31 / 1.96 | 0.934 [0.915, 0.940] | 3 | 3 | 3 | 0.128 |
| 8 s | g0.85s1.04 | 3 | 0.025 / 0.087 | 0.22 / 1.23 | 1.068 [1.066, 1.074] | 3 | 3 | 3 | 0.090 |
| 8 s | g1.15s0.96 | 3 | 0.042 / 0.107 | 0.82 / 1.79 | 0.935 [0.920, 0.938] | 3 | 3 | 3 | 0.176 |
| 8 s | g1.15s1.04 | 3 | 0.076 / 0.151 | 2.32 / 2.91 | 1.072 [1.066, 1.080] | 3 | 3 | 3 | 0.314 |
| 8 s | g1.3s0.96 | 3 | 0.086 / 0.228 | 2.04 / 4.75 | 0.926 [0.923, 0.944] | 3 | 3 | 3 | 0.347 |
| 8 s | g1.3s1.04 | 3 | 0.134 / 0.284 | 4.07 / 5.71 | 1.075 [1.052, 1.079] | 2 | 3 | 3 | 0.687 |

## Registered lines

- FRONTIER: at some Delta >= 8 failed units with p-hat in [0.2, 0.8] and the controls' mean p-hat >= 0.7 (same Delta; pooled over Delta when no control is rewindable there). Delta 3 s: 8 units in [0.2, 0.8], controls 0.55; Delta 5 s: 5 units in [0.2, 0.8], controls 0.55 (pooled); Delta 8 s: 1 units in [0.2, 0.8], controls 0.55 (pooled)
- NONE: every failed unit p-hat < 0.1 at every Delta: False
- SLOW-ONLY: no frontier
- controls pooled over Delta: 0.55 [0.25, 0.89]

**Verdict: UNDECIDED**
