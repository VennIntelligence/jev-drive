# Route-night Amendment 6 read (2026-10-13, prereg `b5e71c76`)

Box: chain `$DATA_DIR/runs/route_follow/a6chain.sh`, reads `a6read.sh nomem | public`, job ids `a6jobs.txt`; public rollouts
`runs/route_follow/guard/read10/RBSG3-L18-s{0,1}/b0..b9`, no-memory reads `runs/route_mem/read/{RSG,RBSG2}-L18-s0-nomem/c0..c5/run`.

## Experiment 1: iteration (RBSG3-L18 = round 3 on RBSG2-L18-s0's own rollouts)

Pool gates (all pass): rows L18RBSG2 (train 1031) R1 0.9993, wrong-seed 0.027, interpreted control 1.0, R2 pass, 1 rollout without time
alignment; rows L18RBSG2allheld (held 460) R1 0.9985, wrong-seed 0.026, R2 pass, 5 rollouts unaligned; on1_L18RBSG2all 25 372 rows, route
ok 1.0, cov20 0.486, route0_share_005 0.995; **on1_L18R3 60 181 rows**, route ok 0.9999, cov20 0.472, route0_share_005 0.994.

G-rely (L1): **pass, both seeds**: junction dADE +0.019 / +0.024 m, junction - straight +0.027 [+0.011, +0.043] / +0.028 [+0.014, +0.042]
(RBSG2 failed: +0.004 / +0.008). Dev ADE (memory on) 0.536 / 0.535 (RB0 0.528, RSG 0.528, RBSG2 0.549).

Public 1491 at 10 decisions (`all10_RBSG3.md`, 1479 scenes / 136 logs, seed means, log-clustered B = 10 000):

| checkpoint | mean | zeros | collision / offroad / corridor | at-fault | GT m | inside-right offroad (per seed) |
|:--|--:|--:|:--|--:|--:|:--|
| RBSG3-L18-s0 / s1 | 0.9491 / 0.9474 | 52 / 54 | 11/36/6 · 14/33/9 | 47 / 47 | 0.879 / 0.866 | 6 / 7 |
| RBSG3-L18 | 0.9482 | 53 | 12.5 / 34.5 / 7.5 | 47 | 0.872 | |
| RBSG2-L18 | 0.9319 | 77.5 | 15.5 / 53.5 / 8.5 | 69 | 0.900 | 12 / 12 |
| RSG-L18 | 0.9480 | 45 | 8 / 30.5 / 7.5 | 38.5 | 0.796 | 4 / 4 |
| YS2G-L18 | 0.9511 | 43 | 9.5 / 24 / 10 | 33.5 | 0.885 | |
| YR10m10-F-s0 (base) | 0.9323 | 68 | 7 / 43 / 20 | 50 | 0.941 | |

| a - b | mean | zeros | at-fault | GT m |
|:--|:--|:--|:--|:--|
| RBSG3 - RBSG2 | +0.0163 [+0.0100, +0.0229] | **-24.5 [-34.3, -15.2]** | -22.0 [-31.6, -12.9] | -0.028 [-0.049, -0.005] |
| RBSG3 - RSG | +0.0002 [-0.0061, +0.0060] | +8.0 [-0.5, +17.0] | +8.5 [+0.0, +17.7] | +0.077 [+0.039, +0.116] |
| RBSG3 - YS2G | -0.0028 [-0.0111, +0.0049] | +10.0 [-1.4, +22.4] | +13.5 [+2.0, +25.7] | -0.013 [-0.058, +0.034] |
| RBSG3 - base | +0.0160 [+0.0079, +0.0242] | -15.0 [-27.7, -2.3] | -3.0 [-15.1, +9.1] | -0.068 [-0.113, -0.024] |
| RBSG3-s0 - RBSG2-s0 | +0.0205 [+0.0124, +0.0295] | -30.0 [-43.1, -18.1] | -27.0 [-39.6, -15.5] | -0.017 [-0.043, +0.011] |
| RBSG3-s1 - RBSG2-s1 | +0.0120 [+0.0045, +0.0193] | -19.0 [-30.1, -7.6] | -17.0 [-28.4, -5.3] | -0.039 [-0.061, -0.016] |

Inside-right = rf_offroad held on the public rollouts (`offroad_pub_counts.json`; offroad zeros whose ego is inside a >= 20 deg right turn).

- **A6-1 pass**: zeros 53 < 77.5, RBSG3 - RBSG2 -24.5 [-34.3, -15.2] (both seeds alone too).
- **A6-2 fail**: at-fault 47 > RSG 38.5 + 5 = 43.5 (RBSG3 - RSG +8.5 [+0.0, +17.7]).
- Reading: the third round moved most of the way back from RBSG2 toward RSG (zeros 77.5 -> 53, RSG 45; inside-right 12 -> 6-7, RSG 4)
  and kept the route dependence (G-rely back to pass), but it is not at RSG: zeros +8.0 (CI touches 0), at-fault +8.5, GT 0.08 m worse.
  Neither clean "converge" (A6-2 fails) nor "oscillate" (zeros well below RBSG2): damped, not yet converged after one more round.
- Better seed: **RBSG3-L18-s0** (52 zeros, mean 0.9491, inside-right 6; s1 54 / 0.9474 / 7); copied to Tokyo.

## Experiment 2: closed-loop reliance (route memory fed empty, held 460 at 18 decisions)

`nomem.md` (460 scenes each; scene bootstrap CI, log-clustered CI in parentheses in the file):

| label | mean | zeros | offroad / corridor / collision | at-fault | GT m | lat GT m | inside-right |
|:--|--:|--:|:--|--:|--:|--:|--:|
| RSG-L18-s0 stored | 0.8656 | 34 | 17 / 6 / 4 | 21 | 1.135 | 0.869 | 1 |
| RSG-L18-s0 no memory | 0.8092 | 58 | 21 / 24 / 7 | 28 | 1.188 | 0.997 | 3 |
| RBSG2-L18-s0 stored | 0.8703 | 37 | 22 / 3 / 4 | 26 | 1.215 | 0.742 | 4 |
| RBSG2-L18-s0 no memory | 0.8084 | 54 | 21 / 23 / 5 | 26 | 1.192 | 0.960 | 2 |

| nomem - stored | mean | zeros | corridor z | offroad z | at-fault | lat GT m |
|:--|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0 | -0.056 [-0.081, -0.034] | **+24 [+13, +36]** | +18 [+10, +27] | +4 [-4, +13] | +7 [-3, +17] | +0.128 [+0.065, +0.192] |
| RBSG2-L18-s0 | -0.062 [-0.089, -0.036] | **+17 [+5, +30]** | +20 [+12, +29] | -1 [-10, +8] | +0 [-10, +10] | +0.219 [+0.132, +0.308] |
| difference (RSG - RBSG2) | +0.006 [-0.021, +0.031] | +7 [-5, +20] | -2 [-8, +3] | +5 [-5, +16] | +7 [-5, +20] | -0.091 [-0.172, -0.010] |

- **A6-3 passes as written** (RSG Δzeros CI > 0; RBSG2's Δ 17 below RSG's 24), but the difference is not resolved (+7 [-5, +20]) and the
  premise does not hold: RBSG2, which fails the offline G-rely, loses as much in closed loop (zeros +17 with CI > 0, mean -0.062, lateral
  GT +0.22 m, the largest). Both lose mostly by leaving the corridor (+18 / +20). The offline G-rely (dev junction rows, memory masked) does
  not predict closed-loop reliance: RBSG2 relies on the memory in closed loop although its open-loop ADE barely moves without it.
