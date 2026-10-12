# STEER-LONG amendment 4: high-speed steering jumps and the force-GT handoff

## nuplan: |vx wz| > 6 m/s^2 within 1.5 s after the force-GT handoff (t = 0.5 s), by speed at the handoff

| speed m/s | rollouts | violent | of which zero |
|:--|--:|--:|--:|
| (0, 5] | 1116 | 0 | 0 |
| (5, 10] | 1206 | 0 | 0 |
| (10, 15] | 390 | 0 | 0 |
| (15, 20] | 0 | 0 | 0 |
| (20, 25] | 0 | 0 | 0 |
| (25, 40] | 0 | 0 | 0 |

## pai: |vx wz| > 6 m/s^2 within 1.5 s after the force-GT handoff (t = 1.7 s), by speed at the handoff

| speed m/s | rollouts | violent | of which zero |
|:--|--:|--:|--:|
| (0, 5] | 234 | 0 | 0 |
| (5, 10] | 186 | 0 | 0 |
| (10, 15] | 213 | 0 | 0 |
| (15, 20] | 72 | 0 | 0 |
| (20, 25] | 63 | 6 | 6 |
| (25, 40] | 75 | 28 | 27 |

Per label: ybase 12, ys0 13, ys1 9

## Handoff replay (force-GT phase replayed with the logged reference; |u| at the handoff step, rad)

| rollout | logged | default | w_steer 20 | w_steer 50 | iter 200 | GT reference | plan at ego speed |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ybase e97cd14e | -0.222 | -0.578 | -0.035 | +0.001 | -0.578 | +0.006 | -0.276 |
| ybase 67033e50 | +0.110 | +0.533 | +0.033 | -0.000 | +0.533 | -0.011 | +0.212 |
| ybase a3a82422 | +0.038 | +0.504 | +0.015 | -0.006 | +0.504 | -0.013 | +0.232 |
| ybase 6e2dabb4 | -0.529 | -0.759 | -0.375 | -0.007 | -0.759 | +0.001 | -0.271 |
| ybase c05b602e | -0.008 | -0.398 | -0.007 | +0.003 | -0.398 | +0.007 | -0.153 |
| ybase 9462af88 | +0.002 | +0.388 | +0.001 | -0.008 | +0.388 | -0.015 | +0.127 |
| ybase 3308ed56 | -0.139 | -0.525 | -0.034 | +0.015 | -0.525 | +0.023 | -0.222 |
| ybase c2c112f6 | -0.057 | +0.550 | +0.002 | +0.001 | +0.550 | +0.001 | +0.103 |
| ybase 73605c18 | +0.000 | -0.325 | -0.000 | +0.000 | -0.325 | -0.004 | -0.176 |
| ybase 910c6a1c | -0.006 | -0.001 | -0.004 | -0.005 | -0.001 | -0.004 | +0.001 |
| ybase c9251b1a | -0.006 | -0.009 | -0.010 | -0.010 | -0.009 | -0.015 | -0.008 |
| ybase 1a86a372 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 |
| ybase 080ec648 | +0.001 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.000 |
| ybase 543b39c6 | +0.001 | -0.001 | +0.001 | +0.001 | -0.001 | -0.000 | -0.001 |

Controls (15 clean PAI rollouts, v >= 20 m/s, 2 s of steps, step-by-step replay on logged states): default reproduces the logged command to median 0.0005 rad; max |u_arm - u_default| median: w_steer20 0.0025, w_steer50 0.0027 rad.
