# rc-*-pre: 25 B2D junction turns (guard b2d_turns set, zones off, seed 2)

took = the car left through the commanded exit. Paired differences: route-cluster bootstrap (cllib.paired_ci), n 25, one seed, one run per cell.

## desire off

| arm | took (25) | choice (13) | forced (12) | entered | leaves lane (of entered) | collisions in windows | route DS mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| shipped | not run / incomplete (24 / 25 scored) | | | | | | |
| rc-ctl-s0 | 5 | 3 | 2 | 21 | 18 / 21 | 3 | 38.5 |
| rc-bear-s0 | 6 | 1 | 5 | 22 | 18 / 22 | 11 | 30.5 |
| rc-ctl-pre-s0 | 3 | 2 | 1 | 20 | 16 / 20 | 7 | 23.0 |
| rc-bear-pre-s0 | 1 | 1 | 0 | 19 | 18 / 19 | 6 | 22.4 |

| paired difference (took rate) | diff [95% CI] | gained / lost |
|---|---|--:|
| rc-bear-pre-s0 - rc-ctl-pre-s0 | -0.08 [-0.24, +0.08] | 1 / 3 |
| rc-bear-pre-s0 - rc-bear-s0 | -0.20 [-0.41, +0.00] | 1 / 6 |
| rc-ctl-pre-s0 - rc-ctl-s0 | -0.08 [-0.24, +0.08] | 1 / 3 |
| rc-bear-s0 - rc-ctl-s0 | +0.04 [-0.17, +0.26] | 4 / 3 |

## desire on

| arm | took (25) | choice (13) | forced (12) | entered | leaves lane (of entered) | collisions in windows | route DS mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| shipped | 1 | 0 | 1 | 20 | 18 / 20 | 5 | 26.3 |
| rc-ctl-s0 | 6 | 2 | 4 | 21 | 17 / 21 | 7 | 34.0 |
| rc-bear-s0 | 4 | 1 | 3 | 21 | 18 / 21 | 5 | 36.2 |
| rc-ctl-pre-s0 | 1 | 0 | 1 | 20 | 17 / 20 | 7 | 22.0 |
| rc-bear-pre-s0 | 0 | 0 | 0 | 19 | 18 / 19 | 6 | 23.1 |

| paired difference (took rate) | diff [95% CI] | gained / lost |
|---|---|--:|
| rc-bear-pre-s0 - rc-ctl-pre-s0 | -0.04 [-0.12, +0.00] | 0 / 1 |
| rc-bear-pre-s0 - rc-bear-s0 | -0.16 [-0.33, +0.00] | 0 / 4 |
| rc-ctl-pre-s0 - rc-ctl-s0 | -0.20 [-0.39, -0.04] | 0 / 5 |
| rc-bear-pre-s0 - shipped | -0.04 [-0.12, +0.00] | 0 / 1 |
| rc-bear-s0 - rc-ctl-s0 | -0.08 [-0.28, +0.15] | 2 / 4 |

