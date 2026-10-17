# NR1 base pilots: open-loop lines (C = RTB1AR-F-s0, dev ADE 0.5519) and the plain L18 continuations against RTB1ARO-L18-s0

| base | dose | dev ADE (<= 0.5619) | W (pilot line >= 0.6) | G (>= 0.9) | D | into W_L | S_L stop | arrival 4 s arc k=1 (<= 2.0) | launch-cell 4 s bias (+-0.10) | candidate |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| RTB1AR-F-s0 (C) | - | 0.5519 | 0.326 | 0.992 | 0.318 | 0.026 | 0.921 | 0.89 | +0.02 | - |
| NR1B05-F-s0 | 0.05/0.05 | 0.5804 x | 0.836 | 0.973 | 0.809 | 0.005 | 0.930 | 0.96 | +nan | no |
| NR1B10-F-s0 | 0.10/0.10 | 0.5801 x | 0.860 | 0.977 | 0.837 | 0.007 | 0.930 | 0.99 | +nan | no |
| NR1BALL-F-s0 | natural | 0.5931 x | 0.971 | 0.885 x | 0.856 | 0.001 | 0.938 | 1.07 | +nan | no |

## Plain continuations (V2's L18 command from each candidate base), closed loop

| tag | read | scenes | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| RTB1ARO-L18-s0 (V2) | held | 460 | 39 | 0.8729 | 7 | 16 | 25 | 6 / 1 / 31 | 0.817 | 8 / 0 | vs RSG-L18-s0 +0.0073 [-0.0054, +0.0245] (zeros 39 / 34) |
| RTB1ARO-L18-s0 (V2) | pub | 1491 | 65 | 0.9378 | 5 | 36 | 51 | 3 / 2 / 17 | 0.864 | 17 / 0 | vs RSG-L18-s0 -0.0020 [-0.0111, +0.0066] (zeros 65 / 58) |

## Waiting set (closed loop, alpasim_nuplan/waitset@v1; base checkpoints and their continuations)

| tag | a zeros, b median delay (set v2, 312) | the same, held-split | vs RSG: a-zero diff [CI], delay diff (all) | (held-split) |
|---|---|---|---|---|
| RTB1AR-F-s0 | - | - | - | - | - |
| RTB1ARO-L18-s0 | - | - | - | - | - |
| RSG-L18-s0 | - | - | - | - | - |
| NR1B05-F-s0 | - | - | - | - | - |
| NR1B10-F-s0 | - | - | - | - | - |
| NR1BALL-F-s0 | - | - | - | - | - |
| NR1B05O-L18-s0 | - | - | - | - | - |
| NR1B10O-L18-s0 | - | - | - | - | - |
| NR1BALLO-L18-s0 | - | - | - | - | - |

closed-loop admissible (amendment 2: held zeros <= V2 continuation + 3): none
waiting-set eligible (b median delay <= V2 + 0.5 s): none
rule: no waiting-set read usable: highest W among the closed-loop-admissible candidates
pick: none -> the full chain starts from RTB1AR-F-s0
