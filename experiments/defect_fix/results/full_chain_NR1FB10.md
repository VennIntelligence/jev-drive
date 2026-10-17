# NR1 full chain: NR1FB10-L18-s0, NR1FB10-L18-s1 against RSG-L18-s0

| tag | read | scenes | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| NR1FB10-L18-s0 | held | 460 | 47 | 0.8590 | 10 | 17 | 30 | 9 / 1 / 28 | 0.824 | 8 / 0 | vs RSG-L18-s0 -0.0066 [-0.0271, +0.0258] (zeros 47 / 34); vs RTB1ARO-L18-s0 -0.0139 [-0.0306, +0.0104] (zeros 47 / 39) |
| NR1FB10-L18-s0 | pub | 1491 | 59 | 0.9407 | 4 | 34 | 44 | 3 / 1 / 17 | 0.865 | 17 / 0 | vs RSG-L18-s0 +0.0010 [-0.0073, +0.0091] (zeros 59 / 58); vs RTB1ARO-L18-s0 +0.0030 [-0.0029, +0.0084] (zeros 59 / 65) |
| NR1FB10-L18-s1 | held | 460 | 48 | 0.8609 | 7 | 22 | 35 | 6 / 1 / 27 | 0.831 | 8 / 0 | vs RSG-L18-s0 -0.0047 [-0.0304, +0.0280] (zeros 48 / 34); vs RTB1ARO-L18-s0 -0.0120 [-0.0314, +0.0104] (zeros 48 / 39) |
| NR1FB10-L18-s1 | pub | 1491 | 57 | 0.9433 | 2 | 34 | 44 | 2 / 0 / 19 | 0.870 | 16 / 0 | vs RSG-L18-s0 +0.0036 [-0.0051, +0.0122] (zeros 57 / 58); vs RTB1ARO-L18-s0 +0.0056 [+0.0001, +0.0110] (zeros 57 / 65) |

## Waiting set

| tag | a zeros, b median delay (set v2, 312) | the same, held-split | vs RSG: a-zero diff [CI], delay diff (all) | (held-split) |
|---|---|---|---|---|
| RSG-L18-s0 | 7 / 146 (wait ok 0.23) | -0.48 s, never 3 / 166 | 2 / 44 (wait ok 0.23) | -0.68 s, never 1 / 50 | +0 [+0.0, +0.0], delay +0.00 s | +0 [+0.0, +0.0], delay +0.00 s |
| RTB1ARO-L18-s0 | 6 / 146 (wait ok 0.41) | -0.63 s, never 5 / 166 | 2 / 44 (wait ok 0.39) | -0.73 s, never 1 / 50 | -1 [-5.6, +3.1], delay -0.15 s | +0 [+0.0, +0.0], delay -0.05 s |
| NR1FB10-L18-s0 | - | - | - | - | - |
| NR1FB10-L18-s1 | - | - | - | - | - |

| tag | line | met |
|---|---|:-:|
| NR1FB10-L18-s0 | held_zeros | no |
| NR1FB10-L18-s0 | held_ci | no |
| NR1FB10-L18-s0 | pub_ci | no |
| NR1FB10-L18-s0 | pub_rear | yes |
| NR1FB10-L18-s0 | static | no |
| NR1FB10-L18-s0 | W | no |
| NR1FB10-L18-s0 | G | yes |
| NR1FB10-L18-s0 | waitset (proposed) | no |
| NR1FB10-L18-s1 | held_zeros | no |
| NR1FB10-L18-s1 | held_ci | no |
| NR1FB10-L18-s1 | pub_ci | no |
| NR1FB10-L18-s1 | pub_rear | no |
| NR1FB10-L18-s1 | static | no |
| NR1FB10-L18-s1 | W | no |
| NR1FB10-L18-s1 | G | yes |
| NR1FB10-L18-s1 | waitset (proposed) | no |
| PAI NR1FB10-L18-s0 | paired | no |
| PAI NR1FB10-L18-s0 | zeros | no |

PAI 281 (NR1FB10-L18-s0, submission config) vs RSG-L18-s0: -0.0277 [-0.0612, +0.0035], zeros 85 / 76, at-fault 12 / 9

bar 'clearly better than RSG-L18-s0' met: no
