# NR1 full chain: NR1F-L18-s0, NR1F-L18-s1 against RSG-L18-s0

| tag | read | scenes | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| NR1F-L18-s0 | held | 460 | 39 | 0.8800 | 5 | 19 | 28 | 4 / 1 / 29 | 0.832 | 7 / 0 | vs RSG-L18-s0 +0.0144 [-0.0021, +0.0415] (zeros 39 / 34); vs RTB1ARO-L18-s0 +0.0071 [-0.0039, +0.0257] (zeros 39 / 39) |
| NR1F-L18-s0 | pub | 1491 | 57 | 0.9429 | 2 | 31 | 45 | 1 / 1 / 14 | 0.870 | 17 / 1 | vs RSG-L18-s0 +0.0032 [-0.0052, +0.0113] (zeros 57 / 58); vs RTB1ARO-L18-s0 +0.0052 [+0.0005, +0.0101] (zeros 57 / 65) |
| NR1F-L18-s1 | held | 460 | 41 | 0.8785 | 5 | 20 | 28 | 5 / 0 / 27 | 0.836 | 8 / 0 | vs RSG-L18-s0 +0.0130 [-0.0075, +0.0404] (zeros 41 / 34); vs RTB1ARO-L18-s0 +0.0057 [-0.0089, +0.0267] (zeros 41 / 39) |
| NR1F-L18-s1 | pub | 1491 | 57 | 0.9436 | 1 | 31 | 45 | 1 / 0 / 16 | 0.872 | 18 / 2 | vs RSG-L18-s0 +0.0039 [-0.0044, +0.0127] (zeros 57 / 58); vs RTB1ARO-L18-s0 +0.0059 [+0.0003, +0.0114] (zeros 57 / 65) |

## Waiting set

| tag | a zeros, b median delay (set v2, 312) | the same, held-split | vs RSG: a-zero diff [CI], delay diff (all) | (held-split) |
|---|---|---|---|---|
| RSG-L18-s0 | 7 / 146 (wait ok 0.23) | -0.48 s, never 3 / 166 | 2 / 44 (wait ok 0.23) | -0.68 s, never 1 / 50 | +0 [+0.0, +0.0], delay +0.00 s | +0 [+0.0, +0.0], delay +0.00 s |
| RTB1ARO-L18-s0 | 6 / 146 (wait ok 0.41) | -0.63 s, never 5 / 166 | 2 / 44 (wait ok 0.39) | -0.73 s, never 1 / 50 | -1 [-5.6, +3.1], delay -0.15 s | +0 [+0.0, +0.0], delay -0.05 s |
| NR1F-L18-s0 | - | - | - | - | - |
| NR1F-L18-s1 | - | - | - | - | - |

| tag | line | met |
|---|---|:-:|
| NR1F-L18-s0 | held_zeros | no |
| NR1F-L18-s0 | held_ci | no |
| NR1F-L18-s0 | pub_ci | no |
| NR1F-L18-s0 | pub_rear | yes |
| NR1F-L18-s0 | static | yes |
| NR1F-L18-s0 | W | no |
| NR1F-L18-s0 | G | yes |
| NR1F-L18-s0 | waitset (proposed) | no |
| NR1F-L18-s1 | held_zeros | no |
| NR1F-L18-s1 | held_ci | no |
| NR1F-L18-s1 | pub_ci | yes |
| NR1F-L18-s1 | pub_rear | yes |
| NR1F-L18-s1 | static | no |
| NR1F-L18-s1 | W | no |
| NR1F-L18-s1 | G | yes |
| NR1F-L18-s1 | waitset (proposed) | no |
| PAI NR1F-L18-s0 | paired | no |
| PAI NR1F-L18-s0 | zeros | no |

PAI 281 (NR1F-L18-s0, submission config) vs RSG-L18-s0: -0.0131 [-0.0440, +0.0173], zeros 80 / 76, at-fault 12 / 9

bar 'clearly better than RSG-L18-s0' met: no
