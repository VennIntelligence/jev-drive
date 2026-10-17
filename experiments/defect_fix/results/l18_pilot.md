# NR1 L18 pilots (1 000 steps from RTB1AR-F-s0, merged pool on1_L18NR1, history-speed noise on) against the control NR1H0P-L18-s0

| tag | read | scenes | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| NR1H0P-L18-s0 | held | 460 | 47 | 0.8570 | 5 | 22 | 34 | 5 / 0 / 26 | 0.824 | 9 / 0 | vs RTB1ARO-L18-s0 -0.0159 [-0.0334, +0.0066] (zeros 47 / 39); vs RSG-L18-s0 -0.0086 [-0.0336, +0.0239] (zeros 47 / 34) |
| NR1H0P-L18-s0 | pub | 1491 | 57 | 0.9401 | 2 | 32 | 43 | 2 / 0 / 16 | 0.866 | 22 / 4 | vs RTB1ARO-L18-s0 +0.0023 [-0.0032, +0.0081] (zeros 57 / 65); vs RSG-L18-s0 +0.0004 [-0.0093, +0.0095] (zeros 57 / 58) |
| NR1H15P-L18-s0 | held | 460 | 47 | 0.8569 | 6 | 22 | 33 | 6 / 0 / 27 | 0.823 | 9 / 0 | vs NR1H0P-L18-s0 -0.0001 [-0.0060, +0.0048] (zeros 47 / 47); vs RTB1ARO-L18-s0 -0.0160 [-0.0313, +0.0047] (zeros 47 / 39); vs RSG-L18-s0 -0.0087 [-0.0302, +0.0212] (zeros 47 / 34) |
| NR1H15P-L18-s0 | pub | 1491 | 61 | 0.9376 | 2 | 34 | 48 | 2 / 0 / 15 | 0.866 | 21 / 4 | vs NR1H0P-L18-s0 -0.0024 [-0.0071, +0.0011] (zeros 61 / 57); vs RTB1ARO-L18-s0 -0.0001 [-0.0067, +0.0063] (zeros 61 / 65); vs RSG-L18-s0 -0.0021 [-0.0156, +0.0092] (zeros 61 / 58) |
| NR1H30P-L18-s0 | held | 460 | 46 | 0.8595 | 7 | 21 | 33 | 6 / 1 / 24 | 0.824 | 9 / 0 | vs NR1H0P-L18-s0 +0.0025 [-0.0050, +0.0091] (zeros 46 / 47); vs RTB1ARO-L18-s0 -0.0134 [-0.0284, +0.0043] (zeros 46 / 39); vs RSG-L18-s0 -0.0061 [-0.0266, +0.0209] (zeros 46 / 34) |
| NR1H30P-L18-s0 | pub | 299 | 9 | 0.9571 | 0 | 7 | 8 | 0 / 0 / 4 | 0.859 | 1 / 0 | vs NR1H0P-L18-s0 +0.0003 [-0.0000, +0.0007] (zeros 9 / 9); vs RTB1ARO-L18-s0 +0.0074 [-0.0080, +0.0252] (zeros 9 / 12); vs RSG-L18-s0 +0.0075 [-0.0114, +0.0256] (zeros 9 / 10) |
| NR1H15O-L18-s0 | held | 460 | 42 | 0.8672 | 6 | 18 | 28 | 6 / 0 / 25 | 0.823 | 9 / 0 | vs NR1H0P-L18-s0 +0.0103 [+0.0027, +0.0194] (zeros 42 / 47); vs RTB1ARO-L18-s0 -0.0056 [-0.0211, +0.0155] (zeros 42 / 39); vs RSG-L18-s0 +0.0017 [-0.0217, +0.0337] (zeros 42 / 34) |

| tag | W | G | static scene | held lateral zeros |
|---|--:|--:|---|--:|
| NR1H0P-L18-s0 | 0.166 | 0.998 | 0.78 m, no contact | 34 |
| NR1H15P-L18-s0 | 0.169 | 0.998 | 0.78 m, no contact | 33 |
| NR1H30P-L18-s0 | 0.164 | 0.998 | 0.78 m, no contact | 33 |
| NR1H15O-L18-s0 | 0.157 | 0.998 | 0.78 m, no contact | 28 |

## Waiting set

| tag | a zeros, b median delay (set v2, 312) | the same, held-split | vs RSG: a-zero diff [CI], delay diff (all) | (held-split) |
|---|---|---|---|---|
| RSG-L18-s0 | - | - | - | - | - |
| RTB1ARO-L18-s0 | - | - | - | - | - |
| NR1H0P-L18-s0 | - | - | - | - | - |
| NR1H15P-L18-s0 | - | - | - | - | - |
| NR1H30P-L18-s0 | - | - | - | - | - |
| NR1H15O-L18-s0 | - | - | - | - | - |

hinge rule (amendment 2, vs the control): held zeros <= control and held lateral zeros < control (fewest held lateral zeros among the passing)
NR1H15P-L18-s0: {'held_zeros': True, 'held_lateral': True}; NR1H30P-L18-s0: {'held_zeros': True, 'held_lateral': True}; NR1H15O-L18-s0: {'held_zeros': True, 'held_lateral': True}
pick: NR1H15O-L18-s0
