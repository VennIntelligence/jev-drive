| contrast                                            |   mean |    lo |    hi |     n |   units |
|:----------------------------------------------------|-------:|------:|------:|------:|--------:|
| P1: G - N (EPDMS)                                   |   1.92 |  1.17 |  2.73 | 12146 |     136 |
| P1: G - W (EPDMS)                                   |   0.19 | -0.30 |  0.69 | 12146 |     136 |
| P2: G - N (EPDMS)                                   |   4.00 |  3.12 |  4.93 | 12146 |     136 |
| P2: G - W (EPDMS)                                   |   0.90 |  0.47 |  1.31 | 12146 |     136 |
| interaction (P2 - P1)_N - (P2 - P1)_G               |  -2.08 | -2.63 | -1.52 | 12146 |     136 |
| interaction (P2 - P1)_W - (P2 - P1)_G               |  -0.71 | -1.23 | -0.20 | 12146 |     136 |
| P2 - P1 under N                                     |   2.78 |  2.19 |  3.38 | 12146 |     136 |
| seed spread P1 N: s1 - s0                           |   0.48 |  0.29 |  0.69 | 12146 |     136 |
| seed spread P2 N: s1 - s0                           |   0.17 | -0.10 |  0.44 | 12146 |     136 |
| P2 - P1 under W                                     |   4.15 |  3.63 |  4.67 | 12146 |     136 |
| seed spread P1 W: s1 - s0                           |   0.09 | -0.07 |  0.23 | 12146 |     136 |
| seed spread P2 W: s1 - s0                           |   0.12 | -0.14 |  0.37 | 12146 |     136 |
| P2 - P1 under G                                     |   4.86 |  4.18 |  5.52 | 12146 |     136 |
| seed spread P1 G: s1 - s0                           |   0.07 | -0.02 |  0.17 | 12146 |     136 |
| seed spread P2 G: s1 - s0                           |   0.11 | -0.09 |  0.30 | 12146 |     136 |
| P1 N: real-frame test - matched test (1 499 tokens) |   2.12 |  1.81 |  2.45 |  1499 |      78 |
| P2 N: real-frame test - matched test (1 499 tokens) |   0.78 |  0.57 |  0.99 |  1499 |      78 |
| P1 W: real-frame test - matched test (1 499 tokens) |   3.22 |  2.25 |  4.13 |  1499 |      78 |
| P2 W: real-frame test - matched test (1 499 tokens) |   0.73 | -0.52 |  1.95 |  1499 |      78 |
| P1 G: real-frame test - matched test (1 499 tokens) |   3.28 |  2.01 |  4.56 |  1499 |      78 |
| P2 G: real-frame test - matched test (1 499 tokens) |  -0.47 | -1.48 |  0.56 |  1499 |      78 |

CI: percentile bootstrap

per-token EPDMS x 100, seed means; cluster bootstrap over navtest logs, B 10000
