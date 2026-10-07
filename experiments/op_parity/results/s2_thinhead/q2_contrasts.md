| features   | contrast                                                         |      d |     lo |     hi |
|:-----------|:-----------------------------------------------------------------|-------:|-------:|-------:|
| E          | ridge head (F20) - constant per intent (straight / left / right) |  0.085 |  0.013 |  0.171 |
| E          | ridge head (F20) - constant on turn-intent frames only           |  0.059 | -0.012 |  0.140 |
| E          | margin - argmax (F20)                                            | -0.005 | -0.030 |  0.016 |
| E          | F30 - F20                                                        | -0.032 | -0.072 |  0.004 |
| E          | speed only - F20                                                 | -0.030 | -0.079 |  0.022 |
| E          | path only - F20                                                  | -0.138 | -0.232 | -0.052 |
| E          | shipped: transferred - retrained                                 |  0.004 | -0.074 |  0.076 |
| Q1+E       | margin - argmax (F20)                                            |  0.007 | -0.015 |  0.031 |
| Q1+E       | F30 - F20                                                        | -0.033 | -0.076 |  0.008 |
| Q1+E       | speed only - F20                                                 | -0.041 | -0.100 |  0.017 |
| Q1+E       | path only - F20                                                  | -0.113 | -0.210 | -0.024 |
| Q1+E       | shipped: transferred - retrained                                 | -0.049 | -0.140 |  0.035 |
| C+E        | margin - argmax (F20)                                            |  0.024 |  0.003 |  0.046 |
| C+E        | F30 - F20                                                        |  0.004 | -0.034 |  0.039 |
| C+E        | speed only - F20                                                 |  0.001 | -0.044 |  0.048 |
| C+E        | path only - F20                                                  | -0.102 | -0.191 | -0.023 |
| C+E        | shipped: transferred - retrained                                 | -0.038 | -0.136 |  0.050 |
|            | Q1+E - E (F20, margin)                                           | -0.014 | -0.049 |  0.026 |
|            | C+E - E (F20, margin)                                            | -0.027 | -0.050 | -0.003 |
|            | selected shipped vs selected WP2 (absolute RFS, E, margin)       | -0.193 | -0.335 | -0.062 |

CI: percentile bootstrap
