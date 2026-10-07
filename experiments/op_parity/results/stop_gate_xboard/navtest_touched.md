| quantity                                                       |     n |   share % |
|:---------------------------------------------------------------|------:|----------:|
| tokens fed vx < 0.5 m/s (gate fires)                           |   852 |      7.01 |
| of which launch tokens                                         |   751 |      6.18 |
| of which logged 4 s distance <= 5 m (stay)                     |   101 |      0.83 |
| launch tokens (v0 < 2, logged 4 s > 5 m), all                  |  1902 |     15.66 |
| SG plan differs from BASE (max |dxy| > 1e-4 m), seed 0         |   852 |      7.01 |
| SG plan differs from BASE (max |dxy| > 1e-4 m), seed 1         |   852 |      7.01 |
| DN plan differs from BASE, seed 0                              | 12146 |    100    |
| DN plan differs from BASE, seed 1                              | 12146 |    100    |
| SG plans identical to BASE on untouched tokens, seed 0 (check) |     0 |      0    |
| SG plans identical to BASE on untouched tokens, seed 1 (check) |     0 |      0    |

The last two rows count untouched tokens whose SG plan differs from the stored baseline plan: 0 means BASE reuse is exact.
