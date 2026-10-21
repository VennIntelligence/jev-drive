# Harmonizer render-noise check: R6DAG-L18-s1 ON re-read of c000 + c001 vs the stored ON read

Scenes compared 72; served plans compared on 72 scenes, 14233 inferences (14 bit-identical); scenes whose every served plan is identical: 0; first differing inference k: median 0.0, min 0.

| quantity | value |
|:--|--:|
| mean stored ON / re-read ON | 0.784 / 0.801 |
| re-read - stored (paired, 95 % CI) | +0.017 [+0.001, +0.038] |
| scenes with a different score / identical | 14 / 58 |
| mean abs scene diff | 0.0174 |
| sd of scene diff | 0.0785 |
| zero flips (stored 0 -> not / new 0) | 1 (1 / 0) |
| scenes moving > 0.5 | 0 |
| implied per-read noise sd of a 279-scene mean | 0.0033 |
| stored ON - OFF on these scenes | -0.006 [-0.073, +0.062] |
| re-read ON - OFF on these scenes | +0.010 [-0.056, +0.079] |
| share of the stored ON - OFF scene variance that is render noise | 0.036 |

## Scenes whose score moved

| scene | OFF | stored ON | re-read ON | class stored / re-read | first plan diff k |
|:--|--:|--:|--:|:--|--:|
| af85ffdb | 1.00 | 0.57 | 1.00 | rear-hit / ok | 0 |
| 5ddb96bd | 0.62 | 0.62 | 1.00 | rear-hit / ok | 0 |
| 6e6cbe67 | 0.00 | 0.00 | 0.36 | side / rear-hit | 0 |
| bbc53aa8 | 0.74 | 0.76 | 0.80 | slow / slow | 0 |
| 91d84204 | 0.90 | 0.93 | 0.95 | rear-hit / rear-hit | 0 |
| 7cfa81de | 1.00 | 1.00 | 0.99 | ok / ok | 0 |
| b44a4a44 | 0.92 | 0.80 | 0.80 | slow / slow | 0 |
| 9d9e7f39 | 1.00 | 0.93 | 0.92 | slow / slow | 0 |
| 6250146a | 0.94 | 0.94 | 0.94 | rear-hit / rear-hit | 0 |
| 94c1d722 | 0.98 | 0.97 | 0.97 | slow / slow | 0 |
| bd97de34 | 0.48 | 0.48 | 0.48 | rear-hit / rear-hit | 0 |
| b65e55e9 | 1.00 | 1.00 | 1.00 | ok / ok | 0 |
| 596af0c7 | 0.62 | 0.62 | 0.62 | rear-hit / rear-hit | 0 |
| 7c346fd5 | 0.81 | 0.83 | 0.83 | rear-hit / rear-hit | 0 |
