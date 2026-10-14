# RSGSELF-L18 vs RSG-L18: ladder table (rf_selfread.py table)

| arm | held mean s0 / s1 | held zeros | public mean | public zeros | lateral abs dlat2 (m) |
|:--|:--|:--|:--|:--|:--|
| RSG-L18 | 0.8809 / 0.8830 | 26 / 26 | 0.9473 / 0.9487 | 46 / 44 | 0.3024 / 0.2932 |
| RSGSELF-L18 | 0.8758 / 0.8739 | 29 / 30 | 0.9384 / 0.9402 | 57 / 55 | 0.3014 / 0.2933 |

Paired differences RSGSELF-L18 - RSG-L18 (log-clustered 95 % CI; ol_confirm on the common scenes):

| pair | held mean | held zeros | public mean | public zeros | lateral abs dlat2 (m) |
|:--|:--|:--|:--|:--|:--|
| RSGSELF-L18 - RSG-L18 | -0.0070 [-0.0166, +0.0032] | +3.5 [-1.2, +7.9] | -0.0088 [-0.0151, -0.0031] | +11.0 [+2.8, +20.3] | +0.0055 [-0.0060, +0.0170] |
| RSGSELF-L18-s0 - RSG-L18-s0 | -0.0051 [-0.0192, +0.0085] | +3.0 [-3.8, +10.1] | -0.0090 [-0.0158, -0.0031] | +11.0 [+2.5, +20.9] | +0.0030 [-0.0137, +0.0198] |
| RSGSELF-L18-s1 - RSG-L18-s1 | -0.0090 [-0.0213, +0.0045] | +4.0 [-2.2, +9.4] | -0.0086 [-0.0155, -0.0023] | +11.0 [+2.0, +21.2] | +0.0087 [-0.0085, +0.0249] |

| line | value | pass |
|:--|:--|:--|
| held mean (seed mean) >= ref - 0.01 | 0.8749 vs 0.8719 | yes |
| held zeros s0 <= ref s0 + 3 | 29 vs 29 | yes |
| public zeros s0 <= ref (seed mean) + 5 | 57 vs 50 | **no** |
| lateral abs dlat2, seed means, arm - ref <= +0.02 m | +0.0055 | yes |
| exam entry speed (Tokyo) | read later on Tokyo | pending |

Sets: held 452 scenes, public 1479 scenes. Lateral: lat.
