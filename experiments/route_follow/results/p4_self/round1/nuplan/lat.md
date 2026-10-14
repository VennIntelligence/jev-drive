# Lateral guard: the plan's 2 s sideways move relative to the route on non-junction held decisions (rf_selfread.py lat)

Decisions up to the first zero event, fed speed >= 1.0 m/s, anchor class straight / curve (rf_offroad.rclass); per scene mean, then the mean over scenes. Guard: arm - control <= +0.02 m on mean |dlat2|.

| label | scenes | decisions | mean abs dlat2 (m) | mean signed dlat2 (m, left +) | mean abs lateral of the 2 s pose (m) |
|:--|--:|--:|--:|--:|--:|
| RSGSELF-L18-s0 | 363 | 3942 | 0.3014 | -0.0369 | 0.6457 |
| RSGSELF-L18-s1 | 357 | 3928 | 0.2933 | -0.0286 | 0.6237 |
| RSG-L18-s0 | 363 | 3938 | 0.3024 | -0.0412 | 0.6443 |
| RSG-L18-s1 | 361 | 3928 | 0.2932 | -0.0243 | 0.6260 |
| RSGSELF-L18 | 356 | seed mean | 0.2944 | -0.0378 | 0.6323 |
| RSG-L18 | 359 | seed mean | 0.2976 | -0.0310 | 0.6266 |

| A - B | scenes / logs | abs dlat2 [95 % CI] | signed dlat2 [95 % CI] | guard (<= +0.02 m) |
|:--|:--|:--|:--|:--|
| RSGSELF-L18 - RSG-L18 | 349 / 39 | +0.0055 [-0.0060, +0.0170] | -0.0133 [-0.0300, +0.0040] | pass |
| RSGSELF-L18-s0 - RSG-L18-s0 | 357 / 39 | +0.0030 [-0.0137, +0.0198] | -0.0067 [-0.0285, +0.0151] | pass |
| RSGSELF-L18-s1 - RSG-L18-s1 | 352 / 39 | +0.0087 [-0.0085, +0.0249] | -0.0107 [-0.0329, +0.0121] | pass |
