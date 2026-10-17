# SELF-ITER gate pilot: arms vs WTCP-L18-s0 (top decile by RSG-L18-s0 on the held-out trainval rows)

rows: {"dev": 1789, "tv_hold": 7236, "top": 724, "free": 277, "start": 319}

| arm | dev ADE (<= C + 0.01) | top-decile ADE (drop >= 3%) | diff CI | free-road 4 s arc (C +- 0.1) | median |cross 2 s| (<= C + 0.05) | standstill-launch 4 s arc (|.| <= |C| + 0.02) | PASS |
|---|--:|--:|--:|--:|--:|--:|:-:|
| C WTCP-L18-s0 | 0.5291 | 1.998 | - | -0.354 | 0.036 | +0.627 | - |
| SIU1P-L18-s0 | 0.5463 x | 1.080 (+45.9%) | [-1.181, -0.528] | -0.306 | 0.031 | -1.027 x | no |
| SIM3P-L18-s0 | 0.5616 x | 0.989 (+50.5%) | [-1.256, -0.644] | -0.296 | 0.031 | -1.430 x | no |
| SIM6P-L18-s0 | 0.5731 x | 0.932 (+53.3%) | [-1.312, -0.693] | -0.237 x | 0.031 | -1.660 x | no |

passing (best top-decile drop first): none
