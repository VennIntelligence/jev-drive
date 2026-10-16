# RSG-ACC pilot offline gates

Lines: G1 dev ADE <= 0.538; G2 rely read; G3 equal-arc turn vs RSG-L18-s0 >= -0.3 deg (pilot); G4 contact rate on the new nuPlan dev rows vs the same-k control, CI hi < 0 (PAI dev rows reported).

| tag | control | dev ADE | rely (d_junction) | turn equal-arc deg | contact nuPlan dev (tag / ctl, d [CI]) | contact PAI dev | lines |
|:--|:--|--:|:--|:--|:--|:--|:--|
| RSGACCpa-L18-s0 | RSGACCpp0k13-L18-s0 | 0.5286 | True (+0.024) | -0.24 [-0.46, -0.07] | 0.1084 / 0.1057, +0.0027 [+0.0007, +0.0043] (n 21688, 4 logs) | 0.0377 / 0.0394, -0.0017 [-0.0045, +0.0001] | G1 ok G2 ok G3 ok G4 NO |
| RSGACCpb-L18-s0 | RSGACCpp0k13-L18-s0 | 0.5289 | True (+0.024) | -0.15 [-0.28, -0.03] | 0.1066 / 0.1057, +0.0010 [+0.0003, +0.0017] (n 21688, 4 logs) | 0.0378 / 0.0394, -0.0016 [-0.0043, +0.0001] | G1 ok G2 ok G3 ok G4 NO |
| RSGACCpc-L18-s0 | RSGACCpp0k26-L18-s0 | 0.5285 | True (+0.024) | +0.15 [-0.00, +0.30] | 0.1060 / 0.1025, +0.0036 [-0.0002, +0.0069] (n 21688, 4 logs) | 0.0380 / 0.0388, -0.0008 [-0.0024, +0.0004] | G1 ok G2 ok G3 ok G4 NO |

control dev ADE 0.5290424823760986

Rule: passing cells none; selected **none: stop, no full run**
