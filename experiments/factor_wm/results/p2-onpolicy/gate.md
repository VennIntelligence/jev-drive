# Gate (seed 0)

verdict: STOP

- (b) HUGSIM-12 exam stall X 9 > P2 1 + 2
- (c) HUGSIM-12 exam HD X - P2 -0.126 < -0.10
- (b) HUGSIM-12 spec stall X 9 > P2 1 + 2
- (c) HUGSIM-12 spec HD X - P2 -0.101 < -0.10

| read | X - P2 | C - P2 | X - C |
|:--|:--|:--|:--|
| navtest W EPDMS | -0.29 [-0.64, +0.08] | -0.11 [-0.41, +0.20] | -0.17 [-0.46, +0.13] |
| navhard G combined | +2.58 [-0.62, +5.97] | | +0.94 [-1.42, +3.22] |
| HUGSIM-12 exam HD (X / P2 / C) | 0.280 / 0.406 / 0.345 | | |
| HUGSIM-12 exam stall (X / P2 / C) | 9 / 1 / 3 | | |
| HUGSIM-12 exam stuck (X / P2 / C) | 1 / 0 / 0 | | |
| HUGSIM-12 spec HD (X / P2 / C) | 0.313 / 0.414 / 0.402 | | |
| HUGSIM-12 spec stall (X / P2 / C) | 9 / 1 / 3 | | |
| HUGSIM-12 spec stuck (X / P2 / C) | 0 / 0 / 0 | | |

engine: {"P2-F-s0": {"g0b_stall": 0.013812154696132596, "g0b_heading": 0.7071823204419889, "g0b_lane": 0.04696132596685083, "g0b_fail": 0.7679558011049724, "g0b_launch_stall": 0.0125, "g1s_false_go": 0.8083333333333333}, "PC-s0": {"g0b_stall": 0.0055248618784530384, "g0b_heading": 0.6767955801104972, "g0b_lane": 0.07458563535911603, "g0b_fail": 0.7569060773480663, "g0b_launch_stall": 0.0125, "g1s_false_go": 0.675}, "PX-s0": {"g0b_stall": 0.013812154696132596, "g0b_heading": 0.2845303867403315, "g0b_lane": 0.19613259668508287, "g0b_fail": 0.494475138121547, "g0b_launch_stall": 0.0625, "g1s_false_go": 0.016666666666666666}}
