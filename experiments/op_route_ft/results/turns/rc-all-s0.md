# rc-all-s0: 25 B2D junction turns (decision 127 set)

ONNX `/root/autodl-tmp/ujs/runs/op_route_ft/onnx/rc-all-s0.onnx`, route adapter `/root/autodl-tmp/ujs/runs/op_route_ft/onnx/rc-all-s0.adapter.npz`. Arm: op_arb.sh spec, zones off, open-loop camera, desire on, seed 2 (= rig122 olnz). Scored 25 / 25 turns; route DS mean 31.0 over 20 routes.

| turns | n | took exit | shipped took | gained / lost vs shipped | leaves lane (of entered) | collisions in window |
|---|--:|--:|--:|--:|--:|--:|
| choice | 13 | 3 | 0 | 3 / 0 | 11 / 12 | 4 |
| forced | 12 | 3 | 1 | 2 / 0 | 8 / 9 | 4 |
| all | 25 | 6 | 1 | 5 / 0 | 19 / 21 | 8 |

Took-exit rate minus shipped, paired, route-cluster bootstrap: +0.200 [+0.038, +0.400] (n 25, one seed).

| route | turn | forced | kind | angle | R_min m | entered | branch | took | shipped took | leaves | peak m | head peak 1/m | need 1/m | coll |
|---|--:|--:|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 10255 | 0 | 0 | junction | 90 | 6.4 | 1 | lost | 0 | 0 | 1 | 22.13 | 0.118 | 0.157 | 0 |
| 15102 | 0 | 0 | junction | -80 | 11.5 | 1 | lost | 0 | 0 | 1 | 6.46 | 0.079 | 0.087 | 0 |
| 24416 | 0 | 1 | curve | 51 | 36.6 | 1 | yes | 1 | 0 | 1 | 2.47 | 0.2 | 0.027 | 1 |
| 24758 | 0 | 1 | curve | -79 | 33.3 | 1 | yes | 1 | 0 | 0 | 1.70 | 0.062 | 0.030 | 0 |
| 24758 | 1 | 0 | curve | 35 | 31.2 | 1 | yes | 1 | 0 | 0 | 1.55 | 0.034 | 0.032 | 0 |
| 24944 | 0 | 1 | curve | -89 | 10.7 | 1 | lost | 0 | 0 | 1 | 7.38 | 0.165 | 0.093 | 0 |
| 24944 | 1 | 0 | junction | -89 | 8.7 | 0 | never | 0 | 0 | - | nan | nan | 0.116 | 0 |
| 25051 | 0 | 1 | junction | 90 | 6.2 | 1 | lost | 0 | 0 | 1 | 5.60 | 0.619 | 0.160 | 0 |
| 26153 | 0 | 1 | curve | 37 | 41.0 | 1 | lost | 0 | 0 | 1 | 3.96 | 0.154 | 0.024 | 0 |
| 26153 | 1 | 1 | curve | -112 | 15.4 | 0 | never | 0 | 0 | - | nan | nan | 0.065 | 0 |
| 26365 | 0 | 1 | curve | 85 | 42.5 | 0 | never | 0 | 0 | - | nan | nan | 0.024 | 0 |
| 26723 | 0 | 1 | curve | 69 | 12.0 | 1 | lost | 0 | 0 | 1 | 6.55 | 0.383 | 0.083 | 1 |
| 26723 | 1 | 1 | curve | 82 | 11.3 | 0 | never | 0 | 0 | - | nan | nan | 0.089 | 0 |
| 26872 | 0 | 0 | junction | -61 | 12.6 | 1 | lost | 0 | 0 | 1 | 25.02 | 0.016 | 0.079 | 0 |
| 27297 | 0 | 0 | junction | 88 | 6.6 | 1 | lost | 0 | 0 | 1 | 25.04 | 0.132 | 0.151 | 2 |
| 27994 | 0 | 1 | junction | -101 | 7.0 | 1 | lost | 0 | 0 | 1 | 25.05 | 0.892 | 0.144 | 1 |
| 28008 | 0 | 1 | curve | -53 | 33.7 | 1 | yes | 1 | 1 | 1 | 4.54 | 0.114 | 0.030 | 0 |
| 28008 | 1 | 0 | junction | -90 | 8.1 | 1 | lost | 0 | 0 | 1 | 25.02 | 0.107 | 0.124 | 0 |
| 28147 | 0 | 0 | junction | -88 | 15.2 | 1 | lost | 0 | 0 | 1 | 25.04 | 0.119 | 0.066 | 0 |
| 28180 | 0 | 1 | junction | 87 | 4.7 | 1 | lost | 0 | 0 | 1 | 4.20 | 0.973 | 0.213 | 1 |
| 334 | 0 | 0 | junction | -88 | 7.4 | 1 | yes | 1 | 0 | 1 | 5.52 | 1.21 | 0.134 | 1 |
| 34183 | 0 | 0 | junction | -89 | 13.4 | 1 | yes | 1 | 0 | 1 | 5.08 | 1.139 | 0.074 | 1 |
| 5423 | 0 | 0 | junction | 90 | 6.2 | 1 | lost | 0 | 0 | 1 | 9.04 | 0.547 | 0.161 | 0 |
| 6999 | 0 | 0 | junction | -90 | 6.1 | 1 | lost | 0 | 0 | 1 | 25.03 | 0.597 | 0.165 | 0 |
| 9196 | 0 | 0 | junction | -89 | 8.1 | 1 | lost | 0 | 0 | 1 | 8.42 | 0.167 | 0.123 | 0 |
