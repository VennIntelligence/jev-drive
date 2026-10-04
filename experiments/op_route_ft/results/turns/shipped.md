# shipped: 25 B2D junction turns (decision 127 set)

ONNX `None`, route adapter `None`. Arm: op_arb.sh spec, zones off, open-loop camera, desire on, seed 2 (= rig122 olnz). Scored 25 / 25 turns; route DS mean 26.3 over 20 routes.

| turns | n | took exit | shipped took | gained / lost vs shipped | leaves lane (of entered) | collisions in window |
|---|--:|--:|--:|--:|--:|--:|
| choice | 13 | 0 | 0 | 0 / 0 | 11 / 11 | 4 |
| forced | 12 | 1 | 1 | 0 / 0 | 7 / 9 | 1 |
| all | 25 | 1 | 1 | 0 / 0 | 18 / 20 | 5 |

| route | turn | forced | kind | angle | R_min m | entered | branch | took | shipped took | leaves | peak m | head peak 1/m | need 1/m | coll |
|---|--:|--:|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 10255 | 0 | 0 | junction | 90 | 6.4 | 1 | lost | 0 | 0 | 1 | 25.07 | 0.014 | 0.157 | 0 |
| 15102 | 0 | 0 | junction | -80 | 11.5 | 1 | lost | 0 | 0 | 1 | 25.03 | 0.23 | 0.087 | 1 |
| 24416 | 0 | 1 | curve | 51 | 36.6 | 1 | lost | 0 | 0 | 0 | 1.39 | 0.081 | 0.027 | 0 |
| 24758 | 0 | 1 | curve | -79 | 33.3 | 1 | lost | 0 | 0 | 0 | 1.70 | 0.046 | 0.030 | 0 |
| 24758 | 1 | 0 | curve | 35 | 31.2 | 0 | never | 0 | 0 | - | nan | nan | 0.032 | 0 |
| 24944 | 0 | 1 | curve | -89 | 10.7 | 1 | lost | 0 | 0 | 1 | 8.48 | 0.015 | 0.093 | 0 |
| 24944 | 1 | 0 | junction | -89 | 8.7 | 0 | never | 0 | 0 | - | nan | nan | 0.116 | 0 |
| 25051 | 0 | 1 | junction | 90 | 6.2 | 1 | lost | 0 | 0 | 1 | 20.36 | 0.057 | 0.160 | 0 |
| 26153 | 0 | 1 | curve | 37 | 41.0 | 1 | lost | 0 | 0 | 1 | 6.76 | 0.271 | 0.024 | 0 |
| 26153 | 1 | 1 | curve | -112 | 15.4 | 0 | never | 0 | 0 | - | nan | nan | 0.065 | 0 |
| 26365 | 0 | 1 | curve | 85 | 42.5 | 0 | never | 0 | 0 | - | nan | nan | 0.024 | 0 |
| 26723 | 0 | 1 | curve | 69 | 12.0 | 1 | lost | 0 | 0 | 1 | 12.14 | 0.034 | 0.083 | 0 |
| 26723 | 1 | 1 | curve | 82 | 11.3 | 0 | never | 0 | 0 | - | nan | nan | 0.089 | 0 |
| 26872 | 0 | 0 | junction | -61 | 12.6 | 1 | lost | 0 | 0 | 1 | 25.02 | 0.013 | 0.079 | 0 |
| 27297 | 0 | 0 | junction | 88 | 6.6 | 1 | lost | 0 | 0 | 1 | 1.90 | 0.101 | 0.151 | 0 |
| 27994 | 0 | 1 | junction | -101 | 7.0 | 1 | lost | 0 | 0 | 1 | 25.01 | 0.106 | 0.144 | 0 |
| 28008 | 0 | 1 | curve | -53 | 33.7 | 1 | yes | 1 | 1 | 1 | 2.73 | 0.159 | 0.030 | 1 |
| 28008 | 1 | 0 | junction | -90 | 8.1 | 1 | lost | 0 | 0 | 1 | 25.00 | 0.009 | 0.124 | 0 |
| 28147 | 0 | 0 | junction | -88 | 15.2 | 1 | lost | 0 | 0 | 1 | 25.03 | 0.042 | 0.066 | 0 |
| 28180 | 0 | 1 | junction | 87 | 4.7 | 1 | lost | 0 | 0 | 1 | 18.55 | 0.16 | 0.213 | 0 |
| 334 | 0 | 0 | junction | -88 | 7.4 | 1 | lost | 0 | 0 | 1 | 2.56 | 0.042 | 0.134 | 1 |
| 34183 | 0 | 0 | junction | -89 | 13.4 | 1 | lost | 0 | 0 | 1 | 14.86 | 0.022 | 0.074 | 0 |
| 5423 | 0 | 0 | junction | 90 | 6.2 | 1 | lost | 0 | 0 | 1 | 23.14 | 0.179 | 0.161 | 1 |
| 6999 | 0 | 0 | junction | -90 | 6.1 | 1 | lost | 0 | 0 | 1 | 25.05 | 0.048 | 0.165 | 1 |
| 9196 | 0 | 0 | junction | -89 | 8.1 | 1 | lost | 0 | 0 | 1 | 25.02 | 0.016 | 0.123 | 0 |
