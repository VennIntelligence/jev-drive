# (iv) ANTI-COPY pilot lines (RSG-L18 recipe + history-speed noise, 1 000 steps, seed 0; open loop)

low-v0 dev rows: 0.5 <= v0 < 8.0 m/s, n 1394 (32 logs)

| arm | dev ADE (<= C + 0.01) | 4 s arc bias, history x 0.8 (arm / C) | abs bias diff [CI] (hi < 0) | 4 s bias, own history (arm / C; within 0.05) | history gain d arc4 / d s (arm / C) | lat 2 s vs C (<= 0.05 m) | wait W | pass |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|
| WTCP-L18-s0 (C) | 0.5290 | - | - | - | - | - | 0.007 | - |
| ACS2P-L18-s0 (M 0.2) | 0.5301 | -0.62 / -0.93 | -0.165 [-0.203, -0.125] | +0.03 / +0.00 | 3.12 / 4.67 | 0.005 | 0.008 | PASS |
| ACS4P-L18-s0 (M 0.4) | 0.5399 | -0.30 / -0.93 | -0.291 [-0.368, -0.209] | +0.07 / +0.00 | 1.68 / 4.67 | 0.009 | 0.008 | fail |

verdict: PASS; passing ACS2P-L18-s0
