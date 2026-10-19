# PNX: frozen board-mix PCS on the native arena held read (secondary, not a gate)

slow_rescore.py's definitions (decision 382 fits, items frozen; one read = one more submission; CI = log-clustered bootstrap of the paired PCS difference, 2 000 draws). Slow = 0 < score < 0.98, type slow / creep / no-launch, non-static, non-launch stratum; launch-short = the same in the launch stratum; near-1 = 0.98 <= score < 1.

| A - B | windows | mean diff [CI] | PCS board-mix A / B, diff [CI] | PCS g15 A / B, diff [CI] | zeros A / B | slow A / B | launch-short A / B | near-1 A / B | exact 1 A / B |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTVXS-L18-s0@off - NTV0-L18-s0@off | 2228 | +0.0103 [+0.0034, +0.0186] | 1109 / 1104, +5 [-2, +14] | 1122 / 1116, +6 [-5, +19] | 469 / 494 | 116 / 108 | 26 / 23 | 30 / 26 | 1580 / 1568 |
| JTXS-L18-s1@off - JTN-L18-s1@off | 2228 | +0.0016 [-0.0129, +0.0197] | 1237 / 1247, -9 [-33, +10] | 1297 / 1317, -20 [-59, +8] | 204 / 210 | 105 / 98 | 42 / 44 | 19 / 18 | 1847 / 1847 |
| JTXS-L18-s1@off - NTV0-L18-s0@off | 2228 | +0.1226 [+0.0793, +0.1740] | 1237 / 1104, +133 [+102, +166] | 1297 / 1116, +181 [+141, +224] | 204 / 494 | 105 / 108 | 42 / 23 | 19 / 26 | 1847 / 1568 |
| NTVXS-L18-s0@off - JTN-L18-s1@off | 2228 | -0.1107 [-0.1554, -0.0728] | 1109 / 1247, -138 [-173, -105] | 1122 / 1317, -195 [-242, -152] | 469 / 210 | 116 / 98 | 26 / 44 | 30 / 18 | 1580 / 1847 |

Items scored: 2106 of 2106 frozen items (MAP on the read's items).
