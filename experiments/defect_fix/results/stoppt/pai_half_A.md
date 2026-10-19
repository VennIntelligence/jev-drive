# PRACTICE-PAI gate: FAIL

Chunks c000, c002, c004, c006 of the PAI 281 exam (144 scenes); scored by both arms without an RPC error: 144.

| arm | mean | zeros |
|:--|--:|--:|
| pilot (+ PAI pool) | 0.6657 | 41 |
| control (no new pool) | 0.6733 | 40 |

- zeros: 41 <= 40 + 2: True
- paired mean pilot - control: -0.0076 [-0.0332, +0.0128] (scene-clustered bootstrap 95 % CI, 10 000 draws, seed 0); line mean >= 0: False

Reported, not gating: against the stored reference RSG-L18-s0 + ACC + GAP (acc_gap/read/s2) on the same scenes:

- pilot - reference: -0.0101 [-0.0553, +0.0350], zeros 41 vs 43 (n 144)
- control - reference: -0.0024 [-0.0526, +0.0469], zeros 40 vs 43 (n 144)
