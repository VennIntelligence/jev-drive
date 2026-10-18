# PRACTICE-PAI gate: FAIL

Chunks c000, c002, c004, c006 of the PAI 281 exam (144 scenes); scored by both arms without an RPC error: 144.

| arm | mean | zeros |
|:--|--:|--:|
| pilot (+ PAI pool) | 0.6608 | 45 |
| control (no new pool) | 0.6707 | 44 |

- zeros: 45 <= 44 + 2: True
- paired mean pilot - control: -0.0099 [-0.0531, +0.0328] (scene-clustered bootstrap 95 % CI, 10 000 draws, seed 0); line mean >= 0: False

Reported, not gating: against the stored reference RSG-L18-s0 + ACC + GAP (acc_gap/read/s2) on the same scenes:

- pilot - reference: -0.0150 [-0.0542, +0.0214], zeros 45 vs 43 (n 144)
- control - reference: -0.0051 [-0.0336, +0.0235], zeros 44 vs 43 (n 144)
