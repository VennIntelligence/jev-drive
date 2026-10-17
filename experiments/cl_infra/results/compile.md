# rm_train torch.compile vs eager (RSG-L18 recipe, 2 500 steps from RB0-F-s0; prereg plans/2026-10-17-compile-prereg.md)

Eager identity RSGE-L18-s0 (today's code, flag off) vs RSG-L18-s0: identical True, weight max |diff| 0.0, on_hash_60 ['ea8cacc3e451c577', 'ea8cacc3e451c577']

| tag | git | card | wall s | loop s | steady ms/step | warm-up s | dev ADE |
|---|---|--:|--:|--:|--:|--:|--:|
| RSGE-L18-s0 | 0f4edd6c | 1 | 459 | 449 | 197.1 | -1 | 0.5284 |
| RSG-L18-s0 | d4864575 | 0 | 382 | 372 | 156.3 | 0 | 0.5284 |
| RSG-L18-s1 | d4864575 | 1 | 377 | 367 | 152.1 | 0 | 0.5270 |
| RSGC-L18-s0 | 0f4edd6c | 2 | 264 | 253 | 103.7 | 6 | 0.5286 |
| RSGC-L18-s1 | 0f4edd6c | 3 | 272 | 260 | 106.4 | 7 | 0.5269 |

saved per training: 195 s, 187 s; wall speed-up 1.74x, 1.69x; steady 1.90x, 1.85x

weights: RSGC-L18-s0 vs RSG-L18-s0: max 1.45e-02, L2 0.972; RSGC-L18-s1 vs RSG-L18-s1: max 1.78e-02, L2 0.946; RSG-L18-s1 vs RSG-L18-s0 (seed): max 3.47e-02, L2 2.998

| line | value E0 / E1 / C0 / C1 | spread (floor) | C - E (s0, s1) | paired C - E (s0; s1; seed) | pass |
|---|---|--:|--:|---|:-:|
| dev ADE | 0.5284 / 0.5270 / 0.5286 / 0.5269 | 0.0014 | +0.0001, -0.0001 | - | True |
| wait_W | 0.0050 / 0.0050 / 0.0046 / 0.0050 | 0.0000 (0.0004) | -0.0004, +0.0000 | -0.000 [-0.001, 0.000]; 0.000 [0.000, 0.000]; 0.000 [0.000, 0.000] | False |
| go_G | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 0.0000 (0.0021) | +0.0000, +0.0000 | 0.000 [0.000, 0.000]; 0.000 [0.000, 0.000]; 0.000 [0.000, 0.000] | True |
| S_L | 0.3798 / 0.3765 / 0.3765 / 0.3748 | 0.0034 (0.0017) | -0.0034, -0.0017 | -0.003 [-0.009, 0.000]; -0.002 [-0.008, 0.004]; -0.003 [-0.015, 0.008] | True |
| desire4 | 0.6849 / 0.6890 / 0.6889 / 0.6801 | 0.0041 (0.0000) | +0.0040, -0.0090 | - | False |
| lateral 2 s med abs diff (m) | C vs E: 0.0006, 0.0010 | E1 vs E0: 0.0055 | - | - | True |

| closed loop | scenes | zeros J / C | mean J / C | paired J - C [session CI] | at-fault J / C | offroad J / C | identical scores | pass |
|---|--:|--:|--:|--:|--:|--:|:-:|:-:|
| held RSGC-L18-s0 vs RSG-L18-s0 | 460 | 34 / 34 | 0.8657 / 0.8656 | 0.000 [-0.004, 0.007] | 4 / 4 | 17 / 17 | False | True |
| held RSGC-L18-s1 vs RSG-L18-s1 | 460 | 33 / 34 | 0.8719 / 0.8676 | 0.004 [0.001, 0.012] | 6 / 6 | 13 / 13 | False | False |
| held seed s1 vs s0 | 460 | 34 / 34 | 0.8676 / 0.8656 | 0.002 [-0.009, 0.015] | 6 / 4 | 13 / 17 | False | - |
| pai RSGC-L18-s0 vs RSG-L18-s0 | 281 | 73 / 76 | 0.6829 / 0.6736 | 0.009 [-0.000, 0.021] | 8 / 9 | 14 / 16 | False | True |
| pai RSGC-L18-s1 vs RSG-L18-s1 | 281 | 72 / 71 | 0.6902 / 0.6910 | -0.001 [-0.017, 0.016] | 9 / 7 | 15 / 17 | False | True |
| pai seed s1 vs s0 | 281 | 71 / 76 | 0.6910 / 0.6736 | 0.017 [0.001, 0.035] | 7 / 9 | 17 / 16 | False | - |

PAI E0: today's re-read vs the stored read: 0.000 [0.000, 0.000], identical scores True; E0 used: stored

lines: dev_ade True, ledger False, held False, pai True; **KEEP eager default**
