# CFT (on1_CFT) registered reads, 2026-10-13

Prereg: ../../plans/2026-10-13-cft-prereg.md. RCFT-L18-s{0,1} (2500 steps, init RB0-F-s0, `--on on1_CFT --on-mode both`), control RNONEG-L18-s{0,1}.

| line | RCFT s0 / s1 | reference | verdict |
|:--|:--|:--|:--|
| 1 G-swap junction median ratio (>= 0.50) | 0.314 / 0.317 (side share 0.75 / 0.75) | on1_CF RB-F 0.55 / 0.56 (side 0.94 / 0.95) | FAIL |
| 2 dev ADE memory on (<= RB0-F 0.528 / 0.531 + 0.01) | 0.5348 / 0.5341 | RB0-F 0.528 / 0.531 | PASS (+0.007 / +0.003); G-rely diff 0.0156 [0.004, 0.028] / 0.0148 [0.003, 0.026] |
| 3 public 1491 @10, zeros <= RSG-L18 + 5 | 59 zeros, mean 0.9352 (s0 only) | RSG-L18 45 (46 s0), mean 0.9480; YS-L18 47.5 / 0.9450 | FAIL (+13 vs RSG, paired +13 [2, 25]; mean -0.0122 [-0.0204, -0.0047] vs RSG-L18-s0; -0.0101 vs YS-L18-s1) |
| 4 PAI hard 131 (Tokyo) | not run here | | open |

Held 460 @18 (common scenes): see held_raw.txt.

| tag | mean | zeros |
|:--|--:|--:|
| RCFT-L18-s0 / s1 | 0.8528 / 0.8486 | 41 / 42 |
| RNONEG-L18-s0 / s1 (no rows) | 0.8156 / 0.8123 | 60 / 61 |
| RSG-L18-s0 / s1 | 0.8656 / 0.8676 | 34 / 34 |
| RB0-F-s0 / s1 | 0.8220 / 0.8252 | 59 / 59 |

RCFT beats the no-row controls by about +0.036 and -19 zeros, and is worse than RSG by about -0.016 and +7 zeros.

Held turn completion (rf_follow, 24 turns): RCFT 62.5 / 62.5 %, RSG 66.7 / 58.3 %, RNONEG 41.7 / 50.0 %, RB0 54.2 / 62.5 %. RCFT s0 - RSG-L18-s0: -4.2 pp [-30.4, +20.7]; RCFT - RNONEG same seed: +20.8 [-4.8, +42.3] / +12.5 [-11.1, +35.0]. See follow_held.md.
