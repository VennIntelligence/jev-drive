# RECOVERY-ENVELOPE RCVL (L): registered lines vs RSG-L18

Tags RCVL-L18-s0, RCVL-L18-s1; reference RSG-L18-s0 / -s1 (stored reads). Held: 452 paired scenes / 39 logs; public: 1479 scenes / 136 logs; log-clustered 95 % CI, 10000 draws, seed 0.

**Verdict: NOT ADOPTED** (H1 FAIL, guard PASS, own lines pending: PAI_cor_route_end, PAI_rear_ended, PAI_zeros)

| line | value [95 % CI] | registered | result |
|:--|:--|:--|:--|
| H1 held zeros d | +4.0 [-2.9, +10.7] | <= +3 (point) | FAIL |
| H1 held at-fault zeros d | +0.5 [-5.1, +5.9] | <= +3 (point) | PASS |
| H1 held GT distance d (m) | -0.027 [-0.117, +0.052] | <= +0.03 m (point) | PASS |
| held mean score d | +0.0038 [-0.0136, +0.0212] | reported | - |
| guard public mean score d | -0.0016 [-0.0079, +0.0049] | CI low > -0.01 | PASS |
| guard public zeros d | -5.0 [-13.1, +2.6] | <= +5 (point) | PASS |
| H2 held turns completed d (of ~24) | -2.0 [-4.5, +0.5] | >= +3 (seed-mean completed turns) | FAIL |
| PAI all-281 zeros, arm vs ref | pending | | pending |
| PAI cor_route_end, arm vs ref | pending | | pending |
| PAI rear-ended, arm vs ref | pending | | pending |
| PAI strict 36 completed, arm vs ref | pending | | pending |

Held turns (rf_follow, 452 x 18 decisions): RCVL-L18-s0 12 / 24; RCVL-L18-s1 14 / 24; RSG-L18-s0 16 / 24; RSG-L18-s1 14 / 24; paired on the 24 turn scenes common to every label (14 logs): -2.0 [-4.5, +0.5].

Lines (prereg 全量读数与线, verbatim): H1 zeros <= +3, at-fault <= +3, GT <= +0.03 m (points; CIs reported); H2 held 24 turns completion seed mean >= +3 for R and RL; PAI 281 (seed 0): strict 36 completion >= 10 (reference 4), all-281 zeros <= reference + 3; L / RL: PAI cor_route_end <= reference - 3 and rear-ended <= reference + 3; public guard: score CI low > -0.01, zeros <= +5. Adopt only if H1 + guard pass and the arm's own H2 line passes. Own lines as read here: R = held turns + PAI strict 36 + PAI zeros; L = PAI cor_route_end + rear-ended + PAI zeros; RL = all five. PAI reference = RSG-L18-s0 read on the box (box and Tokyo PAI reads are deterministic per host but not equal, results/p4_self/box_det.md); the strict-36 threshold stays the registered absolute 10.
Sources: /root/autodl-tmp/ujs/runs/route_follow/recovery_envelope/RCVL/held18, confirm, follow, pai_londiag.json, pai_turns.json. Held complete: RCVL-L18-s0 RCVL-L18-s1; public complete: RCVL-L18-s0 RCVL-L18-s1.
