# PAIT80 pilot reads (on1_PAIT80, on1_PAITC80), seed 0, 2026-10-13

Prereg: ../../plans/2026-10-13-turning-school-prereg.md (80-scene pilot pools). RPAIT80-L18-s0 = `--on on1_PAIT80 --on-k 12 --on-mode both --on-weight topo=straight_exit:2,topo=straight_road:2`;
RPAITC80-L18-s0 = `--on on1_PAITC80` (combined pool, the prereg's `on1_PAITC` at 80 scenes; that is the pool name that exists on the box) with `pool=a:9,` added to the weights. Both: init RB0-F-s0, 2500 steps, recipe as RCFT. Chain: scripts/rf_pait80.sh.

Held 460 @18 (held_raw.txt), public 1491 @10 guard (guard_public1491_10.md), held turn completion (follow_held.md; 24 turns).

| tag | held mean | held zeros | turn completion (24) | public mean | public zeros | public at-fault zeros |
|:--|--:|--:|:--|--:|--:|--:|
| RPAIT80-L18-s0 | 0.7997 | 43 | 54.2 % | 0.9062 | 69 | 65 |
| RPAITC80-L18-s0 | 0.8016 | 44 | 58.3 % | 0.9109 | 66 | 58 |
| RSG-L18-s0 | 0.8656 | 34 | 66.7 % | 0.9473 | 46 | 39 |
| RNONEG-L18-s0 | 0.8156 | 60 | 41.7 % | not read here | | |
| RB0-F-s0 | 0.8220 | 59 | 54.2 % | | | |

Verdicts (single seed, all CIs wide):
- Held 460: both PAIT80 arms are below RSG (-0.066 / -0.064 mean) and below the no-row control RNONEG (-0.016 / -0.014) and RB0-F (-0.022 / -0.020) on the mean, but have fewer zeros than RNONEG / RB0 (43 / 44 vs 60 / 59); at-fault collision zeros 8 each (RSG 4, RNONEG 6).
- Turn completion: PAIT80 54.2 % (-12.5 pp [-31.6, +4.5] vs RSG; +12.5 [-5.9, +34.5] vs RNONEG; 0.0 vs RB0); PAITC80 58.3 % (-8.3 [-31.0, +14.8] vs RSG; +16.7 [0.0, +33.3] vs RNONEG). Neither beats RSG; both sit between RNONEG and RSG.
- Public 1491 @10 guard: FAIL for both. Mean -0.0261 [-0.0364, -0.0154] (PAIT80) and -0.0214 [-0.0298, -0.0129] (PAITC80) vs YR10m10-F-s0 (guard limit -0.02); vs RSG-L18-s0 -0.0412 [-0.0509, -0.0309] / -0.0365 [-0.0455, -0.0278], zeros +23 [11, 35] / +20 [10, 31]; at-fault zeros +26 / +19. Worse than RCFT-L18-s0 too (0.9352).
- Conclusion: the 80-scene PAIT pools do not reproduce the RSG gain; the guard is clearly failed (offroad zeros 53 / 47 vs RSG 31). The held turn read is not distinguishable from the controls at n = 24. Tokyo exam reads requested (EXAM_READ_REQUEST_PILOT).
