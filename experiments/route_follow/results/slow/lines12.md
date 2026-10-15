# SLOW S1 / S2: lines 1 (held 460 @18) and 2 (public 1491 @10 guard), seed means vs RSG-L18 (prereg 读数与线)

| arm | held zeros d (<= +3) | held GT d m (<= +0.03) | held score d (>= 0) | rear-ended arm / RSG (<= 0.7) | line 1 | guard score d (CI lo > -0.01) | guard zeros d (<= +5) | plan 2 s short d (CI lo > 0) | line 2 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| S1 | +5.0 [-1.6, +11.6] | +0.081 [+0.018, +0.153] | +0.0098 [-0.0055, +0.0260] | 22.5 / 35.5 = 0.63 | **FAIL** | -0.0009 [-0.0052, +0.0030] | +6.5 [+1.5, +12.2] | +0.091 [+0.079, +0.103] | **FAIL** |
| S3 | +3.0 [-0.5, +6.8] | +0.062 [-0.012, +0.159] | -0.0042 [-0.0129, +0.0036] | 32.5 / 35.5 = 0.92 | **FAIL** | -0.0019 [-0.0039, -0.0001] | +2.5 [+0.0, +5.2] | +0.031 [+0.023, +0.038] | **PASS** |
| S2 | +7.5 [+0.6, +13.9] | +0.254 [+0.109, +0.418] | +0.0042 [-0.0113, +0.0215] | 24.5 / 35.5 = 0.69 | **FAIL** | -0.0007 [-0.0054, +0.0036] | +5.5 [-0.5, +11.8] | +0.127 [+0.114, +0.141] | **FAIL** |

Interaction readings (Amendment 1): RSGS1-L18 - RSGSall-L18: not read (Sall stopped at the pilot, prereg Amendment 3); RSGS2-L18 - RSGS3-L18: guard zeros +3.0 [-2.6, +9.2], held zeros +4.5 [-1.1, +9.9]

Line 1: zeros <= RSG + 3 (point), GT <= RSG + 0.03 m (point; CI shown), rear-ended (collision_rear > 0, seed mean on the common scenes) <= 0.7 x RSG, mean score >= RSG (point; CI shown). Line 2: public mean-score difference CI low > -0.01, zeros <= RSG + 5 (point), plan 2 s short (rf_deficit, positive = less short) CI low > 0. Sources: held18/confirm.json, confirm/confirm.json, deficit/deficit.json under /root/autodl-tmp/ujs/runs/route_follow/slow

