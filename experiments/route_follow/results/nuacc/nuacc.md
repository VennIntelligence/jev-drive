# nuPlan JEV_ACC: held 460 @18, 452 paired scenes, 39 logs; acc vs ref (same-code re-read of RSG-L18-s0)

Arms: ref = RSG-L18-s0 no switch (re-read 2026-10-16), acc = JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1, vl = JEV_VCONT=1.0 JEV_LEAD=1, stored = the 2026-10-13 RSG-L18-s0 read. Official camera set (ALPASIM_FAST unset). Log-clustered 95 % CI, 10 000 draws.

| registered line | value | result |
|:--|:--|:--|
| held mean paired d, CI lower > -0.01 | -0.0357 [-0.0654, -0.0079] | FAIL |
| zeros <= ref + 2 | 43 vs 26 (d +17 [+8.0, +25.9]) | FAIL |
| cross-track GT paired d (point) <= +0.05 m | +0.081 [+0.000, +0.166] m | FAIL |
| at-fault km >= 0.9 x ref (point) | 0.827 vs 0.988 km (ratio 0.84) | FAIL |

**Verdict: FAIL (held); no public read.**

| arm | mean score | zeros | at-fault zeros | GT (m) | cross-track GT (m) | past-end GT (m) | at-fault km |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ref (RSG-L18-s0-re1016) | 0.8809 | 26 | 21 | 1.135 | 0.845 | 0.290 | 0.988 |
| acc (RSG-L18-s0-acc) | 0.8452 | 43 | 29 | 2.465 | 0.927 | 1.538 | 0.827 |
| vl (RSG-L18-s0-vl) | 0.8364 | 25 | 18 | 0.974 | 0.833 | 0.141 | 1.108 |
| stored (RSG-L18-s0) | 0.8809 | 26 | 21 | 1.135 | 0.845 | 0.290 | 0.988 |

| a - b | mean score | zeros | at-fault zeros | GT (m) | cross-track GT (m) | past-end GT (m) | at-fault km |
|:--|:--|:--|:--|:--|:--|:--|:--|
| acc - ref | -0.0357 [-0.0654, -0.0079] | +17.0 [+8.0, +25.9] | +8.0 [+0.0, +15.1] | +1.330 [+1.056, +1.577] | +0.081 [+0.000, +0.166] | +1.248 [+0.963, +1.512] | -0.161 |
| vl - ref | -0.0445 [-0.0724, -0.0193] | -1.0 [-7.3, +4.6] | -3.0 [-8.7, +2.0] | -0.161 [-0.248, -0.077] | -0.012 [-0.051, +0.022] | -0.148 [-0.237, -0.063] | +0.120 |
| acc - vl | +0.0088 [-0.0137, +0.0319] | +18.0 [+9.0, +27.2] | +11.0 [+3.3, +17.7] | +1.491 [+1.233, +1.719] | +0.094 [+0.010, +0.178] | +1.397 [+1.125, +1.644] | -0.281 |
| stored - ref | +0.0000 [+0.0000, +0.0000] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 |

New zeros under acc (23): c1e1661e600c570e (cor, v_hand 0.14), f42258b638fc5b64 (cor, v_hand 3.191), b24575ef9d575fb4 (cor, v_hand 5.414), 41da8ea7c14754d2 (off, v_hand 3.582), 9baf3c1d42b25070 (cor, v_hand 4.812), de4ace24443b5b2d (col, v_hand 1.677), 83e895f722575452 (off, v_hand 5.172), b7c35b12755c5d44 (cor, v_hand 7.816), fc0fc7649a335d27 (cor, v_hand 7.003), eeab709ca6e05d14 (cor, v_hand 9.095), 09cd7b3746d65a79 (off, v_hand 10.964), 06fbdeb141965cca (off, v_hand 7.434), 9b1bea0cc0d75583 (off, v_hand 8.546), bd70654aee0f54a6 (off, v_hand 0.05), f3512ae175b45844 (cor, v_hand 0.0), fc0a60a9d8245aea (cor, v_hand 8.125), 6b68c09f87b85dc8 (off, v_hand 5.964), ddc1271ea57154bc (off, v_hand 2.054), f082cf21642c5cbc (cor, v_hand 0.424), d81c4398a40b52b9 (cor, v_hand 6.6), b227439005885966 (col, v_hand 5.933), 078bc1027dde5d1a (off, v_hand 3.069), 5e02e80df7fe5f5b (off, v_hand 7.959)
Fixed by acc (6): 5b61b9072cad5dbc (off), 7bbb6cfc4f135e50 (col), 4faa4706a50958e2 (cor), e14711ef4d2f5000 (off), f9ed38d9ddfa531e (off), f4495a91b9fb505d (off)

ACC terms at 1.5 s over 8129 decisions: won {'set': 4415, 'lead': 2163, 'ramp': 666, 'model': 635, 'curve': 250}; nearest {'set': 4740, 'lead': 2480, 'model': 637, 'curve': 272}; guard {'none': 6965, 'brake': 639, 'stop': 525}.
Handoff speed v_hand (m/s) over 452 sessions: p10 / p50 / p90 [0.04, 6.01, 10.96], share < 3 m/s 0.26; route cover > 0 on 1.00 of decisions.

Turn completion (rf_follow): /root/autodl-tmp/ujs/runs/route_follow/nuacc/report/follow/ (descriptive).
