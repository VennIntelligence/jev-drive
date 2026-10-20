# LON-ANCHOR offline gates (G1 FAIL)

G1: the human's station-matched path (points on his logged path at s_e + s_t,j, in the row frame) as the plan, the trainer's term: 6656 of 143144 labelled rows active (max |d| > 0.5 m) = 4.65 % (line <= 1 %); term quantiles (50 / 90 / 99 %) [0.0003, 0.1789, 8.8652]; nuPlan 4.82 %, PAI 4.43 %.

Stored plans against the anchored target (reported only):

| rows | n | active | plan too fast | plan too slow | d4 mean | d4 50 / 90 / 99 % | target ends standing | of those: human moving at the clock | target moves >= 2 m, clock human stands | dt_h 50 / 90 / 99 % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all | 143144 | 96.4 % | 32.2 % | 67.8 % | -3.64 | [-2.3207, 5.4713, 21.6945] | 13047 | 1591 | 4120 | [-0.0446, 0.9695, 6.2372] |
| nuplan | 80850 | 95.6 % | 23.0 % | 74.7 % | -4.52 | [-3.1435, 3.1688, 10.3251] | 5127 | 1117 | 917 | [-0.0994, 0.2494, 2.7443] |
| pai | 62294 | 97.5 % | 44.1 % | 58.8 % | -2.51 | [-0.9637, 8.6427, 30.5076] | 7920 | 474 | 3203 | [-0.0001, 3.1897, 6.7519] |
| bend | 6950 | 97.2 % | 27.4 % | 71.5 % | -8.02 | [-4.2996, 6.5874, 25.2222] | 11 | 0 | 0 | [-0.0042, 0.4785, 3.4773] |
| launch | 8038 | 97.0 % | 38.4 % | 61.8 % | -5.81 | [-2.1416, 5.5365, 13.6252] | 694 | 46 | 931 | [0.0163, 5.5756, 7.6135] |
| stop_nolead | 22826 | 97.4 % | 57.1 % | 48.3 % | +1.88 | [0.9268, 12.3466, 40.9094] | 6521 | 369 | 1262 | [-0.0, 0.9782, 6.2444] |
| turn_sharp | 24480 | 97.9 % | 38.6 % | 64.0 % | -3.95 | [-2.2048, 7.369, 20.6048] | 694 | 59 | 1010 | [-0.0005, 3.7, 6.7271] |

Label outcomes (S rows): {"nuplan": {"ok": 80850, "lead": 6091, "no_reach": 47, "log_end": 24}, "pai": {"ok": 62294, "lead": 10564, "log_end": 8249, "no_reach": 2771, "lead_unknown": 164}}

stored plans = nuPlan rows' served plans (rows meta `poses`), cp rows' model plans (`poses_model`, NTV0-L18-s0); d4 = plan 4 s arc - anchored target (m); active = max |d| > 0.5 m; reported, not gated
