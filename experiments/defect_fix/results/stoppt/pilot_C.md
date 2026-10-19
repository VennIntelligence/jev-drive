# NR1STOPPT pilot (STOPPTF1K-L18-s0 against JTNS1K-L18-s0; NR1 prereg supplement 18; native reads)

| | JTNS1K-L18-s0 (control) | STOPPTF1K-L18-s0 | line |
|:--|--:|--:|:--|
| held 460 zeros | 42 | 33 | <= control + 2: ok |
| b median launch delay other_lead (s) | +0.57 | +0.57 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.77 | +0.62 | <= control + 0.15: ok |
| b launched other_lead | 0.729 | 0.688 | >= 0.95 x control: FAIL |
| b launched red_queue | 0.657 | 0.600 | >= 0.95 x control: FAIL |
| held 460 front collisions | 7 | 2 | <= control + 2: ok |
| arena subset AF-front windows | 2 | 1 | <= control + 2: ok |
| PAI half (4 chunks) mean / zeros | 0.6733 / 40 | 0.6236 / 42 | paired -0.0497 [-0.1036, +0.0024] >= +0.01: FAIL |
| PAI half d381 overrun zeros (cor_route_end) | 11 | 11 | description |
| arena subset (753 windows) mean / zeros | 0.8618 / 96 | 0.8690 / 85 | paired +0.0072 [-0.0079, +0.0211] >= 0: ok |

Description:

- a wait other_lead: 0.944 -> 0.944
- a wait red_queue: 0.923 -> 0.923
- overrun scenes fixed: []; new: []
- held 460 mean 0.8268 -> 0.7949, paired -0.0319 [-0.0400, -0.0214]
- arena subset k = 9 speed ratio: 0.8411 -> 0.7704, diff -0.0707 [-0.1035, -0.0503]
- arena subset strata (control -> arm, zeros): braking (141) 0.953 -> 0.955, 5 -> 3; cruise (23) 0.974 -> 0.917, 0 -> 1; launch (66) 0.883 -> 0.827, 4 -> 6; other (122) 0.946 -> 0.933, 5 -> 5; static (331) 0.761 -> 0.792, 79 -> 69; turn (70) 0.950 -> 0.974, 3 -> 1
- arena subset by cue (arm - control): static: red light at the line, no lead (96) 0.323 -> 0.385, +0.062 [+0.000, +0.087]; static: light of unknown state, no lead (34) 0.882 -> 0.941, +0.059 [+0.000, +0.065]; static: lead <= 8 m (188) 0.989 -> 1.000, +0.011 [+0.000, +0.033]; static: lead 8-30 m (12) 0.333 -> 0.333, +0.000 [+0.000, +0.000]; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan]; red-line launch (render-red per d380) (59) 0.945 -> 0.914, -0.031 [-0.082, -0.004]
- arena subset zero types (control / arm): {"ok": 603, "corridor": 86, "slow": 50, "offroad": 8, "rear-hit": 2, "no-launch": 2, "AF-front": 1, "AF-front+corridor": 1} / {"ok": 582, "slow": 76, "corridor": 74, "offroad": 10, "rear-hit": 7, "no-launch": 2, "creep": 1, "AF-front": 1}
- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): k5_10 -0.005 [-0.016, +0.005] (2510), k11_17 -0.030 [-0.056, -0.002] (2659), turn_k11_17 -0.062 [-0.207, +0.116] (146)

PAI half overrun zeros (cor_route_end) by scene, score control / arm:

- clipgt-054b5901-fdcb-4146-b125-eb2bb333cf02: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-095cf563-c5d6-4f4f-9e98-7f179e733560: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-127c4f85-a5db-4920-9813-7970dcd6ebdb: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-3fe2bac8-5502-4626-a8cc-b52d644c7da5: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-44df49fe-0729-4c30-98d0-5a40f43b7b09: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-47db2eb6-a299-4469-a3ca-e5539e8763a0: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-4ecc114d-4c76-49bb-874c-601561d738bb: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-68c3be44-7f95-46ce-ab72-ee05aeb986bc: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-7750265d-c41f-42a3-8b49-ee128b71fa55: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-7d994ed3-d15f-4c0d-a0ff-fd36ba005726: 0.000 / 0.000 (control overrun) (arm overrun)
- clipgt-fc9fde87-6b71-49ff-baad-2ac992e7e034: 0.000 / 0.000 (control overrun) (arm overrun)

Gates: held_zeros ok, delay_other_lead ok, delay_red_queue ok, launched_other_lead FAIL, launched_red_queue FAIL, held_front ok, arena_af_front ok, pai_half FAIL, arena ok

**Verdict: FAIL**
