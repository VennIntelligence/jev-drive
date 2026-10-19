# NR1STOPPT pilot (STOPPTP1K-L18-s0 against JTNS1K-L18-s0; NR1 prereg supplement 18; native reads)

| | JTNS1K-L18-s0 (control) | STOPPTP1K-L18-s0 | line |
|:--|--:|--:|:--|
| held 460 zeros | 42 | 39 | <= control + 2: ok |
| b median launch delay other_lead (s) | +0.57 | +0.44 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.77 | +0.77 | <= control + 0.15: ok |
| b launched other_lead | 0.729 | 0.667 | >= 0.95 x control: FAIL |
| b launched red_queue | 0.657 | 0.571 | >= 0.95 x control: FAIL |
| held 460 front collisions | 7 | 5 | <= control + 2: ok |
| arena subset AF-front windows | 2 | 3 | <= control + 2: ok |
| PAI half (4 chunks) mean / zeros | 0.6733 / 40 | 0.6676 / 39 | paired -0.0057 [-0.0544, +0.0434] >= +0.01: FAIL |
| PAI half d381 overrun zeros (cor_route_end) | 11 | 11 | description |
| arena subset (753 windows) mean / zeros | 0.8618 / 96 | 0.8676 / 90 | paired +0.0058 [-0.0104, +0.0211] >= 0: ok |

Description:

- a wait other_lead: 0.944 -> 0.944
- a wait red_queue: 0.923 -> 0.897
- overrun scenes fixed: []; new: []
- held 460 mean 0.8268 -> 0.8195, paired -0.0074 [-0.0176, +0.0006]
- arena subset k = 9 speed ratio: 0.8411 -> 0.8377, diff -0.0034 [-0.0309, +0.0129]
- arena subset strata (control -> arm, zeros): braking (141) 0.953 -> 0.958, 5 -> 4; cruise (23) 0.974 -> 0.935, 0 -> 1; launch (66) 0.883 -> 0.851, 4 -> 5; other (122) 0.946 -> 0.944, 5 -> 5; static (331) 0.761 -> 0.782, 79 -> 72; turn (70) 0.950 -> 0.950, 3 -> 3
- arena subset by cue (arm - control): static: red light at the line, no lead (96) 0.323 -> 0.375, +0.052 [-0.062, +0.125]; static: light of unknown state, no lead (34) 0.882 -> 0.912, +0.029 [-0.333, +0.065]; static: lead <= 8 m (188) 0.989 -> 0.995, +0.005 [-0.012, +0.031]; static: lead 8-30 m (12) 0.333 -> 0.333, +0.000 [+0.000, +0.000]; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan]; red-line launch (render-red per d380) (59) 0.945 -> 0.945, -0.000 [-0.002, +0.001]
- arena subset zero types (control / arm): {"ok": 603, "corridor": 86, "slow": 50, "offroad": 8, "rear-hit": 2, "no-launch": 2, "AF-front": 1, "AF-front+corridor": 1} / {"ok": 602, "corridor": 78, "slow": 55, "offroad": 8, "rear-hit": 3, "AF-front": 3, "no-launch": 3, "offroad+corridor": 1}

Gates: held_zeros ok, delay_other_lead ok, delay_red_queue ok, launched_other_lead FAIL, launched_red_queue FAIL, held_front ok, arena_af_front ok, pai_half FAIL, arena ok

**Verdict: FAIL**
