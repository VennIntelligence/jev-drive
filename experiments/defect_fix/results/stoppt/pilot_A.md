# NR1STOPPT pilot (STOPPT1K-L18-s0 against JTNS1K-L18-s0; NR1 prereg supplement 18; native reads)

| | JTNS1K-L18-s0 (control) | STOPPT1K-L18-s0 | line |
|:--|--:|--:|:--|
| held 460 zeros | 42 | 42 | <= control + 2: ok |
| b median launch delay other_lead (s) | +0.57 | +0.52 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.77 | +0.84 | <= control + 0.15: ok |
| b launched other_lead | 0.729 | 0.729 | >= 0.95 x control: ok |
| b launched red_queue | 0.657 | 0.686 | >= 0.95 x control: ok |
| held 460 front collisions | 7 | 5 | <= control + 2: ok |
| arena subset AF-front windows | 2 | 2 | <= control + 2: ok |
| PAI half (4 chunks) mean / zeros | 0.6733 / 40 | 0.6657 / 41 | paired -0.0076 [-0.0332, +0.0128] >= +0.01: FAIL |
| PAI half d381 overrun zeros (cor_route_end) | 11 | 11 | description |
| arena subset (753 windows) mean / zeros | 0.8618 / 96 | 0.8636 / 94 | paired +0.0018 [-0.0004, +0.0047] >= 0: ok |

Description:

- a wait other_lead: 0.944 -> 0.944
- a wait red_queue: 0.923 -> 0.923
- overrun scenes fixed: []; new: []
- held 460 mean 0.8268 -> 0.8204, paired -0.0064 [-0.0123, -0.0027]
- arena subset k = 9 speed ratio: 0.8411 -> 0.8366, diff -0.0045 [-0.0285, +0.0013]
- arena subset strata (control -> arm, zeros): braking (141) 0.953 -> 0.952, 5 -> 5; cruise (23) 0.974 -> 0.972, 0 -> 0; launch (66) 0.883 -> 0.876, 4 -> 4; other (122) 0.946 -> 0.946, 5 -> 5; static (331) 0.761 -> 0.767, 79 -> 77; turn (70) 0.950 -> 0.950, 3 -> 3
- arena subset by cue (arm - control): static: red light at the line, no lead (96) 0.323 -> 0.323, +0.000 [+0.000, +0.000]; static: light of unknown state, no lead (34) 0.882 -> 0.941, +0.059 [+0.000, +0.065]; static: lead <= 8 m (188) 0.989 -> 0.989, +0.000 [+0.000, +0.000]; static: lead 8-30 m (12) 0.333 -> 0.333, +0.000 [+0.000, +0.000]; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan]; red-line launch (render-red per d380) (59) 0.945 -> 0.944, -0.001 [-0.003, -0.000]
- arena subset zero types (control / arm): {"ok": 603, "corridor": 86, "slow": 50, "offroad": 8, "rear-hit": 2, "no-launch": 2, "AF-front": 1, "AF-front+corridor": 1} / {"ok": 602, "corridor": 84, "slow": 53, "offroad": 8, "rear-hit": 2, "no-launch": 2, "AF-front": 1, "AF-front+corridor": 1}

Gates: held_zeros ok, delay_other_lead ok, delay_red_queue ok, launched_other_lead ok, launched_red_queue ok, held_front ok, arena_af_front ok, pai_half FAIL, arena ok

**Verdict: FAIL**
