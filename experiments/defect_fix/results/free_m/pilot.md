# NR1FREE-M pilot (FREEM1K-L18-s0 against JTNS1K-L18-s0; NR1 prereg supplement 16; native reads)

| | JTNS1K-L18-s0 (control) | FREEM1K-L18-s0 | line |
|:--|--:|--:|:--|
| held 460 zeros | 42 | 56 | <= control + 2: FAIL |
| a wait other_lead | 0.944 | 0.889 | >= control - 0.05: FAIL |
| a wait red_queue | 0.923 | 0.795 | >= control - 0.05: FAIL |
| held 460 front collisions | 7 | 13 | <= control + 2: FAIL |
| arena subset AF-front windows | 2 | 8 | <= control + 2: FAIL |
| arena subset k = 9 speed ratio (median v / v_h) | 0.8411 | 1.0252 | diff +0.1841 [+0.1455, +0.2449] >= +0.02: ok |
| PAI half (4 chunks) mean / zeros | 0.6733 / 40 | 0.6677 / 43 | paired -0.0056 [-0.0389, +0.0259] >= 0, zeros <= control + 2: FAIL |

Description:

- b other_lead: launched 0.729 -> 0.771, median delay +0.57 -> +0.32 s
- b red_queue: launched 0.657 -> 0.743, median delay +0.77 -> +0.52 s
- held 460 mean 0.8268 -> 0.8568, paired +0.0300 [+0.0056, +0.0486]
- arena subset (753 windows) mean 0.8618 -> 0.8290, paired -0.0328 [-0.0666, -0.0006] (log-clustered), zeros 96 -> 125
- arena subset strata (control -> pilot, zeros): braking (141) 0.953 -> 0.961, 5 -> 5; cruise (23) 0.974 -> 1.000, 0 -> 0; launch (66) 0.883 -> 0.910, 4 -> 3; other (122) 0.946 -> 0.957, 5 -> 5; static (331) 0.761 -> 0.674, 79 -> 108; turn (70) 0.950 -> 0.942, 3 -> 4
- speed ratio by k (control / pilot): k2 0.985 / 0.998, k3 0.962 / 1.001, k4 0.940 / 1.007, k5 0.912 / 1.018, k6 0.882 / 1.020, k7 0.873 / 1.026, k8 0.859 / 1.034, k9 0.841 / 1.025

Gates: held_zeros FAIL, wait_other_lead FAIL, wait_red_queue FAIL, held_front FAIL, arena_af_front FAIL, ratchet_k9 ok, pai_half FAIL

**Verdict: FAIL**
