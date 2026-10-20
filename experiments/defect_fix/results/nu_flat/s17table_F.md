# NR1STAND pilot (B2F-L18-s1 against B2C-L18-s1; NR1 prereg supplement 17; native reads)

| | B2C-L18-s1 (control) | B2F-L18-s1 | line |
|:--|--:|--:|:--|
| held 460 zeros | 39 | 37 | <= control + 2: ok |
| a wait other_lead | 0.944 | 0.944 | >= control - 0.03: ok |
| a wait red_queue | 0.949 | 0.949 | >= control - 0.03: ok |
| b median launch delay other_lead (s) | +0.72 | +0.77 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.79 | +0.62 | <= control + 0.15: ok |
| b launched other_lead | 0.583 | 0.542 | >= 0.95 x control: FAIL |
| b launched red_queue | 0.400 | 0.371 | >= 0.95 x control: FAIL |
| held 460 front collisions | 5 | 4 | <= control + 2: ok |
| arena subset AF-front windows | 1 | 1 | <= control + 2: ok |
| arena subset static stratum (331 windows) mean / zeros | 0.9698 / 10 | 0.9698 / 10 | paired >= 0: ok |
| arena subset (753 windows) mean / zeros | 0.9445 / 28 | 0.9483 / 25 | paired +0.0038 [-0.0024, +0.0115] >= +0.01: FAIL |

Description:

- arena subset partials (control -> arm; d382): slow (progress < 0.8) 14 -> 16, near-1 (0.98-0.999) 3 -> 1; frozen-pool PCS: pcs_local.py score --subset on arena_windows.csv.gz (Mac)
- arena subset k = 9 speed ratio (median v / v_h): 1.0083 -> 1.0011, diff -0.0072 [-0.0266, +0.0050]
- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): k5_10 +0.005 [+0.001, +0.010] (2518), k11_17 +0.014 [+0.002, +0.028] (2639), turn_k11_17 -0.000 [-0.054, +0.047] (155)
- held 460 mean 0.8496 -> 0.8548, paired +0.0052 [-0.0059, +0.0159]
- arena subset strata (control -> arm, zeros): braking (141) 0.954 -> 0.962, 6 -> 5; cruise (23) 1.000 -> 1.000, 0 -> 0; launch (66) 0.731 -> 0.747, 6 -> 5; other (122) 0.963 -> 0.961, 3 -> 3; static (331) 0.970 -> 0.970, 10 -> 10; turn (70) 0.957 -> 0.971, 3 -> 2
- speed ratio by k (control / arm): k2 0.993 / 0.988, k3 0.991 / 0.979, k4 0.996 / 0.984, k5 0.997 / 0.985, k6 1.003 / 0.990, k7 0.995 / 0.987, k8 1.016 / 0.998, k9 1.008 / 1.001
- arena subset by cue (arm - control paired, log-clustered; zeros control -> arm): static: red light at the line, no lead (96) 0.906 -> 0.906, +0.000 [-0.043, +0.021], 9 -> 9; static: light of unknown state, no lead (34) 0.971 -> 0.971, +0.000 [+0.000, +0.000], 1 -> 1; static: lead <= 8 m (188) 1.000 -> 1.000, +0.000 [+0.000, +0.000], 0 -> 0; static: lead 8-30 m (12) 1.000 -> 1.000, +0.000 [+0.000, +0.000], 0 -> 0; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan], 0 -> 0; red-line launch (render-red per d380) (59) 0.865 -> 0.865, -0.000 [-0.004, +0.004], 1 -> 1
- arena subset zero types (control / arm): {"ok": 697, "corridor": 15, "slow": 14, "offroad": 11, "no-launch": 9, "creep": 3, "rear-hit": 2, "AF-front": 1, "offroad+corridor": 1} / {"ok": 699, "slow": 16, "corridor": 15, "no-launch": 8, "offroad": 8, "creep": 3, "rear-hit": 2, "AF-front": 1, "offroad+corridor": 1}

Gates: held_zeros ok, wait_other_lead ok, wait_red_queue ok, delay_other_lead ok, delay_red_queue ok, launched_other_lead FAIL, launched_red_queue FAIL, held_front ok, arena_af_front ok, static ok, arena_win FAIL, drift ok

**Verdict: FAIL**
