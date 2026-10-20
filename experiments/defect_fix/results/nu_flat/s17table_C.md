# NR1STAND pilot (B2C-L18-s1 against STANDFMXH-L18-s1; NR1 prereg supplement 17; native reads)

| | STANDFMXH-L18-s1 (control) | B2C-L18-s1 | line |
|:--|--:|--:|:--|
| held 460 zeros | 41 | 39 | <= control + 2: ok |
| a wait other_lead | 0.944 | 0.944 | >= control - 0.03: ok |
| a wait red_queue | 0.949 | 0.949 | >= control - 0.03: ok |
| b median launch delay other_lead (s) | +0.57 | +0.72 | <= control + 0.15: FAIL |
| b median launch delay red_queue (s) | +0.92 | +0.79 | <= control + 0.15: ok |
| b launched other_lead | 0.646 | 0.583 | >= 0.95 x control: FAIL |
| b launched red_queue | 0.486 | 0.400 | >= 0.95 x control: FAIL |
| held 460 front collisions | 4 | 5 | <= control + 2: ok |
| arena subset AF-front windows | 0 | 1 | <= control + 2: ok |
| arena subset static stratum (331 windows) mean / zeros | 0.9728 / 9 | 0.9698 / 10 | paired >= 0: FAIL |
| arena subset (753 windows) mean / zeros | 0.9522 / 24 | 0.9445 / 28 | paired -0.0076 [-0.0183, +0.0013] >= +0.01: FAIL |

Description:

- arena subset partials (control -> arm; d382): slow (progress < 0.8) 14 -> 14, near-1 (0.98-0.999) 4 -> 3; frozen-pool PCS: pcs_local.py score --subset on arena_windows.csv.gz (Mac)
- arena subset k = 9 speed ratio (median v / v_h): 0.9932 -> 1.0083, diff +0.0150 [+0.0005, +0.0270]
- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): k5_10 +0.015 [+0.006, +0.026] (2511), k11_17 +0.026 [+0.005, +0.048] (2621), turn_k11_17 -0.034 [-0.139, +0.058] (142)
- held 460 mean 0.8527 -> 0.8496, paired -0.0031 [-0.0302, +0.0177]
- arena subset strata (control -> arm, zeros): braking (141) 0.969 -> 0.954, 4 -> 6; cruise (23) 0.998 -> 1.000, 0 -> 0; launch (66) 0.768 -> 0.731, 5 -> 6; other (122) 0.957 -> 0.963, 4 -> 3; static (331) 0.973 -> 0.970, 9 -> 10; turn (70) 0.971 -> 0.957, 2 -> 3
- speed ratio by k (control / arm): k2 0.993 / 0.993, k3 0.989 / 0.991, k4 0.991 / 0.996, k5 0.988 / 0.997, k6 0.989 / 1.003, k7 0.984 / 0.995, k8 1.000 / 1.016, k9 0.993 / 1.008
- arena subset by cue (arm - control paired, log-clustered; zeros control -> arm): static: red light at the line, no lead (96) 0.917 -> 0.906, -0.010 [-0.065, +0.052], 8 -> 9; static: light of unknown state, no lead (34) 0.971 -> 0.971, +0.000 [+0.000, +0.000], 1 -> 1; static: lead <= 8 m (188) 1.000 -> 1.000, +0.000 [+0.000, +0.000], 0 -> 0; static: lead 8-30 m (12) 1.000 -> 1.000, +0.000 [+0.000, +0.000], 0 -> 0; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan], 0 -> 0; red-line launch (render-red per d380) (59) 0.880 -> 0.865, -0.015 [-0.023, -0.005], 1 -> 1
- arena subset zero types (control / arm): {"ok": 705, "slow": 14, "corridor": 14, "offroad": 9, "no-launch": 6, "rear-hit": 2, "creep": 2, "offroad+corridor": 1} / {"ok": 697, "corridor": 15, "slow": 14, "offroad": 11, "no-launch": 9, "creep": 3, "rear-hit": 2, "AF-front": 1, "offroad+corridor": 1}

Gates: held_zeros ok, wait_other_lead ok, wait_red_queue ok, delay_other_lead FAIL, delay_red_queue ok, launched_other_lead FAIL, launched_red_queue FAIL, held_front ok, arena_af_front ok, static FAIL, arena_win FAIL, drift ok

**Verdict: FAIL**
