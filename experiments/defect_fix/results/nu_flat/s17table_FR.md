# NR1STAND pilot (B2FR-L18-s1 against B2C-L18-s1; NR1 prereg supplement 17; native reads)

| | B2C-L18-s1 (control) | B2FR-L18-s1 | line |
|:--|--:|--:|:--|
| held 460 zeros | 39 | 38 | <= control + 2: ok |
| a wait other_lead | 0.944 | 0.972 | >= control - 0.03: ok |
| a wait red_queue | 0.949 | 1.000 | >= control - 0.03: ok |
| b median launch delay other_lead (s) | +0.72 | +0.77 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.79 | +0.82 | <= control + 0.15: ok |
| b launched other_lead | 0.583 | 0.438 | >= 0.95 x control: FAIL |
| b launched red_queue | 0.400 | 0.257 | >= 0.95 x control: FAIL |
| held 460 front collisions | 5 | 5 | <= control + 2: ok |
| arena subset AF-front windows | 1 | 1 | <= control + 2: ok |
| arena subset static stratum (331 windows) mean / zeros | 0.9698 / 10 | 0.9909 / 3 | paired >= 0: ok |
| arena subset (753 windows) mean / zeros | 0.9445 / 28 | 0.9517 / 17 | paired +0.0072 [-0.0053, +0.0272] >= +0.01: FAIL |

Description:

- arena subset partials (control -> arm; d382): slow (progress < 0.8) 14 -> 17, near-1 (0.98-0.999) 3 -> 8; frozen-pool PCS: pcs_local.py score --subset on arena_windows.csv.gz (Mac)
- arena subset k = 9 speed ratio (median v / v_h): 1.0083 -> 0.9941, diff -0.0142 [-0.0385, +0.0013]
- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): k5_10 +0.004 [-0.002, +0.009] (2518), k11_17 -0.001 [-0.035, +0.028] (2621), turn_k11_17 +0.039 [-0.022, +0.114] (154)
- held 460 mean 0.8496 -> 0.8170, paired -0.0326 [-0.0561, -0.0181]
- arena subset strata (control -> arm, zeros): braking (141) 0.954 -> 0.968, 6 -> 4; cruise (23) 1.000 -> 1.000, 0 -> 0; launch (66) 0.731 -> 0.679, 6 -> 5; other (122) 0.963 -> 0.956, 3 -> 3; static (331) 0.970 -> 0.991, 10 -> 3; turn (70) 0.957 -> 0.966, 3 -> 2
- speed ratio by k (control / arm): k2 0.993 / 0.987, k3 0.991 / 0.976, k4 0.996 / 0.979, k5 0.997 / 0.976, k6 1.003 / 0.985, k7 0.995 / 0.983, k8 1.016 / 0.992, k9 1.008 / 0.994
- arena subset by cue (arm - control paired, log-clustered; zeros control -> arm): static: red light at the line, no lead (96) 0.906 -> 0.969, +0.062 [+0.000, +0.141], 9 -> 3; static: light of unknown state, no lead (34) 0.971 -> 1.000, +0.029 [+0.000, +0.064], 1 -> 0; static: lead <= 8 m (188) 1.000 -> 1.000, +0.000 [+0.000, +0.000], 0 -> 0; static: lead 8-30 m (12) 1.000 -> 1.000, +0.000 [+0.000, +0.000], 0 -> 0; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan], 0 -> 0; red-line launch (render-red per d380) (59) 0.865 -> 0.818, -0.047 [-0.092, -0.014], 1 -> 2
- arena subset zero types (control / arm): {"ok": 697, "corridor": 15, "slow": 14, "offroad": 11, "no-launch": 9, "creep": 3, "rear-hit": 2, "AF-front": 1, "offroad+corridor": 1} / {"ok": 700, "slow": 17, "no-launch": 11, "offroad": 10, "corridor": 6, "rear-hit": 4, "creep": 4, "AF-front": 1}

Gates: held_zeros ok, wait_other_lead ok, wait_red_queue ok, delay_other_lead ok, delay_red_queue ok, launched_other_lead FAIL, launched_red_queue FAIL, held_front ok, arena_af_front ok, static ok, arena_win FAIL, drift ok

**Verdict: FAIL**
