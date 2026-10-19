# NR1STAND pilot (STANDRL2H1K-L18-s0 against STANDFMXH1K-L18-s0; NR1 prereg supplement 17; native reads)

| | STANDFMXH1K-L18-s0 (control) | STANDRL2H1K-L18-s0 | line |
|:--|--:|--:|:--|
| held 460 zeros | 42 | 38 | <= control + 2: ok |
| a wait other_lead | 0.917 | 0.917 | >= control - 0.03: ok |
| a wait red_queue | 0.923 | 0.949 | >= control - 0.03: ok |
| b median launch delay other_lead (s) | +0.37 | +0.47 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.42 | +0.67 | <= control + 0.15: FAIL |
| b launched other_lead | 0.812 | 0.604 | >= 0.95 x control: FAIL |
| b launched red_queue | 0.657 | 0.429 | >= 0.95 x control: FAIL |
| held 460 front collisions | 3 | 2 | <= control + 2: ok |
| arena subset AF-front windows | 1 | 0 | <= control + 2: ok |
| arena subset static stratum (331 windows) mean / zeros | 0.8701 / 43 | 0.9789 / 7 | paired >= 0: ok |
| arena subset (753 windows) mean / zeros | 0.9144 / 56 | 0.9513 / 20 | paired +0.0369 [+0.0070, +0.0688] >= +0.01: ok |

Description:

- arena subset partials (control -> arm; d382): slow (progress < 0.8) 36 -> 40, near-1 (0.98-0.999) 7 -> 9; frozen-pool PCS: pcs_local.py score --subset on arena_windows.csv.gz (Mac)
- arena subset k = 9 speed ratio (median v / v_h): 0.9068 -> 0.8778, diff -0.0290 [-0.0491, -0.0090]
- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): k5_10 +0.001 [-0.007, +0.009] (2519), k11_17 +0.005 [-0.009, +0.019] (2680), turn_k11_17 -0.019 [-0.083, +0.022] (139)
- held 460 mean 0.8514 -> 0.8099, paired -0.0415 [-0.0611, -0.0229]
- arena subset strata (control -> arm, zeros): braking (141) 0.965 -> 0.971, 4 -> 3; cruise (23) 0.988 -> 0.987, 0 -> 0; launch (66) 0.863 -> 0.766, 3 -> 3; other (122) 0.959 -> 0.948, 4 -> 4; static (331) 0.870 -> 0.979, 43 -> 7; turn (70) 0.967 -> 0.949, 2 -> 3
- speed ratio by k (control / arm): k2 0.989 / 0.988, k3 0.974 / 0.973, k4 0.959 / 0.957, k5 0.946 / 0.940, k6 0.926 / 0.911, k7 0.908 / 0.900, k8 0.909 / 0.891, k9 0.907 / 0.878
- arena subset by cue (arm - control paired, log-clustered; zeros control -> arm): static: red light at the line, no lead (96) 0.615 -> 0.927, +0.312 [+0.149, +0.463], 37 -> 7; static: light of unknown state, no lead (34) 0.941 -> 1.000, +0.059 [+0.000, +0.128], 2 -> 0; static: lead <= 8 m (188) 0.995 -> 1.000, +0.005 [+0.000, +0.020], 1 -> 0; static: lead 8-30 m (12) 0.750 -> 1.000, +0.250 [+0.000, +0.545], 3 -> 0; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan], 0 -> 0; red-line launch (render-red per d380) (59) 0.943 -> 0.882, -0.060 [-0.096, -0.020], 0 -> 0
- arena subset zero types (control / arm): {"ok": 656, "corridor": 48, "slow": 36, "offroad": 7, "no-launch": 4, "rear-hit": 1, "AF-front+corridor": 1} / {"ok": 679, "slow": 40, "corridor": 11, "offroad": 9, "no-launch": 6, "rear-hit": 4, "creep": 4}

Gates: held_zeros ok, wait_other_lead ok, wait_red_queue ok, delay_other_lead ok, delay_red_queue FAIL, launched_other_lead FAIL, launched_red_queue FAIL, held_front ok, arena_af_front ok, static ok, arena_win ok, drift ok

**Verdict: FAIL**
