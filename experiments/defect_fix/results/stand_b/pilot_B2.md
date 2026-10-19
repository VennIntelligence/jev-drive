# NR1STAND pilot (STANDFMXH1K-L18-s0 against JTX1K-L18-s0; NR1 prereg supplement 17; native reads)

| | JTX1K-L18-s0 (control) | STANDFMXH1K-L18-s0 | line |
|:--|--:|--:|:--|
| held 460 zeros | 37 | 42 | <= control + 2: FAIL |
| a wait other_lead | 0.944 | 0.917 | >= control - 0.03: ok |
| a wait red_queue | 0.923 | 0.923 | >= control - 0.03: ok |
| b median launch delay other_lead (s) | +0.52 | +0.37 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.72 | +0.42 | <= control + 0.15: ok |
| b launched other_lead | 0.688 | 0.812 | >= 0.95 x control: ok |
| b launched red_queue | 0.629 | 0.657 | >= 0.95 x control: ok |
| held 460 front collisions | 2 | 3 | <= control + 2: ok |
| arena subset AF-front windows | 1 | 1 | <= control + 2: ok |
| arena subset static stratum (331 windows) mean / zeros | 0.7764 / 74 | 0.8701 / 43 | paired >= 0: ok |
| arena subset (753 windows) mean / zeros | 0.8700 / 88 | 0.9144 / 56 | paired +0.0444 [+0.0036, +0.1093] >= +0.01: ok |

Description:

- arena subset partials (control -> arm; d382): slow (progress < 0.8) 48 -> 36, near-1 (0.98-0.999) 12 -> 7; frozen-pool PCS: pcs_local.py score --subset on arena_windows.csv.gz (Mac)
- arena subset k = 9 speed ratio (median v / v_h): 0.8453 -> 0.9068, diff +0.0616 [+0.0429, +0.1048]
- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): k5_10 -0.013 [-0.020, -0.006] (2511), k11_17 -0.025 [-0.043, -0.005] (2676), turn_k11_17 -0.176 [-0.296, -0.032] (146)
- held 460 mean 0.8292 -> 0.8514, paired +0.0222 [-0.0030, +0.0400]
- arena subset strata (control -> arm, zeros): braking (141) 0.951 -> 0.965, 5 -> 4; cruise (23) 0.974 -> 0.988, 0 -> 0; launch (66) 0.877 -> 0.863, 3 -> 3; other (122) 0.954 -> 0.959, 4 -> 4; static (331) 0.776 -> 0.870, 74 -> 43; turn (70) 0.963 -> 0.967, 2 -> 2
- speed ratio by k (control / arm): k2 0.985 / 0.989, k3 0.961 / 0.974, k4 0.939 / 0.959, k5 0.908 / 0.946, k6 0.880 / 0.926, k7 0.871 / 0.908, k8 0.859 / 0.909, k9 0.845 / 0.907
- arena subset by cue (arm - control paired, log-clustered; zeros control -> arm): static: red light at the line, no lead (96) 0.354 -> 0.615, +0.260 [-0.023, +0.435], 62 -> 37; static: light of unknown state, no lead (34) 0.941 -> 0.941, +0.000 [+0.000, +0.000], 2 -> 2; static: lead <= 8 m (188) 0.989 -> 0.995, +0.005 [-0.014, +0.027], 2 -> 1; static: lead 8-30 m (12) 0.333 -> 0.750, +0.417 [+0.000, +0.812], 8 -> 3; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan], 0 -> 0; red-line launch (render-red per d380) (59) 0.949 -> 0.943, -0.006 [-0.018, +0.006], 0 -> 0
- arena subset zero types (control / arm): {"ok": 610, "corridor": 79, "slow": 48, "offroad": 8, "no-launch": 4, "rear-hit": 3, "AF-front": 1} / {"ok": 656, "corridor": 48, "slow": 36, "offroad": 7, "no-launch": 4, "rear-hit": 1, "AF-front+corridor": 1}

Gates: held_zeros FAIL, wait_other_lead ok, wait_red_queue ok, delay_other_lead ok, delay_red_queue ok, launched_other_lead ok, launched_red_queue ok, held_front ok, arena_af_front ok, static ok, arena_win ok, drift ok

**Verdict: FAIL**
