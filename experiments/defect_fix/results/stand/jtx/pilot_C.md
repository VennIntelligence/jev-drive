# NR1STAND pilot (STANDFRX1K-L18-s0 against JTX1K-L18-s0; NR1 prereg supplement 17; native reads)

| | JTX1K-L18-s0 (control) | STANDFRX1K-L18-s0 | line |
|:--|--:|--:|:--|
| held 460 zeros | 37 | 52 | <= control + 2: FAIL |
| a wait other_lead | 0.944 | 0.889 | >= control - 0.03: FAIL |
| a wait red_queue | 0.923 | 0.872 | >= control - 0.03: FAIL |
| b median launch delay other_lead (s) | +0.52 | +0.27 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.72 | +0.24 | <= control + 0.15: ok |
| b launched other_lead | 0.688 | 0.875 | >= 0.95 x control: ok |
| b launched red_queue | 0.629 | 0.800 | >= 0.95 x control: ok |
| held 460 front collisions | 2 | 9 | <= control + 2: FAIL |
| arena subset AF-front windows | 1 | 4 | <= control + 2: FAIL |
| arena subset static stratum (331 windows) mean / zeros | 0.7764 / 74 | 0.7190 / 93 | paired >= 0: FAIL |
| arena subset (753 windows) mean / zeros | 0.8700 / 88 | 0.8464 / 113 | paired -0.0236 [-0.0429, -0.0015] >= +0.01: FAIL |

Description:

- arena subset partials (control -> arm; d382): slow (progress < 0.8) 48 -> 13, near-1 (0.98-0.999) 12 -> 3; frozen-pool PCS: pcs_local.py score --subset on arena_windows.csv.gz (Mac)
- arena subset k = 9 speed ratio (median v / v_h): 0.8453 -> 0.9479, diff +0.1027 [+0.0779, +0.1487]
- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): k5_10 -0.019 [-0.031, -0.007] (2503), k11_17 -0.009 [-0.041, +0.024] (2639), turn_k11_17 -0.018 [-0.124, +0.105] (146)
- held 460 mean 0.8292 -> 0.8710, paired +0.0419 [+0.0209, +0.0768]
- arena subset strata (control -> arm, zeros): braking (141) 0.951 -> 0.943, 5 -> 8; cruise (23) 0.974 -> 1.000, 0 -> 0; launch (66) 0.877 -> 0.917, 3 -> 3; other (122) 0.954 -> 0.951, 4 -> 6; static (331) 0.776 -> 0.719, 74 -> 93; turn (70) 0.963 -> 0.956, 2 -> 3
- speed ratio by k (control / arm): k2 0.985 / 0.992, k3 0.961 / 0.981, k4 0.939 / 0.974, k5 0.908 / 0.961, k6 0.880 / 0.955, k7 0.871 / 0.951, k8 0.859 / 0.948, k9 0.845 / 0.948
- arena subset by cue (arm - control paired, log-clustered; zeros control -> arm): static: red light at the line, no lead (96) 0.354 -> 0.344, -0.010 [-0.236, +0.076], 62 -> 63; static: light of unknown state, no lead (34) 0.941 -> 0.500, -0.441 [-0.796, -0.062], 2 -> 17; static: lead <= 8 m (188) 0.989 -> 0.973, -0.016 [-0.060, +0.000], 2 -> 5; static: lead 8-30 m (12) 0.333 -> 0.333, +0.000 [+0.000, +0.000], 8 -> 8; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan], 0 -> 0; red-line launch (render-red per d380) (59) 0.949 -> 0.969, +0.020 [+0.001, +0.052], 0 -> 0
- arena subset zero types (control / arm): {"ok": 610, "corridor": 79, "slow": 48, "offroad": 8, "no-launch": 4, "rear-hit": 3, "AF-front": 1} / {"ok": 626, "corridor": 99, "slow": 13, "offroad": 10, "AF-front+corridor": 3, "no-launch": 1, "AF-front": 1}

Gates: held_zeros FAIL, wait_other_lead FAIL, wait_red_queue FAIL, delay_other_lead ok, delay_red_queue ok, launched_other_lead ok, launched_red_queue ok, held_front FAIL, arena_af_front FAIL, static FAIL, arena_win FAIL, drift ok

**Verdict: FAIL**
