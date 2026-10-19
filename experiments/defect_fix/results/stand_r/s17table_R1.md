# NR1STAND pilot (STANDRH1K-L18-s0 against STANDFMXH1K-L18-s0; NR1 prereg supplement 17; native reads)

| | STANDFMXH1K-L18-s0 (control) | STANDRH1K-L18-s0 | line |
|:--|--:|--:|:--|
| held 460 zeros | 42 | 38 | <= control + 2: ok |
| a wait other_lead | 0.917 | 0.889 | >= control - 0.03: ok |
| a wait red_queue | 0.923 | 0.923 | >= control - 0.03: ok |
| b median launch delay other_lead (s) | +0.37 | +0.37 | <= control + 0.15: ok |
| b median launch delay red_queue (s) | +0.42 | +0.47 | <= control + 0.15: ok |
| b launched other_lead | 0.812 | 0.688 | >= 0.95 x control: FAIL |
| b launched red_queue | 0.657 | 0.571 | >= 0.95 x control: FAIL |
| held 460 front collisions | 3 | 1 | <= control + 2: ok |
| arena subset AF-front windows | 1 | 0 | <= control + 2: ok |
| arena subset static stratum (331 windows) mean / zeros | 0.8701 / 43 | 0.9486 / 17 | paired >= 0: ok |
| arena subset (753 windows) mean / zeros | 0.9144 / 56 | 0.9406 / 32 | paired +0.0262 [+0.0051, +0.0521] >= +0.01: ok |

Description:

- arena subset partials (control -> arm; d382): slow (progress < 0.8) 36 -> 37, near-1 (0.98-0.999) 7 -> 8; frozen-pool PCS: pcs_local.py score --subset on arena_windows.csv.gz (Mac)
- arena subset k = 9 speed ratio (median v / v_h): 0.9068 -> 0.8883, diff -0.0185 [-0.0272, -0.0028]
- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): k5_10 -0.004 [-0.009, +0.000] (2519), k11_17 -0.010 [-0.022, +0.003] (2704), turn_k11_17 -0.069 [-0.122, -0.013] (144)
- held 460 mean 0.8514 -> 0.8396, paired -0.0118 [-0.0217, +0.0024]
- arena subset strata (control -> arm, zeros): braking (141) 0.965 -> 0.958, 4 -> 5; cruise (23) 0.988 -> 0.988, 0 -> 0; launch (66) 0.863 -> 0.796, 3 -> 4; other (122) 0.959 -> 0.954, 4 -> 4; static (331) 0.870 -> 0.949, 43 -> 17; turn (70) 0.967 -> 0.965, 2 -> 2
- speed ratio by k (control / arm): k2 0.989 / 0.989, k3 0.974 / 0.972, k4 0.959 / 0.960, k5 0.946 / 0.940, k6 0.926 / 0.914, k7 0.908 / 0.907, k8 0.909 / 0.898, k9 0.907 / 0.888
- arena subset by cue (arm - control paired, log-clustered; zeros control -> arm): static: red light at the line, no lead (96) 0.615 -> 0.833, +0.219 [+0.085, +0.302], 37 -> 16; static: light of unknown state, no lead (34) 0.941 -> 0.971, +0.029 [+0.000, +0.064], 2 -> 1; static: lead <= 8 m (188) 0.995 -> 1.000, +0.005 [+0.000, +0.020], 1 -> 0; static: lead 8-30 m (12) 0.750 -> 1.000, +0.250 [+0.000, +0.545], 3 -> 0; static: no light, no lead (1) 1.000 -> 1.000, +0.000 [+nan, +nan], 0 -> 0; red-line launch (render-red per d380) (59) 0.943 -> 0.906, -0.036 [-0.066, -0.006], 0 -> 0
- arena subset zero types (control / arm): {"ok": 656, "corridor": 48, "slow": 36, "offroad": 7, "no-launch": 4, "rear-hit": 1, "AF-front+corridor": 1} / {"ok": 676, "slow": 37, "corridor": 24, "offroad": 8, "no-launch": 5, "rear-hit": 2, "creep": 1}

Gates: held_zeros ok, wait_other_lead ok, wait_red_queue ok, delay_other_lead ok, delay_red_queue ok, launched_other_lead FAIL, launched_red_queue FAIL, held_front ok, arena_af_front ok, static ok, arena_win ok, drift ok

**Verdict: FAIL**
