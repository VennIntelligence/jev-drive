# B2D openpilot lateral path: extra readouts (drive = shipped, opc = OP_CTRL)

## 1. Realised c (heading change over 5 ticks per degree of the plan's 1 s direction; lat == op, non-warm, non-zone; magnitude)

| v (m/s) | drive | opc |
|---|---|---|
| 0-1 | 0.014 [0.002, 0.043] (n 131, runs 16) | 0.013 [0.013, 0.028] (n 144, runs 16) |
| 1-2 | 0.059 [0.017, 0.097] (n 271, runs 16) | 0.086 [0.033, 0.201] (n 253, runs 16) |
| 2-3 | 0.157 [0.081, 0.230] (n 380, runs 16) | 0.063 [0.030, 0.235] (n 332, runs 16) |
| 0-3 | 0.071 [0.032, 0.121] (n 782, runs 16) | 0.059 [0.027, 0.146] (n 729, runs 16) |

## 2. Who steers (non-warm ticks) and the heading change each owner delivered

| arm | owner | ticks | share | sum abs dyaw (deg) | share of heading |
|---|---|---|---|---|---|
| drive | op | 19712 | 41.5% | 205 | 5.9% |
| drive | zone | 10264 | 21.6% | 2862 | 82.7% |
| drive | div | 17576 | 37.0% | 392 | 11.3% |
| drive | other | 0 | 0.0% | 0 | 0.0% |
| opc | op | 17991 | 34.9% | 207 | 6.0% |
| opc | zone | 14269 | 27.7% | 2836 | 82.7% |
| opc | div | 19256 | 37.4% | 386 | 11.3% |
| opc | other | 0 | 0.0% | 0 | 0.0% |

## 3. The path on op-owned ticks (opc)

- op-owned ticks with a trace: 13143; v <= 0.3 (modeld hold / controlsd inactive): 43.2%
- clip_curvature binds (des != act, moving): 0.07% of moving ticks
- |kappa_cmd| quantiles on moving ticks (1/m) 50 / 90 / 99 / max: 0.0011 / 0.0052 / 0.0140 / 0.0702
- |kappa_real| same: 0.0011 / 0.0053 / 0.0169 / 0.0847
- lateral-acceleration cap binds where |kappa_cmd| v^2 > 3: 0.000% of moving ticks
- bends (|kappa_cmd| > 0.01, R < 100 m; 233 ticks): realised / commanded kappa slope 1.145 (delay-lagged but not attenuated if ~1)

## 4. Junction turns (zone-owned in both arms): would the action curvature have turned enough? (counterfactual, from the logged act_k)

| arm | zone ticks (v > 1) | slope of act_k on realised kappa | turn ticks (R < 33 m) | summed act_k / realised | same sign | mean v on turns |
|---|---|---|---|---|---|---|
| drive | 1242 | 0.43 | 482 | 0.56 | 93% | 3.5 |
| opc | 1134 | 0.43 | 496 | 0.55 | 94% | 3.4 |
