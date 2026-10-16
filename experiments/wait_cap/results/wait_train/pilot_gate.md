# WAIT-TRAIN pilot gates (1 000 steps, seed 0; open loop on waitcap_test, dev rows of navtrain s0)

| arm | wait W (>= 0.60) | go G (>= 0.85) | into W_L (<= 0.05) | D | dev ADE (<= C + 0.01) | lat 2 s med abs diff vs C (<= 0.05 m) | S_N stop / over med | launch-lock | pass |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| WTCP-L18-s0 (C) | 0.007 | 1.000 | 0.287 | 0.007 | 0.5290 | - | 0.603 / 1.28 | - | - |
| WTJ2P-L18-s0 | 0.861 | 0.958 | 0.010 | 0.819 | 0.6080 | 0.011 | 0.766 / 0.53 | no | fail |
| WTJ5P-L18-s0 | 0.905 | 0.958 | 0.010 | 0.863 | 0.7623 | 0.020 | 0.842 / 0.17 | no | fail |

secondary (not gates), decision 331's ledger cells on the F read (capledger_test, success share, J vs C paired):
- WTJ2P-L18-s0: S_L 0.943 vs 0.380 (n 595), diff 0.563 [0.481, 0.642]; Y_P 0.500 vs 0.063 (n 142), diff 0.437 [0.305, 0.578]; Y_V 0.484 vs 0.093 (n 161), diff 0.391 [0.292, 0.507]
- WTJ5P-L18-s0: S_L 0.980 vs 0.380 (n 595), diff 0.600 [0.514, 0.682]; Y_P 0.704 vs 0.063 (n 142), diff 0.641 [0.536, 0.754]; Y_V 0.671 vs 0.093 (n 161), diff 0.578 [0.471, 0.684]

paired vs C (log-clustered bootstrap): WTJ2P-L18-s0: wait W 0.854 [0.816, 0.886], wait G 0.042 [0.024, 0.062]; WTJ5P-L18-s0: wait W 0.898 [0.870, 0.920], wait G 0.042 [0.025, 0.061]

verdict: STOP: both doses fail
