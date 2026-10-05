# Guard set: dg1 (subset mode)

Verdict: **PASS** (1 pass, 0 fail, 3 not evaluable). 2026-10-05 12:38:42

| line | readout | candidate | shipped | delta | rule | result | note |
|---|---|---:|---:|---:|---|:--|---|
| hugsim (subset) | spins (heading err >= 60 deg) | 0 | 0 | 0 | not up vs shipped | pass | 11 scenes |
| hugsim (subset) | stuck (end max_steps) | 4 | 6 | -2 | reported (stuck: reduction is the goal) | n/a |  |
| hugsim (subset) | completes | 3 | 3 | 0 | reported | n/a |  |
| hugsim (subset) | HD-Score mean (paired over scenes) | 0.532 | 0.544 | -0.012 [-0.2, 0.189] | reported | n/a | paired mean diff -0.012 |

Runtime per line (min): hugsim 9.1; sum 9.1
