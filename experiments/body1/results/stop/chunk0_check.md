| check | value | verdict |
|:--|:--|:--|
| rollouts complete | 234 / 234; sessions 234, drive calls 2340, inference errors 0, input errors 0 | pass |
| `drive` total ms p50 / p90 (target <= 100) | 29.4 / 47.8 (p99 69.9, max 126); baseline run 21.5 / 34.5; hook alone 9.4 / 20.3 | pass |
| flag rate per decision (1-5 %, expected 2-2.6 %) | 86 / 2340 = 3.68 %; scenes with a flag 36 / 234; `body` on 2340 decisions; by decision index [8, 9, 6, 4, 8, 9, 12, 14, 9, 7] | pass |
| ego slower 0.5 s after a flagged decision that removes speed (>= 80 %) | 57 / 78 = 73 %; median speed change -0.27 m/s for a median commanded 1.00 m/s^2 | FAIL |
| acceleration command after a flagged decision never under -8.5 m/s^2 | min -8.53 in the 0.5 s after a flag; min -8.53 anywhere in flagged scenes | FAIL |
| steering reversals in flagged scenes (switch on / same scenes of the baseline run) | sum 3 / 2; max per scene 1 / 1 | see the traces |

Flagged decisions: 86, of them removing speed 85, at the 6 m/s^2 cap 11, ego under 0.5 m/s 8; medians: stop point 5.6 m, deceleration 1.00 m/s^2, metres removed in 2 s 1.08, ego speed 3.6 m/s.

| run | n | mean | score 1 | zeros | collision | offroad | corridor | slow |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| switch on | 234 | 0.9451 | 179 | 5 | 2 | 2 | 1 | 50 |
| baseline run | 234 | 0.9518 | 184 | 7 | 4 | 2 | 1 | 43 |

Traces: check_traces.png, scenes 794b439e9922527b, d8338aefbb73570f, 994ab680895a55d2.
