| check | value | verdict |
|:--|:--|:--|
| rollouts complete, no driver error, a re-plan record on every decision | 233 / 233; sessions 233, drive calls 2330, inference errors 0, input errors 0; records 2330 / 2330 | pass |
| `drive` total ms p50 / p90 (<= 100) | 35.4 / 55.1 (p99 82.6, max 119); baseline run 17.8 / 30.7; hook alone 13.7 / 25.6 | pass |
| share of decisions re-planned (0.40-5.00 %) | 19 / 2330 = 0.82 %; scenes re-planned 19 / 233; by decision index [0, 1, 1, 3, 3, 1, 3, 2, 3, 2]; by size {'0.3': 1, '0.6': 6, '0.9': 5, '1.2': 3, '1.5': 4}; left / right 10 / 9; flagged without a clear candidate 72 | pass |
| no side change within 2 decisions | 0 (scenes served both sides at any distance: 0) | pass |
| new zeros in re-planned scenes <= zeros removed | new zeros 2 (in re-planned scenes 2: offroad, offroad); zeros removed 1 (corridor; in re-planned scenes 1) | FAIL |
| steering reversals in re-planned scenes (switch on / same scenes of the baseline run): no scene with >= 3 more | sum 10 / 2; max per scene 2 / 1; scenes with >= 3 more than the baseline run: 0 | pass |
| heading at decision 9 minus the baseline run's, re-planned scenes (deg) | n 19, sd 12.24, mean abs 6.62, max abs 43.37 | reported |

| run | n | mean | score 1 | zeros | collision | offroad | corridor | slow |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| switch on | 233 | 0.9368 | 185 | 11 | 2 | 7 | 2 | 37 |
| baseline run | 233 | 0.9411 | 186 | 10 | 2 | 5 | 3 | 37 |

Re-planned scenes: 19, mean score 0.7764 against 0.8293; scenes never re-planned: 214, with a different score 0.

Traces: check_traces.png, scenes a04628cdd3f25947, 9fdd329b72e85179, 60e8f2447c205324.
