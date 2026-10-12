# STEER-LONG speed-history counterfactual (amendment 2 item 1)

G-rep: replayed F plan within 0.10 m of the stored plan at 4 s: 5685 / 5685 (100.00 %)

Decisions read (zero rollouts, k >= 3, before the event, log moving >= 2 m/s 1.5 s ahead): 2479, scenes 103

| label | n | ego / log now | F plan 1-2 s / log | A | B | recovery A | recovery B |
|:--|--:|--:|--:|--:|--:|--:|--:|
| YR10m10-F-s0 | 613 | 0.910 | 0.836 | 0.908 | 0.962 | +0.44 | +0.77 |
| YR10m10-F-s1 | 629 | 0.908 | 0.852 | 0.901 | 0.969 | +0.33 | +0.79 |
| YS-L18-s0 | 339 | 0.884 | 0.865 | 0.927 | 0.978 | +0.46 | +0.83 |
| YS-L18-s1 | 316 | 0.957 | 0.958 | 0.975 | 1.012 | +0.39 | +1.30 |
| YS2G-L18-s0 | 308 | 0.999 | 1.008 | 0.993 | 1.009 | +nan | +nan |
| YS2G-L18-s1 | 274 | 0.975 | 0.960 | 0.976 | 0.994 | +0.40 | +0.85 |
| all | 2479 | 0.930 | 0.898 | 0.939 | 0.982 | +0.40 | +0.83 |

Subset ego < 0.8 x log now (805 decisions): F 0.537, A 0.764, B 0.936; recovery A +0.49, B +0.86

Registered line on B (all): recovery +0.83 -> history feedback explains the slowness
