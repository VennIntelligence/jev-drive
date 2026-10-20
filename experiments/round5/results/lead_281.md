# LEAD rule on the stored PAI 281 human states (Mac, NTV0's tax record; the logged traffic is the same in every read)

Rule: r5.py LEAD (addendum 3). Launches: NTV0's 37 launch scenes (handoff < 1 m/s, log > 5 m), the human's state at its launch t_L (d1_launch.csv; log 0.5 m past the handoff pose). Overshoot zeros: the human's state at its rest (d1 overshoot rule) and 4 s before.

Launch scenes: 37; lead at t_L 26, at the handoff 28; NTV0's serve chose the LEAD mode 21 (main session's 21); both 20.

| read | launched / lead scenes | launched / no-lead scenes |
|:--|--:|--:|
| NTV0 281 ACC + GAP | 23 / 26 | 10 / 11 |
| R2C 281 ACC + GAP | 23 / 26 | 10 / 11 |
| R2J 281 ACC + GAP | 15 / 26 | 6 / 11 |

| read | overshoot zeros | lead at rest | lead 4 s before rest | lead at either | no lead |
|:--|--:|--:|--:|--:|--:|
| NTV0 | 23 | 3 | 3 | 3 | 20 |
| R2C | 22 | 3 | 3 | 3 | 19 |
| R2J | 21 | 4 | 4 | 4 | 17 |

4 s before rest outside the rollout's traffic snapshots: 1 of 25 scenes (counted no lead).
