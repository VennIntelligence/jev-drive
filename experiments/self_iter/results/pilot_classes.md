# SELF-ITER pilot: per-class open-loop reads (decision 338)

These are the same dev rows as `gate_pilot.md`: navtrain dev (1 789 rows) and the held-out trainval logs (7 236 rows). Rows are grouped by the human's
own motion class (lib/self_iter.classes). Each cell gives the mean ADE (m) and the mean signed 4 s arc-length bias (plan − human, m). C is WTCP-L18-s0
(the RSG-L18 recipe at 1 000 steps). The arms are SIU1P (trainval rows added, uniform draw) and SIM3P / SIM6P (top 10 % × 3 / × 6). Computed from
`runs/self_iter/scan/<tag>.dev.npz` on the box.

| set | class | rows | C | SIU1P | SIM3P | SIM6P |
|---|---|--:|--:|--:|--:|--:|
| nav_dv | starting | 382 | 0.520 / −0.08 | 0.595 / −0.56 | 0.653 / −0.81 | 0.691 / −1.01 |
| nav_dv | turning | 349 | 0.646 / +0.05 | 0.650 / +0.04 | 0.651 / −0.00 | 0.665 / −0.10 |
| nav_dv | braking | 543 | 0.489 / +0.31 | 0.497 / +0.40 | 0.497 / +0.41 | 0.499 / +0.43 |
| nav_dv | cruising | 515 | 0.499 / −0.39 | 0.492 / −0.39 | 0.502 / −0.43 | 0.502 / −0.47 |
| tv_hold | stopped | 2 434 | 1.186 / +3.45 | 0.111 / +0.36 | 0.095 / +0.29 | 0.088 / +0.25 |
| tv_hold | starting | 364 | 0.813 / +0.46 | 0.740 / −0.88 | 0.771 / −1.19 | 0.806 / −1.45 |
| tv_hold | turning | 511 | 0.709 / −0.13 | 0.662 / −0.26 | 0.660 / −0.33 | 0.664 / −0.45 |
| tv_hold | braking | 836 | 0.684 / +0.93 | 0.652 / +0.96 | 0.632 / +0.89 | 0.621 / +0.83 |
| tv_hold | cruising | 3 091 | 0.658 / −0.08 | 0.586 / −0.12 | 0.576 / −0.17 | 0.575 / −0.18 |

What to look at: the stopped rows are fixed: the model now holds where the human holds. The starting rows show the cost: in both sets the
plan falls behind the human's launch, and the gap grows with the mining strength.
