# ONPOL-NUPLAN pilots: training-side gates (3000 steps, seed 0)

Dev ADE gate: <= ON-NONE (0.532) + 0.01 m; archived YR10m10-F-s0 0.539; dev_drift_off <= 0.3. On-policy rows identical across the arms (running sha1 of the drawn indices): True.

| Tag | Mode | Rows per batch (normal / yaw-rate / hinge-only / on-policy) | Finite | Dev ADE at 0 | Dev ADE | - ON-NONE | drift_off | it/s | Gate |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|:--|
| YNONE-ON-P-s0 | off | 115 / 13 / 0 / 0 | True | 0.539 | 0.532 | +0.000 | 0.045 | 7.77 | pass |
| YSELFnoL-ON-P-s0 | self | 102 / 13 / 0 / 13 | True | 0.539 | 0.535 | +0.004 | 0.045 | 7.23 | pass |

Descriptive, on-policy rows (means of the trainer's 25-step means; `pos` = share of rows with a non-zero hinge, `mean` over all rows, `posmean` over the active ones; imit = the imitation loss, ade = mean displacement to the target in m):

| Tag | Steps | imit | ade | agent_pos | agent_mean | agent_posmean | road_pos | road_mean | road_posmean |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| YSELFnoL-ON-P-s0 | 201-400 | - | - | 0.0496 | 0.0074 | 0.1612 | 0.1669 | 0.0263 | 0.1243 |
| YSELFnoL-ON-P-s0 | 2701-3000 | - | - | 0.0456 | 0.0070 | 0.1556 | 0.1418 | 0.0246 | 0.2155 |
