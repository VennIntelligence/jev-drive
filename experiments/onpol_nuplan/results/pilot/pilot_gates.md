# ONPOL-NUPLAN pilots: training-side gates (3000 steps, seed 0)

Dev ADE gate: <= ON-NONE (0.540) + 0.01 m; archived P2H10S-F-s0 0.545; dev_drift_off <= 0.3. On-policy rows identical across the arms (running sha1 of the drawn indices): True.

| Tag | Mode | Rows per batch (normal / hinge-only / on-policy) | Finite | Dev ADE at 0 | Dev ADE | - ON-NONE | drift_off | it/s | Gate |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|:--|
| SNONE-ON-P-s0 | off | 115 / 13 / 0 | True | 0.545 | 0.540 | +0.000 | 0.046 | 5.94 | pass |
| SIMIT-ON-P-s0 | imit | 102 / 13 / 13 | True | 0.545 | 0.557 | +0.018 | 0.049 | 7.02 | FAIL: dev ADE 0.557371199131012 > ON-NONE 0.5396775007247925 + 0.01 |
| SSELF-ON-P-s0 | self | 102 / 13 / 13 | True | 0.545 | 0.560 | +0.020 | 0.051 | 5.66 | FAIL: dev ADE 0.5596268177032471 > ON-NONE 0.5396775007247925 + 0.01 |
| SBOTH-ON-P-s0 | both | 102 / 13 / 13 | True | 0.545 | 0.568 | +0.028 | 0.052 | 5.41 | FAIL: dev ADE 0.5681552290916443 > ON-NONE 0.5396775007247925 + 0.01 |
| SSELFnoL-ON-P-s0 | self | 102 / 13 / 13 | True | 0.545 | 0.542 | +0.003 | 0.046 | 5.82 | pass |

Descriptive, on-policy rows (means of the trainer's 25-step means; `pos` = share of rows with a non-zero hinge, `mean` over all rows, `posmean` over the active ones; imit = the imitation loss, ade = mean displacement to the target in m):

| Tag | Steps | imit | ade | agent_pos | agent_mean | agent_posmean | road_pos | road_mean | road_posmean |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SIMIT-ON-P-s0 | 201-400 | 3.1938 | 1.6680 | - | - | - | - | - | - |
| SIMIT-ON-P-s0 | 2701-3000 | 2.3120 | 1.2720 | - | - | - | - | - | - |
| SSELF-ON-P-s0 | 201-400 | - | - | 0.0492 | 0.0080 | 0.1509 | 0.1550 | 0.0238 | 0.1698 |
| SSELF-ON-P-s0 | 2701-3000 | - | - | 0.0421 | 0.0049 | 0.1132 | 0.1438 | 0.0171 | 0.1101 |
| SBOTH-ON-P-s0 | 201-400 | 3.1162 | 1.6094 | 0.0465 | 0.0076 | 0.1556 | 0.1581 | 0.0229 | 0.1629 |
| SBOTH-ON-P-s0 | 2701-3000 | 2.3305 | 1.2780 | 0.0315 | 0.0037 | 0.1140 | 0.1387 | 0.0153 | 0.1164 |
| SSELFnoL-ON-P-s0 | 201-400 | - | - | 0.0546 | 0.0091 | 0.1616 | 0.1554 | 0.0242 | 0.1726 |
| SSELFnoL-ON-P-s0 | 2701-3000 | - | - | 0.0531 | 0.0068 | 0.1319 | 0.1385 | 0.0169 | 0.1396 |
