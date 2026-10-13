# ROUTE-ONLY summary (2 common scenes)

| label | mean | zeros | collision / offroad / corridor / sanity / other | GT (m) | lateral GT (m) | progress_rel | travelled / log (m) | route served (decisions) | mean poses on route | gap median (m) | lateral change 1 / 2 / 4 s median (m) |
|:--|--:|--:|:--|--:|--:|--:|:--|--:|--:|--:|:--|
| off | 0.5000 | 1 | 0 / 1 / 0 / 0 / 0 | 3.183 | 3.183 | 0.9978 | 28.2 / 38.8 | nan | nan | - | - |
| R-own | 0.5000 | 1 | 0 / 1 / 0 / 0 / 0 | 3.183 | 3.183 | 0.9978 | 28.2 / 38.8 | 0.167 | 0.53 | 28.7 | 0.0 / 0.0 / 1.114 |
| R-bridge | 0.0000 | 2 | 0 / 2 / 0 / 0 / 0 | 1.469 | 1.469 | 1.0000 | 11.0 / 38.8 | 1.000 | 8.00 | 22.2 | 0.062 / 0.224 / 0.452 |
| R-log-priv | 0.5000 | 1 | 0 / 1 / 0 / 0 / 0 | 2.252 | 2.252 | 0.9405 | 27.5 / 38.8 | 0.278 | 0.97 | 29.9 | 0.001 / 0.012 / 0.481 |

R-own: decisions 36, kept-plan reasons {'gap': 30, 'ok': 6}, served share by decision index {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5, 17: 0.5}

R-bridge: decisions 36, kept-plan reasons {'bridge': 33, 'ok': 3}, served share by decision index {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0}

R-log-priv: decisions 36, kept-plan reasons {'gap': 26, 'ok': 10}, served share by decision index {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0}
