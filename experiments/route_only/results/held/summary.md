# ROUTE-ONLY summary (460 common scenes)

| label | mean | zeros | collision / offroad / corridor / sanity / other | GT (m) | lateral GT (m) | progress_rel | travelled / log (m) | route served (decisions) | mean poses on route | gap median (m) | lateral change 1 / 2 / 4 s median (m) |
|:--|--:|--:|:--|--:|--:|--:|:--|--:|--:|--:|:--|
| RSG-L18-s0 | 0.8656 | 34 | 4 / 17 / 5 / 8 / 0 | 1.135 | 0.869 | 0.7876 | 45.9 / 58.5 | nan | nan | - | - |
| R-own | 0.8628 | 35 | 5 / 17 / 5 / 8 / 0 | 1.148 | 0.873 | 0.7879 | 45.7 / 58.5 | 0.476 | 2.53 | 19.7 | 0.0 / 0.016 / 0.181 |
| R-log-priv | 0.9166 | 38 | 8 / 17 / 5 / 8 / 0 | 0.916 | 0.913 | 0.9405 | 54.2 / 58.5 | 0.558 | 3.25 | 18.6 | 0.012 / 0.061 / 0.25 |
| R-bridge | 0.7275 | 98 | 12 / 74 / 4 / 8 / 0 | 1.128 | 0.865 | 0.7995 | 41.5 / 58.5 | 0.996 | 7.97 | 18.7 | 0.023 / 0.066 / 0.175 |

R-own: decisions 8129, kept-plan reasons {'gap': 4235, 'ok': 3866, 'no_route': 9, 'behind_end': 19}, served share by decision index {0: 0.091, 1: 0.164, 2: 0.237, 3: 0.299, 4: 0.343, 5: 0.383, 6: 0.409, 7: 0.44, 8: 0.465, 9: 0.487, 10: 0.533, 11: 0.566, 12: 0.597, 13: 0.642, 14: 0.67, 15: 0.713, 16: 0.749, 17: 0.777}

R-log-priv: decisions 8129, kept-plan reasons {'gap': 3508, 'ok': 4600, 'no_route': 9, 'behind_end': 12}, served share by decision index {0: 0.088, 1: 0.162, 2: 0.248, 3: 0.334, 4: 0.383, 5: 0.467, 6: 0.515, 7: 0.555, 8: 0.58, 9: 0.615, 10: 0.65, 11: 0.681, 12: 0.728, 13: 0.77, 14: 0.796, 15: 0.818, 16: 0.824, 17: 0.842}

R-bridge: decisions 8129, kept-plan reasons {'bridge': 6799, 'ok': 1301, 'no_route': 9, 'behind_end': 20}, served share by decision index {0: 0.993, 1: 0.996, 2: 0.996, 3: 0.996, 4: 0.993, 5: 0.993, 6: 0.996, 7: 0.996, 8: 0.998, 9: 0.998, 10: 0.998, 11: 0.998, 12: 0.998, 13: 0.998, 14: 0.998, 15: 0.998, 16: 0.998, 17: 0.998}
