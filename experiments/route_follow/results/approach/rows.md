# Approach course rows (rf_apppool.py pack)

Pool `on1_APP`: 17429 S rows (on1_L18RB0, RSG-L18's) + 3864 approach rows (3091 turn, 773 straight negatives); on-k 20, --on-weight app=turn:2.4288,app=neg:2.4288 (~7 approach rows per batch of 128).

| source | pool | turn rows (train) | scenes / logs | 40-20 / 20-0 m | left / right | speed term | negatives (train) | dev turn rows (logs) | dev negatives |
|:--|:--|--:|:--|:--|:--|--:|--:|:--|--:|
| nup | on1_L18RSG | 2379 | 300 / 63 | 924 / 1455 | 1472 / 907 | 1255 | 595 | 249 (11) | 249 |
| pai | on1_PAIonp | 712 | 17 / 17 | 151 / 561 | 270 / 442 | 365 | 178 | 57 (2) | 64 |

Turn rows in total (train): 3091 (line 1 500: supplement not needed).

Human entry state (train turn rows, q25 / median / q75): nup: v 3.3, 4.1, 5.4 m/s, lateral to the route (+ inside) -0.17, +0.11, +0.41 m, heading to the route (+ inside) -0.8, +2.9, +6.8 deg; pai: v 2.2, 3.3, 3.5 m/s, lateral to the route (+ inside) -0.71, -0.52, -0.20 m, heading to the route (+ inside) -0.6, +1.8, +5.3 deg

Drops: nup: {"crawl": 1120, "far": 61, "no_class": 6063, "route_bends": 1071, "route_mismatch": 155}; pai: {"crawl": 470, "far": 67, "no_class": 5522, "no_path": 1163, "route_bends": 4029, "route_mismatch": 139}
