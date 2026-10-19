# PAI 281: ACC+GAP against VCONT+LEAD, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`, reference = `/root/autodl-tmp/ujs/runs/route_follow/pai_input_loop/N0`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J ['1059fc66'] ['<AioRpcError of RPC that terminated with:'], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |
| VCONT+LEAD | 280 | 0.6100 | 59 | 104 | 0.769 | 0.613 |

Paired mean (ACC+GAP - VCONT+LEAD): **+0.1342 [+0.0940, +0.1742]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0000 [-0.0393, +0.0393] (+0 zeros).


## Zero types

| type | ACC+GAP | VCONT+LEAD |
|---|--:|--:|
| AF-front | 3 | 2 |
| AF-side | 0 | 3 |
| corridor | 44 | 39 |
| offroad | 12 | 15 |

## Partial scores (0 < s < 0.98)

| type | ACC+GAP | VCONT+LEAD |
|---|--:|--:|
| hit | 0 | 0 |
| rear-hit | 26 | 75 |
| no-launch | 0 | 0 |
| creep | 0 | 0 |
| slow | 11 | 42 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | ACC+GAP mean | VCONT+LEAD mean | diff | contribution | ACC+GAP zeros | VCONT+LEAD zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| launch | 37 | 0.7738 | 0.5028 | +0.2710 | +0.0358 | 3 | 8 | 0 | 5 |
| moving | 241 | 0.7418 | 0.6274 | +0.1144 | +0.0984 | 55 | 50 | 17 | 12 |

## New zeros (zero for ACC+GAP, scored for VCONT+LEAD): 17

| scene | stratum | ACC+GAP | VCONT+LEAD |
|---|---|---|---|
| 787c8ada | moving | 0.000 AF-front | 0.277 rear-hit |
| fde5cd20 | moving | 0.000 corridor | 0.449 rear-hit |
| 6c88b571 | moving | 0.000 offroad | 0.303 rear-hit |
| 1e139d7e | moving | 0.000 corridor | 0.314 rear-hit |
| 62ec6f99 | moving | 0.000 AF-front | 1.000 ok |
| 975e6436 | moving | 0.000 offroad | 0.421 rear-hit |
| ffa52af8 | moving | 0.000 corridor | 0.455 slow |
| 11250b18 | moving | 0.000 corridor | 1.000 ok |
| 9e8763a0 | moving | 0.000 corridor | 1.000 ok |
| 75784c2a | moving | 0.000 offroad | 0.466 slow |
| e20ef839 | moving | 0.000 corridor | 0.947 slow |
| 88b2c556 | moving | 0.000 corridor | 0.660 rear-hit |
| 90f1c66d | moving | 0.000 corridor | 1.000 ok |
| 3c1893bb | moving | 0.000 offroad | 0.514 slow |
| 97ac3ade | moving | 0.000 corridor | 1.000 ok |
| 8ca1f1e0 | moving | 0.000 offroad | 1.000 ok |
| 7f699124 | moving | 0.000 offroad | 0.882 slow |

## Fixed zeros (zero for VCONT+LEAD, scored for ACC+GAP): 17

| scene | stratum | ACC+GAP | VCONT+LEAD |
|---|---|---|---|
| ed392963 | moving | 1.000 ok | 0.000 corridor |
| 90f9cd38 | launch | 1.000 ok | 0.000 AF-front |
| 272b7bf0 | moving | 0.825 slow | 0.000 corridor |
| 1f5c9e1f | moving | 0.565 rear-hit | 0.000 offroad |
| 28bde89a | moving | 0.355 rear-hit | 0.000 AF-side |
| 8b28b8f9 | launch | 1.000 ok | 0.000 AF-front |
| ff7ccca9 | moving | 0.559 rear-hit | 0.000 corridor |
| 3fd457e2 | moving | 0.819 slow | 0.000 corridor |
| 0fdf46a2 | launch | 0.766 slow | 0.000 AF-side |
| 7c0fb633 | moving | 1.000 ok | 0.000 offroad |
| 6f6aaecc | moving | 1.000 ok | 0.000 offroad |
| 9f243b4a | moving | 0.491 rear-hit | 0.000 corridor |
| 75c7d11b | moving | 1.000 ok | 0.000 corridor |
| fd7779a6 | launch | 0.462 rear-hit | 0.000 offroad |
| 9b3ca23d | launch | 1.000 ok | 0.000 AF-side |
| a3bad8dd | moving | 1.000 ok | 0.000 offroad |
| d5527d43 | moving | 1.000 ok | 0.000 corridor |
