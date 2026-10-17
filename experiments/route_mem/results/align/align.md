## Gates

served = offline: the logged `rmem` against the rebuild from the regenerated messages (max point distance, mask agreement); msg lead: AlpaSim's own route anchor against the tracked ego arc; fold: rmem.merge's polyline doubles back by > 1 m (decisions, rollouts).

| set | rollouts (errors) | decisions | served = offline max m | mask agree | msg lead > 1 m | fold decisions (rollouts) | served empty, ordered not |
|:--|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0, held 460 @18 | 452 (0) | 8129 | 2.7e-04 | 1.0 | 0.0033 (27) | 32 (5) | 13 |
| RTB1ARO-L18-s0 (V2), held 460 @18 | 452 (0) | 8129 | 2.2e-04 | 1.0 | 0.0036 (29) | 31 (5) | 15 |
| RSG-L18-s0, public 1491 @10 | 1479 (0) | 14790 | 2.4e-04 | 1.0 | 0.0018 (26) | 16 (2) | 40 |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | 1479 (0) | 14790 | 2.4e-04 | 1.0 | 0.0016 (24) | 16 (2) | 40 |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | 1021 (0) | 18352 | 2.2e-02 | 1.0 | 0.0037 (67) | 38 (4) | 45 |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | 1479 (0) | 14790 | 3.1e-04 | 1.0 | 0.0016 (24) | 16 (2) | 40 |

## T1 Lead of the served memory (projected layout) against the car's route arc

lead = true route offset of the slot's point minus the slot's label (m); first = the first valid slot; near = slots <= +20 m.

| set | stratum | decisions | near slots valid | first lead median | p90 | >= 5 m | >= 10 m | >= 30 m | ahead (near >= 10 m) | early turn |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0, held 460 @18 | all | 8129 | 0.505 | 0.0 | 7.8 | 0.138 | 0.075 | 0.007 | 0.041 (337) | 0.019 (156) |
| RSG-L18-s0, held 460 @18 | route turn ahead: no turn in 60 m | 2 | 1.000 | 0.1 | 0.2 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RSG-L18-s0, held 460 @18 | route turn ahead: turn 44-60 m | 5472 | 0.527 | 0.0 | 0.2 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RSG-L18-s0, held 460 @18 | route turn ahead: turn within 42 m | 2655 | 0.461 | 3.6 | 16.5 | 0.421 | 0.231 | 0.021 | 0.127 (337) | 0.059 (156) |
| RSG-L18-s0, held 460 @18 | rollout geometry: bend | 1454 | 0.558 | 0.0 | 0.7 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RSG-L18-s0, held 460 @18 | rollout geometry: straight | 3039 | 0.497 | 0.0 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RSG-L18-s0, held 460 @18 | rollout geometry: turn | 3420 | 0.472 | 0.9 | 12.7 | 0.294 | 0.154 | 0.009 | 0.077 (264) | 0.038 (130) |
| RSG-L18-s0, held 460 @18 | rollout geometry: uturn | 216 | 0.806 | 5.7 | 31.7 | 0.528 | 0.398 | 0.106 | 0.338 (73) | 0.120 (26) |
| RTB1ARO-L18-s0 (V2), held 460 @18 | all | 8129 | 0.514 | 0.0 | 7.8 | 0.134 | 0.075 | 0.007 | 0.041 (332) | 0.018 (144) |
| RTB1ARO-L18-s0 (V2), held 460 @18 | route turn ahead: no turn in 60 m | 9 | 1.000 | -0.0 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1ARO-L18-s0 (V2), held 460 @18 | route turn ahead: turn 44-60 m | 5521 | 0.535 | 0.0 | 0.2 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1ARO-L18-s0 (V2), held 460 @18 | route turn ahead: turn within 42 m | 2599 | 0.466 | 3.7 | 16.8 | 0.420 | 0.234 | 0.021 | 0.128 (332) | 0.055 (144) |
| RTB1ARO-L18-s0 (V2), held 460 @18 | rollout geometry: bend | 1454 | 0.569 | 0.0 | 0.8 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1ARO-L18-s0 (V2), held 460 @18 | rollout geometry: straight | 3039 | 0.497 | 0.0 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1ARO-L18-s0 (V2), held 460 @18 | rollout geometry: turn | 3438 | 0.489 | 0.8 | 12.9 | 0.288 | 0.154 | 0.009 | 0.078 (267) | 0.035 (121) |
| RTB1ARO-L18-s0 (V2), held 460 @18 | rollout geometry: uturn | 198 | 0.788 | 5.4 | 32.5 | 0.520 | 0.394 | 0.116 | 0.328 (65) | 0.116 (23) |
| RSG-L18-s0, public 1491 @10 | all | 14790 | 0.247 | 0.0 | 10.3 | 0.189 | 0.103 | 0.007 | 0.041 (609) | 0.019 (283) |
| RSG-L18-s0, public 1491 @10 | route turn ahead: no turn in 60 m | 58 | 0.000 | 0.0 | 0.6 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RSG-L18-s0, public 1491 @10 | route turn ahead: turn 44-60 m | 9817 | 0.251 | 0.0 | 0.3 | 0.003 | 0.001 | 0.000 | 0.000 (0) | 0.000 (3) |
| RSG-L18-s0, public 1491 @10 | route turn ahead: turn within 42 m | 4915 | 0.243 | 6.2 | 18.7 | 0.562 | 0.310 | 0.022 | 0.124 (609) | 0.057 (280) |
| RSG-L18-s0, public 1491 @10 | rollout geometry: bend | 2710 | 0.234 | 0.1 | 1.4 | 0.019 | 0.000 | 0.000 | 0.000 (0) | 0.001 (2) |
| RSG-L18-s0, public 1491 @10 | rollout geometry: straight | 6830 | 0.257 | 0.0 | 0.1 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RSG-L18-s0, public 1491 @10 | rollout geometry: turn | 4970 | 0.216 | 5.2 | 16.5 | 0.504 | 0.268 | 0.009 | 0.092 (456) | 0.045 (223) |
| RSG-L18-s0, public 1491 @10 | rollout geometry: uturn | 280 | 0.679 | 17.9 | 46.3 | 0.843 | 0.700 | 0.218 | 0.546 (153) | 0.207 (58) |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | all | 14790 | 0.254 | 0.0 | 10.2 | 0.187 | 0.102 | 0.007 | 0.041 (605) | 0.019 (287) |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | route turn ahead: no turn in 60 m | 58 | 0.017 | 0.0 | 0.6 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | route turn ahead: turn 44-60 m | 9818 | 0.259 | 0.0 | 0.3 | 0.003 | 0.001 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | route turn ahead: turn within 42 m | 4914 | 0.246 | 6.1 | 18.7 | 0.557 | 0.306 | 0.022 | 0.123 (605) | 0.058 (287) |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | rollout geometry: bend | 2800 | 0.244 | 0.1 | 1.4 | 0.019 | 0.000 | 0.000 | 0.000 (0) | 0.001 (3) |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | rollout geometry: straight | 6780 | 0.265 | 0.0 | 0.1 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | rollout geometry: turn | 4920 | 0.220 | 5.2 | 16.5 | 0.502 | 0.265 | 0.010 | 0.090 (445) | 0.045 (222) |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | rollout geometry: uturn | 290 | 0.672 | 17.7 | 45.4 | 0.841 | 0.700 | 0.210 | 0.552 (160) | 0.214 (62) |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | all | 18352 | 0.476 | 0.0 | 6.5 | 0.120 | 0.063 | 0.007 | 0.028 (517) | 0.013 (245) |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | route turn ahead: no turn in 60 m | 75 | 0.493 | 0.0 | 0.3 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | route turn ahead: turn 44-60 m | 12808 | 0.492 | 0.0 | 0.2 | 0.001 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | route turn ahead: turn within 42 m | 5469 | 0.438 | 3.2 | 15.2 | 0.401 | 0.210 | 0.023 | 0.095 (517) | 0.045 (245) |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | rollout geometry: bend | 3400 | 0.505 | 0.0 | 0.8 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | rollout geometry: straight | 7636 | 0.480 | 0.0 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | rollout geometry: turn | 6722 | 0.434 | 0.8 | 11.6 | 0.277 | 0.132 | 0.002 | 0.044 (296) | 0.018 (124) |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | rollout geometry: uturn | 594 | 0.724 | 9.1 | 44.1 | 0.589 | 0.446 | 0.189 | 0.372 (221) | 0.204 (121) |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | all | 14790 | 0.254 | 0.0 | 10.2 | 0.187 | 0.103 | 0.007 | 0.041 (607) | 0.019 (287) |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | route turn ahead: no turn in 60 m | 58 | 0.017 | 0.0 | 0.6 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | route turn ahead: turn 44-60 m | 9816 | 0.259 | 0.0 | 0.3 | 0.003 | 0.001 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | route turn ahead: turn within 42 m | 4916 | 0.248 | 6.2 | 18.8 | 0.557 | 0.307 | 0.022 | 0.123 (607) | 0.058 (287) |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | rollout geometry: bend | 2780 | 0.243 | 0.1 | 1.4 | 0.022 | 0.000 | 0.000 | 0.000 (0) | 0.001 (2) |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | rollout geometry: straight | 6790 | 0.264 | 0.0 | 0.1 | 0.000 | 0.000 | 0.000 | 0.000 (0) | 0.000 (0) |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | rollout geometry: turn | 4930 | 0.222 | 5.1 | 16.6 | 0.499 | 0.266 | 0.010 | 0.090 (446) | 0.045 (222) |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | rollout geometry: uturn | 290 | 0.676 | 17.7 | 46.1 | 0.841 | 0.703 | 0.214 | 0.555 (161) | 0.217 (63) |

## T1b First-slot lead by decision index (all sets pooled per benchmark)

| benchmark | k | decisions | near valid | first lead p90 | >= 10 m | early |
|:--|--:|--:|--:|--:|--:|--:|
| held 460 @18 | 0 | 904 | 0.05 | 14.3 | 0.184 | 30 |
| held 460 @18 | 1 | 904 | 0.05 | 13.7 | 0.155 | 30 |
| held 460 @18 | 2 | 904 | 0.05 | 12.7 | 0.144 | 28 |
| held 460 @18 | 3 | 904 | 0.05 | 11.8 | 0.128 | 26 |
| held 460 @18 | 4 | 904 | 0.14 | 10.9 | 0.119 | 22 |
| held 460 @18 | 5 | 904 | 0.24 | 10.0 | 0.100 | 21 |
| held 460 @18 | 6 | 904 | 0.40 | 8.7 | 0.079 | 18 |
| held 460 @18 | 7 | 904 | 0.50 | 7.6 | 0.071 | 17 |
| held 460 @18 | 8 | 904 | 0.58 | 6.2 | 0.066 | 19 |
| held 460 @18 | 9 | 904 | 0.63 | 5.0 | 0.054 | 14 |
| held 460 @18 | 10 | 904 | 0.67 | 4.4 | 0.044 | 12 |
| held 460 @18 | 11 | 904 | 0.71 | 3.4 | 0.043 | 13 |
| held 460 @18 | 12 | 904 | 0.74 | 2.7 | 0.037 | 11 |
| held 460 @18 | 13 | 904 | 0.79 | 2.1 | 0.035 | 9 |
| held 460 @18 | 14 | 904 | 0.83 | 1.6 | 0.027 | 7 |
| held 460 @18 | 15 | 900 | 0.89 | 1.1 | 0.023 | 7 |
| held 460 @18 | 16 | 900 | 0.91 | 0.6 | 0.022 | 10 |
| held 460 @18 | 17 | 898 | 0.93 | 0.4 | 0.020 | 6 |
| public 1491 @10 | 0 | 2958 | 0.04 | 14.5 | 0.181 | 84 |
| public 1491 @10 | 1 | 2958 | 0.04 | 13.4 | 0.166 | 86 |
| public 1491 @10 | 2 | 2958 | 0.05 | 12.2 | 0.137 | 82 |
| public 1491 @10 | 3 | 2958 | 0.05 | 11.2 | 0.116 | 74 |
| public 1491 @10 | 4 | 2958 | 0.11 | 10.4 | 0.105 | 63 |
| public 1491 @10 | 5 | 2958 | 0.22 | 9.4 | 0.090 | 46 |
| public 1491 @10 | 6 | 2958 | 0.37 | 8.2 | 0.071 | 36 |
| public 1491 @10 | 7 | 2958 | 0.47 | 6.7 | 0.062 | 36 |
| public 1491 @10 | 8 | 2958 | 0.55 | 5.8 | 0.055 | 34 |
| public 1491 @10 | 9 | 2958 | 0.61 | 4.8 | 0.044 | 29 |

## T2 The same decisions in the ordered layout (d302, `memory_o`, progress anchor)

| set | decisions | first lead median | p90 | |lead| > 1 m | ahead | early turn | of these on a folded merge | AlpaSim msg lead > 1 m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0, held 460 @18 | 8129 | -0.00 | 0.00 | 58 | 30 | 30 | 30 | 27 |
| RTB1ARO-L18-s0 (V2), held 460 @18 | 8129 | -0.00 | 0.00 | 64 | 28 | 28 | 28 | 29 |
| RSG-L18-s0, public 1491 @10 | 14790 | -0.00 | 0.00 | 48 | 13 | 13 | 13 | 26 |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | 14790 | -0.00 | 0.00 | 47 | 12 | 12 | 12 | 24 |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | 18352 | -0.00 | 0.00 | 150 | 29 | 30 | 29 | 67 |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | 14790 | -0.00 | 0.00 | 44 | 11 | 11 | 11 | 24 |

## T3 Zeros against early-turned memory

with early = an early-turn decision before the failure (any decision for a rollout without one); early-before = within 3 s before the failure; pulled = plan and failure on the memory's side (module docstring).

ahead-before (looser) = a decision with a near slot >= 10 m ahead within 3 s before the failure; side = that decision's pull, the plan and the failure on one side. RR = zero rate with early / without, scene-bootstrap CI; '-' where a cell has no zero.

| set | geometry | rollouts | zeros | rollouts with early | zeros with early-before | pulled | zeros with ahead-before | side | zero rate with / without early | RR [CI] |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| RSG-L18-s0, held 460 @18 | all | 452 | 26 | 21 | 2 | 1 | 4 | 3 | 0.143 / 0.053 | 2.68 [0.00, 6.75] |
| RSG-L18-s0, held 460 @18 | bend | 81 | 2 | 0 | 0 | 0 | 0 | 0 | nan / 0.025 | - |
| RSG-L18-s0, held 460 @18 | straight | 169 | 4 | 0 | 0 | 0 | 0 | 0 | nan / 0.024 | - |
| RSG-L18-s0, held 460 @18 | turn | 190 | 18 | 16 | 1 | 1 | 3 | 3 | 0.125 / 0.092 | 1.36 [0.00, 4.08] |
| RSG-L18-s0, held 460 @18 | uturn | 12 | 2 | 5 | 1 | 0 | 1 | 0 | 0.200 / 0.143 | 1.40 [0.00, 99.00] |
| RTB1ARO-L18-s0 (V2), held 460 @18 | all | 452 | 31 | 19 | 1 | 0 | 4 | 3 | 0.105 / 0.067 | 1.57 [0.00, 4.36] |
| RTB1ARO-L18-s0 (V2), held 460 @18 | bend | 81 | 3 | 0 | 0 | 0 | 0 | 0 | nan / 0.037 | - |
| RTB1ARO-L18-s0 (V2), held 460 @18 | straight | 169 | 2 | 0 | 0 | 0 | 0 | 0 | nan / 0.012 | - |
| RTB1ARO-L18-s0 (V2), held 460 @18 | turn | 191 | 24 | 14 | 0 | 0 | 3 | 3 | 0.071 / 0.130 | 0.55 [0.00, 2.05] |
| RTB1ARO-L18-s0 (V2), held 460 @18 | uturn | 11 | 2 | 5 | 1 | 0 | 1 | 0 | 0.200 / 0.167 | 1.20 [0.00, 99.00] |
| RSG-L18-s0, public 1491 @10 | all | 1479 | 46 | 57 | 1 | 0 | 2 | 1 | 0.018 / 0.032 | 0.55 [0.00, 1.98] |
| RSG-L18-s0, public 1491 @10 | bend | 271 | 8 | 1 | 0 | 0 | 0 | 0 | 0.000 / 0.030 | - |
| RSG-L18-s0, public 1491 @10 | straight | 683 | 7 | 0 | 0 | 0 | 0 | 0 | nan / 0.010 | - |
| RSG-L18-s0, public 1491 @10 | turn | 497 | 29 | 42 | 0 | 0 | 1 | 1 | 0.000 / 0.064 | - |
| RSG-L18-s0, public 1491 @10 | uturn | 28 | 2 | 14 | 1 | 0 | 1 | 0 | 0.071 / 0.071 | 1.00 [0.00, 99.00] |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | all | 1479 | 53 | 57 | 2 | 1 | 5 | 3 | 0.035 / 0.036 | 0.98 [0.00, 2.65] |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | bend | 280 | 9 | 1 | 0 | 0 | 0 | 0 | 0.000 / 0.032 | - |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | straight | 678 | 3 | 0 | 0 | 0 | 0 | 0 | nan / 0.004 | - |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | turn | 492 | 39 | 41 | 0 | 0 | 3 | 2 | 0.000 / 0.086 | - |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | uturn | 29 | 2 | 15 | 2 | 1 | 2 | 1 | 0.133 / 0.000 | - |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | all | 1021 | 136 | 38 | 5 | 3 | 13 | 5 | 0.368 / 0.124 | 2.97 [1.73, 4.40] |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | bend | 189 | 15 | 0 | 0 | 0 | 0 | 0 | nan / 0.079 | - |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | straight | 425 | 21 | 0 | 0 | 0 | 0 | 0 | nan / 0.049 | - |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | turn | 374 | 86 | 22 | 2 | 2 | 5 | 2 | 0.318 / 0.224 | 1.42 [0.53, 2.44] |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | uturn | 33 | 14 | 16 | 3 | 1 | 8 | 3 | 0.438 / 0.412 | 1.06 [0.42, 2.71] |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | all | 1479 | 51 | 58 | 2 | 1 | 3 | 1 | 0.034 / 0.034 | 1.00 [0.00, 2.71] |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | bend | 278 | 7 | 1 | 0 | 0 | 0 | 0 | 0.000 / 0.025 | - |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | straight | 679 | 4 | 0 | 0 | 0 | 0 | 0 | nan / 0.006 | - |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | turn | 493 | 37 | 42 | 0 | 0 | 1 | 0 | 0.000 / 0.082 | - |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | uturn | 29 | 3 | 15 | 2 | 1 | 2 | 1 | 0.133 / 0.071 | 1.87 [0.00, 99.00] |

## T4 Scenes: U-turn rollouts and every zero with early memory before the failure

| scene | set | geometry | score | zero | wrong lane | early decisions (before) | max near lead m | ordered max near lead m | pulled |
|:--|:--|:--|--:|:--|:-:|--:|--:|--:|:-:|
| 00c48999 | a_pub | uturn (165 deg) | 1.00 | - | - | 0 (0) | 9.6 | -0.1 | - |
| 00c48999 | pool_rb0 | uturn (165 deg) | 1.00 | - | - | 0 (0) | 9.8 | -0.1 | - |
| 00c48999 | rsg_pub | uturn (165 deg) | 1.00 | - | - | 0 (0) | 9.8 | -inf | - |
| 00c48999 | v2_pub | uturn (165 deg) | 1.00 | - | - | 0 (0) | 9.7 | -0.1 | - |
| 00ff8eeb | pool_rb0 | uturn (181 deg) | 0.00 | off | - | 6 (0) | 82.5 | -0.3 | - |
| 02f75336 | pool_rb0 | uturn (177 deg) | 0.00 | cor | - | 0 (0) | -0.0 | 0.6 | - |
| 04daf401 | a_pub | uturn (145 deg) | 1.00 | - | - | 0 (0) | 33.3 | -inf | - |
| 04daf401 | pool_rb0 | uturn (179 deg) | 0.00 | cor | - | 12 (4) | 52.3 | -inf | - |
| 04daf401 | rsg_pub | uturn (140 deg) | 0.87 | - | - | 0 (0) | 33.3 | -inf | - |
| 04daf401 | v2_pub | uturn (140 deg) | 1.00 | - | - | 0 (0) | 33.3 | -inf | - |
| 0bd98f90 | pool_rb0 | uturn (167 deg) | 1.00 | - | - | 0 (0) | 0.8 | -0.0 | - |
| 10909749 | a_pub | uturn (180 deg) | 0.00 | coll | - | 0 (0) | -inf | -inf | - |
| 10909749 | rsg_pub | uturn (180 deg) | 0.00 | coll | - | 0 (0) | -inf | -inf | - |
| 10909749 | v2_pub | uturn (180 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| 2115a02d | a_pub | uturn (140 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| 2115a02d | pool_rb0 | uturn (180 deg) | 0.00 | coll | y | 0 (0) | 0.4 | -0.1 | - |
| 2115a02d | rsg_pub | uturn (136 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| 2115a02d | v2_pub | uturn (136 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| 257d737f | pool_rb0 | uturn (172 deg) | 0.00 | cor | y | 0 (0) | -inf | -inf | - |
| 30f09e00 | pool_rb0 | uturn (185 deg) | 0.00 | off | y | 0 (0) | 83.2 | -0.2 | - |
| 3b4031de | a_pub | uturn (208 deg) | 0.00 | off | y | 8 (5) | 57.1 | -inf | - |
| 3b4031de | rsg_held | uturn (210 deg) | 0.00 | off | y | 10 (3) | 57.1 | -inf | - |
| 3b4031de | rsg_pub | uturn (208 deg) | 0.00 | off | y | 7 (3) | 57.1 | -inf | - |
| 3b4031de | v2_held | uturn (210 deg) | 0.00 | off | y | 8 (4) | 57.1 | -inf | - |
| 3b4031de | v2_pub | uturn (208 deg) | 0.00 | off | y | 7 (4) | 57.1 | -inf | - |
| 3f2cec87 | pool_rb0 | uturn (185 deg) | 0.00 | off | y | 0 (0) | 84.8 | -0.3 | - |
| 40df1973 | rsg_held | uturn (181 deg) | 1.00 | - | - | 0 (0) | 35.3 | 63.2 | - |
| 40df1973 | v2_held | uturn (181 deg) | 1.00 | - | - | 0 (0) | 35.3 | 63.7 | - |
| 41da8ea7 | a_pub | uturn (198 deg) | 1.00 | - | - | 2 (0) | 65.9 | -inf | - |
| 41da8ea7 | rsg_held | uturn (204 deg) | 0.97 | - | - | 2 (0) | 65.9 | 64.2 | - |
| 41da8ea7 | rsg_pub | uturn (201 deg) | 1.00 | - | - | 2 (0) | 65.9 | -inf | - |
| 41da8ea7 | v2_held | uturn (204 deg) | 0.94 | - | - | 2 (0) | 65.9 | 65.3 | - |
| 41da8ea7 | v2_pub | uturn (198 deg) | 1.00 | - | - | 2 (0) | 65.9 | -inf | - |
| 42629dec | pool_rb0 | uturn (180 deg) | 1.00 | - | - | 18 (0) | 77.1 | 4.2 | - |
| 42c7526f | a_pub | uturn (180 deg) | 0.00 | off | y | 10 (6) | 82.4 | -inf | y |
| 42c7526f | rsg_pub | uturn (180 deg) | 1.00 | - | - | 10 (0) | 82.4 | -inf | - |
| 42c7526f | v2_pub | uturn (180 deg) | 1.00 | - | y | 10 (0) | 82.4 | -inf | - |
| 47359dd2 | a_pub | uturn (146 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| 47359dd2 | pool_rb0 | uturn (180 deg) | 1.00 | - | y | 0 (0) | 0.3 | -0.1 | - |
| 47359dd2 | rsg_pub | uturn (150 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| 47359dd2 | v2_pub | uturn (146 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| 4abb45d9 | a_pub | uturn (138 deg) | 1.00 | - | - | 0 (0) | 30.2 | -inf | - |
| 4abb45d9 | rsg_pub | uturn (138 deg) | 1.00 | - | - | 0 (0) | 30.2 | -inf | - |
| 4abb45d9 | v2_pub | uturn (138 deg) | 1.00 | - | - | 0 (0) | 30.2 | -inf | - |
| 4e0667aa | pool_rb0 | uturn (185 deg) | 0.00 | off | y | 0 (0) | 83.6 | -0.3 | - |
| 4fd76c8d | pool_rb0 | uturn (179 deg) | 1.00 | - | - | 0 (0) | 0.4 | 60.7 | - |
| 53d37873 | a_pub | uturn (181 deg) | 1.00 | - | - | 4 (0) | 75.8 | -inf | - |
| 53d37873 | rsg_held | uturn (210 deg) | 1.00 | - | - | 5 (0) | 75.8 | 64.3 | - |
| 53d37873 | rsg_pub | uturn (181 deg) | 1.00 | - | - | 5 (0) | 75.8 | -inf | - |
| 53d37873 | v2_held | uturn (210 deg) | 1.00 | - | - | 5 (0) | 75.8 | 64.4 | - |
| 53d37873 | v2_pub | uturn (181 deg) | 1.00 | - | - | 5 (0) | 75.8 | -inf | - |
| 56b80cb2 | a_pub | uturn (164 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| 56b80cb2 | pool_rb0 | uturn (173 deg) | 0.98 | - | y | 0 (0) | 0.8 | -0.1 | - |
| 56b80cb2 | rsg_pub | uturn (157 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| 56b80cb2 | v2_pub | uturn (164 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| 5d478633 | a_pub | uturn (155 deg) | 1.00 | - | - | 5 (0) | 19.8 | -0.0 | - |
| 5d478633 | pool_rb0 | uturn (157 deg) | 0.00 | cor | - | 4 (2) | 19.8 | -0.0 | y |
| 5d478633 | rsg_pub | uturn (156 deg) | 1.00 | - | - | 4 (0) | 19.8 | -0.0 | - |
| 5d478633 | v2_pub | uturn (155 deg) | 1.00 | - | - | 5 (0) | 19.8 | -0.0 | - |
| 5e9e8c31 | pool_rb0 | uturn (180 deg) | 1.00 | - | y | 4 (0) | 82.7 | 46.8 | - |
| 653ebb37 | pool_rb0 | uturn (180 deg) | 1.00 | - | - | 9 (0) | 81.0 | 4.5 | - |
| 657ea528 | a_pub | uturn (152 deg) | 1.00 | - | y | 3 (0) | 29.3 | -0.1 | - |
| 657ea528 | pool_rb0 | uturn (181 deg) | 1.00 | - | y | 3 (0) | 29.3 | -0.1 | - |
| 657ea528 | rsg_pub | uturn (141 deg) | 1.00 | - | - | 3 (0) | 29.3 | -0.1 | - |
| 657ea528 | v2_pub | uturn (146 deg) | 1.00 | - | y | 3 (0) | 29.3 | -0.1 | - |
| 661ce644 | rsg_held | uturn (158 deg) | 0.84 | - | - | 8 (0) | 36.0 | -0.0 | - |
| 661ce644 | v2_held | uturn (181 deg) | 1.00 | - | y | 7 (0) | 36.0 | -0.0 | - |
| 6a85f61b | pool_rb0 | uturn (143 deg) | 1.00 | - | - | 0 (0) | 3.4 | -0.0 | - |
| 6d2178f2 | pool_rb0 | turn (94 deg) | 0.00 | off | y | 3 (3) | 10.3 | 0.0 | y |
| 708d081d | a_pub | uturn (180 deg) | 1.00 | - | - | 5 (0) | 53.5 | -inf | - |
| 708d081d | rsg_pub | uturn (180 deg) | 1.00 | - | - | 5 (0) | 53.5 | -inf | - |
| 708d081d | v2_pub | uturn (180 deg) | 1.00 | - | - | 5 (0) | 53.5 | -inf | - |
| 77218354 | a_pub | uturn (174 deg) | 0.97 | - | - | 4 (0) | 67.4 | -inf | - |
| 77218354 | pool_rb0 | uturn (205 deg) | 0.86 | - | - | 3 (0) | 67.4 | -inf | - |
| 77218354 | rsg_pub | uturn (178 deg) | 1.00 | - | - | 3 (0) | 67.4 | -inf | - |
| 77218354 | v2_pub | uturn (178 deg) | 1.00 | - | - | 3 (0) | 67.4 | -inf | - |
| 7be1e4f2 | pool_rb0 | uturn (180 deg) | 1.00 | - | - | 9 (0) | 47.7 | 4.2 | - |
| 84ec7e45 | a_pub | uturn (154 deg) | 0.95 | - | y | 0 (0) | 22.2 | -inf | - |
| 84ec7e45 | pool_rb0 | uturn (165 deg) | 1.00 | - | - | 0 (0) | 22.2 | -inf | - |
| 84ec7e45 | rsg_pub | uturn (154 deg) | 0.95 | - | - | 0 (0) | 22.2 | -inf | - |
| 84ec7e45 | v2_pub | uturn (154 deg) | 0.93 | - | - | 0 (0) | 22.2 | -inf | - |
| 8898650e | pool_rb0 | uturn (187 deg) | 1.00 | - | - | 0 (0) | 34.5 | -inf | - |
| 975c802f | a_pub | uturn (186 deg) | 1.00 | - | - | 0 (0) | 0.1 | -0.2 | - |
| 975c802f | rsg_pub | uturn (186 deg) | 1.00 | - | - | 0 (0) | 0.2 | -0.2 | - |
| 975c802f | v2_pub | uturn (186 deg) | 1.00 | - | - | 0 (0) | 0.1 | -0.2 | - |
| 9b1bea0c | rsg_held | uturn (142 deg) | 0.87 | - | - | 0 (0) | 0.3 | -0.1 | - |
| 9b1bea0c | v2_held | uturn (142 deg) | 0.86 | - | - | 0 (0) | 0.3 | -0.1 | - |
| a454e18c | a_pub | uturn (177 deg) | 1.00 | - | y | 5 (0) | 45.3 | -inf | - |
| a454e18c | pool_rb0 | uturn (177 deg) | 0.00 | cor | - | 5 (0) | 45.3 | -inf | - |
| a454e18c | rsg_pub | uturn (177 deg) | 1.00 | - | - | 5 (0) | 45.3 | -inf | - |
| a454e18c | v2_pub | uturn (177 deg) | 0.00 | off | y | 5 (2) | 45.3 | -inf | y |
| a567ea67 | pool_rb0 | turn (92 deg) | 0.00 | cor | y | 8 (2) | 31.5 | -0.0 | y |
| b762ea96 | a_pub | uturn (180 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| b762ea96 | rsg_pub | uturn (180 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| b762ea96 | v2_pub | uturn (180 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| bf896d50 | a_pub | uturn (168 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| bf896d50 | rsg_held | uturn (168 deg) | 1.00 | - | - | 0 (0) | 0.0 | -0.0 | - |
| bf896d50 | rsg_pub | uturn (168 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| bf896d50 | v2_held | uturn (168 deg) | 1.00 | - | - | 0 (0) | 0.0 | -0.0 | - |
| bf896d50 | v2_pub | uturn (168 deg) | 1.00 | - | - | 0 (0) | -inf | -inf | - |
| c2ae8e85 | rsg_held | uturn (211 deg) | 0.91 | - | y | 0 (0) | 3.5 | -0.2 | - |
| c2ae8e85 | v2_held | uturn (211 deg) | 1.00 | - | y | 0 (0) | 3.4 | -0.2 | - |
| cbbf59be | pool_rb0 | uturn (182 deg) | 0.73 | - | - | 18 (0) | 80.6 | 4.2 | - |
| ced5e9f4 | a_pub | uturn (185 deg) | 1.00 | - | - | 1 (0) | 40.0 | -inf | - |
| ced5e9f4 | pool_rb0 | uturn (185 deg) | 1.00 | - | y | 1 (0) | 40.1 | 63.6 | - |
| ced5e9f4 | rsg_pub | uturn (185 deg) | 1.00 | - | - | 1 (0) | 40.0 | -inf | - |
| ced5e9f4 | v2_pub | uturn (185 deg) | 1.00 | - | - | 1 (0) | 40.0 | -inf | - |
| d50b69be | pool_rb0 | uturn (138 deg) | 0.78 | - | - | 3 (0) | 26.4 | -0.0 | - |
| d5ece623 | a_pub | uturn (180 deg) | 0.70 | - | y | 0 (0) | 29.7 | -inf | - |
| d5ece623 | rsg_pub | uturn (180 deg) | 0.35 | - | y | 0 (0) | 29.7 | -inf | - |
| d5ece623 | v2_pub | uturn (180 deg) | 0.64 | - | y | 0 (0) | 29.7 | -inf | - |
| d743862c | a_pub | uturn (150 deg) | 1.00 | - | y | 5 (0) | 33.0 | -0.0 | - |
| d743862c | pool_rb0 | uturn (181 deg) | 0.00 | off | y | 5 (0) | 33.0 | -0.0 | - |
| d743862c | v2_pub | uturn (150 deg) | 1.00 | - | y | 5 (0) | 33.0 | -0.0 | - |
| d88fdf29 | pool_rb0 | uturn (138 deg) | 1.00 | - | - | 0 (0) | 10.3 | -0.0 | - |
| daf38cec | a_pub | uturn (188 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| daf38cec | rsg_held | uturn (188 deg) | 0.00 | coll | y | 0 (0) | 0.0 | -0.0 | - |
| daf38cec | rsg_pub | uturn (188 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| daf38cec | v2_held | uturn (188 deg) | 0.00 | coll | y | 0 (0) | 0.0 | 0.0 | - |
| daf38cec | v2_pub | uturn (188 deg) | 1.00 | - | y | 0 (0) | -inf | -inf | - |
| e051405f | a_pub | uturn (140 deg) | 1.00 | - | - | 6 (0) | 44.7 | -inf | - |
| e051405f | pool_rb0 | uturn (141 deg) | 0.00 | cor | - | 8 (0) | 44.7 | -inf | - |
| e051405f | rsg_pub | uturn (140 deg) | 1.00 | - | - | 8 (0) | 44.7 | -inf | - |
| e051405f | v2_pub | uturn (140 deg) | 1.00 | - | - | 6 (0) | 44.7 | -inf | - |
| e6056b57 | rsg_held | turn (89 deg) | 0.00 | cor | y | 1 (1) | 26.2 | 0.0 | y |
| e7295df6 | a_pub | uturn (180 deg) | 1.00 | - | - | 2 (0) | 38.7 | -inf | - |
| e7295df6 | pool_rb0 | uturn (264 deg) | 0.00 | cor | - | 11 (1) | 70.9 | 0.0 | - |
| e7295df6 | rsg_pub | uturn (180 deg) | 1.00 | - | - | 2 (0) | 38.7 | -inf | - |
| e7295df6 | v2_pub | uturn (180 deg) | 1.00 | - | - | 2 (0) | 38.7 | -inf | - |
| ebfb953d | a_pub | uturn (188 deg) | 1.00 | - | - | 2 (0) | 57.6 | -inf | - |
| ebfb953d | pool_rb0 | uturn (188 deg) | 0.00 | cor | - | 2 (0) | 57.6 | 66.6 | - |
| ebfb953d | rsg_pub | uturn (186 deg) | 1.00 | - | - | 2 (0) | 57.6 | -inf | - |
| ebfb953d | v2_pub | uturn (188 deg) | 1.00 | - | - | 2 (0) | 57.6 | -inf | - |
| f2592c08 | a_pub | uturn (177 deg) | 1.00 | - | y | 1 (0) | 25.2 | -0.1 | - |
| f2592c08 | rsg_held | uturn (181 deg) | 1.00 | - | y | 1 (0) | 25.2 | -0.0 | - |
| f2592c08 | rsg_pub | uturn (163 deg) | 1.00 | - | y | 1 (0) | 25.2 | -0.1 | - |
| f2592c08 | v2_held | uturn (181 deg) | 1.00 | - | y | 1 (0) | 25.2 | -0.0 | - |
| f2592c08 | v2_pub | uturn (173 deg) | 1.00 | - | y | 1 (0) | 25.2 | -0.1 | - |
| f976da77 | pool_rb0 | uturn (185 deg) | 1.00 | - | y | 0 (0) | 85.1 | -0.1 | - |
| fc0a60a9 | a_pub | uturn (168 deg) | 1.00 | - | - | 0 (0) | 6.2 | -0.1 | - |
| fc0a60a9 | rsg_held | uturn (169 deg) | 1.00 | - | - | 0 (0) | 6.3 | -0.1 | - |
| fc0a60a9 | rsg_pub | uturn (167 deg) | 1.00 | - | - | 0 (0) | 6.3 | -0.1 | - |
| fc0a60a9 | v2_held | uturn (169 deg) | 1.00 | - | - | 0 (0) | 6.2 | -0.0 | - |
| fc0a60a9 | v2_pub | uturn (168 deg) | 1.00 | - | - | 0 (0) | 6.2 | -0.1 | - |
| fe4e5fdf | rsg_held | uturn (138 deg) | 1.00 | - | y | 0 (0) | 29.0 | 63.5 | - |

## T5 Training rows against serving (same rule, same truth)

NAVSIM rows rebuilt as rm_build nav builds them (store = share of rows within 0.05 m of the stored float16 feature row); pool = the on-policy rows of on1_L18RB0 (RB0-F-s0's rollouts, rm_build pool = the served Tracker). label turns = the logged 4 s yaw change >= 30 deg on the early rows.

| rows | n | rebuilt = stored | near slots valid | first lead median | p90 | >= 10 m | ahead | early turn | fold | early rows whose label turns |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NAVSIM navtrain | 101313 | 0.9939 | 0.430 | 0.0 | 9.9 | 0.098 | 0.0436 | 0.0185 (1872) | 0.0133 | 247 |
| RSG-L18-s0, held 460 @18 | 8129 | nan | 0.505 | 0.0 | 7.8 | 0.075 | 0.0415 | 0.0192 (156) | 0.0039 | - |
| RTB1ARO-L18-s0 (V2), held 460 @18 | 8129 | nan | 0.514 | 0.0 | 7.8 | 0.075 | 0.0408 | 0.0177 (144) | 0.0038 | - |
| RSG-L18-s0, public 1491 @10 | 14790 | nan | 0.247 | 0.0 | 10.3 | 0.103 | 0.0412 | 0.0191 (283) | 0.0011 | - |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | 14790 | nan | 0.254 | 0.0 | 10.2 | 0.102 | 0.0409 | 0.0194 (287) | 0.0011 | - |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | 18352 | nan | 0.476 | 0.0 | 6.5 | 0.063 | 0.0282 | 0.0134 (245) | 0.0021 | - |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | 14790 | nan | 0.254 | 0.0 | 10.2 | 0.103 | 0.0410 | 0.0194 (287) | 0.0011 | - |

## T6 Open-loop replay: the same logged inputs with the memory served, re-anchored (ordered layout's slots) or empty

At each target's decisions that were `ahead` in the 3 s before its failure: the plan's lateral from the route at s_e + 10 m (m) and its 2 s heading error (deg), both signed toward the side of the misplaced memory (+ = toward it). Identity: own checkpoint, served memory, 4 s plan point against the stored plan.

| target | checkpoint | decisions | served p10 | ordered p10 | empty p10 | served dpsi | ordered dpsi | empty dpsi |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| rsg_held f9ed38d9 | RSG-L18-s0 (own) | 2 | +0.52 | +0.52 | +0.48 | -5.3 | -5.2 | -6.0 |
| rsg_held f9ed38d9 | RTB1ARO-L18-s0 | 2 | +0.66 | +0.66 | +0.50 | -4.6 | -4.6 | -5.5 |
| rsg_held f9ed38d9 | RTB1O-L18-s0 | 2 | +0.57 | +0.58 | +0.43 | -5.1 | -5.0 | -5.7 |
| rsg_held e6056b57 | RSG-L18-s0 (own) | 4 | +4.59 | +4.58 | +4.66 | +2.9 | +2.5 | +6.0 |
| rsg_held e6056b57 | RTB1ARO-L18-s0 | 4 | +4.64 | +4.63 | +4.67 | +3.7 | +3.4 | +6.0 |
| rsg_held e6056b57 | RTB1O-L18-s0 | 4 | +4.66 | +4.64 | +4.68 | +3.5 | +3.2 | +6.2 |
| rsg_held 5ddfd1fe | RSG-L18-s0 (own) | 7 | +4.04 | +4.01 | +4.07 | +12.4 | +12.1 | +13.1 |
| rsg_held 5ddfd1fe | RTB1ARO-L18-s0 | 7 | +3.99 | +3.96 | +4.05 | +12.6 | +12.2 | +13.5 |
| rsg_held 5ddfd1fe | RTB1O-L18-s0 | 7 | +3.97 | +3.94 | +4.06 | +12.1 | +11.7 | +13.2 |
| rsg_held 3b4031de | RSG-L18-s0 (own) | 7 | -2.90 | -2.96 | -2.88 | -7.8 | -8.2 | -8.5 |
| rsg_held 3b4031de | RTB1ARO-L18-s0 | 7 | -3.14 | -3.08 | -2.95 | -9.2 | -8.9 | -8.6 |
| rsg_held 3b4031de | RTB1O-L18-s0 | 7 | -3.10 | -3.14 | -3.01 | -8.5 | -8.8 | -8.8 |
| v2_held f9ed38d9 | RSG-L18-s0 | 2 | +5.79 | +5.72 | +6.00 | +28.6 | +28.3 | +30.2 |
| v2_held f9ed38d9 | RTB1ARO-L18-s0 (own) | 2 | +5.48 | +5.40 | +5.68 | +27.9 | +27.6 | +29.5 |
| v2_held f9ed38d9 | RTB1O-L18-s0 | 2 | +5.73 | +5.69 | +5.97 | +28.9 | +28.8 | +30.9 |
| v2_held 5ddfd1fe | RSG-L18-s0 | 6 | +3.06 | +3.02 | +3.00 | +9.0 | +8.5 | +8.9 |
| v2_held 5ddfd1fe | RTB1ARO-L18-s0 (own) | 6 | +2.99 | +2.95 | +2.98 | +9.1 | +8.5 | +9.0 |
| v2_held 5ddfd1fe | RTB1O-L18-s0 | 6 | +2.99 | +2.95 | +2.99 | +8.9 | +8.3 | +9.1 |
| v2_held 3b4031de | RSG-L18-s0 | 7 | -3.71 | -3.79 | -3.99 | -16.1 | -16.5 | -18.1 |
| v2_held 3b4031de | RTB1ARO-L18-s0 (own) | 7 | -3.98 | -3.87 | -4.03 | -17.6 | -17.2 | -18.1 |
| v2_held 3b4031de | RTB1O-L18-s0 | 7 | -4.07 | -4.13 | -4.16 | -17.2 | -17.5 | -18.2 |
| v2_held 67049536 | RSG-L18-s0 | 4 | +4.57 | +4.56 | +4.64 | +2.4 | +2.3 | +6.0 |
| v2_held 67049536 | RTB1ARO-L18-s0 (own) | 4 | +4.58 | +4.58 | +4.66 | +2.9 | +2.7 | +5.0 |
| v2_held 67049536 | RTB1O-L18-s0 | 4 | +4.58 | +4.58 | +4.64 | +3.1 | +3.1 | +5.4 |
| rsg_pub 3b4031de | RSG-L18-s0 (own) | 6 | -2.88 | -2.96 | -2.88 | -8.1 | -8.6 | -9.2 |
| rsg_pub 3b4031de | RTB1ARO-L18-s0 | 6 | -3.17 | -3.09 | -3.00 | -9.8 | -9.4 | -9.5 |
| rsg_pub 3b4031de | RTB1O-L18-s0 | 6 | -2.93 | -2.97 | -2.89 | -9.0 | -9.2 | -9.5 |
| rsg_pub f9ed38d9 | RSG-L18-s0 (own) | 2 | +0.60 | +0.59 | +0.55 | -4.7 | -4.7 | -5.4 |
| rsg_pub f9ed38d9 | RTB1ARO-L18-s0 | 2 | +0.74 | +0.74 | +0.56 | -3.9 | -3.9 | -5.0 |
| rsg_pub f9ed38d9 | RTB1O-L18-s0 | 2 | +0.64 | +0.66 | +0.50 | -4.5 | -4.4 | -5.1 |
| v2_pub 67049536 | RSG-L18-s0 | 4 | +4.55 | +4.53 | +4.62 | +2.5 | +2.4 | +6.0 |
| v2_pub 67049536 | RTB1ARO-L18-s0 (own) | 4 | +4.57 | +4.57 | +4.65 | +3.1 | +2.9 | +5.2 |
| v2_pub 67049536 | RTB1O-L18-s0 | 4 | +4.56 | +4.57 | +4.64 | +3.2 | +3.2 | +5.6 |
| v2_pub 3b4031de | RSG-L18-s0 | 7 | -3.74 | -3.81 | -4.03 | -16.3 | -16.6 | -18.1 |
| v2_pub 3b4031de | RTB1ARO-L18-s0 (own) | 7 | -4.00 | -3.89 | -4.06 | -17.6 | -17.2 | -18.1 |
| v2_pub 3b4031de | RTB1O-L18-s0 | 7 | -4.11 | -4.16 | -4.19 | -17.3 | -17.6 | -18.3 |
| v2_pub f9ed38d9 | RSG-L18-s0 | 2 | +5.86 | +5.80 | +6.10 | +29.3 | +29.0 | +30.9 |
| v2_pub f9ed38d9 | RTB1ARO-L18-s0 (own) | 2 | +5.58 | +5.50 | +5.78 | +28.7 | +28.4 | +30.2 |
| v2_pub f9ed38d9 | RTB1O-L18-s0 | 2 | +5.81 | +5.78 | +6.05 | +29.6 | +29.5 | +31.6 |
| v2_pub a454e18c | RSG-L18-s0 | 7 | +1.63 | +1.51 | +1.96 | +10.9 | +10.1 | +11.9 |
| v2_pub a454e18c | RTB1ARO-L18-s0 (own) | 7 | +2.19 | +2.14 | +2.16 | +11.8 | +11.5 | +11.3 |
| v2_pub a454e18c | RTB1O-L18-s0 | 7 | +2.02 | +1.88 | +2.16 | +12.0 | +10.8 | +11.4 |
| v2_pub 30f09e00 | RSG-L18-s0 | 1 | -1.15 | -1.18 | -1.23 | -14.4 | -14.9 | -15.6 |
| v2_pub 30f09e00 | RTB1ARO-L18-s0 (own) | 1 | -1.23 | -1.31 | -1.24 | -15.3 | -16.0 | -15.4 |
| v2_pub 30f09e00 | RTB1O-L18-s0 | 1 | -1.20 | -1.23 | -1.23 | -15.1 | -15.4 | -15.5 |
| a_pub 3b4031de | RSG-L18-s0 | 7 | -3.81 | -3.88 | -4.12 | -16.3 | -16.7 | -18.1 |
| a_pub 3b4031de | RTB1ARO-L18-s0 | 7 | -4.09 | -4.00 | -4.17 | -17.8 | -17.4 | -18.3 |
| a_pub 3b4031de | RTB1O-L18-s0 (own) | 7 | -4.22 | -4.29 | -4.33 | -17.5 | -17.8 | -18.5 |
| a_pub 42c7526f | RSG-L18-s0 | 6 | +1.83 | +1.14 | +2.28 | +13.3 | +10.1 | +15.0 |
| a_pub 42c7526f | RTB1ARO-L18-s0 | 6 | +1.85 | +1.09 | +2.21 | +13.7 | +10.2 | +16.1 |
| a_pub 42c7526f | RTB1O-L18-s0 (own) | 6 | +2.06 | +1.46 | +2.35 | +15.5 | +12.4 | +16.6 |
| a_pub 30f09e00 | RSG-L18-s0 | 1 | -1.14 | -1.16 | -1.22 | -14.5 | -14.9 | -15.6 |
| a_pub 30f09e00 | RTB1ARO-L18-s0 | 1 | -1.22 | -1.29 | -1.21 | -15.3 | -16.0 | -15.3 |
| a_pub 30f09e00 | RTB1O-L18-s0 (own) | 1 | -1.18 | -1.20 | -1.20 | -15.0 | -15.4 | -15.4 |
| ctl_pub 42c7526f | RSG-L18-s0 (own) | 10 | -0.39 | -0.50 | -0.51 | -3.2 | -4.1 | -4.4 |
| ctl_pub 42c7526f | RTB1ARO-L18-s0 | 10 | -0.33 | -0.42 | -0.39 | -2.4 | -3.3 | -3.3 |
| ctl_pub 42c7526f | RTB1O-L18-s0 | 10 | -0.22 | -0.29 | -0.29 | -2.2 | -2.6 | -2.9 |

Identity (own checkpoint, served): rsg_held__f9ed38d9 1.00 of 18 within 10 cm; rsg_held__e6056b57 1.00 of 18 within 10 cm; rsg_held__5ddfd1fe 1.00 of 18 within 10 cm; rsg_held__3b4031de 1.00 of 18 within 10 cm; rsg_pub__3b4031de 1.00 of 10 within 10 cm; rsg_pub__f9ed38d9 1.00 of 10 within 10 cm; ctl_pub__42c7526f 1.00 of 10 within 10 cm; v2_held__f9ed38d9 1.00 of 18 within 10 cm; v2_held__5ddfd1fe 1.00 of 18 within 10 cm; v2_held__3b4031de 1.00 of 18 within 10 cm; v2_held__67049536 1.00 of 18 within 10 cm; v2_pub__67049536 1.00 of 10 within 10 cm; v2_pub__3b4031de 1.00 of 10 within 10 cm; v2_pub__f9ed38d9 1.00 of 10 within 10 cm; v2_pub__a454e18c 1.00 of 10 within 10 cm; v2_pub__30f09e00 1.00 of 10 within 10 cm; a_pub__3b4031de 1.00 of 10 within 10 cm; a_pub__42c7526f 1.00 of 10 within 10 cm; a_pub__30f09e00 1.00 of 10 within 10 cm
