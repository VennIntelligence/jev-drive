# Repeats: spec vs spec_plan_smooth on HUGSIM-64

64 scenarios, turning 23, straight 41; arms P2-F-s0, P2-F-s1, P2H10-F-s0, P2H10-F-s1. Sources: bench /root/autodl-tmp/ujs/runs/bench/hugsim/P2-F-s0_spec; bench /root/autodl-tmp/ujs/runs/bench/hugsim/P2-F-s0_spec-rr; bench /root/autodl-tmp/ujs/runs/bench/hugsim/P2-F-s0_spec_pl; bench /root/autodl-tmp/ujs/runs/bench/hugsim/P2-F-s1_spec-rr; bench /root/autodl-tmp/ujs/runs/bench/hugsim/P2-F-s1_spec_pl; bench /root/autodl-tmp/ujs/runs/bench/hugsim/P2H10-F-s0_spec; bench /root/autodl-tmp/ujs/runs/bench/hugsim/P2H10-F-s1_spec; stored runs/op_parity/hugsim/results.csv

## Paired contrast smooth - spec (HD-Score; scenario = unit, arms and repeats averaged, bootstrap over scenarios; wins / losses / ties at |d| 0.02)

| pool | set | spec | smooth | smooth - spec [95% CI] | W / L / T |
|---|---|---|---|---|---|
| new repeats r1 + r2 | all64 | 0.395 | 0.429 | +0.034 [+0.003, +0.072] | 16 / 8 / 40 |
| new repeats r1 + r2 | turning23 | 0.279 | 0.334 | +0.055 [+0.006, +0.124] | 8 / 3 / 12 |
| new repeats r1 + r2 | straight41 | 0.459 | 0.482 | +0.022 [-0.013, +0.069] | 8 / 5 / 28 |
| all three r0 + r1 + r2 | all64 | 0.394 | 0.429 | +0.034 [+0.004, +0.072] | 16 / 8 / 40 |
| all three r0 + r1 + r2 | turning23 | 0.279 | 0.334 | +0.055 [+0.006, +0.122] | 8 / 3 / 12 |
| all three r0 + r1 + r2 | straight41 | 0.459 | 0.482 | +0.022 [-0.013, +0.070] | 8 / 5 / 28 |
| stored r0 only | all64 | 0.394 | 0.428 | +0.034 [+0.004, +0.071] | 16 / 8 / 40 |
| stored r0 only | turning23 | 0.279 | 0.333 | +0.054 [+0.007, +0.117] | 8 / 3 / 12 |
| stored r0 only | straight41 | 0.459 | 0.482 | +0.023 [-0.012, +0.070] | 8 / 5 / 28 |
| r1 only | all64 | 0.395 | 0.427 | +0.033 [+0.003, +0.069] | 16 / 8 / 40 |
| r1 only | turning23 | 0.279 | 0.331 | +0.052 [+0.006, +0.115] | 8 / 3 / 12 |
| r1 only | straight41 | 0.459 | 0.481 | +0.022 [-0.013, +0.069] | 8 / 5 / 28 |
| r2 only | all64 | 0.395 | 0.430 | +0.035 [+0.003, +0.075] | 16 / 8 / 40 |
| r2 only | turning23 | 0.279 | 0.338 | +0.059 [+0.006, +0.134] | 8 / 3 / 12 |
| r2 only | straight41 | 0.459 | 0.482 | +0.022 [-0.013, +0.070] | 8 / 5 / 28 |

## Per arm (3 repeats averaged)

| arm | set | spec | smooth | smooth - spec |
|---|---|---|---|---|
| P2-F-s0 | all64 | 0.380 | 0.422 | +0.042 [+0.011, +0.083] |
| P2-F-s0 | turning23 | 0.280 | 0.326 | +0.045 [+0.008, +0.090] |
| P2-F-s0 | straight41 | 0.436 | 0.476 | +0.041 [-0.004, +0.097] |
| P2-F-s1 | all64 | 0.407 | 0.428 | +0.021 [-0.011, +0.062] |
| P2-F-s1 | turning23 | 0.280 | 0.309 | +0.029 [-0.029, +0.110] |
| P2-F-s1 | straight41 | 0.479 | 0.495 | +0.016 [-0.015, +0.062] |
| P2H10-F-s0 | all64 | 0.388 | 0.418 | +0.030 [-0.020, +0.078] |
| P2H10-F-s0 | turning23 | 0.278 | 0.348 | +0.070 [+0.008, +0.151] |
| P2H10-F-s0 | straight41 | 0.450 | 0.457 | +0.007 [-0.058, +0.066] |
| P2H10-F-s1 | all64 | 0.403 | 0.446 | +0.043 [+0.011, +0.084] |
| P2H10-F-s1 | turning23 | 0.277 | 0.352 | +0.075 [+0.010, +0.161] |
| P2H10-F-s1 | straight41 | 0.473 | 0.498 | +0.026 [-0.002, +0.069] |

## End classes (mean count per arm x repeat, 12 runs per preset; cls = spin, else the end)

| preset | set | complete | fg_coll | bg_coll | off_route | stuck | spin |
|---|---|---|---|---|---|---|---|
| spec | all64 | 20.25 | 28.00 | 11.42 | 2.83 | 0.00 | 1.50 |
| spec | turning23 | 2.00 | 7.50 | 9.17 | 2.83 | 0.00 | 1.50 |
| spec | straight41 | 18.25 | 20.50 | 2.25 | 0.00 | 0.00 | 0.00 |
| spec_plan_smooth | all64 | 24.67 | 29.75 | 7.75 | 1.83 | 0.00 | 0.00 |
| spec_plan_smooth | turning23 | 5.33 | 10.25 | 6.00 | 1.42 | 0.00 | 0.00 |
| spec_plan_smooth | straight41 | 19.33 | 19.50 | 1.75 | 0.42 | 0.00 | 0.00 |

## Spins over all 64 (per repeat r0 / r1 / r2)

| preset | arm | r0 | r1 | r2 |
|---|---|---|---|---|
| spec | P2-F-s0 | 1 | 1 | 1 |
| spec | P2-F-s1 | 1 | 1 | 1 |
| spec | P2H10-F-s0 | 2 | 2 | 2 |
| spec | P2H10-F-s1 | 2 | 2 | 2 |
| spec_plan_smooth | P2-F-s0 | 0 | 0 | 0 |
| spec_plan_smooth | P2-F-s1 | 0 | 0 | 0 |
| spec_plan_smooth | P2H10-F-s0 | 0 | 0 | 0 |
| spec_plan_smooth | P2H10-F-s1 | 0 | 0 | 0 |

## Between-repeat noise per scenario (3 repeats of the same arm and preset)

| preset | arm | HD r0 / r1 / r2 | sd mean | sd median | sd max | max range | identical HD | same end | range > 0.05 | range > 0.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| spec | P2-F-s0 | 0.379 / 0.380 / 0.380 | 0.001 | 0.000 | 0.010 | 0.018 | 48 / 64 | 64 / 64 | 0 | 0 |
| spec | P2-F-s1 | 0.407 / 0.408 / 0.408 | 0.001 | 0.000 | 0.008 | 0.015 | 43 / 64 | 63 / 64 | 0 | 0 |
| spec | P2H10-F-s0 | 0.388 / 0.388 / 0.388 | 0.001 | 0.000 | 0.014 | 0.025 | 48 / 64 | 64 / 64 | 0 | 0 |
| spec | P2H10-F-s1 | 0.403 / 0.402 / 0.403 | 0.001 | 0.000 | 0.011 | 0.020 | 41 / 64 | 64 / 64 | 0 | 0 |
| spec_plan_smooth | P2-F-s0 | 0.419 / 0.419 / 0.428 | 0.006 | 0.000 | 0.330 | 0.578 | 49 / 64 | 63 / 64 | 1 | 1 |
| spec_plan_smooth | P2-F-s1 | 0.428 / 0.428 / 0.429 | 0.001 | 0.000 | 0.019 | 0.033 | 48 / 64 | 63 / 64 | 0 | 0 |
| spec_plan_smooth | P2H10-F-s0 | 0.418 / 0.417 / 0.418 | 0.001 | 0.000 | 0.014 | 0.026 | 43 / 64 | 62 / 64 | 0 | 0 |
| spec_plan_smooth | P2H10-F-s1 | 0.448 / 0.444 / 0.445 | 0.002 | 0.000 | 0.038 | 0.068 | 44 / 64 | 64 / 64 | 2 | 0 |

Per-repeat contrast (smooth - spec, 4-arm mean, one repeat at a time): stored r0 only all64 +0.034; stored r0 only turning23 +0.054; r1 only all64 +0.033; r1 only turning23 +0.052; r2 only all64 +0.035; r2 only turning23 +0.059.

Noise of the 4-arm scenario value (the unit of the bootstrap): spec: scenario-level sd of the 4-arm mean over repeats mean 0.001, max 0.004; spec_plan_smooth: scenario-level sd of the 4-arm mean over repeats mean 0.002, max 0.082.

Largest scenario contrasts (3 repeats, 4 arms): worst scene-090-hard-01 -0.24, scene-2800_3000-easy-00 -0.14, scene-8440_8640-easy-00 -0.07, scene-322492347634-easy-00 -0.06, scene-3000_3200-medium-00 -0.05; best scene-0013-medium-00 +0.75, scene-152217047339-medium-00 +0.64, scene-095-medium-01 +0.29, scene-164701907483-easy-00 +0.27, scene-053-medium-02 +0.23.
