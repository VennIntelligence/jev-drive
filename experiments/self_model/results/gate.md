# SELF-FWD offline gate: readout (2026-10-18 23:50 box time)

Prereg: [../plans/2026-10-19-fwd-gate-prereg.md](../plans/2026-10-19-fwd-gate-prereg.md) (committed 8ea7e6de before any output). Code
`scripts/sm_gate.py` (build: pool job 1018-233959-1c61; gate: 1018-233959-bd98). Full numbers: [gate.json](gate.json).

Data: decision k <= 9 of every rollout; executed = the driver anchors of decisions k+1..k+8 in decision k's frame. Train 45 512 rows / 24 log
units (RB0, RTB1, YS, Y pools, onpol-train), val 5 249 / 4 logs (crc32 % 7). T1 = RSG-L18-s0 held rollouts (4 461 rows, 16 held logs, rasters);
T2 = JTN-L18-s0 held 460 default-serve reads (4 513 decisions, 16 logs). Pre-check: executed vs plan at 0.5 s median 0.04-0.06 m (line 0.10): met.

## Gate (a): forward-model accuracy, NOT MET

| set | plan 4 s end median (m) | ridge | MLP | MLP / plan | paired mean gain (log CI) | lon / lat abs median, MLP (plan) |
|---|--:|--:|--:|--:|--|--|
| T1 RSG held | 1.708 | 1.42 | **1.144** | 0.67 | +0.75 [+0.63, +0.91] | 1.01 / 0.17 (1.61 / 0.17) |
| T2 JTN held | 1.714 | 1.55 | **1.337** | 0.78 | +0.43 [+0.33, +0.57] | 1.24 / 0.18 (1.62 / 0.17) |
| val logs | 2.056 | 1.56 | 1.337 | 0.65 | +0.83 [+0.53, +1.05] | 1.17 / 0.21 (1.90 / 0.24) |

Line: median < 0.20 m and <= 0.5 x plan. Missed by 6x. Error by horizon (T1, plan -> MLP, median m): 0.5 s 0.04 -> 0.02, 1 s 0.06 -> 0.02,
2 s 0.28 -> 0.15, 3 s 0.83 -> 0.52, 4 s 1.71 -> 1.14. The controller follows the plan to 2-4 cm for 1 s; what grows after that is the next
replans, and it is longitudinal (lateral 0.17 m at 4 s already on the plan itself). The plan runs ahead of the car (mean +0.84 m along heading
on T1); the MLP removes that bias on T1 (-0.05) but over-corrects on JTN (-0.51): the forward model is policy-specific.
Strata (T1 MLP 4 s median): straight 1.04, turn 1.30; v<1 1.28, 1-5 1.08, 5-10 1.11, >=10 1.27 m. No stratum near 0.2 m.

## Gate (b): consequence judgement, MET

T1, events within 4 s judged on the executed sweep (same footprint code as the S hinges; road raster corners < 0 or AgentHinge distance < 0);
score = -min clearance on the plan vs on the MLP's predicted executed poses. Log-clustered bootstrap over 16 held logs, B = 10 000.

| event | rows | events | AUC plan | AUC predicted | gain [95 % CI] | ridge AUC | TPR / FPR at 0, plan -> pred |
|---|--:|--:|--:|--:|--|--:|--|
| **any (primary)** | 4 443 | 188 | 0.859 | **0.939** | **+0.080 [+0.052, +0.121]** | 0.890 | 0.69 / 0.024 -> 0.84 / 0.031 |
| contact | 4 461 | 81 | 0.863 | 0.962 | +0.100 [+0.055, +0.181] | 0.923 | 0.65 / 0.008 -> 0.88 / 0.010 |
| offroad | 4 443 | 115 | 0.940 | 0.962 | +0.022 [+0.003, +0.038] | 0.940 | 0.72 / 0.018 -> 0.83 / 0.023 |

Strata (any, AUC plan -> pred): straight 0.81 -> 0.93, turn 0.87 -> 0.94, mild 0.88 -> 0.88; v5-10 0.80 -> 0.96, v1-5 0.88 -> 0.92.
Val logs (4 train-pool sources, no CI): any +0.03 to +0.06, contact +0.05 to +0.11, offroad +0.00 to +0.02, same sign everywhere.

## Reading

- Prereg branch "(a) fails, (b) passes": the mechanism has room (the plan misjudges contact, chiefly longitudinal: the car does not end
  where the plan says), but this forward model is not accurate enough to be written into training. No training design is made.
- The gain is contact-led and longitudinal; offroad gains little, consistent with lateral plan-vs-execution being only 0.17 m at 4 s.
  So d308's lateral under-turn is not what this mechanism would feed; the 4 s lateral sweep is already close to the plan's.
- The 0.2 m line is not reachable by any model of this shape: past 1 s the executed path is the next replans' answer to the world,
  not the controller's. A self model that is "body only" is the 0.5-1 s part, and there plan = execution to 2-4 cm (nothing to learn).
- Caveat: events are labelled with the same 2 Hz judge on the executed anchors, so any model closer to the executed poses gains AUC by
  construction; the readout says how much of the judge's error the predictable part removes, not that the simulator's own events agree.
