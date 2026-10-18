# JTN / JTB / JT00 / JT10 on the 24 held turns vs RSG-L18-s0 (2026-10-18)

Instrument: `experiments/route_follow/scripts/rf_follow.py`, unchanged (decision 286 / 302 / 312 turn read: route turn >= 60 deg in 30 m, completed =
crosses exit + 10 m before any zero event and <= 2.5 m from the route). Turn list = the 24 strict turns of RSG-L18-s0's stored read
(`report/follow_a3/scenes.csv`), passed with `--scenes`. Every label finds the same 24 turns. Turn shape: `rf_tax.py extract/report` + `rf_ledger.py
report` (geometry on the human's path). Open-loop equal-arc read: `nx_turns_report.dev_rows` (explore_g section 1a) on `nx.py hist` plans. G-rely:
the trainer's own `route_mem/rely/<tag>.json`.

**Serve.** Every read is a held 460 @18 run with no serve switches: no `JEV_*` in any run config, and no hold. On nuPlan, `JEV_TRACK` defaults to off;
the gate default from d309 is set only in `pai_driver.py`. RSG's stored 10-13 read gives its reference numbers again exactly: 16 / 1 / 0 / 7, 66.7 % /
65.2 %. RSG's 10-16 re-read at a later code revision matched it on all 460 scenes (`zero_overlap.md`). So RSG was not re-read.

## Completion (24 turns)

| label | all [95 % CI] | memory not covered (23) | left 16 / right 8 | completed / event / off / short | pair vs RSG (pp) [CI] | fixed / broken vs RSG |
|:--|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0 | 66.7 % [45, 86] | 65.2 % | 68.8 / 62.5 % | 16 / 1 / 0 / 7 | | |
| **JTN-L18-s0** | **83.3 % [72, 95]** | **82.6 %** | 81.2 / 87.5 % | 20 / 1 / 2 / 1 | +16.7 [+0.0, +42.3] | **4 / 0** |
| JTB-L18-s0 | 87.5 % [75, 100] | 87.0 % | 87.5 / 87.5 % | 21 / 1 / 1 / 1 | +20.8 [+0.0, +45.0] | 5 / 0 |
| JT00-L18-s0 | 83.3 % [68, 96] | 82.6 % | 81.2 / 87.5 % | 20 / 2 / 0 / 2 | +16.7 [+0.0, +40.0] | 5 / 1 |
| JT10-L18-s0 | 75.0 % [57, 89] | 73.9 % | 68.8 / 87.5 % | 18 / 5 / 0 / 1 | +8.3 [-18.2, +34.8] | 5 / 3 |

References (decisions 286 / 302): RBSG2 70.8-75 %, YS-L18 54.2 %. Junction |lateral to route| median / p90: RSG 0.52 / 1.90 m, JTN 0.49 / 1.67 m.

## Per turn (RSG vs JTN; C completed, E zero event, O crossed > 2.5 m off the route, S short = not there by decision 17)

Failed turns show the rf_tax primary class and the scene score. Entry = the last decision before the entry station. JTB / JT00 / JT10 show outcome only.

| turn | side / deg | RSG | JTN | JTN vs RSG | JTB | JT00 | JT10 | entry speed m/s (RSG / JTN) | entry lateral m, + inside (RSG / JTN) | peak outside m (RSG / JTN) | lateral at exit + 10 m (RSG / JTN) |
|:--|:--|:--|:--|:--|:-:|:-:|:-:|--:|--:|--:|--:|
| 430589fe | L 86 | S ok 0.97 | C | fixed | C | C | C | 4.5 / 4.5 | +0.27 / +0.27 | 0.08 / 0.39 | - / 1.09 |
| aec02ec2 | L 73 | S low_slow_free 0.86 | C | fixed | C | S | E | 2.9 / 3.0 | -0.50 / -0.49 | 2.32 / 1.90 | - / 0.62 |
| b958e696 | R 92 | S low_other 0.89 | C | fixed | C | C | C | 2.9 / 2.9 | +0.55 / +0.54 | 2.52 / 2.09 | - / 1.17 |
| ffe92084 | R 79 | S ok 1.00 | C | fixed | C | C | C | 3.8 / 5.0 | -0.23 / +0.07 | 2.02 / 0.45 | - / 0.49 |
| 7ae0a03b | L 74 | S ok 1.00 | O ok 1.00 | both fail | O | C | C | 4.0 / 4.1 | +0.12 / +0.13 | 1.44 / 3.30 | - / 3.13 |
| d1ef1e42 | R 90 | S ok 1.00 | S low_other 0.78 | both fail | S | C | C | 2.5 / 2.3 | +0.92 / +0.95 | 1.01 / 1.18 | - / - |
| e14711ef | L 96 | E off_junc_outside 0.00 | O ok 1.00 | both fail | E | E | E | 2.6 / 2.6 | +0.47 / +0.46 | 4.09 / 3.56 | - / 3.38 |
| e9cfc98f | L 90 | S ok 0.95 | E off_junc_outside 0.00 | both fail | C | S | S | 4.8 / 4.6 | +0.05 / +0.06 | 0.00 / 0.34 | - / 0.44 |
| 16 others | | C | C | | C x 16 | 14 C, 1 E (8c9e3fd9) | 13 C, 3 E (365c0937, 8c9e3fd9, c2ae8e85) | median 4.4 / 4.3 | | | |

Turn shape, 24 strict turns, paired JTN - RSG with log-clustered CIs (`rf_ledger`):
- Entry speed: 4.11 vs 4.12 m/s, -0.01 [-0.11, +0.06]. The human enters at 4.16.
- Mid-turn speed: +0.38 [+0.09, +0.60] m/s.
- Exit + 10 m speed: 7.26 vs 5.54 m/s, +1.72 [+1.09, +2.68]. The human exits at 6.98. Time lag behind the human at that point: -0.27 vs +0.34 s.
- Onset: +0.23 [-0.04, +0.59] m later.
- Offset at 100 % of the turn: +0.22 [-0.09, +0.49] m wider.
- Peak inside: -0.33 vs -0.49 m, +0.16 [+0.04, +0.31] (cuts less).
- Wide > 1 m: 33 % vs 50 %, -16.7 [-29, -5] pp.
- Correct lane at exit + 10 m: 82 % vs 89 %.
- Peak yaw rate against the human: JTN +1.2 [+0.0, +2.4] deg/s, RSG -0.8 [-1.9, +0.5].

**Turn-related zeros, all 460 held scenes** (rf_tax classes, the ledger's grouping), RSG -> JTN:
- Outside (wide / under-turned): 7 -> 10.
- Inside (cut): 5 -> 3.
- Straight through the junction: 2 -> 0.
- Straight at the turn: 1 -> 0.
- Total: 15 -> 13.

Scene by scene, the 7 / 7 turn zeros that only one side has:
- JTN only, all outside: off_curve_outside 3380d56c, 13ae6351, ddc1271e; off_junc_outside e9cfc98f, 078bc102, e072351f; cor_turn_under 8d3ac506.
  Event speeds: 8.8, 5.1, 4.6, 4.9 m/s, ...
- RSG only: 5 outside (5b61b907, e6056b57, a3c4e4bf, 4faa4706, e14711ef) and 2 straight-through (20728d3e, b628faa4).

Six of JTN's seven new outside zeros are on bends outside the strict 24. The exception, e9cfc98f, is a strict turn that RSG fails as short.

## Open loop and memory reliance

- **Equal-arc turning (logged frames, 458 dev rows turning >= 15 deg, 23 logs), JTN - RSG.**
  - All turning rows: out@10 m +0.01 [-0.01, +0.03] m, lag@10 m +0.06 [-0.10, +0.20] deg. No under-turn.
  - Sharp rows >= 45 deg (22 rows reach 20 m): out@20 m +0.14 [-0.00, +0.30] m, lag@20 m +0.62 [-0.02, +1.35] deg. A borderline hint of slightly
    less turning late in sharp bends.
  - JTB: +0.13 m / +0.58 deg. JT10: +0.16 [+0.01, +0.34] m / +0.85 [+0.19, +1.62] deg.
- **G-rely** (junction dev ADE with memory masked minus memory on; junction - straight difference with CI):

  | label | junction dev ADE | junction reliance | difference [CI] | read |
  |:--|--:|--:|:--|:--|
  | RSG | 0.554 | +0.025 | +0.026 [+0.012, +0.038] | true |
  | JTN | 0.587 | +0.015 | +0.014 [-0.007, +0.033] | false |
  | JTB | | +0.017 | | false |
  | JT00 | | +0.010 | | false |
  | JT10 | | +0.001 | | false |

  The joint pool roughly halves offline route-memory reliance and costs +0.033 m junction dev ADE. Decision 286 found that offline G-rely does
  not predict closed-loop reliance.

## Reading

The held turns do not show JTN turning worse than RSG. JTN completes 20 / 24 against 16 / 24: 4 fixed, 0 broken, CI lower bound at 0.
All four fixed turns are ones RSG failed as "short": RSG was still on the route but too slow to clear exit + 10 m by decision 17. Entry speed is
identical (4.1 m/s), and so is entry lateral offset. JTN carries more speed through and out of the turn (+0.4 m/s mid-turn, +1.7 m/s at exit +
10 m, close to the human) and cuts less. What it trades for that is a different failure type. Its failed turns are wide: 2 "off" with peak outside
3.3-3.6 m but score 1.0, and 1 junction-outside offroad zero. RSG's failures were slowness. Across all 460 scenes, turn-related zeros net out
(15 vs 13, 7 / 7 flips), but every one of JTN's own new ones is an outside / under-turn zero, often at higher speed. The sharp-bend open-loop hint
(+0.14 m outside at 20 m) points the same way. This agrees with the arena turn stratum: a tie, +0.004 [-0.032, +0.031], 5 new / 6 fixed. If the
user's impression comes from watching JTN's failures, it reflects a real shift toward going wide in bends when faster. It is not a loss of turning
completion. One seed and one read per label; 24 turns give wide CIs.

Box outputs: `$DATA_DIR/runs/defect_fix/jt/turns24/` (`follow/`, `tax/`, `ledger/`, `devturn.json`, job wrappers `job.sh` / `job2.sh`).
