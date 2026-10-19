# TURN-CAP offline gate (NR1 prereg supplement 21): FAIL, lane stopped (2026-10-20)

Pre-registration: [plans/2026-10-20-nr1-supp21-turncap.md](../../plans/2026-10-20-nr1-supp21-turncap.md), registered before any label, gate
output or training. Code: `lib/nr1_free.py` (`turn_range`, `turncap_rows`, `turncap_cap`, `turncap_load`, `turncap_term`), `lib/nr1_lat.py`,
`rm_train --on-turncap W`, [scripts/turn_cap.py](../../scripts/turn_cap.py), [scripts/turn_cap.sh](../../scripts/turn_cap.sh).
Decision 396.

**The term.** It acts on moving S rows: the ego at 2 m/s or more, the human's logged 4 s arc at least 4 m, and the human's logged path with a
heading range of at least 45 deg over -10 .. +20 m around the ego's station. On those rows it penalises the part of the plan's arc beyond
1.05 x max(arc(`fut`), arc(`fut_reg`)). `fut` gives the ego 2 s to blend from its own speed to the human's, so the cap is never below either
human arc. The gradient moves poses only along their own heading. The switch is off by default (W = 0 loads nothing and adds nothing).

**Code checks.** The `nr1_free.py` known-answer checks pass: turn range on straight / 90 deg / clamped paths, the row limits, the cap
(4 -> 3 m/s gives 13.65 m), fire / no fire, the human's paths do not fire, and the gradient is along the heading only and shortens the plan.
The bit-identical switch-off check (`TCGG-old` / `TCGG-new`) was queued, then cancelled when the gate failed. It never ran.

## Gate (box CPU, stored reads; [gate.json](gate.json))

| check | measured | line | result |
|:--|--:|:--|:--|
| G1 human paths as the plan, on1_JTX domain rows (13 965) | 0 / 0 fires (`fut` / `fut_reg`) | 0 | pass |
| G2 B2 s1 target windows, pre-exit decisions (k >= 2, last 3 s before the exit) | 36 / 41 = 88 %, 9 / 9 windows | >= 50 % and >= 6 / 9 | pass |
| G3 B2 s1 held windows scored 1: decisions that fire | 967 / 16 272 = **5.9 %** | <= 1 % | **FAIL** |
| G3 same: windows with any fire | 238 / 2 034 = **11.7 %** | <= 5 % | **FAIL** |

**Reading.** The cap covers the failing turns, but it does not separate them from turns B2 drives well.
- On passing windows, 62 % of the in-domain decisions fire.
- The fires there fall by stratum as follows: turn 542, other 154, braking 144, launch 122, static 5.
- The plan / cap ratio at the fires is about the same on passing windows (median 1.21, 10-90 % 1.04-1.85) as on the target pre-exit decisions (median 1.17, 1.04-1.64).

Across all held windows of B2 s1, 283 windows fire:
- 145 of them are outside d395's sharp-turn set;
- of the turn-set windows, 116 fire and pass, and 22 fire and fail.

The s0 read gives the same picture: 273 windows fire, 140 of them outside the turn set, and of the turn-set windows 109 fire and pass while 24 fire and fail.

This fits d395's own caveat that speed alone does not force the exit. B2 is faster than the human in many turns and gets through most of them. A cap at 1.05 x the human would therefore act mostly on turns that already pass.

**Pool description** ([label.json](label.json)).
- Domain size: 13 965 of the 128 798 S rows in on1_JTX (10.8 %), 5 531 right and 8 434 left. That is 4.7 % of the JTX00 draw, about 3.3 rows per 70-row batch.
- On the domain rows, the stored served plans would fire on 30.8 % (1.45 % of the draw).
- Plan / human arc on those rows, deciles 10 / 25 / 50 / 75 / 90 %: 0.57 / 0.75 / 0.93 / 1.10 / 1.39.

## Post-hoc sensitivity (descriptive only, written after the gate; not a gate re-run)

This uses the same stored decision table (`runs/defect_fix/turn_cap/gate/decisions.parquet`, B2 s1), with the cap factor varied and an optional right-turn-only row limit:

| variant | G2 decision share | G2 windows | G3 decision share (line 1 %) | G3 window share (line 5 %) |
|:--|--:|--:|--:|--:|
| both turns, K 1.05 (registered) | 0.878 | 9 / 9 | 5.9 % | 11.7 % |
| both, K 1.15 | 0.634 | 9 / 9 | 4.4 % | 9.1 % |
| both, K 1.30 | 0.366 | 5 / 9 | 2.7 % | 6.0 % |
| right only, K 1.05 | 0.878 | 9 / 9 | 1.9 % | 4.2 % |
| right only, K 1.15 | 0.634 | 9 / 9 | 1.5 % | 3.1 % |
| right only, K 1.30 | 0.366 | 5 / 9 | 1.0 % | 2.2 % |

**What the table shows.** No cap factor separates the targets from passing turns.
- Raising K costs target coverage faster than it removes collateral.
- A right-turn-only limit cuts the collateral by about 3x and keeps 9 / 9 coverage up to K 1.15. Even so, it still fires on 1.5-1.9 % of the passing decisions, above the registered 1 % line.
- Any revised variant would need a new registration and the user's approval. None was run.

## Jobs and cost

| item | count / time | card-hours |
|:--|:--|--:|
| CPU jobs that ran | 2: `tc-label` and `tc-gate`, each about 1 min | 0 |
| jobs cancelled before they ran | 10: identity pair, bitcmp, dose, 6 control turn-set reads | 0 |
| arm jobs failed by dependency, no training | the rest (both arms' train, reads and psum) | 0 |
| **GPU total** | | **0** |

Box outputs are in `runs/defect_fix/turn_cap/` (gate/, pool/, stages.txt), and the label file is `runs/op_parity/cache/on1_JTX/turn_cap.npz`.

## Files
- [turn_set.csv](turn_set.csv): the 170 turn-set windows (d395's 169 sharp-turn held windows plus 202f583c), for the stage-1 read that did not run.
- [gate.json](gate.json), [label.json](label.json).
