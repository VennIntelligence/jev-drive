# DAG-LANE: R6DAG + heading command + the DAgger teacher's human-lane rule (ROUND9, 2026-10-22)

Pre-registration (Chinese): [../../plans/2026-10-22-dag-lane.md](../../plans/2026-10-22-dag-lane.md).
Code: [../../scripts/dl.py](../../scripts/dl.py), [../../scripts/dl.sh](../../scripts/dl.sh), and teacher `DAG_TV=b3L` in
`experiments/round6/scripts/dag.py`. Box: `$DATA_DIR/runs/round9/dag_lane/`.

**Arm R9DL** (2 seeds, both from NTV0-L18-s0, 9 500 steps). It is R6DAG-L18-s0's recorded argv with two changes:
- The pool is `on1_JTPHDL`. Its DAgger rows (`pd`, the same iteration-1 states) are relabelled by teacher b3L. The draw file is R6DAG.npz,
  unchanged.
- The heading command is used, as in R7J: `--cmd-rule heading` in training and `JEV_CMD=heading` at serve.

**Teacher b3L.** It equals b3, except where the human drove in the route lane's parallel neighbour lane on a straight stretch. There the
reference is the human's own lane centre plus the human's offset within that lane (saturated at ±0.5 m), not the route's lane. Scenes
without such a stretch are bit-identical to b3.

**Status (2026-10-22 14:00 box time).** All gates pass. Training and the PAI 281 reads (OFF and ON) are queued in one chain. The verdict
will be appended below when it lands.

## Gates (before training)

| gate | result | file |
|---|---|---|
| G0 code identity: R6DAG-L18-s0 argv, 60 steps eager, R6DAG's commit 4d8d1a33 vs HEAD | identical (274 scalars, every weight, dev ADE 0.53824, on_hash f5368348b47e2019) | [identity.json](identity.json) |
| Stage 1 (6 scenes of c000) | 6 / 6 files, b3 reproduced exactly, finite | [teach_stage1.json](teach_stage1.json) |
| G-L0 label identity | b3 reproduced max \|dxy\| 0 over 100 % of the 90 300 drawable rows; b3L = b3 bit for bit on the scenes without a qualifying stretch | [rows.md](rows.md) |
| G-L1 target states | 4 039 drawable rows at >= 15 m/s whose b3 label ends > 1.75 m from the human's path and that b3L moves by > 1 m (line 300); b3L brings 99.9 % of them back within 1.75 m (line 90 %). 134 of 718 scenes have a qualifying stretch. | [rows.md](rows.md) |
| G-L2 (T1, 18 745 human states) | median \|error\| at 2 / 4 s: b3 0.075 / 0.089 m, b3L 0.057 / 0.068 m. On the 2 172 states (11.6 %) where b3 misses the human by > 1.75 m: b3 2.53 m, b3L 0.20 m. | [teacher_gates.md](teacher_gates.md) |
| G-L3 (T2, in canvas) | teacher inside 62 / 65 for both b3 and b3L | [teacher_gates.md](teacher_gates.md) |
| G-L4 (T4, position estimator) | p99 3.68 m/s² (b3 3.63), 0 rows over the line. 1 734 changed rows over 4 m/s² reverted to b3, 618 of them target rows. | [rows.md](rows.md) |
| G-S smoke (300 steps vs R6DAG control) | batch 51 / 13 / 72, shares equal, gradient ratio 0.988, dev ADE 0.5569, command rule on. pd / ph rows L/R 41.6-45.6 % -> 21.7-25.5 %; 22 % of the CMD_PARTS rows change command. | [smoke.json](smoke.json) |
| G-C command | offline and serve gates are R7J's: same pool features, same `rmem.cmd_heading`, and `pai_driver.py` is unchanged since R7J's serve gate (7082 / 7082 inferences agree). The post-read agreement check on this arm's own reads is in the chain. | `experiments/round7/results/joint/` |

What the relabel changes: 14 475 of 90 300 drawable DAgger labels (16 %), 6 157 of them at >= 15 m/s.

Under b3, the share of rows whose label ends in a lane the human never used rises with speed: 10 % below 10 m/s and 18-21 % at >= 15 m/s.
Under b3L the high-speed share falls to 0.7-4 %. This is the mechanism route_check named for the high-speed drift block.

## Attempt 1 (stopped, no training)

The first submission (234b6da4) failed in `dl.py teach`. A diagnostic line read the lane-rule weight on the wrong station grid, which
crashed the 125 scenes that have a qualifying stretch. No label had been written from those scenes. The gates that ran (stage 1, T1, T2,
identity) passed with the same numbers. The fix is a3cc61bc, and the chain was resubmitted whole. Attempt 1's logs are in
`$DATA_DIR/runs/round9/dag_lane/attempt1/`.

## Reads

Pending: PAI 281 ACC + GAP + `JEV_CMD=heading`, Harmonizer OFF and ON, for both seeds. Then `r2.py r281` against R6DAG s0 / s1 / s2 and
NTV0, the per-class table and adoption rule (`dl.py verdict`), and the local PCS ruler.
