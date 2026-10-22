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

**Verdict (2026-10-22 18:31 box time): not adopted.** Both seeds fail rule 1 ((OFF + ON) / 2 0.7795 / 0.7828 against R6DAG's 3-seed
0.7890) and rule 3 (other zeros 42.0 / 41.5 against the line 38.2 + 2). They pass rule 2 by one zero (high-speed lane-keeping zeros 6.0 /
5.0 against 6.5). Details are in [Reads and verdict](#reads-and-verdict) below.

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

## Reads and verdict

PAI 281 ACC + GAP + `JEV_CMD=heading`, NTV0's 8-chunk table, Harmonizer OFF and ON (`HARMONIZER=1`), 279 common scenes. Files:
[pai281_R9DL.md](pai281_R9DL.md) (paired CIs, `r2.py r281`), [verdict.md](verdict.md) (per-class table, rule), [pcs_dl.md](pcs_dl.md)
(local PCS ruler), [cmdagree_s0.json](cmdagree_s0.json) / [cmdagree_s1.json](cmdagree_s1.json).

**Train / serve command agreement (G-C): pass.** On every inference of all four reads (4 x 55 693), the served command equals
`rmem.cmd_heading` of the fed float16 memory, 100 %.

| read | mean | zeros | lane-keep zeros | high-speed | other zeros | turn inside | collisions | offroad (fclass) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| R6DAG s0 / s1 / s2 OFF | 0.775 / 0.815 / 0.791 | 53 / 43 / 51 | 12 / 8 / 12 | 8 / 6 / 9 | 41 / 35 / 39 | 10 / 7 / 8 | 6 / 6 / 6 | 20 / 12 / 18 |
| **R9DL s0 / s1 OFF** | 0.786 / 0.800 | 51 / 47 | 10 / 7 | 7 / 4 | 41 / 40 | 9 / 7 | 5 / 7 | 20 / 15 |
| R6DAG s0 / s1 / s2 ON | 0.792 / 0.780 / 0.780 | 46 / 49 / 48 | 8 / 10 / 11 | 4 / 6 / 6 | 38 / 39 / 37 | 6 / 8 / 6 | 5 / 6 / 7 | 15 / 15 / 15 |
| **R9DL s0 / s1 ON** | 0.773 / 0.765 | 53 / 54 | 10 / 11 | 5 / 6 | 43 / 43 | 9 / 10 | 6 / 7 | 20 / 20 |
| NTV0 OFF / ON | 0.743 / 0.745 | 59 / 58 | 12 / 13 | 8 / 8 | 47 / 45 | 3 / 5 | 3 / 5 | 12 / 17 |

Paired against the same seed of R6DAG:

| seed | OFF | ON |
|---|---|---|
| s0 | +0.011 [-0.016, +0.039], zeros -2 | -0.019 [-0.045, +0.004], zeros +7 [+1, +14], offroad +5 [+1, +10] |
| s1 | -0.015 [-0.041, +0.010], zeros +4 | -0.015 [-0.041, +0.010], zeros +5 |

Against NTV0, OFF is +0.043 / +0.057 (CIs exclude 0) and ON is +0.028 / +0.020 (CIs include 0). R9DL's ON - OFF gap is -0.013 / -0.035.
R6DAG's is +0.017 / -0.035 / -0.011.

**Target scenes (route_check A, the route in the lane next to the human's).**
- 7dc59d0d passes in all four R9DL reads. R6DAG passes it in 2 of its 6 reads (s0 and s2 ON). R7J, which used the same heading command,
  also passed it on both seeds.
- 26004ca8 and 4cb05d65 fail in every read of every arm.
- 0e05cb0c passes in the ON reads only, as for R6DAG.

**Local PCS ruler (OFF, out of pool).** R9DL scores -184 / -182 against NTV0, CI [-286, -56] / [-276, -22]. R6DAG scores -213 / -155 /
-165 (3-seed mean -178). Paired with the same draws: R9DL-s0 - DAG-s0 +25 [-27, +83], R9DL-s1 - DAG-s1 -25 [-127, -1].

**Adoption rule (plan section 6).** R6DAG's 3-seed (OFF + ON) / 2 is mean 0.7890, high-speed lane-keeping zeros 6.50, other zeros 38.17.

| seed | rule 1: (OFF + ON) / 2 >= 0.7890 | rule 2: high-speed lane-keeping zeros < 6.50 | rule 3: other zeros <= 40.17 |
|---|---|---|---|
| s0 | 0.7795, fail | 6.0, pass | 42.0, fail |
| s1 | 0.7828, fail | 5.0, pass | 41.5, fail |

**Not adopted.**

How to read it:
1. The two changes do what they were built to do, but only in a small part of the score.
   - The command rule recovers 7dc59d0d on every read.
   - High-speed lane-keeping zeros fall by about 1 per read: 5.5 against 6.5.
   - That block was 4-5 points per seed (route_check). The two widest-offset scenes (26004ca8, 4cb05d65) are not recovered: the route
     geometry still pulls the car across.
2. The loss is in the Harmonizer-ON reads and in offroad.
   - ON offroad is 20 / 20 against R6DAG's 15 / 15 / 15, and ON zeros are +5 / +7.
   - OFF is within R6DAG's seed spread.
3. The cause cannot be separated: the arm bundles (a) and (b) by design.
   - R7J also used the heading command. It also lost lateral (s1 offroad +9) and lost in the ON reads.
   - R8K dropped the command and its offroad returned to R6DAG's level.
   - That pattern is consistent with the heading command costing offroad and ON robustness. This has not been tested here.
4. Untested next step: (b) alone, the same pool `on1_JTPHDL` without `--cmd-rule`, 2 seeds, about 6 card-h.

Cost: about 13 card-h by VRAM share (2 trainings of about 40 min each, 2 x 8 OFF chunks at 3.4 job-h, 2 x 8 ON chunks at about 10 job-h).
The chain ran 13:51-18:31 box time, sharing cards with other lanes from about 14:30.
