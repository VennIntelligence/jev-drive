# NTV1 step 2a: P1L's own DAgger pool (ROUND9, 2026-10-24)

Pre-registration (Chinese): [../../plans/2026-10-24-ntv1-step2.md](../../plans/2026-10-24-ntv1-step2.md), amendments 1-2 (two mfix
check-script faults, see below). Code: [../../scripts/dgc.sh](../../scripts/dgc.sh), [../../scripts/dgc.py](../../scripts/dgc.py). The roll,
rows, teacher, pool, label and input-path job bodies are `experiments/round6/scripts/dag.sh` / `dag.py`, unchanged, run under this lane's
env with `DAG_TV=b3`. Box: `$DATA_DIR/runs/round9/ntv1_step2/`. Full summary: [build.json](build.json). No policy was trained.

**What it is.** Decision 434's R6DAG iteration 1 rebuilt with the student P1L = PCL1L-F-s0 (decision 461) in place of R5LC-L18-s0.
- Serve: the submission serve (ACC + GAP, g10, FAST, CONC 4, frame dumps, KEEP_ASL), Harmonizer off.
- Scenes: the same 719 non-held curated train scenes in R6DAG's chunk lists. They are verified disjoint from test281 and curated-val
  (441 scenes) and from the 207 scenes excluded for exam drive groups.
- Teacher: round 1's teacher b3. Laterally it is the route-lane centre plus the human's offset (sigma 10 m, saturated at +-0.5 m), with a
  3 s merge. Longitudinally it keeps the student's own served-plan arc lengths.

## Pool

| | on1_PAIDC (P1L) | on1_PAID (R5LC, d434) |
|---|--:|--:|
| rows | 112 620 | 111 k |
| drawable (fed >= 2 m/s, teacher label, T4 fallback applied) | **92 561** | 90 300 |
| scenes (rows / drawable) | 719 / 707 | 718 |

- Drawable rows by class: stop_nolead 18 703, turn_sharp 20 609, launch 6 547, other 46 702.
- Signed d0 (left +): median +0.03 m. |d0| > 1 m on 9 450 rows to the left and 7 103 to the right, so both drift sides are covered.
- Per class: [rows_on1_PAIDC.md](rows_on1_PAIDC.md).

## Gates

| gate | line | result |
|---|---|---|
| chk1 (c000, 6 scenes) | scored / rows >= 5, teacher on >= 80 %, <= 6 card-h projected | 6 / 6, 100 %, 3.1 card-h: pass |
| T1 teacher geometry (18 745 human states) | lane centre median 2 s <= 0.30, 4 s <= 0.50 m | 0.299 / 0.369 (= d434 exactly): **pass**. [t1.md](t1.md) |
| T2 in canvas | teacher inside >= 90 % | 62 / 65 = 95 % (served 55): **pass**. [t2.md](t2.md) |
| T3 (a) weight 0 vs the stored P1L | every scalar <= step 100 bit-identical | P1L's argv on on1_JTDC / NTV1D0: 131 / 131 identical, steps 0-100: **pass**. [t3a.json](t3a.json) |
| T3 (b) weight 0, pool drawn | T = 30 pair, every scalar <= 60 identical | on1_JT / JT00S-rb0 vs on1_JTDC / NTV1D0: 231 / 231: **pass**. [t3b.json](t3b.json) |
| T4 label feasibility | position p99 <= 4 m/s², else the fallback if <= 3 % over | p99 5.30 (served 3.53), 2 449 rows over (2.6 %): **fallback applied** as registered, those rows at weight 0 (d434 2.1 %). Curvature read p99 706, route curvature alone 3.83. |
| input path (rf_recov fwd) | median <= 0.02 m, over <= 10 %, wrong ckpt >= 90 % | P1L 0.011 m, 3.7 %. RB0-F-s1 over on 99.5 %: pass |
| mfix (on1_JTDC prefix = on1_JT) | common keys identical | 0 mismatches. On1_JT's extra metadata keys are listed: `agent.cls/log`, `road.log/pose_global`, `tab.acc/cmd/pose/vel`. on1_JTPH lacks the same keys. |
| smoke A / B (60 steps, T = 30) | composition, shares, finite | A 83 / 13 / 40 and B 75 / 13 / 40 (mixed). Self share after T 0.813 (expected 0.8). Draw shares S 0.8 / pd 0.2. Both pass. [smoke_a.json](smoke_a.json), [smoke_b.json](smoke_b.json) |

## Two process faults, both in the check script, fixed and rerun

1. dag.py mfix raised KeyError on `agent.cls`. On1_JT carries keys that rf_pait merge never writes.
2. My first fix still compared dtypes: `tab.log` is `<U43` in the merge and `<U38` in on1_JT. The values are equal, and on1_JTPH is `<U43` too.

`dgc.py mfix` now compares the common keys, with unicode keys compared by value. Only the tail from mfix onward was resubmitted
(`dgc.sh tail2`); the merged pool was kept. The failed job records are in `jobs.attempt{1,2}.txt`. T3 (a) / (b) show that the missing keys
do not matter to the trainer.

## Command-rule fields (for a later `--cmd-rule heading-all`, decision 459)

The DAgger rows carry:
- the fed route memory: `runs/route_mem/feat/on1_PAIDC.npz` `f`, 40 slots x 5, float16, the old projected layout. In on1_JTDC it is the
  feat tail, identical bit for bit.
- the served shipped-rule command: `tab.ego[:, 1:4]`.
- `tab.speed`.

The memory is valid on 100 % of the rows. On the drawable rows, `rmem.cmd_heading` changes 20.9 % of the commands (L / S / R stored
0.17 / 0.63 / 0.20 -> heading 0.09 / 0.82 / 0.10). Smoke A ran `--cmd-rule heading-all` over the merged pool: part `pd` 112 620 rows, 0 %
kept for lack of memory, 19.4 % changed. Details: [cmd_fields.json](cmd_fields.json).

## By-product: train-scene roll scores (old AlpaSim scorer, rf_tax + pai_fix facts, no RPC errors)

| student | scenes | mean | zeros | offroad | overshoot | route-turn | zero-other | AF-front | side |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| P1L | 719 | 0.674 | 198 | 56 | 65 | 29 | 19 | 17 | 12 |
| R5LC-L18-s0 (round 6, stored) | 718 | 0.663 | 209 | 71 | 65 | 10 | 33 | 18 | 12 |

Paired P1L - R5LC on 718 scenes: +0.011 [-0.021, +0.043]. P1L has fewer offroad / zero-other zeros and more route-turn zeros (R5LC was
trained on PAI human rows). The new scorer was not run: the rollout.asl files are deleted after each chunk's rows are built.

## Cost and disk

- Rolls: 2.6 card-h (7.9 stack-h, 3 stacks a card).
- All GPU jobs: 8.6 job-h (pai chunks 8.2, gates / smokes 0.46). Chain wall time 2.2 h.
- Disk (GiB, on the box): rows 37.9, cache 26.2, on1_PAIDC 3.5 + tokens 28.2, on1_JTDC 15.4 + 1.7 (zero-copy tokens), roll / tax / feat
  about 1.6. **Total about 114 GiB.** The frame dumps and rollout.asl are deleted, and no staging dir is left. Box free space is 622 GB.

## Step-3 flags (not run)

Put these on N1's argv (decision 465's base, pool on1_JT). On1_JTDC is on1_JT plus the DAgger rows, so it is the step-3 pool as it stands:

- **A (R6DAG's way, 8 rows added):** `--on on1_JTDC --on-wfile runs/round9/ntv1_step2/w/NTV1D.npz --on-k 40 --batch 136 --yr-mass 0.095`.
  - After T: 83 navtrain + 13 yaw + 40 pool (expected 32 S + 8 DAgger).
  - Before T: 123 navtrain + 13 yaw (P1L / N1 have 115), so steps before T are not identical to N1.
- **B (fixed batch):** `--on on1_JTDC --on-wfile runs/round9/ntv1_step2/w/NTV1D.npz --on-k 40`.
  - Before T: identical to N1.
  - After T: 75 navtrain + 13 yaw + 40 pool (the 8 DAgger rows replace 8 navtrain rows).

NTV1D = JT00S-rb0 normalised x 32 / 40, plus 8 / 40 uniform over the 92 561 drawable DAgger rows. NTV1D0 is the weight-0 identity file.
Before T, the DAgger rows are not drawn in either variant. Both variants passed the smoke, A with `--cmd-rule heading-all`.
