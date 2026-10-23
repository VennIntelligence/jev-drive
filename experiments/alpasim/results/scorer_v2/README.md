# scorer_v2: stored PAI 281 reads rescored with AlpaSim's fixed corridor scorer (PR #215)

Date: 2026-10-23. Script: [scripts/scorer_v2.py](../../scripts/scorer_v2.py). Box outputs (mirror trees, per-rollout csv, logs):
`$DATA_DIR/runs/alpasim/scorer_v2/` (`old/`, `new/`, `report/`, `end_shape.csv`). Decision 453.

**What changed upstream.** NVlabs/alpasim `e2e_challenge` 1b024f0 "Bugfix: Lateral Corridor Exit" (PR #215, issue #198, merged 10-20),
only `src/eval/src/eval/scorers/ground_truth.py`: the end direction is taken over the last >= 1 m of GT motion (not the last, often
micrometre-long, segment of a stopped log), and the lateral distance is min(plain, perpendicular to the forward ray from the GT end) whenever
the ego is ahead of the end. A human log that ends stationary (jittering tail) plus an ego that drives past the end used to read as
`left_corridor_laterally` (score 0); now it does not. The organisers recomputed the board with it.

**How.** Fixed scorer in a separate clone `$DATA_DIR/third_party/alpasim-e2e-1b024f0` (the live `third_party/alpasim` at 0bb4c4b is
untouched; its eval code was only imported, read-only, for the old-scorer pass). Upstream unit tests `test_ground_truth_corridor.py`: 15 / 21
pass on the new scorer; the other 6 fail on both scorers because the test file uses a proto field (`SessionMetadata.scene`) and a
`ScenarioEvalInput(recorded_metrics=...)` argument that this branch does not have (upstream test drift, not our env).
Stored reads keep `rollout.asl` only for zero-score rollouts; those were re-evaluated with AlpaSim's own eval code (the calls of
`eval.main` / `reeval.py`: ASL -> `ScenarioEvaluator` -> metrics.parquet -> `_run_aggregation_core`) in a mirror tree where every other
rollout keeps its stored metrics.parquet. CPU only, 40 subprocess shards, about 15 min per scorer for 47 reads.

**Reproduction gate (old scorer through this path).** First 3 NTV0-s0 scenes, then all: 13 186 / 13 186 stored rollouts reproduce the stored
scene score (max |diff| 1.1e-16, an aggregation ulp) and failure reason; 2 309 / 2 315 re-evaluated metrics.parquet files are identical at every
step, 6 differ at one step by < 2e-6 m (scores equal). 21 rollouts were RPC failures with no metrics (not rescored, excluded as before).
Old-scorer pair sums reproduce decision 443 (R8K - R6DAG 3-seed +0.006 / -0.007 / -0.001) and decision 426 (NTV0 - RSG +0.0545).

## Result

**Every changed scene goes 0 -> 1.0, all from `left_corridor_laterally`, and every one with a facts row was `cor_route_end` (overshoot).**
761 scene-reads change, 32 distinct scenes (18 of them flip in nearly every read). No scene gets worse among the re-evaluated rollouts.
All complete reads gain +0.064 to +0.089 (18-25 scenes); the ranking between our models barely moves, because the same stopped-log
scenes flip for everyone.

| read (281, ACC + GAP; agH = Harmonizer on) | zero ASLs kept | scenes | old mean | new mean | new - old | zeros old / new | flipped 0 -> 1 | nonzero without ASL that hit 4 m |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-L18-s0-ag | 59/59 | 280 | 0.744 | 0.816 | +0.071 | 59 / 39 | 20 | 78 |
| NTV0-L18-s0-agH | 58/58 | 281 | 0.747 | 0.811 | +0.064 | 58 / 40 | 18 | 77 |
| RSG-L18-s0-ag | 76/76 | 281 | 0.691 | 0.755 | +0.064 | 76 / 58 | 18 | 76 |
| R6DAG-L18-s0-ag | 53/53 | 281 | 0.777 | 0.844 | +0.068 | 53 / 34 | 19 | 101 |
| R6DAG-L18-s0-agH | 46/46 | 280 | 0.793 | 0.868 | +0.075 | 46 / 25 | 21 | 92 |
| R6DAG-L18-s1-ag | 43/43 | 281 | 0.817 | 0.881 | +0.064 | 43 / 25 | 18 | 104 |
| R6DAG-L18-s1-agH | 49/49 | 280 | 0.781 | 0.852 | +0.071 | 49 / 29 | 20 | 91 |
| R6DAG-L18-s2-ag | 51/51 | 281 | 0.792 | 0.860 | +0.068 | 51 / 32 | 19 | 101 |
| R6DAG-L18-s2-agH | 48/48 | 281 | 0.782 | 0.853 | +0.071 | 48 / 28 | 20 | 96 |
| R8K-L18-s0-ag | 50/50 | 281 | 0.794 | 0.858 | +0.064 | 50 / 32 | 18 | 108 |
| R8K-L18-s0-agH | 53/53 | 280 | 0.775 | 0.843 | +0.068 | 53 / 34 | 19 | 101 |
| R8K-L18-s1-ag | 47/47 | 281 | 0.802 | 0.866 | +0.064 | 47 / 29 | 18 | 114 |
| R8K-L18-s1-agH | 54/54 | 281 | 0.770 | 0.837 | +0.068 | 54 / 35 | 19 | 103 |
| R8K-L18-s2-ag | 47/47 | 281 | 0.806 | 0.874 | +0.068 | 47 / 28 | 19 | 118 |
| R8K-L18-s2-agH | 47/47 | 280 | 0.795 | 0.867 | +0.071 | 47 / 27 | 20 | 104 |
| R3MV-L18-s0-ag | 62/62 | 280 | 0.743 | 0.818 | +0.075 | 62 / 41 | 21 | 97 |
| R3MV-L18-s1-ag | 70/70 | 278 | 0.715 | 0.784 | +0.068 | 70 / 51 | 19 | 95 |
| R5LC-L18-s0-ag | 62/62 | 280 | 0.747 | 0.819 | +0.071 | 62 / 42 | 20 | 102 |
| R5LC-L18-s1-ag | 68/68 | 281 | 0.727 | 0.801 | +0.075 | 68 / 47 | 21 | 100 |
| R6DAG2-L18-s0-ag | 50/50 | 281 | 0.790 | 0.858 | +0.068 | 50 / 31 | 19 | 99 |
| R6DAG2-L18-s1-ag | 50/50 | 281 | 0.796 | 0.863 | +0.068 | 50 / 31 | 19 | 106 |
| R6FIX-L18-s0-ag | 70/70 | 280 | 0.714 | 0.792 | +0.079 | 70 / 48 | 22 | 87 |
| R6FIX-L18-s1-ag | 64/64 | 281 | 0.741 | 0.823 | +0.082 | 64 / 41 | 23 | 100 |
| R6REC-L18-s0-ag | 66/66 | 281 | 0.734 | 0.812 | +0.078 | 66 / 44 | 22 | 96 |
| R6REC-L18-s1-ag | 68/68 | 281 | 0.725 | 0.803 | +0.078 | 68 / 46 | 22 | 100 |
| R7J-L18-s0-ag | 52/52 | 280 | 0.787 | 0.863 | +0.075 | 52 / 31 | 21 | 110 |
| R7J-L18-s0-agH | 52/52 | 281 | 0.776 | 0.847 | +0.071 | 52 / 32 | 20 | 107 |
| R7J-L18-s1-ag | 54/54 | 280 | 0.783 | 0.851 | +0.068 | 54 / 35 | 19 | 115 |
| R7J-L18-s1-agH | 53/53 | 278 | 0.768 | 0.833 | +0.065 | 53 / 35 | 18 | 101 |
| R9DL-L18-s0-ag | 51/51 | 281 | 0.788 | 0.859 | +0.071 | 51 / 31 | 20 | 99 |
| R9DL-L18-s0-agH | 53/53 | 281 | 0.775 | 0.846 | +0.071 | 53 / 33 | 20 | 91 |
| R9DL-L18-s1-ag | 47/47 | 281 | 0.802 | 0.869 | +0.068 | 47 / 28 | 19 | 109 |
| R9DL-L18-s1-agH | 54/54 | 281 | 0.767 | 0.838 | +0.071 | 54 / 34 | 20 | 95 |
| RB0-F-s0-ag | 88/88 | 281 | 0.649 | 0.717 | +0.068 | 88 / 69 | 19 | 74 |
| RB0-P30-s0-ag | 105/105 | 281 | 0.593 | 0.664 | +0.071 | 105 / 85 | 20 | 101 |
| RB0-P50-s0-ag | 102/102 | 281 | 0.605 | 0.676 | +0.071 | 102 / 82 | 20 | 102 |
| RB0-P70-s0-ag | 114/114 | 281 | 0.564 | 0.653 | +0.089 | 114 / 89 | 25 | 97 |

Partial reads (zero-score ASLs deleted after the read; only the kept ones rescored, so their new mean is a lower bound, about -0.06 short):
NTV0-L18-s1-ag 8/62, JTN-L18-s0-ag 8/65, JTN-L18-s1-ag 8/68, BNC2K-L18-s1-ag 8/65, JTXS-L18-s1-ag 8/65, NTVX-L18-s0-ag 7/64, NTVXS-L18-s0-ag 8/66, R2C-L18-s0-ag 8/60, R2J-L18-s0-ag 8/64, R5A-L18-s0-ag 8/70.

R6DAG-s0 / s1 (decision 436's overshoot class 20 / 19): 19 / 18 flip, all `cor_route_end` / overshoot, all to 1.0; the one overshoot zero left is 3fe2bac8 in both (4.27 / 4.07 m off the GT at 72 % progress, before the log end: a real lateral exit, unchanged by the fix). R6DAG's
new zeros by facts class: s0 offroad 20, zero-other 5, AF-front 4, route-turn 2, side 2, overshoot 1 (34); s1 offroad 12, zero-other 4,
AF-front 3, side 3, route-turn 2, overshoot 1 (25). Off-road is now the largest loss class.

**Paired, new scorer** (277 scenes common to the 14 reads, no RPC error; paired scene bootstrap, 10 000 draws, as decision 443):

| | OFF | ON | (OFF + ON) / 2 |
|---|--:|--:|--:|
| NTV0-s0 mean | 0.814 | 0.808 | 0.811 |
| R6DAG 3-seed mean | 0.860 | 0.856 | 0.858 |
| R8K 3-seed mean | 0.864 | 0.847 | 0.856 |
| R6DAG 3-seed - NTV0-s0 | +0.046 [+0.009, +0.083] | +0.048 [+0.010, +0.087] | +0.047 [+0.015, +0.080] |
| R8K 3-seed - NTV0-s0 | +0.050 [+0.010, +0.090] | +0.039 [+0.002, +0.077] | +0.045 [+0.013, +0.077] |
| R8K - R6DAG 3-seed | +0.005 [-0.015, +0.024] | -0.009 [-0.025, +0.005] | -0.002 [-0.015, +0.010] |
| old scorer, same scenes: R6DAG / R8K - NTV0 | +0.051 / +0.056 | +0.040 / +0.033 | +0.045 / +0.045 |

Single checkpoints, new - NTV0-s0 (OFF / ON): R6DAG s0 +0.028 / +0.059, s1 +0.065 / +0.043, s2 +0.044 / +0.043; R8K s0 +0.042 / +0.033,
s1 +0.051 / +0.027, s2 +0.058 / +0.057. Best by (OFF + ON) / 2 on its own scenes: R8K-L18-s2 0.870 (0.874 / 0.867), R6DAG-L18-s1 0.867
(0.881 / 0.852).

**Submitted-pair calibration (decision 426).** NTV0-s0 - RSG-s0, ACC + GAP OFF, 280 common scenes: old +0.054 [+0.024, +0.088]
(0.744 / 0.690), **new +0.062 [+0.029, +0.097] (0.816 / 0.754)**. Board numbers under the new scorer are for the user to supply.

**Can a scene get worse?** Not among the 2 315 re-evaluated rollouts (all were zeros). A nonzero rollout without an ASL can only change at its one
kept truncation step (first step with plain GT distance >= 4 m), and only if the new lateral there reaches 4 m while the old one was < 4 m. Over
the re-evaluated rollouts that had such a step with old lateral < 4 m, that crossing happened 0 / 340 times (the per-rollout max lateral rose
in 6 / 2 315, by <= 0.11 m, all already zero for another reason). There are 74-118 such unverified nonzero rollouts per read (`report/unverified.csv`
on the box); at the observed rate (95 % upper bound about 1 %) at most about one scene per read could be wrong, the same for every read.

Files: [reads.csv](reads.csv) (per read), [changed_scenes.csv](changed_scenes.csv) (every changed scene-read with old failure reason and the
pai_fix facts `primary` / `fclass`), [per_scene.csv](per_scene.csv) (old / new score per scene and read), [summary.json](summary.json)
(pairs, CIs, crossing check), [gate.json](gate.json) (reproduction gate, RPC-failed rollouts).

Rerun (box, repo root):
```bash
PY=/root/autodl-tmp/ujs/third_party/alpasim/.venv/bin/python; R=$DATA_DIR/runs/alpasim/scorer_v2; S=experiments/alpasim/scripts/scorer_v2.py
$PY $S score --variant old --root $R --workers 40
PYTHONPATH=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src $PY $S score --variant new --root $R --workers 40
.venv/bin/python $S report --root $R --out $R/report
```
