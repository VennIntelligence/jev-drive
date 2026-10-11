# JOINT-1: one joint training for the nuPlan submission #2 candidate, read with drop-one arms (pre-registration, 2026-10-11)

Lane JOINT, phase 1. Follows decisions 189, 199, 201, 205, 209, 210, 212, 213, 216, 226, 232, 233, 244, 247. Written and pushed before any
closed-loop or board score of any arm of this lane is read (pilot smokes included). Local closed loop only; nothing is submitted to AlpaSim. No
WA-JEPA weights or features. No privileged input at inference: GT boxes, map rasters and logged futures are training labels only.

## Question

Four ingredients have same-direction evidence on the AlpaSim nuPlan track, each from its own trainer: the lambda-10 drivable hinge base `P2H10-F`
(the registered servable, decision 205); the AlpaSim input standard `AP2` (decision 189, 212); the body lesson `P2H10S` (decisions 232, 233, 244:
shape-only agent and road hinges on the own plan, hinge-only off-track rows, road-and-lane raster); yaw-rate rows `YR10m10` (decisions 213, 216;
`APY10m10-AB` = AP2 + yaw rows, all 1491 scenes 0.9335 against P2H10 0.9196). Decision 244 shows ingredients do not add, and the user agreed on
2026-10-11 that final weights come from ONE joint policy training read with drop-one arms, by fixed-vs-broken counts and zero composition, never by
summing separately measured gains. Questions: (1) which arm of the joint family is the submission #2 candidate by the line below; (2) inside the joint
training, does each ingredient carry, is it not needed, or does it hurt.

## Arms (two seeds each, 10 000 steps, batch 128, the P2H10 recipe otherwise unchanged)

All new arms come from one trainer, `scripts/joint_train.py` (every recipe flag; nothing it imports is edited). Checkpoints under
`$DATA_DIR/runs/op_parity/runs/<tag>/`.

| Arm | Tags | Recipe | Trainer flags (besides the defaults = P2H10) | Rows per batch: normal / yaw-rate / hinge-only |
|:--|:--|:--|:--|:--|
| **APS** (the candidate) | `JAPS-s{0,1}` | AP2 standard + S + yaw-rate rows | `--std alpasim --yr-mass 0.1` + S | 102 / 13 / 13 |
| APS - yaw rows | `JAPSnoY-s{0,1}` | AP2 standard + S | `--std alpasim` + S | 115 / 0 / 13 |
| APS - AP2 standard | `JAPSnoA-s{0,1}` | P2H10S + yaw-rate rows (NAVSIM standard) | `--yr-mass 0.1` + S | 102 / 13 / 13 |
| APS - S | `APY10m10-AB-s{0,1}` (stored, decision 212) | AP2 standard + yaw-rate rows | (ot3_rows.py `--std alpasim --ot-mass 0.1`) | 115 / 13 / 0 |
| reference | `P2H10S-F-s{0,1}` (stored, decision 232) | S | (bd4_train.py) | 115 / 0 / 13 |
| reference, the line's base | `P2H10-F-s{0,1}` (stored, decision 205) | lambda-10 hinge | (pp_train.py) | 128 / 0 / 0 |

S = `--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape` (Amendment 6 of
experiments/body1/plans/2026-10-10-body1-prereg.md, as `P2H10S-F` was trained). Every new run adds `--label-bank` (same values bit for bit).

How the ingredients join (stated now, not tuned):
1. Each ingredient keeps its own per-batch count: 13 yaw-rate rows (round(0.1 x 128), as APY10m10) and 13 hinge-only rows (ot1 4, yr1 4, bd4 5, as
   P2H10S); the normal rows fill the rest (102 when both are present); anchor rows are 25 % of the normal rows, as before.
2. Yaw-rate rows are imitation rows (target = the logged future in the perturbed frame, never anchor rows). Under the AlpaSim standard they are
   m = 4 decisions with the AlpaSim ego definitions and the route re-expressed in their frame (as in APY10m10). The lambda-10 drivable hinge acts on
   them in their own frame (ot_rows.off_hinge, as in APY10m10 / YR10m10). As imitation rows of the body1 train logs they also carry S's agent hinge
   (shape-only), in their own frame (loss43.OffAgentHinge); they do not carry the road hinge, which belongs to hinge-only rows.
3. Hinge-only rows are unchanged from P2H10S; under the AlpaSim standard they are m = 4 decisions fed the AlpaSim ego / route re-expressed in their
   frame, and a hinge-only row whose source token has no rebuilt m = 4 route is not drawn (about 1.8 % of tokens have none).
4. The same yr1 state can be drawn as a yaw-rate imitation row and, in another draw, as a hinge-only row.
5. Every rng stream is the original trainer's: normal rows [seed, 0], decision types [seed, 0, 23], AlpaSim-standard yaw-rate rows [seed, 0, 41],
   hinge-only rows [seed, 0, 43]. The learning-rate cosine is written as each original trainer writes it.

Trainer identity (gate before any full run): 60 steps, seed 0, one shard, `joint_train.py` against pp_train.py (P2H10, and P2H10 + a memory bank),
bd4_train.py (P2H10S), ap2_train.py (AP2H10-AB), ot3_rows.py `--std alpasim` (APY10m10-AB) and `--std navsim` (YR10m10-F): every trained net and
adapter weight equal bit for bit (`joint_train.py ident`, results in `results/joint1/ident_*.json`). A difference stops the lane until it is
explained; with all six equal, the stored APY10m10-AB / P2H10S-F / P2H10-F checkpoints are what the joint trainer gives for those flags (up to
run-to-run determinism of the box, which the S-DROP lane measured as bit for bit for 60-step runs).

Steps are eager (`--compile` off): every reused checkpoint is eager, a compiled step is not bit-identical to an eager one (pp_train.py docstring), and
phase 2's identity rule needs eager runs; the saving would be about 10 minutes per run (experiments/body1/results/tokyo_3090_train.md).

## Pilot (before the fan-out)

Per new family (JAPS, JAPSnoY, JAPSnoA): seed 0, 300 steps on all 12 shards (`-P300` tags), then a 48-scene closed-loop smoke (the 48 public part001
scenes of decision 185, `c0b/lists/smoke48.txt`). Checklist; one failed item stops that family:
- training: finite losses; the batch composition of the table above in the log line; the terms of each ingredient logged (`hinge`, `agent`,
  `agent_ho`, `road_ho`; yaw-rate rows: `ot_` dev rows); step speed and card peak recorded for the declarations;
- smoke: 48 / 48 rollouts, 0 driver errors (inference or input), the driver family as registered (`ap2` for JAPS / JAPSnoY, `sh30` for JAPSnoA,
  from the driver's version line), every served plan finite. The smoke score is a pipeline check of an under-trained checkpoint and is not compared
  with anything.

## Closed loop

- Scenes: all 1491 public scenes, five lists under `$DATA_DIR/runs/alpasim/c0b/lists/`: chunk0, chunk1, chunk2 (the 700 of decision 201), cf1fresh
  (391), ot3new (400). One pool job per (checkpoint, list) through `scripts/ot2_loop.py` (`scripts/joint1_chain.sh`); the simulator is deterministic
  only for a fixed scene list (decision 211), so every checkpoint runs on exactly these lists.
- Driver: the box's default path (compiled model passes, nvJPEG off; the path of every closed-loop read since the evening of 2026-10-09: CF1, TR1,
  FIX1, BODY1), no serving switch; the driver family follows the checkpoint's record as `JEV_DRIVER=auto` does (`ap2` record with std alpasim ->
  `ap2_driver.py`, else `sh30_driver.py`).
- Stored reads reused, all on the compiled path: P2H10-F s0 / s1 on the 700 (TR1, box 894bf6e0) and on cf1fresh (CF1); APY10m10-AB s0 / s1 on
  cf1fresh and ot3new (CF1); P2H10S-F s0 / s1 on the 700 (BODY1, TR1's lists). Stored reads on the uncompiled path are rerun (stage `ref`):
  P2H10-F s0 / s1 on ot3new and APY10m10-AB s0 / s1 on the 700 (OT3, 2026-10-09, before the compiled default). Reproducibility checks in the same
  stage: P2H10-F-s0 on chunk0 (against TR1) and on cf1fresh (against CF1), APY10m10-AB-s0 on cf1fresh (against CF1); pass = every scene's score and
  zero class equal. If a check fails, every stored compiled read of that driver family is rerun (stage `refx`) and only reruns are used.
- P2H10S-F s0 / s1 on the 791 scenes outside the BODY1 read: lane nuplan1491's runs (same lists and driver; seeds 2 and 3 run there too and are
  reported only descriptively).
- New arms: 6 checkpoints x 5 lists.

## Reads (per arm; a recipe = the per-scene mean of its two seeds; 1491 scenes, 44 logs)

1. Mean scene score with a 95 % by-log bootstrap CI (B 10 000, seed 0); per-seed means.
2. Zeros by class: at-fault collision, offroad, left corridor (a zero takes the first of these flags it carries, in that order; route-fold zeros
   without a decision are listed apart); seed means.
3. At-fault events (`offroad_or_collision_at_fault`), slow scenes (0 < score < 1), mean progress.
4. The > 45 deg bucket: scenes whose navtest token's logged 4 s heading change exceeds 45 deg (experiments/body1/scripts/stop_report.turns, as
   decision 233): zeros by class, mean.
5. Fixed-vs-broken per class against P2H10-F and against each component arm: per scene the number of zero seeds of each side (0, 1, 2); fixed =
   reference 2 -> arm 0, broken = 0 -> 2, partial = any other change.
6. Guardrails through `python -m jevdrive.bench` for every new checkpoint: navtest EPDMS (12 146 tokens) and navhard two-stage on G frames
   (`<tag>@gimm`, 225 groups); HUGSIM 64 `spec` for JAPS-s0 / s1. WOD-E2E is skipped: no read path exists for the AlpaSim input standard
   (decision 217: "AP2H10-AB not run (no WOD mapping for its input standard)"), and writing one is new code.

## Registered lines (from the lane brief; not changed after any read)

**Candidate for submission #2** = all three:
- **L1** zeros (at-fault collision + offroad + corridor) on the 1491 fewer than P2H10-F with a paired scene bootstrap CI excluding 0: D = sum over
  scenes of (z_arm - z_P2H10F), z = the seed-mean zero indicator of the three classes; D < 0 and the 95 % CI of D (scenes resampled with
  replacement, the same draw for both sides, B 10 000, seed 0) has upper bound < 0.
- **L2** mean scene score not below P2H10-F by more than 0.002: point estimate of the difference of the two-seed means >= -0.002 (its by-log CI is
  reported).
- **L3** no guardrail drop: navtest EPDMS (two-seed means) minus P2H10S-F >= -0.2 (point estimate); navhard combined (two-seed means) against
  P2H10-F not below with CI (fails only if the whole by-log CI of the difference is below 0); HUGSIM 64 HD (`spec`, two-seed means) minus P2H10-F
  >= -0.03 (point estimate).

**Evaluated arms**: JAPS, JAPSnoY, JAPSnoA, APY10m10-AB and P2H10S-F (the S-only arm is a two-drop arm of the same family). HUGSIM is read for JAPS
as registered; if the rule below picks an arm without a HUGSIM read, its HUGSIM 64 `spec` (two seeds) is run before the recommendation is written
and L3's HUGSIM clause is applied to it (P2H10S-F has one: decision 236).

**Which arm**: the recommended arm is the one with the fewest at-fault collision + offroad zeros (seed mean, 1491) among the arms meeting L1-L3; a
tie goes to the higher mean scene score. If a drop-one arm meets the line and APS does not, that arm is the recommendation (the rule gives this). If
no arm meets the line: no candidate; the report names the closest arm and the line it misses. The user decides what is submitted.

**Non-additivity read** (decision 244's wording, applied to APS's effects over P2H10-F): for each quantity Q in {zeros of the three classes,
at-fault collision + offroad zeros, corridor zeros, mean scene score} let E = APS - P2H10-F; for ingredient c with drop-one arm X_c:
- **carries** Q: X_c - APS gives back at least half of E (opposite sign to E, |X_c - APS| >= |E| / 2) with a CI excluding 0;
- **is not needed** for Q: X_c - P2H10-F keeps at least half of E (same sign, >= |E| / 2) with a CI excluding 0;
- **hurts** Q: X_c - APS lies in E's direction (dropping c improves Q beyond APS) with a CI excluding 0;
- otherwise **unresolved**; if E's own CI includes 0, the read for Q is "no effect of APS to attribute" and X_c - APS is reported.
Zero counts use L1's scene bootstrap, the mean score its by-log CI. One sentence per ingredient states the result. Zero composition (zeros by class,
overlap of zero sets of APS and each drop-one arm) and fixed-vs-broken (APS against each drop-one arm) are reported next to it.

## How this differs from earlier decisions

- **212 (OT3)**: the AlpaSim standard and off-track / yaw rows were separate recipes on the lambda-10 base, each against P2H10 on 700 scenes with a
  mean-score line; here they are joined with the body lesson in one trainer, read on all 1491 scenes with drop-one arms and a zero-count line.
- **213**: yaw-rate rows alone, NAVSIM standard; here they enter under the AlpaSim standard and next to S, and as S-carrying imitation rows.
- **216 (CF1)**: APY10m10 against P2H10 by mean score on 791 / 1491; here APY10m10-AB is the "minus S" arm (stored checkpoints, its 700 rerun on the
  compiled path), and the registered quantity is the zero count with a guardrail set.
- **233**: P2H10S read descriptively (4 seeds, 700 scenes); here S is an ingredient of the joint training and P2H10S-F is read on all 1491.
- **244 (S-DROP)**: drop-ones inside S, open-loop boards; here the drop-ones are across ingredients (AP2, S, yaw rows), read in closed loop.

## Known limits, stated before the data

- Every one of the 1491 public scenes has been read by recipes of this family (decision 216, amendment 1): this is a development set; no fresh
  confirmation is possible locally.
- Two seeds per arm (the base's seed spread on 700 scenes: mean 0.9446 to 0.9498, zeros 23 to 26, decision 232); five evaluated arms against one line
  without multiplicity correction.
- Local render; the driver's compiled path, while the submission image serves the interpreted path (tolerance-level difference, decision 215 /
  LAT1: one scene of 700).
- WOD-E2E not read (above).

## Cost, before running (card-hours; a simulator stack books 32 GB, two per card)

| Item | Card-hours |
|:--|--:|
| identity (12 x 60 steps) and 3 pilots (300 steps) + 3 smokes (48 scenes) | 1.0 |
| 6 full trainings (eager; AlpaSim standard about 40 min, NAVSIM about 30 min alone) | 4.0 |
| closed loop: new arms 6 x 1491, `ref` 3 216 scenes, P2H10S-F s0-s3 on 791 (3 164) = 15 326 scenes at about 3.7 s per scene and stack | 8.0 (refx: +2) |
| bench: navtest + navhard for 6 checkpoints, HUGSIM 64 for 2 | 3.0 |
| **phase 1** | **16 (worst case 18)** |

With phase 2 (its own pre-registration, about 9) the lane stays inside the 35 card-hours of the brief.

## Execution

`scripts/joint_ident.sh` (identity), `scripts/joint1_chain.sh <stage>` (closed loop: `ref`, `refx`, `arms`, `smoke <tag> <driver>`), trainings as pool
jobs (`--train`, one job per arm and seed, closed-loop jobs of an arm wait for its checkpoint). State under `$DATA_DIR/runs/alpasim/joint1/`. Read-out
`scripts/joint1_report.py` (CPU; c0b_report / stop_report helpers, imported) -> `results/joint1/`; write-up `results/joint1.md`, decision draft.
