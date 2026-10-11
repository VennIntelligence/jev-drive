# ONPOL-PAI pre-registration: on-policy self-consequence rows against on-policy imitation rows, AlpaSim PAI track

Written 2026-10-11, 20:35 to 20:50 JST, by the main session of lane ONPOL-PAI (task: `tmp/onpol-pai-prompt.md`), before any score of a
new checkpoint exists and before any score of the training-scene rollouts was read (the 4-scene stage had finished; only its completion
and driver-error counts were looked at). The user is asleep: every choice below that the task left open is marked **(assumed)** and is
the main session's. Thresholds are design choices, not measurements. Wall-clock figures are estimates unless a source is named.

## 1. Question

The frozen production driver (openpilot Cinque, encoder frozen in every arm) with the body lesson S (`P2H10S-F`, decision 247: PAI
0.374 -> 0.484 on 281 scenes) has never been trained on the PAI serving input or on states its own closed loop visits. At the states
its own rollouts visit, does labelling **the consequence of its own plan** (ON-SELF) do better on the PAI board than labelling the same
states with **the path back to the log** (ON-IMIT, the RoaD / DAgger baseline)? ON-BOTH has both labels.

## 2. Arms

All arms start from the archived weights of `P2H10S-F-s0` (adapter + plan path as its `ckpt-final.pt` holds them), fresh optimizer, the
S recipe unchanged (`--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape`, warmup 300,
`--label-bank --compile` as `experiments/body1/results/tokyo_3090_train.md`), 10 000 steps, batch 128. One seed per arm tonight (seed 0).

| Arm | Tag | Batch: normal / S hinge-only / on-policy | Terms on the on-policy rows |
|---|---|---|---|
| ON-IMIT | `P2H10S-IMIT-ON-F-s0` | 99 / 13 / 16 | the recipe's imitation loss, target = recovery path (section 4) |
| ON-SELF | `P2H10S-SELF-ON-F-s0` | 99 / 13 / 16 | hinges on the own plan (section 5); no imitation target |
| ON-BOTH | `P2H10S-BOTH-ON-F-s0` | 99 / 13 / 16 | both |
| FT-CTRL **(assumed, added)** | `P2H10S-FT-F-s0` | 115 / 13 / 0 | none: the S recipe continued for the same steps from the same weights |

- The on-policy rows are the same states, drawn by the same rng stream, in the three ON arms; only the terms differ.
- 16 on-policy rows per batch **(assumed)**: 12 from pass 1 (visited states) and 4 from pass 2 (on-log states) if pass 2 exists, else 16
  from pass 1. On-policy rows are never anchor rows. The S offline rows (99 + 13) are the anchor the task asks for.
- FT-CTRL is not in the task. It is trained tonight (23 min of one card) and read on the 281 last, if the night allows, because the
  archived control cannot tell "10 000 more steps" from "on-policy rows". The contrast ON-SELF against ON-IMIT does not need it.
- Tags: the family suffix `-ON` marks checkpoints that saw PAI rollouts; they are reported on PAI and the challenge boards only.
  Pilot tags have `-P-` in place of `-F-`. If the driver's loader rejects these names, the nearest accepted names with `-ON` are used
  and written under "As run".
- Control: archived `P2H10S-F` seeds 0 to 3 on the 281 (`experiments/alpasim/results/pai_seeds281.md`), read before this lane existed.

## 3. Split and data

- Test = the 281 scenes of `/data/runs/alpasim/pai_full/s_*.tsv` (`results/split/test281.tsv`). Never rolled out for training, never trained on.
- Train pool = the other 160 scenes of the 441 curated-validation scenes (`results/split/train160.tsv`; resolved as `pai_scenes.py`
  does; all 160 `.usdz` are on the Tokyo box). Launch order = `random.Random(0).shuffle(sorted(ids))`: `stageA.txt` 4, `stageB.txt` 12,
  `stageC.txt` 144.
- **Held scenes (assumed)**: stage A + stage B (16 scenes, a random prefix) give no training row; they are the "held rows" of the
  pilot's open-loop read. Training rows come from the 144 of stage C.
- Pass 1: `P2H10S-F-s0`, served configuration `JEV_VCONT=1.0 JEV_LEAD=1`, CONC 4, image and scripts as the stored 281 rows
  (frozen clone at c58f7136). **The driver is not changed (assumed, deviation from the task's suggestion):** every `rollout.asl` is kept
  on Tokyo and holds the driver-side messages, so the model's inputs at every decision are reconstructed by replaying them through the
  driver's own functions; `PAI_DUMP` (existing switch, one dump per second, on for every session) gives the bit-for-bit check of that
  replay (gate G-in). No new flag, so the stored 281 rows stay reproducible by construction.
- Pass 2 (states along the logged path, a log-replay driver variant): opportunistic. It runs on a card that would otherwise idle while
  the row pipeline and trainer are written; the scenes finished when the full trainings are launched are used (same for every arm), the
  count is reported. If the variant is not working within about an hour of work, pass 2 is skipped (task rule).
- Rows: one per driver decision (10 Hz, about 200 per scene). Excluded in every arm: rows at or after the first scorer event of the
  rollout (at-fault collision, offroad, left corridor): the states after a failure are not states to learn from. Cold-start rows (the
  first 1.4 s, slots filled by back-warp) are kept: hand-over is part of the track.
- Reported before training: row count, rows per scene, class composition, share of visited states more than 0.5 m off the log.

## 4. ON-IMIT target (RoaD / DAgger recovery; fixed here)

Pose k (t = 0.5 k s, k = 1..8) lies on the logged path at arc `s_ego + (s_log(t0 + t) - s_log(t0))`: the ego's own projection on the
path, advanced by what the logged driver advanced over the same scene time (speed from the log). It is displaced laterally by
`e0 * max(0, 1 - d_k / D)`, `d_k` the arc advanced, `D = max(8 m, 2 s * v0)`, `e0` the ego's signed offset at t0; heading = direction of
that curve. On pass-2 rows the target is the log's own future. Loss and weight = the recipe's imitation term.
Known property, accepted: the target ignores the simulator's other actors (that is the baseline's definition).

## 5. ON-SELF terms (the model's own plan against privileged labels of the same rollout; fixed here)

Weights as S's hinge-only rows: lam 10 per term, row weight `ho_w` 3. Shape-only / timing-only projections as S and the SELF-CON design
(`research/self-consequence/design.md` 3.1): lateral terms cannot change the speed profile, longitudinal terms cannot change the path.

| Term | Label | Definition | Projection |
|---|---|---|---|
| agent | simulator actor boxes of the rollout at t0 + 0..4 s | S's swept-footprint agent hinge, PAI ego dimensions, margin 0.3 m inside the ego corridor, 0 outside | shape-only |
| road edge | the map's road-edge polylines, 1 m segments as static thin boxes, the 96 nearest within 40 m | the same hinge, margin 0.3 m | shape-only |
| corridor | the logged path (finite) | route hinge: relu(distance of the plan poses to the path - 3.0 m); the scorer's band is 4 m | shape-only |
| B stop line | the map's static stop lines crossing the logged path | design 3.2's stop-sign rule: the plan's front bumper must not pass the line (minus 1.0 m, relaxed to the log's own furthest point) at plan times before the logged driver passed it; never passed in the clip = the whole window | timing-only |
| C speed into turn | first turn peak >= 45 deg of the logged path ahead | design 3.2: v(t) <= max(v_log + 1.0, 1.15 v_log) at the same arc, window s_pk - 10 m .. s_pk + 3 m | timing-only |
| P guard | the log's same-time progress | design 3.2: relu(0.95 s_log - 0.5 m - s_plan); only on "open" rows (no stop line, no turn peak within 60 m, no actor box within 2 m of the path in the next 60 m) | timing-only |

- L item A (lead in lane) is not a term: the task lists B and C as the labelable L items and the served configuration carries
  openpilot's lead limit (decision 226). The agent term sees the same boxes with a shape-only gradient.
- If the stop lines cannot be read from the map on the Tokyo box tonight, ON-SELF runs without B and the report says so (task rule).
  No term is dropped for any other reason.
- P is one-sided and relative to the log; it is the progress guard of the L package, not an imitation target. **(assumed)** that this
  is inside "no imitation target on the on-policy rows"; the held read reports arc-length ratios so its effect is visible.

## 6. Gates before training

- **G-in** (rows): replayed model frames, slot validity and ego inputs equal the driver's `PAI_DUMP` dumps bit for bit on every dumped
  decision (frames) and to the log's rounding (ego). Fail = fix the replay; no rows are used before it passes.
- **G-id** (trainer input path): the training-path forward of `P2H10S-F-s0` on the rows reproduces the driver's logged adapter plan
  (`poses_model`), first 4 s, share of rows over 0.03 m, with a wrong-checkpoint control (`docs/long-runs.md`, fp16 equality gates).
  Line: <= 1 % of rows over 0.03 m and the wrong checkpoint clearly apart.
- **G-log**: every ON-SELF term is 0 on >= 99 % of rows when the log's own future is put in as the plan (pass-2 rows, or pass-1 rows
  within 0.3 m of the log); exceptions are listed.
- **G-load**: a saved checkpoint loads through the unchanged driver image and plans.
- Opus audit of the row pipeline and the trainer (a second agent that did not write them) before the full trainings.
- Staged launch of pass 1: 4 scenes -> 12 -> 144. Checklist per stage: every scene has a result, `inference_error = input_error = 0`,
  198 to 200 decisions per scene, a dump per second, every `rollout.asl` present.

## 7. Pilot (3 000 steps, seed 0, one run per ON arm; tags `-P-`)

Read on the 16 held scenes' pass-1 rows (not after an event) and on the trainer's dev rows, against `P2H10S-F-s0`:

- hygiene: finite losses; every term logged.
- forgetting: `dev_drift_off` <= 0.30; navtrain dev ADE <= 0.575 (S final 0.545 + 0.03).
- learnable: ON-IMIT and ON-BOTH: ADE to the recovery target on held rows below S's. ON-SELF and ON-BOTH: the summed rate of rows
  with an active term (agent, road edge, corridor, B, C) on held rows below S's.
- reported, no line: 4 s arc-length ratio against the served plan on held rows (does an arm drive shorter or longer).
- **Clear negative = stop that arm**: NaN; either forgetting line missed; or its "learnable" quantity not below S's. No retry tonight.
  The other arms continue. If ON-SELF or ON-IMIT stops, the lane's contrast cannot be read and the report says so.

## 8. Full runs and the 281 read

- Trainings: SELF and IMIT in parallel (one per card), then BOTH and FT-CTRL. Then one `pai_queue.sh` queue over the 281 scenes,
  served configuration, CONC 4, CHUNK 36 (the chunks of the control's s2 / s3 runs), order ON-SELF, ON-IMIT, ON-BOTH, FT-CTRL.
  Estimate: 2.1 h per checkpoint on two cards (lane PAI-281: 704 scenes in 5 h 51 min), so SELF + IMIT about 4.2 h, all four about 8.5 h.
- Read-out: `pai_eval_report.py --detail` (rows with zero kinds; pairs against each control seed and against `P2H10S-F-s0`, the parent;
  scene-bootstrap 95 % CI, 10 000 draws, seed 0; zero -> scored and scored -> zero counts).
- Definitions. Zero kinds = the scorer's flags on zero-score scenes (at-fault collision, offroad, left corridor). **At-fault zeros** =
  zero-score scenes with the at-fault collision or the offroad flag (a scene with both counts once). Fixed = zero for the control seed and
  scored for the arm; broken = the reverse.
- Control numbers these lines use (already known): S seeds 0 to 3 on the 281: zeros 119 / 110 / 109 / 107; at-fault collision
  18 / 14 / 17 / 12; offroad 33 / 29 / 28 / 28; left corridor 70 / 72 / 70 / 70. Seed-pair null of fixed minus broken (the six S seed
  pairs, `results/pai281/seeds281.md`): +9, +10, +12, +1, +3, +2, so **null = 12** (largest in absolute value, either direction). The
  control's at-fault zero count per seed (union of the two flags) is computed from `seeds281_scenes.tsv` / the run dirs before the new
  rows are read and written under "As run".

### Lines

- **H (hard gate, every arm)**: the arm's at-fault zeros are not more than the parent's (`P2H10S-F-s0`) and not more than the largest of
  the four control seeds. More than either = **negative for that arm whatever its mean** (OPTED's at-fault collisions rose under
  on-policy training). The mean over control seeds is reported next to it.
- **V "self beats imitation"** = all of: (a) ON-SELF has fewer zeros of each kind (at-fault collision, offroad, left corridor) than
  ON-IMIT; (b) ON-SELF has fewer at-fault zeros than ON-IMIT; (c) ON-SELF's fixed minus broken against the parent `P2H10S-F-s0` exceeds
  12; (d) ON-SELF passes H. Ties in (a) or (b) = not met.
- Reported next to V, no line: ON-SELF minus ON-IMIT paired mean difference with CI; each arm minus each control seed and minus the
  four-seed mean; fixed minus broken of each arm against each control seed; ON-BOTH on every table; FT-CTRL if it was read.
- Standing: one seed per arm. Whatever the outcome, the entry is "one seed, descriptive" until seed 1 (tomorrow) and FT-CTRL are read;
  if FT-CTRL alone reaches (c), the fixed-minus-broken part of V is credited to fine-tuning, not to the rows.

## 9. Failure branches

- G-in or G-id cannot be passed tonight: no training; the report gives the rollouts, the row pipeline state and what blocks.
- Pass 2 skipped: every arm uses 16 pass-1 rows per batch; ON-IMIT has imitation labels only at visited states.
- B not labelable: ON-SELF without B, said in the first line of the results.
- An arm stops at the pilot: section 7.
- H missed by ON-SELF: negative for the paper's version on this track, reported as such; no rescue variant tonight.
- V not met with H passed: "no separation of self from imitation at one seed"; the components are reported; nothing is promoted.
- ON-IMIT misses H and ON-SELF passes it: reported as the OPTED pattern repeating for imitation only; still one seed.
- Time runs out: the queue keeps running in tmux; arms are read in the order of section 8, and the morning state file says which rows exist.
- ENOSPC: report and stop that step (the disk is the user's; nothing is deleted). Card 0 idle above 300 W: `CARDS=1` and write it down.

## 10. Budget (estimates)

| Step | Cards | Wall |
|---|---|---|
| Pass 1, 160 scenes | 2 | about 1.3 h (133 scenes / h, docs/tokyo-box.md) |
| Pass 2, up to 160 scenes | 1, idle time only | about 2.3 h on one card, cut at training launch |
| Row extraction (CPU) and token cache (GPU) | 1 | under 30 min (to be measured on 4 scenes first) |
| Pilots, 3 x 3 000 steps | 1 to 2 | about 25 min |
| Full trainings, 4 x 10 000 steps | 2 | about 50 min (23 min each, two at a time) |
| 281 read, per checkpoint | 2 | 2.1 h; four checkpoints 8.5 h |

Disk: about 0.5 GB per rolled scene (0.47 measured on stage A + 38 MB of dumps): pass 1 80 GB, pass 2 up to 80 GB, the 281 reads 130 GB
each; 1.5 TB free.

## As run

(filled as the night goes)
