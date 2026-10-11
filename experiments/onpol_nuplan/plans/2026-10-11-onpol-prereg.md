# ONPOL-NUPLAN: self-consequence labels against imitation labels at the same on-policy states (pre-registration, 2026-10-11)

Lane ONPOL-NUPLAN (main GPU box, AlpaSim nuPlan track). Follows decisions 220, 232, 233, 244, 247, 253, 254, 257 and
`research/self-consequence/design.md` (sections 3 to 5). Written and pushed before any closed-loop score, pilot read or row statistic of
this lane is read. Local closed loop only; nothing is submitted. The encoder stays frozen in every arm; no privileged input at inference
(simulator boxes, map rasters and logged paths are training labels only). The user is asleep: every choice below that the task sheet
(`tmp/onpol-nuplan-prompt.md`) left open is a design proposal of the lane's main session, marked (P), not a measurement.
A sister lane ONPOL-PAI runs the same design on the PAI track (Tokyo box, `experiments/onpol_pai/`).

## Question

Decision 220: the closed-loop failures happen at states 0.5 to 2.4 m off the log or 13 to 39 m ahead of it, which the offline rows of
lesson S never cover. The generic fix is RoaD / DAgger: roll out, label the visited states with an expert path, imitate. The paper's
version changes the label only: at the same visited states the label is the consequence of the model's own plan (the S hinges, and the L
items the labels allow), with no imitation target. **At identical on-policy states, do self-consequence labels beat imitation labels,
add to them, or lose?**

## Data

**Split.** The 1491 public nuPlan scenes (`c0b` chunk0-2 = 700, `cf1fresh` 391, `ot3new` 400), scene id `<log>-<token>`, unit = log
(`cf1_report.log_of`, the unit of every log-clustered CI of this track). Sort the logs, permute with `numpy.random.default_rng(0)`, take
logs in that order into `train` until it holds at least 1000 scenes; the rest is `held`. The two lists are written to
`experiments/onpol_nuplan/results/split/{train,held}.txt` and registered (`jevdrive.data.splits`, `alpasim_nuplan/onpol-train`,
`alpasim_nuplan/onpol-held`) before any rollout. `held` is the only closed-loop read of this lane; no row comes from it.

**Rollouts.** `P2H10S-F-s0` and `-s1` on `train`, unchanged `sh30` driver path (compiled passes, no serving switch), with a per-decision
dump of what the driver fed the model (the 8 history slots: MTGS 2 Hz keyframes and the synthesised warp slots, cold start at decision 0
included), the ego vector, command, camera and the served plan. Staged 1 -> 10 -> all. Estimate: 2 x ~1000 scenes, 0.5 to 0.85
card-hours each (0.7 to 1.25 per 1491, design.md section 11).

**Rows.** One per visited decision (10 per scene), both seeds' rollouts pooled into one pool that every training of every arm draws from.
Inputs = exactly the dumped model inputs. Gate R1 (inputs): the trainer's forward of `P2H10S-F-s<seed>` on a row reproduces the served
plan of that decision (first 15 plan points / the 8 served poses within 0.03 m on at least 99 % of rows, plus a wrong-seed control;
docs/long-runs.md equality rule). Gate R2 (frames): at the log-replay decisions the simulator's ego pose equals the logged pose at that
time within 0.3 m, and the simulator's actor boxes equal the logged boxes of the same tracks within 0.5 m (alignment of simulator time
and frame with the nuPlan log and map). A failed R1 or R2 stops the lane.

A row is excluded, for every arm alike, when (a) the driver aborted the call, (b) at its time the ego footprint already overlaps an
actor box or the rear axle is more than 4.0 m from the logged path (the scorer's corridor; the scene is already a zero and the labels
are degenerate), or (c) one of its label sets is missing (logged path to t + 4 s, actor boxes, raster). All arms draw the same rows.
Reported before any training read: rows, exclusions by reason, rows more than 0.5 m off the logged path, rows more than 5 m ahead of or
behind the log's position at the same time, per-decision-index counts, and per label the share of rows whose stored own plan activates it.

## Labels at a visited state (time t, rear-axle pose p; all in the row's own frame)

**Imitation (RoaD / DAgger baseline) (P).** G = the logged rear-axle path of the scene's nuPlan log (dense, arc length s). Project p on G:
arc s0, signed lateral offset e0, heading error psi_e (clipped to +-30 deg). Progress "from the log": D_k = s_log(t + 0.5 k) - s_log(t),
k = 1..8, the log's own travel over the same clock interval. Lateral return, critically damped in travelled distance d:
`e(d) = (e0 + (tan(psi_e) + e0 / Lr) d) exp(-d / Lr)`, `Lr = max(4 m, 1.0 s x D_8 / 4 s)`. Target pose k = the point of G at s0 + D_k,
moved by e(D_k) along G's left normal, heading = the tangent of that curve. The path is recovered; the longitudinal offset to the log's
timeline is not (the task sheet: "recovery path back to the logged path, speed from the log"). The alternative of the offline yaw-rate
rows (the logged future itself re-expressed in the perturbed frame) is not used: at 2 m or 20 m off it is a jump in the first 0.5 s.

**Self-consequence S.** Exactly S's two hinges on the model's own plan, as on S's hinge-only rows (`loss43.Losses43`, shape-only,
`--agent-lam 10`, margin 0.3 m inside the ego corridor and 0 outside, `--road-lam 10`, `--road-margin 0.3`): the swept footprint against
the simulator's actor boxes over t .. t + 4 s (the vehicles the scorer collides against; same array layout as
`runs/op_parity/agent_labels`, K = 32), and the road-and-lane raster (layers ROADBLOCK, ROADBLOCK_CONNECTOR, INTERSECTION, LANE,
LANE_CONNECTOR, the `runs/body1/labels_road` format) built at the visited pose from the scene's map.

**Self-consequence L (only if ready, rule below) (P).** design.md section 3.2 through `timing_only`, lambda 10 each, with the changes a
state off the log's timeline needs, since "log-consistent" has no meaning there: route = G from the projection point on; A (lead contact)
on the simulator's actors with margin `m_j = min(2.0 m, max(g_j(t) - 0.5 m, 0))` (no closer than 2 m, or than now); B (stop line) for
traffic-light lines only, red at the nearest keyframe, `theta = max(d_l - 1.0 m, s_f(t))`, stop-sign lines masked (their rule is the
log driver's crossing time); C (speed into a >= 45 deg turn) unchanged (the cap is by arc position); P (progress guard) on open rows
with D_k as the reference. None of this has been trained before, offline or on-policy.

L inclusion rule: a separate agent builds the L labels and loss while the critical path (rows, S, imitation, trainer) proceeds. L enters
ON-SELF and ON-BOTH only if, **before the pilots are submitted**, (i) its code passed the audit, (ii) `timing_only` passes a numerical
gradient check (no cross-heading component), (iii) each item is active on at least 0.2 % of the pool rows on the stored own plans (an
item below that is left out and named). L never delays the pilots. If it misses them, ON-SELF tonight is S only, said so in every
result, and an L arm may follow later as exploratory with its own pilot.

## Arms (two seeds each; tags carry the family suffix `-ON`)

All arms start from the archived `P2H10S-F-s<seed>` weights (net and adapter; seed 0 and 1) and train with the S recipe unchanged
(`--ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank`, eager, 10 000 steps x 128,
warmup 300, cosine, fresh optimiser) through `experiments/onpol_nuplan/scripts/on_train.py`, a wrapper that imports `joint_train.py`
unedited and adds the on-policy rows. 13 rows of every batch (round(0.1 x 128), the yaw-rate rows' share) are on-policy rows drawn
uniformly from the pool with their own rng stream `[seed, 0, 53]`: the same row indices at every step in every arm. Batch = 102 normal
(25 % anchor, as before) / 13 S hinge-only / 13 on-policy.

| Arm | Tags | On-policy rows carry | How the loss counts them |
|:--|:--|:--|:--|
| ON-IMIT | `SIMIT-ON-s{0,1}` | the imitation target | imitation rows (weight of one imitation row); no hinge of any kind |
| ON-SELF | `SSELF-ON-s{0,1}` | S hinges (and L if included) | hinge-only rows, x `ho_w` 3, as S's own hinge-only rows; no imitation target |
| ON-BOTH | `SBOTH-ON-s{0,1}` | both | imitation row + the hinges x 3 |
| ON-NONE (P) | `SNONE-ON-s{0,1}` | (no on-policy rows; 115 / 13 / 0) | the S recipe continued from the same start with the same schedule |
| control | `P2H10S-F-s{0,1,2,3}` (stored) | | |

ON-NONE separates "the rows" from "10 000 more steps"; it is not in the task sheet. Eight trainings fit the budget: if L is included,
the two ON-NONE runs are replaced by `SSELFnoL-ON-s{0,1}` (ON-SELF with the S hinges only, the drop-one arm of L as a group) and the
missing continuation control is stated as a limit. Per-item drop-one arms of L do not fit tonight.

Trainer gates before any pilot: T1 identity, 60 steps, seed 0, one shard, on-policy rows off and no `--init`: bit for bit `joint_train.py`
with the S flags; T2 `--init P2H10S-F-s0` loads every tensor of the checkpoint (dev ADE at step 0 = the archived 0.545 to 3 decimals);
T3 the three arms draw identical on-policy row indices (logged hash over the first 60 steps).

These checkpoints have seen renders of navtest logs: **no `-ON` checkpoint is ever reported on navtest or navhard**.

## Pilot (3 000 steps, seed 0, all shards, tags `<family>-ON-P-s0`; four pilots including ON-NONE)

Gates per arm, read on the trainer's dev set (held navtrain logs) and logs: finite losses; composition 102 / 13 / 13 in the log line;
dev ADE <= the ON-NONE pilot's + 0.010 m (the archived S, 0.545, is shown beside it); `dev_drift_off` <= 0.30; then a 48-scene
closed-loop smoke on scenes of `train` (48 / 48 rollouts, 0 non-finite plans, 0 driver errors; its score is not a read). Descriptive:
on the on-policy rows, hinge activation rates and the imitation error at steps 2701-3000 against 201-400.
A failed gate stops that arm, with no retry tonight unless the cause is mechanical. If ON-IMIT or ON-SELF stops, the contrast cannot be
read and the pilots are the result. Full runs: one first (`SSELF-ON-s0`), its first 300 steps inspected, then the rest.

## Read (the `held` scenes only)

Each arm's two seeds and the control's four seeds are run on the `held` list in this lane, one pool job per checkpoint through
`ot2_loop.py`, same driver path and code version. The stored reads of the control on these scenes are compared with the reruns
(descriptive reproducibility check). Zero classes, route-fold scenes, slow scenes, fixed / broken: as `joint1_report.py`.
Reported per arm: per-seed and seed-mean score, zeros by kind (at-fault collision, offroad, corridor), at-fault zeros (collision +
offroad), at-fault events, slow scenes, progress; paired differences (per-scene seed means) with scene-bootstrap and log-clustered
CIs (10 000 draws) against the control, against ON-NONE and between arms; fixed and broken scenes (zero in both seeds of one side, in
neither of the other).

Null of fixed - broken: the control's own seed pairs, the three 2-against-2 splits of s0-s3 in both directions (6 values).

- **H (hard gate, every arm):** seed-mean at-fault zeros above the control's four-seed mean = the arm is a negative, whatever its mean.
- **"Self beats imitation":** ON-SELF has fewer seed-mean zeros than ON-IMIT in each of the three kinds and in at-fault zeros, ON-SELF
  passes H, and ON-SELF's fixed - broken against the control (s0, s1) exceeds the maximum of the null.
- **"Imitation beats self":** the same sentence with the two arms exchanged.
- **"Adds":** ON-BOTH passes H, has fewer seed-mean at-fault zeros and fewer zeros in total than each of ON-IMIT and ON-SELF, and its
  fixed - broken against the control exceeds the maximum of the null.
- Anything else: **"not separated"**, reported with the paired CIs of ON-SELF - ON-IMIT (zeros in total, at-fault zeros, mean score).
- Descriptive: every arm against ON-NONE (rows against steps); the > 45 deg bucket and corridor zeros (decision 233's wide turns);
  rollout clips of every fixed and broken scene of the three arms against the control.

Known before the read: `held` is about 490 scenes and the control has about 25 zeros per seed there (collision about 2.5, offroad 12,
corridor 9 by proportion of nuplan1491_pick.md), so "fewer in each kind" needs differences of single scenes to line up; two seeds per
arm; no multiplicity correction over three verdict sentences. All 1491 scenes are development scenes of this family (JOINT-1 limits),
`held` included: it is held out from the on-policy rows, not from earlier recipe selection.

## Failure branches (fixed now)

- R1 or R2 fails and cannot be fixed as a mechanical error: no rows, the lane stops with the gate tables as its result.
- L is not ready at pilot time: ON-SELF = S hinges only (stated in the README and the decision).
- An arm fails a pilot gate: stopped, see Pilot. An arm fails H: negative, still reported in full.
- ENOSPC on the data disk: stop that step and report; nothing is deleted or moved.
- Budget (about 20 card-hours for the night, step 0 included; step 0 is estimated at 6 to 7). If the tally projects above 22, cut in
  this order: ON-NONE's closed-loop read; ON-NONE's trainings; the reruns of control s2 / s3 (then their stored reads are used, provided
  the reruns of s0 / s1 reproduce the stored reads with no zero flip).

## Budget (estimates before the runs; card-hours)

| Item | Estimate | Basis |
|:--|--:|:--|
| Step 0 (separate section of the report): 4 trainings + 4 x 1491 scenes | 6 - 7 | JOINT-1: 0.6 per training, 0.7 - 1.25 per 1491 scenes |
| Smoke and staged rollouts, 2 x ~1000 train scenes with dump | 1.2 - 1.8 | same |
| Row forward (gate R1, shipped-model teacher) | < 0.2 | BODY1 forwards, 203 s for 485 k states |
| Trainer gates, 4 pilots (3 000 steps) + 48-scene smokes | 1.0 | joint1.md: 31 - 47 min per 10 000 steps |
| 8 trainings | 4.5 - 5.5 | same |
| `held` read: 8 arm + 4 control checkpoints x ~490 scenes | 3 - 5 | 0.7 - 1.25 per 1491 scenes |
| Clips | < 0.3 | COL1 clips |
| **Total** | **16 - 21** | |

## Deliverables

`experiments/onpol_nuplan/README.md` (from experiments/TEMPLATE.md), one INDEX.md line, `results/` (split, row report, gates, pilot
table, read-out, per-scene csv, clips index), a decision draft with the standing stated as it is, `tmp/onpol-nuplan-state.md`.

## Amendment 1 (2026-10-11, after the 10-scene smoke, before the full rollout, any training and any closed-loop read)

Gate R1 as written failed on its second clause and is rewritten; the smoke numbers that prompted this are in
`results/rows/smoke10*/`. Measured on 100 smoke rows: first 15 plan points within 0.03 m on 0.99 of rows, the 8 served poses on 0.80
(maximum 0.045 m); against the same scenes served with interpreted passes (`SH30_COMPILE=0`, smoke10e) 1.00 / 0.96, and forwarded one
row at a time with the driver's batch shape the interpreted plans are reproduced bit for bit on 100 of 100 rows; the wrong-seed control
(`-s1` weights) is within 0.03 m on 0.04 of rows (median 0.123 m). So the stored inputs are the driver's; what the pose clause measured
is fp16 rounding of the compiled passes and of the batch size, the case docs/long-runs.md's equality rule warns about.

R1 (amended): (a) first 15 plan points within 0.03 m on at least 99 % of the pool rows against the served (compiled) plans; (b) on a
10-scene interpreted control, rows forwarded with the driver's batch shape reproduce the served plan bit for bit on at least 99 % of
rows; (c) the wrong-seed control stays below 20 %. The 8-pose share is reported as a diagnostic. (b) is met by the smoke; (a) and (c)
are read again on the full pool. The pool keeps rows of the compiled path (the path every read of this lane and the image-independent
box reads use).
