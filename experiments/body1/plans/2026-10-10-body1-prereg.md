# BODY1: teach the student its own body, stage 1 on navtrain (pre-registration, 2026-10-10)

Lane BODY1. Written and pushed before any number of this lane is computed. Goal: raise the AlpaSim nuPlan-track score of the
P2H10 driver by adding one lesson: the student predicts whether the swept footprint of its own plan will touch an object or
leave the road, and acts on it (re-plan or stop). Score first: every stage ends with a servable checkpoint and a closed-loop
read; a stage that does not help is dropped and the previous checkpoint stays.

Fixed: base model openpilot Cinque; no WA-JEPA or other borrowed strong weights or features (a small generic initialisation
for a light branch is allowed and named where used). Test-time inputs: front camera, ego state, command. Agent boxes, the map,
the drivable raster, logged futures and simulator state are labels only. The lesson is added to P2H10 (then to its
AlpaSim-input-standard sibling): hinge 10 / 0.3 m, anchor, fed acceleration and the ego-state speed profile all stay.
Training rows come from navtrain only, never from AlpaSim public scenes (navtest logs). Nothing is registered or submitted.

## 1. How this differs from settled reads (none is re-tested)

| Decision | What it settled | What BODY1 does instead |
|---|---|---|
| 220 | every at-fault collision is already in the served plan's sweep; no frozen geometric head separates it (<= 0.72; simulator current boxes 0.73); privileged stop 2 m before the first intersection avoids 20 / 22 at 1 % clean cost | takes its label, its 122 / 1 827 decisions and its AUC reader as the offline gate; trains the missing signal |
| 219, 218, FIX1 | lead outputs are discarded; a 1-D lead cap is a net loss on nuPlan (-0.086, slow 112 -> 271); the speed profile is a function of ego state | no lead cap, no change of the speed profile's source; the stop acts only on a predicted 2-D sweep contact and is held to the slow-scene line |
| 158 | agent hinge on the plan head over frozen features: loss flat at pilot scale | the consequence is a predicted output queried per trajectory (read by a stop / re-plan), trained on dense synthetic queries and off-track states; a plan-side loss is only a conditional later arm (4.3) |
| 203 | slow-down selector on frozen features: about 1 000 positives of "fails and slowing rescues", AUC 0.66; gains were DAC horizon effects | label = geometric fact of a queried trajectory (contact, time, arc length, gap), millions of queries from perturbations; no score-regression label; the action is not a gear choice |
| 197, 200, 204 | ground-truth occupancy through the memory channel is read but worth < +0.3; a branch must be pre-trained with its own head and carry information the policy lacks | nothing privileged enters the policy; branch arms are pre-trained with the contact head first; the memory-channel route is used only after the offline gate |
| 192 | frozen features carry the boundary to 0.475 m (AUC 0.739), still falling with labels; the information is lost in the frozen encoder | the boundary half of the lesson reuses that head design (implicit field + footprint query) and adds off-track states; the object half and the encoder arms are new |
| 160, 145 | unfreezing stage 4 alone or the whole encoder under the imitation loss: <= +0.11 | the encoder arms are trained by the contact label (new supervision), with the native-frame guardrail of 137 |
| 198, 212, 205 | +-0.5 m off-track imitation rows add nothing on lambda 10; +-1.5 m rows diverge in closed loop; strong hinge triggers drift | off-track states are inputs to the contact predictor, not imitation rows; the adapter's imitation targets are unchanged in 4.1 / 4.2 |
| 170, 207, 166 | > 45 deg "does not make the turn" 2.47 %, untouched by every recipe; the turn gap is path shape | reported as its own bucket at every read; no claim that the contact lesson fixes it; the turn-gain perturbation family (2.2) gives it labels |
| 194 | a selector on frozen features: +0.53 open loop, 0 in HUGSIM closed loop | the re-plan arm picks by predicted contact (not predicted score) and is read in closed loop only |

## 2. Deliverable 1: scene taxonomy and row generator (CPU, no line)

### 2.1 Taxonomy
Every navtrain token (103 288; `agent_labels/navtrain_all-k32.npz`, drivable raster `op_probe/labels/navtrain_all.npz`, token
tables `cache/navtrain_full.s*of12/tab.npz`) gets two tags from logged quantities only.

- Manoeuvre: launch (v0 < 1 m/s and 4 s arc > 2 m), stop (4 s end speed < 1 m/s from v0 > 2), turn 20-45 deg, turn > 45 deg
  (bench `TURN_BINS` on logged 4 s heading change), go-around (heading change < 20 deg and the logged sweep passes an object
  that lies inside the lane-following straight-ahead sweep), straight (the rest).
- Hazard (nearest over the 4 s logged sweep, ego box 5.176 x 2.297 m, agents at matching times): in-path object (an object box
  within the logged sweep widened by 0.3 m ahead of the ego at any time), side object (lateral gap < 1.0 m), boundary (drivable
  margin of the logged sweep < 0.5 m), none.

User classes: 1 "obstacle ahead" = in-path or go-around; 2 "turn" = turn 20-45 / > 45 with a side object or boundary hazard,
plus every > 45 deg token; 3 "leaving the road" = straight / launch with boundary hazard, and straight with no hazard (drift).
Mapped onto decision 220: (i) -> class 1 in-path, (ii) -> class 1 go-around + class 2, (iv) -> class 3 with a side object;
offroad / left-corridor zeros -> class 3 boundary and class 2 boundary. Counts per cell, per city and per split are reported,
and 3-10 representative tokens per class are shown as BEV with the labels drawn before any full generation.

### 2.2 Rows
A row = (state, query trajectory, labels). State = a navtrain token on the log, or off it. Query = 8 poses at 0.5 s.

- States: on-log; off-track by plane reprojection (existing `ot1_*` +-0.5 m / +-2 deg and `yr1_*` yaw-rate caches reused as
  they are; a new `bd1_*` family up to +-1.0 m / +-4 deg built with `ot_rows.cmd_prep` only for the classes that lack rows).
- Queries per state: the student's own plan (P2H10-F on the state's tokens), the shipped plan, the logged future, and
  perturbations of the student's plan: rigid lateral ramp (to +-0.3 ... 2.5 m at 4 s), heading drift (+-1 ... 6 deg linear),
  turn gain (curvature x 0.5 ... 1.5), arc-length scale (x 0.5 ... 1.5), stop at a sampled arc length.
- Labels (numpy, 41 steps at 0.1 s, agent boxes interpolated from 2 Hz, velocity by differencing; a new shared module
  `experiments/body1/lib/sweep.py`, checked against `swv1_lib.Rollout.sweep` and `AgentHinge.distances` on common cases):
  agent contact 0 / 1, time and arc length of first contact, class of the struck object, minimum clearance and signed lateral
  gap; boundary contact 0 / 1 (any footprint corner at SDF < 0), time and arc length, minimum margin. Agents are logged
  (non-reactive); a contact by an agent that runs into the ego from behind (first contact on the rear half, agent faster) is
  flagged and excluded from the positives.
- Balance: sampling weights so that each user class and each contact type (agent / boundary / none) has equal mass in a batch;
  the composition before and after weighting is reported.

Splits (`jevdrive.data.splits`, new, by log): `navsim/body1-hold-logs` = navtrain logs with `sha256(log) % 10 == 0` (contains
the op-parity-full dev logs), `navsim/body1-train-logs` = the rest. The student plans are P2H10-F's, which was trained on all
of these logs: on-log student plans are in-sample for the adapter (stated limit; the predictor never sees hold logs).

## 3. Deliverable 2: the contact predictor (offline gate)

Head: implicit field + footprint query (decision 192's design): scene tokens -> for each query trajectory, logits for agent
contact and boundary contact within 4 s, first-contact arc length (regression on positives) and clearance / margin.

| Arm | Near-range signal | Differs from |
|---|---|---|
| S0 | frozen Cinque `view_39` tokens (cached, 8 frames) | 203 / 158 by label and scale; it is the floor |
| S1 | LoRA r16 on the Cinque vision encoder (`pp_unfreeze` U1L path), trained by the contact loss with distillation to the frozen encoder on the same frames | 145 / 160: supervision is the contact label |
| S2 | light front-view branch (ImageNet-initialised ResNet-18 class, two front frames at the camera's native crop) with the same head | 197 / 200: input is the image, not ground truth; pre-trained with its own head (204) |

Order and scale: S0 at full scale first (cached tokens). S1 and S2 at pilot scale (shards s2-s4, 25 415 tokens) and only then
full. Seeds: 1 at pilot, 2 for the arm that goes to closed loop.

**Lines (tighten only).** G1: on `body1-hold-logs`, AUC of the predicted agent contact for the student's own plan (on-log and
off-track states pooled, clustered by log) >= 0.80, per user class where the class has >= 30 positives (else descriptive),
and the boundary contact AUC >= 0.80 likewise. G2: on decision 220's nuPlan P2H10-F decisions (122 intersecting decisions of
collision rollouts against 1 827 clean ones, `swv1_bc.read_auc`: speed-matched weights, cluster bootstrap by log), AUC >= 0.80
with the lower bound > 0.70; references there: simulator current boxes 0.728, best frozen head 0.718, plan lateral std 0.786.
G2 needs the replay re-run with the predictor in the hook (COL1 messages on the box); those decisions are navtest logs and
are used for reading only, never for fitting or threshold choice. An arm passes with G1 and G2. Pilot stop: an arm whose
pilot AUC is < 0.70 on G1 (pooled) ends; 0.70-0.80 gets one scale-up; S0 is judged at full scale. The > 45 deg bucket and
perturbed-query AUCs are reported but carry no line (perturbed queries are easy by construction).

## 4. Deliverable 3: acting on it, closed loop

### 4.1 Stop (serving, no adapter retraining)
In the driver hook after `serve_fix.apply`: if the predicted agent-contact probability of the served plan exceeds a threshold,
the plan is re-timed along its own path to stop `m` metres before the predicted first-contact arc length. No latch.
Threshold and `m` are set on `body1-hold-logs` only (target: false-stop share of clean hold decisions <= 2 %, `m` in
{1, 2} m), before any closed-loop score is read. With the switch off the driver output is bit-identical (checked on a replay).

### 4.2 Re-plan (serving)
Candidates = the plan plus lateral ramps of +-0.3 ... 1.5 m (the generator's family); the served plan is the candidate of the
smallest shift whose predicted agent and boundary contact are both below threshold; none clear -> 4.1. Read after 4.1.

### 4.3 Loss (conditional, training)
Only if an arm passes G1 + G2 and 4.1 / 4.2 leave collisions of classes (ii) / (iv): adapter retrained with the recipe
unchanged plus the predictor's branch through the memory channel (pre-trained, 204) and a contact hinge on student-plan rows
including off-track states. Needs the navtest guardrail. Its own amendment before it runs.

### 4.4 Read and lines
AlpaSim nuPlan public, the 700 scenes of `c0b/lists/chunk{0,1,2}.txt`, existing loop scripts (`ot2_loop.py`, `run.sh`),
P2H10-F-s0 and -s1 each with the predictor, paired per scene with P2H10-F (reference 0.9468 / 0.9483; 26 zeros = 8 at-fault
collision, 10 offroad, 8 left corridor; 112 slow scenes; the baseline runs of the same code state are reused or rerun in the
same chunks). Recipe = mean of the two seeds per scene.

- L1: zeros of the taught class go down (4.1: at-fault collisions; 4.2: collisions + offroad + left corridor).
- L2: mean scene-score difference >= 0 and the log-clustered 95 % lower bound > -0.005.
- L3: slow scenes (0 < score < 1) <= 1.1 x base.
- All three -> the arm is the new servable checkpoint and goes to the AlpaSim-input-standard sibling. All 1 491 public scenes
  were used for selection before: this is a regression check, not a held-out estimate.
- Staged: `pilot8.txt` (plumbing, identity with the switch off), one chunk of one seed, then the rest.

Guardrails: serving-only arms do not change navtest (open-loop plans are untouched; stated, not scored). Any arm that changes
weights producing the plan: navtest >= base - 0.3 through `python -m jevdrive.bench`, the > 45 deg bucket reported
(EPDMS, inside-cut %, cannot-make-turn %); native comma1M straight-road ADE <= 1.2 x base if base weights are touched (137).
Every closed-loop read also reports the zeros and the mean on scenes whose log turns > 45 deg. PAI track: only if PAI2 has
made it runnable on the GPU box; otherwise the exact command is written into the results doc.

## 5. Stage 2 (design only in this document)
CARLA rows with our student in the loop, by scene class (real contacts, the states the student reaches). Designed after a
stage-1 arm passes; its cost (servers, card-hours, GB) is reported before anything runs. b2dc-train@v2 is not built on.
Per dataset we hold (WOD-E2E, PhysicalAI-AV, comma1M, HUGSIM sources), whether it carries boxes and a boundary for this
lesson is reported with the taxonomy.

## 6. Budget and conduct
Stage 1: about 3 days, 60 card-hours, 60 GB (>= 150 GB kept free on the box). Estimates: generator CPU only (hours);
S0 < 2 card-h; S1 / S2 pilots about 3 card-h each, full about 10 each; one closed-loop arm (700 x 2 seeds) about 1.5 card-h.
Over budget: cut S1 or S2 and the re-plan arm, keep the navtrain S0 / best arm with 4.1. Every GPU job goes through the pool;
card 2 (given to this lane by the user) is used through the pool's own mechanism; other lanes' jobs are never touched.
Limits known in advance: logged agents are non-reactive; 2 Hz boxes interpolated; in-sample student plans on navtrain;
plane reprojection distorts objects above the road at large offsets; the AlpaSim scenes are 5 s with 10 decisions.

## Amendment 1 (2026-10-10, after the taxonomy and row set, before any predictor is trained or read)

Read so far: label-side counts only ([results/taxonomy.md](../results/taxonomy.md)); no predictor exists.

1. **Class 3 includes drift beside a standing object.** Section 2.1 mapped decision 220's class (iv) to "class 3 with a side
   object", but the written class 3 needed the boundary flag, which left straight / launch tokens with only a side object
   (7 474) in "other". Class 3 is now: straight / launch with a boundary or a side hazard, and straight with no hazard. The
   G1 class-3 read uses this set. Training uses the `w4` weights (the remaining "other" tokens as a fourth class at equal
   mass), so no generated row is unused; "other" is reported, without a line.
2. **Boundary positive needs depth.** The logged future itself has a footprint corner at SDF < 0 on 2.0 % of tokens (median
   depth 0.10 m, a quarter already at t = 0): raster resolution (0.5 m cells), not road departures. For the G1 boundary read
   and the boundary logit, positive = minimum margin < -0.20 m with first contact after t = 0; rows between -0.20 and 0 m are
   neither positive nor negative for the AUC; the margin itself is regressed. The agent label is unchanged, with rear-end
   contacts by a faster object excluded as written.
3. **Go-around tag.** It uses a constant-curvature continuation of the t0 motion as lane reference and has false positives on
   curved lanes (seen in the BEV panels). It only enters the class-1 membership and the balance weights; no line depends on
   it alone.
4. G1 positives of the student's own plan on hold logs (on-log + off-track pooled): agents 127 / 70 / 98 for classes 1 / 2 / 3
   (before item 1), boundary 133 / 297 / 433 after removing contacts at t = 0: every class is a gated read. Agent positives
   sit in 19-39 logs per class, so the log-clustered intervals will be wide; the pooled read (462 positives, 64 logs) is
   reported first.
5. G2 harness: the replay reproduces SWV1's served plans bit for bit (2 220 decisions) and its reader reproduces
   `plan_std2_FT` 0.786 [0.738, 0.865] and the road-edge margin 0.718 [0.643, 0.822] (`lib/g2.py`). The policy sees 8 token
   slots per decision.

## Amendment 2 (2026-10-10, the stop arm 4.1: fixed after S0 passed G1 / G2 and before any closed-loop run with the switch)

Read so far: [results/s0_gate.md](../results/s0_gate.md) (hold logs and the G2 decisions, as reported there); the hold-log rule
table of this amendment (`scripts/bd1_stop.py hold`, own-plan decisions of `body1-hold-logs` only); the replay plumbing check
(`bd1_stop.py replay`, item 7). No closed-loop run with the switch exists and no closed-loop score of it has been read.
Code: `lib/serve_body.py` (its docstring is the specification), hooked into `experiments/alpasim/lib/sh30_driver.py`.

1. **Switch.** `JEV_STOP=<m>` (metres), read once at driver start, carried in `get_version` (`-stop<m>`). Unset or 0: the
   module is not imported and the driver returns what it returned before. On: at every `drive` call, after `serve_fix.apply`,
   the served plan (8 poses) is scored by the two full-scale `step` checkpoints (seed 0 `full-step/20261010-024906`, seed 1
   `full-step/20261010-024909`) with the gate's input standard (fp16 tokens of the valid slots, slot mask, standardised ego
   vector); score = mean of the two agent-contact logits. The boundary outputs are not used (4.2).
2. **Threshold.** Flag = score >= 0.4080 (p = 0.60): the 98th percentile of the 50 265 clean own-plan decisions of the hold logs
   (`results/s0/stop.json`). Hold recall 0.51; by ego speed the head is nearly silent at standstill (0-1 m/s: flag rate 0.13 %
   of 2 981 clean decisions, recall 3 of 22; 3-10 m/s: 2.2 %, recall 0.50), which is where the hold logs have few positives.
3. **Contact arc length.** s_c = min(regressed first-contact arc length, arc length at the start of the first 0.5 s interval of
   the plan whose predicted clearance is under 0.25 m), two-seed means. Reason (463 flagged true positives of the hold logs): the
   regression is unbiased overall (median signed error 0.00 m, |error| median 1.83 m) but over-estimates near contacts (true arc
   0-3 m: median +1.42 m, n 63; 20 m and more: -1.74 m), and a stop placed behind the contact is no stop. Share of the true
   positives whose stop point lies before the true contact, m = 2 m: regression alone 0.747; with intervals under 0 / 0.25 /
   0.5 m: 0.790 / 0.875 / 0.914; mean speed removed by one false flag: 0.89 / 0.92 / 1.02 / 1.19 m/s. 0.25 m is kept: the
   0.5 m variant buys 0.04 more for 17 % more false-flag cost and puts 10 % of the false flags' stop points within 0.5 m of the ego.
4. **m = 2 m** (primary, the registered configuration; the lines apply to it alone). On the hold logs, open loop, one decision
   at a time, with the rule of item 5: the ego stands still before the true first contact in 355 of 463 flagged positives
   (0.767) with m = 2 against 322 (0.695) with m = 1; a false flag removes 1.02 against 0.91 m/s. The arc-length error (median
   1.8 m) is larger than 1 m. **Secondary, labelled: m = 1 m**, same rule, run after the primary's one-chunk check, reported
   next to it, never substituted for it.
5. **Speed profile of a flagged plan.** The path is unchanged; d = max(s_c - m, 0). Served speed v(t) = min(plan's own speed,
   sqrt(2 a (D - s(t)))) with D = max(d, v0^2 / 12) and a = clip(v0^2 / (2 D), 1, 6) m/s^2 (v0 = the ego speed the simulator
   reports). Reasons, from the controller (docs/alpasim.md, "Serving switches"), not from scores: the MPC tracks positions
   1.0-2.0 s ahead and no speed, so a stop reference must be reachable from the ego's own speed: a constant deceleration
   from v0 to a standstill at d is the reference it meets with that deceleration (FIX1's speed-continuous serving, applied to
   a stop). Cap 6 m/s^2: two thirds of the controller's limit (-9); the organisers report zig-zag steering for braking
   references the vehicle cannot follow; when d needs more, the reference brakes at 6 m/s^2 and stands past d (7 % of the
   false flags and 18 % of the flagged positives on hold logs). Floor 1 m/s^2: when d is far the plan is followed until the
   1 m/s^2 envelope reaches it, so a slow or standing ego may still roll up to the stop point (24 % of the false flags remove
   no speed at the flagged decision). Speed is only removed.
6. **What else the rule does.** No minimum speed: a flagged plan of a standing ego is served under the same envelope (it
   stays, or creeps to d). No latch and no filter: each decision stands alone; when the flag clears, the plan is served
   unchanged from that decision on (the launch is the adapter's own). A latch would need a release rule, which could only be
   tuned on closed-loop outcomes. Every decision is logged (`body` in `drive.jsonl`: p, both logits, flag, ego speed, hook ms;
   when flagged: both arc estimates, d, D, a, metres removed in the first 2 s, the plan before re-timing).
7. **What was looked at on the navtest replay (plumbing only; nothing chosen there).** Switch off: the driver's plans against
   SWV1's stored replay of the same messages, bit for bit. Switch on: the hook's two logits against the gate's stored
   predictions for the same decisions (max |difference| 0.06 on the collision groups, flags identical: 34 / 34 and 31 / 31),
   the plan before the stop against the stored replay (0.0 m), geometry checks of the stop trajectories (on the path, never
   ahead of the plan, never past D) and the hook's latency (9.5 ms median). The replay also printed the median deceleration
   and metres removed of the flagged decisions of the collision groups (1.84 m/s^2, 2.1 to 2.4 m); every constant above was
   already committed (33badb5a) and none changed afterwards.
8. **Read (section 4.4, tightened).** Scenes: `c0b/lists/chunk{0,1,2}.txt` (700), P2H10-F-s0 and -s1 with `JEV_STOP=2`, through
   `ot2_loop.py` / `run.sh` / the pool. Baseline: TR1's P2H10-F-s0 / -s1 runs on the same chunks (2026-10-10 01:22, box HEAD
   894bf6e0); the files the driver imports (`experiments/alpasim/docker/closure.txt`) are unchanged between that commit and
   this one except `sh30_driver.py`, whose switch-off path is the identity of item 7. Recipe = per-scene mean of the two seeds.
   - L1: at-fault collision zeros (first failing flag of the scene) go down in the two-seed mean, and in neither seed up.
   - L2: mean per-scene difference >= 0 and the log-clustered 95 % lower bound > -0.005 (`jevdrive.stats.paired(groups=log)`).
   - L3: slow scenes (0 < score < 1), two-seed mean count, <= 1.1 x base.
   - Reported without a line: zeros by class, every scene whose zero / non-zero state changes with the reason seen in the logs,
     flag rate per decision and per scene, driver latency, the scenes whose logged path turns > 45 deg (count, mean, zeros;
     logged heading change over the scene), the secondary m = 1.
   - Staged. (a) `pilot8.txt`, P2H10-F-s0: switch off twice and switch on once. Off must give the same per-scene scores twice
     (or the difference is the repeatability every later comparison is read against); on must complete with a `body`
     record on every decision. (b) chunk0 of P2H10-F-s0 with the switch, against this checklist: 234 / 234 rollouts and no
     driver error; `drive` total p50 and p90 <= 100 ms; flags on 1-5 % of the decisions (expected 2-2.6 %); the ego is slower
     0.5 s after a flagged decision that removes speed in >= 80 % of them; the controller's acceleration command never under
     -8.5 m/s^2 after a flagged decision and no steering reversal train in the steering / acceleration traces of 3 flagged
     scenes (drawn; a zig-zag fails the check). A failed check stops the arm before the rest runs. (c) the remaining five
     chunk jobs, then the secondary.
   - It is a regression check: all public scenes took part in selecting P2H10 before. Open-loop plans are untouched (the
     switch acts on the trajectory handed to AlpaSim's tracker), so navtest is stated unchanged and not scored.
