# STACK cheap checks (prereg `plans/2026-10-16-stack-prereg.md` section 1)

## C1: can the Cinque plan head be sampled?

- No. `net.slices["plan"]` = outputs 917 .. 1907, 990 numbers = ONE MDN component: 495 means (33 x 15, `jevdrive.op_adapt.plan_index`) + 495 spreads.
  No hypotheses, no hypothesis probabilities (the 5 x 991 plan with a probability column in `jevdrive/openpilot/model.py` is the old supercombo).
- The spread is never trained by our recipe: the distillation index keeps 495 positions inside the plan slice, all of them the means
  (`op_adapt.distill_index` takes the first half of every MDN head), and the imitation term reads the means only. Decision 220 read it once
  (2 s lateral std, AUC 0.786 for collisions): a weak uncertainty signal, not a source of candidates.
- Candidates are therefore structured perturbations of the model's own plan, `lib/stack_cand.py` (K = 8): lateral offset d in
  {-1.0, -0.5, +0.5, +1.0} m along each pose's normal, ramped 0 -> d over 0-2 s; curvature gain g in {0.8, 1.2} and arc scale s in {0.75, 0.9}
  (`perturb.branch_poses`, copied as `branch_np`). Box selftest: torch vs numpy max |diff| 3.2e-14 m over 300 plans x 8 candidates.

## Reflex: self-state data and serving

Code: `lib/reflex_mem.py` (the efference memory, one rule for rows and drivers), `experiments/route_follow/scripts/rf_stack_self.{py,sh}`,
`rmem.load_pmodel` / `rmem.reflex_of`, `sh30_core.Core.plan(selfstate=)`, `pai_core.plan(selfstate=)`, `sh30_driver.py`, `pai_driver.py`.
`lib/reflex.py` unchanged. Box outputs `$DATA_DIR/runs/route_follow/stack/self/` (summary.json, replay_checks.json, smoke_checks*.json).

**Rule.** The driver stores every served decision (t0, the plan frame = rear-axle anchor in the local frame, the poses actually served:
after every serve-side switch, the JEV_TRACK reference when on). At a decision the previous one is the newest stored decision at least
0.45 s back (LAG 0.5 s): decision k - 1 on the nuPlan track (2 Hz); on the PAI track (10 Hz inference) the decision 0.5 s back, so dt
matches the rows, and the first 0.5 s has valid 0. rel = the anchor now in that decision's frame, dt from the t0s, v = the measured speed;
`reflex.selfstate`. Under JEV_TRACK with a tilted pose the frame and the pose now both take `track_ref.anchor`'s planar yaw.

**Offline rows** (`self/<P>.npz`: names = the tab's, same order; f (N, 14) float32; source pools also k, dt, k_prev). Each session's
drive.jsonl replayed through `reflex_mem.Memory`; previous served poses from its SH30_DUMP (unrounded), anchors / t0 from drive.jsonl,
v = the tab's `speed`. RCVL maps back by prefix (s -> on1_L18RB0, e0 -> on1_L18RSG; served, ego, speed identical on every row).

| pool | rows | valid | coverage | \|y_0\| median / p90 (m) | \|yaw_0\| median / p90 (rad) | \|x_0\| median / p90 | validation max \|diff\| |
|:--|--:|--:|--:|--:|--:|--:|--:|
| on1_L18RB0 | 17 429 | 16 409 | 0.9415 | 0.0052 / 0.0187 | 0.0025 / 0.0081 | 0.0186 / 0.0523 | 6.6e-5 |
| on1_L18RSG | 17 474 | 16 454 | 0.9416 | 0.0052 / 0.0186 | 0.0024 / 0.0076 | 0.0187 / 0.0525 | 6.3e-5 |
| on1_L18RSGallheld | 7 837 | 7 390 | 0.9430 | 0.0050 / 0.0183 | 0.0021 / 0.0072 | 0.0185 / 0.0530 | 6.2e-5 |
| on1_RCVL (all; src = human on every row) | 29 922 | 28 152 | 0.9408 | 0.0051 / 0.0183 | 0.0025 / 0.0079 | 0.0188 / 0.0529 | (mapped) |
| on1_RCVL pool s / e0 | 17 429 / 12 493 | 16 409 / 11 743 | 0.9415 / 0.9400 | 0.0052 / 0.0187, 0.0050 / 0.0178 | 0.0025 / 0.0081, 0.0024 / 0.0075 | | |

- Invalid rows = exactly the k = 0 rows (1 020 / 1 020 / 447); no k >= 1 row lacks its previous decision, the previous decision is
  always k - 1, dt = 0.500 s on every valid row, no session unmatched, no feature clipped.
- Validation (every session of the three source pools, not just 2): the features recomputed from drive.jsonl alone, as a served driver's
  log gives them (1e-4 m poses, the logged ego's speed), against the rows: max |diff| 6.6e-5 over all 14 features (limit 1e-4), valid
  flag identical on every row.
- The tab's `pose` (n, 4, 3) is the fed history: rear-axle poses at t0 - 1.5 / 1.0 / 0.5 / 0 s in the t0 frame (1e-4 m). pose[:, -2] is the
  previous decision's anchor in the current frame: it equals the inverse of the replayed rel within 5.0e-5 m / 5.0e-5 rad (rounding).
- Scale: on the nuPlan track the ego tracks the previous intent closely (|y_0| 5 mm median); the tau > 0 numbers carry the shape of the
  previous intent (RCVL |y_1.0| median 0.035 / p90 0.32 m). One PAI scene (STKRX1-test): |y_0| median 0.031 / p90 0.088 m, |x_0| 0.022 /
  0.091: the PAI body deviates ~6x more.

**Serving checks** (test checkpoints, new tags, sidecar = RSG-L18-s0's + `reflex: {"sf": 14, "d": 128}`: STKRX0-test = RSG-L18-s0 +
module at its zero init; STKRX1-test = the same with gate 1 and out weights N(0, 1e-3), seed 7).

- Offline replay (`rf_stack_self.sh replays`): `Core.plan` on 45 stored decisions of 3 held RSG-L18-s0 rollouts (the dumped JPEGs of
  decisions k-3 .. k, logged history / ego / route memory; eager, cpu synth), the self-state from reflex_mem over the stored rollout.
  This is the bitwise instrument (same inputs in every process).
  - (1) RSG-L18-s0 through the old code (worktree at 88a392cc) vs the new code: mu and poses bit-identical (max |diff| 0); new vs new repeat
    bit-identical.
  - (2) STKRX0-test vs RSG-L18-s0: bit-identical; fed row 234 = 20 + 200 + 14 (base 220); valid = 1 on all 45 replayed decisions (k >= 3).
  - (3) STKRX1-test moves the plan: max 0.077 / 0.087 / 0.103 m at 0.5 / 1.0 / 1.5 s (median 0.081 m), 0.14 m beyond; <= 0.5 m.
    STKRX1-test with JEV_REFLEX_OFF=1: bit-identical to RSG-L18-s0 (valid 0 everywhere).
- Closed-loop smoke (`rm_chain.sh readlist`, 3 held scenes x 18 decisions, SH30_COMPILE=0, pool jobs): RSG-L18-s0 repeated = itself
  exactly (54 / 54); STKRX1-test + JEV_REFLEX_OFF=1 = RSG-L18-s0 exactly (54 / 54); STKRX0-test = RSG-L18-s0 exactly in a repeat (54 / 54).
  Its first run differed in one scene from k = 4 (4.8 mm at k = 4 with identical logged ego / history / route / anchor, then the
  rollout drifted, 0.83 m by k 17). The repeat matched and the offline replay of the module is bit-identical, so this is
  closed-loop non-determinism outside the logged inputs (most likely the frames), not the module; it bounds what a "bit-identical"
  closed-loop read can claim. Logged `rx`: valid 0 at k = 0, 1 from k = 1 on (STKRX0 / STKRX1), 0 everywhere with JEV_REFLEX_OFF=1; fed
  row 234 on every decision; rx_dt 0.5 s; logged rx against reflex_mem replayed over the same drive.jsonl max 5.3e-5.
  STKRX1-test live: plans differ from k = 1 on as intended (the rollouts then diverge).
- PAI (pai_box.sh, 1 scene, STKRX1-test, default serving = JEV_TRACK gate + JEV_VCONT / JEV_LEAD): runs without error (199 inferences); fed
  row 234; valid from k = 5 on (the 0.5 s lag); rx_dt 0.5 s. A first run used the full-quaternion yaw for the pose now while the stored
  frame took track_ref's planar yaw on tilted poses (63 / 199 here): logged rx vs replay up to 0.20. Fixed (73dc4649: the pose now in the
  same convention); after the fix the replay matches to 0.018, limited by the log (anchor 1e-4 m, tilt 1e-3 rad). The scene ended
  offroad at 13.5 % of its duration: this is a test checkpoint with random weights, so that says nothing about the module.

Cost: about 1 GPU-hour (ckpt 1 min; replays 2 x ~6 min; 6 three-scene reads ~5 min each; 2 one-scene PAI runs ~4.5 min each) + ~2 CPU-min.

## C2: can a small consequence head learn per-time contact of any candidate plan?

**Verdict: FAIL** (gate 1 misses; gate 2 lands exactly on its threshold). Under prereg 4.1 the `cons` constraint therefore comes **out**
of the joint arm. The head is saved anyway (`$DATA_DIR/runs/route_follow/stack/cons/head.pt`, interface `lib/conshead.py`) and is not
to be used as a training signal. Numbers: [c2.json](c2.json). Code: `scripts/rf_stack_c2.py` (labels / train / eval), `lib/conshead.py`.

Setup. own = pool on1_L18RSG: 17 474 rows, 158 dropped for t0 contact, leaving 17 316 (dev 3 618 rows = **4 logs / 212 scenes**,
train 13 698 rows / 24 logs). Candidate 0 = RSG-L18-s0's forward plan (rf_recov fwd; 6 mm median / 20 mm p95 from the served plan),
then the 8 `stack_cand` candidates. human = navtrain op-parity-full: 38 000 train rows + all 1 789 dev rows (32 logs, none shared with
train), 211 t0 rows dropped. Candidates: teacher plan, + 8, + log future. Box checks: `stack_cand` torch vs numpy 3e-14 m;
`branch_np` == `perturb.branch_poses` bit for bit. Head: 0.84 M params (d 128; slots t and t - 1.0 s + ego make the memory; 8 pose
queries, 2 decoder layers). BCE, 8 000 steps, 128 own + 128 human rows per step, each row with all its candidates + 2 random
candidates (lat / gain / arc drawn at random, labelled on the fly with the same geometry). Seed 0, last step kept. An inner 10 % of
the train logs was held out of the gradient and only watched: its own AUC was 0.808 at step 1k and 0.82 from step 2k to the end, so
it plateaus rather than overfits; train-row AUC 0.985. Cost: about 0.4 card-hours in total (fwd ~6 min, smokes ~3 min, labels 1 min,
train 9 min, eval 2 x ~3 min).

CIs are log-clustered bootstraps (B 10 000, jevdrive.stats resampling). Own dev has only 4 logs, so the scene-clustered CI
(212 scenes) is given next to it as a descriptive second view.

| read (dev) | own: AUC [log CI] | own: [scene CI] | human: AUC [log CI] |
|:--|--:|--:|--:|
| **gate 1: pooled, all candidates, any contact (>= 0.85)** | **0.773 [0.736, 0.813] FAIL** | [0.743, 0.803] | 0.906 [0.880, 0.931] |
| agent only | 0.713 [0.656, 0.784] | [0.670, 0.755] | 0.890 [0.842, 0.927] |
| road only | 0.909 [0.889, 0.945] | [0.875, 0.936] | 0.941 [0.914, 0.966] |
| perturbed candidates only (1-8) | 0.774 [0.739, 0.813] | | 0.902 [0.877, 0.927] |
| executed plan (cand 0, across states; d220 <= 0.72) | 0.740 [0.683, 0.798] | [0.694, 0.786] | 0.917 [0.876, 0.952] |
| within-state (mean over states with both outcomes) | 0.836 [0.814, 0.870], 1 655 states | [0.807, 0.863] | 0.922 [0.890, 0.942], 776 states |
| baseline: candidate type + speed (logistic), pooled | 0.625 [0.615, 0.636] | [0.599, 0.654] | 0.705 [0.666, 0.749] |
| baseline, executed | 0.538 [0.520, 0.589] | | 0.532 [0.468, 0.617] |
| baseline, within-state | 0.799 [0.775, 0.821] | | 0.823 [0.786, 0.854] |
| bin-level AUC (each of the 2 x 8 bins), point | 0.862 | | 0.961 |
| human log future only (27 positives) | | | 0.908 [0.670, 0.951] |

Gate 2 covers the 5 174 dev own candidates (1 848 rows) with true contact. Each takes one xy step of every pose along
-grad(sum of predicted probabilities), yaw kept, scaled to an RMS of h per pose. The read is the share whose true contact depth
strictly decreases:

| h | gradient [log CI] | [scene CI] | random direction [log CI] | mean depth change, grad / random |
|--:|--:|--:|--:|--:|
| 0.05 m | 0.712 [0.664, 0.735] | [0.677, 0.749] | 0.490 [0.482, 0.497] | -0.26 / +0.01 |
| **0.10 m (gate, >= 0.70)** | **0.708 [0.664, 0.729] (point >= 0.70; CI straddles)** | [0.672, 0.744] | 0.473 [0.462, 0.487] | -0.46 / +0.04 |
| 0.20 m | 0.696 [0.647, 0.719] | [0.659, 0.734] | 0.451 [0.438, 0.467] | -0.74 / +0.19 |

Rates (3), the dual-ascent reference inputs of prereg 4.1 (trainer geometry, margin 0.3):

| rows | agent hinge violated (AgentHinge 0.3 / side 0, any of 41 steps) | road hinge violated (Hinge 0.3, JT.ROAD / pool road.npz) | max predicted contact p > 0.5 |
|:--|--:|--:|--:|
| navtrain train rows, log futures (101 499 rows, 1 160 logs) | **0.0005** [0.0002, 0.0007] | **0.127** [0.120, 0.134] | 0.024 [0.022, 0.026] (head-train rows 0.021, others 0.026) |
| on1_L18RSG train rows, RSG-L18-s0's own plans (13 830 rows, 24 logs) | 0.072 [0.055, 0.088] | 0.131 [0.094, 0.162] | 0.091 [0.076, 0.107] |

What to take from this:
- The head reads the scene (it beats the type + speed baseline by 0.15 pooled and 0.20 on the executed plan), and on human rows it
  would pass (0.906). On the policy's own states it does not: road contact is learnable (0.909), **agent contact is not (0.713)**,
  and agent contact is half of the own positives. The executed-plan AUC (0.740) sits at decision 220's frozen-model ceiling (<= 0.72),
  so a dedicated head does not lift it: the frozen tokens of the policy's own states carry little about which actors the plan will
  touch.
- Ranking inside a state is better (0.836), but the baseline that knows only the candidate type already gets 0.799 there, so most of
  it is "a 1 m left offset hits more often".
- The gradient does point away from contact (71 % against 47-49 % for a random step of the same size; mean depth falls), but only
  at the gate level, and it weakens at larger steps.
- The human agent reference rate is about 0 (5 in 10 000 log futures violate the margin-0.3 agent hinge), so a level x human-rate
  target for `agent` sits at the prereg's 0.02 floor of the denominator. The road reference is 12.7 %, close to the policy's own
  13.1 %. The human log futures' road violations come from the road-and-lane raster's margin, not from leaving the road (the
  margin-0 contact of the log future is 1.5 %).
