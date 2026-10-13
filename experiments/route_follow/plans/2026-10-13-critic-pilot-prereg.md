# CRITIC-PILOT prereg (2026-10-13): can a learned consequence critic replace the hand-written closed-loop hinges?

Written and pushed before any critic is trained or any label is read against a prediction. Code: `experiments/critic_pilot/`.
Feasibility pilot, one day, read-only on stored rollouts; GPU only through the pool (`--vram 20 --cpu 8 --ram 60`).

## Question

From one decision's inputs (the frozen Cinque tokens the policy saw `front` (8, 32, 512), the ego vector (20), the route memory it was fed
`rmem` (200, rmem.py layout)) and a candidate plan (8 rear-axle poses, own frame), can a small critic predict the closed-loop consequence of
that decision within the next T s of the same rollout? If yes on a policy it never saw, a learned critic is a candidate to replace S
(agent / road), the route hinge, stop-line B and progress P (research/self-consequence/design.md) with one model trained on the policy's
own decisions and their simulator outcomes.

## Data (all stored; nothing is re-simulated)

- Train pools (train 1031 split `alpasim_nuplan/onpol-train`, 18 decisions): `on1_L18RB0` (RB0-F-s0), `on1_L18RSG` (RSG-L18-s0),
  `on1_L18RBSG` (RBSG-L18-s0). Row inputs from `<pool>/tab.npz` (ego, served), `<pool>@warp/front.npy`, `rows/<N>/meta.npz` (rmem, run,
  scene, k, t_us, anchor, log), `<pool>/route.npz` (full route label polyline, own frame; labels only, never an input).
- Held test pool: `on1_L18RSGallheld` (RSG-L18-s0 on the held 460, `alpasim_nuplan/onpol-held`).
- The policy-unseen held test with tokens has to be RSG: RBSG2's held reads (`runs/route_mem/read/RBSG2-L18-s*/`) carry no token dump
  (no `driver-logs/dump`), so no token critic can be evaluated on them. They are used for the blind arm only (below).
- Counterfactual sanity: `on1_CFdev` (309 rows, dev logs, never trained; real vs counterfactual memory + command exactly as `rf_gates.py`).

## Labels (decision k at simulation time t = t_us; from the same rollout's run dir)

Per rollout: `aggregate/results-summary.json` (score_metrics flags), `rollouts/<scene>/<rid>/metrics.parquet` (evaluator series at the
decision times), `controller/alpasim_controller_<rid>.csv` (10 Hz ego, rollout frame = the frame of meta `anchor`).
- Zero event times (rf_tax.py rule): collision = first collision_front / collision_lateral > 0 if the rollout's collision_at_fault flag > 0;
  offroad = first offroad > 0 if its flag > 0; corridor = first left_corridor_laterally > 0 if its flag > 0.
- (a) `ev[type, T]` = the type's event time e satisfies t < e <= t + T, T in {2, 4} s; `any[T]` = any type. Rows with t >= the earliest
  event of the rollout are dropped (already zeroed); a row whose horizon passes the series end without an event is masked for that T.
- (b) `lat2` = signed lateral offset (m, left +) of the executed rear axle at t + 2 s (controller, interpolated) from the route label
  polyline (`route.npz` poly, own frame via anchor); masked where route not ok or the trace ends.
- (c) `prog4` = executed path length (m) over (t, t + 4 s].
- (d) `wzpk` = peak |yaw rate| (rad/s) over (t, t + 4 s]; `spin` = |vx wz| >= 6 m/s^2 on 3 consecutive 10 Hz samples within 4 s
  (rf_tax spin rule).
- Gate G0 before training: median |controller pose at t - anchor| <= 0.3 m on every pool; label counts per pool written to `labels.json`.

## Critic

Transformer encoder, d = 256, 6 layers, 8 heads: tokens = 256 vision tokens (Linear 512 -> d + slot / index embeddings), 1 ego token,
40 route-point tokens (valid ones; + 1 null token), 8 plan-pose tokens ((x, y) / 20, cos, sin of yaw, + time embedding), 1 CLS.
~5 M params, bf16 autocast, AdamW 3e-4, batch 256, max 12 epochs, early stop on the val loss (10 % of the train logs, log split).
Heads from CLS: 6 event logits (3 types x 2 T) + 2 any-zero logits + spin logit (BCE, masked) and lat2 / prog4 / wzpk (Huber, masked).
Plan input = the served plan (the only plan whose outcome exists); plan diversity comes from three policies on the same scenes.

## Arms (each one training, seed 0)

| arm | trained on | role |
|:--|:--|:--|
| **C2** | RB0 + RBSG | primary: RSG held = unseen policy and unseen scenes |
| C3 | RB0 + RSG + RBSG | same policy, unseen scenes (RSG held) |
| C2-blind | RB0 + RBSG, no vision tokens | what the tokens add; the only arm also read on RBSG2 held reads (zero AUCs; inputs from drive.jsonl) |
| C2-noroute | RB0 + RBSG, route memory zeroed | what the route input adds |
| C2-aug | C2 + route-shift augmentation | memory shifted sideways by U(-3, 3) m on half the rows (rmem.transform_rows) with lat2 relabelled exactly (the executed position is known); event labels unchanged but masked on shifted rows. Tests whether a lateral head can be made to read the route |

## Reads and lines (on RSG held, log-clustered bootstrap B = 10 000, decision-level)

- **L1** any-zero within 2 s AUC >= 0.90 on C2 (unseen policy). Per type (collision / offroad / corridor, 2 and 4 s) reported where >= 20
  positive rows. Reference (not a line): d266's served-plan rule reached 0.95 on corridor (rollout-level max before event, a different
  protocol); here the same-row hand rules are computed on the test rows: `b_plan` = |served plan's lateral offset from the route label at
  the pose nearest 10 m arc| (d266's s10 analogue), `b_now` = |ego offset from the route now|, `b_log` = |e0| (offset from the logged path).
- **L2** lat2 R^2 >= 0.60 on C2.
- **L3** counterfactual flip >= 0.70: on `on1_CFdev` junction rows (j0 / j1) whose two route polylines are >= 1 m apart at the 2 s arc,
  plans P_real / P_cf = the real / counterfactual route polyline sampled at the arc lengths of the row's `fut`, inputs (real ego, real
  memory) and (cf ego, cf memory). Flip = |lat2_hat(P_real | real)| < |lat2_hat(P_cf | real)| and |lat2_hat(P_cf | cf)| < |lat2_hat(P_real | cf)|.
  Share over rows, read on C2 (line) and on C2-aug / C2-noroute (descriptive; C2-noroute should sit near chance of 0.25 for both
  conditions jointly, it gives the floor).
- prog4 R^2, wzpk R^2, spin AUC: descriptive, no line.
- **Gradient sanity** (descriptive, soft line 0.70): on 100 RSG held rows with an offroad or corridor event within 4 s, and on all rows
  with a collision event within 4 s (<= 100), g = -d risk / d(plan x, y), risk = sum of the 4 s event logits + |lat2_hat|. Read: share of
  rows where g at the 1 - 2 s poses points toward the route label (cos with the vector to the nearest route point > 0) for offroad /
  corridor rows, and away from the nearest valid actor box (agent.npz at the same times) for collision rows; median |g| against the median
  gradient norm of the S hinges on the same rows (10 x agent hinge (margin 0.5) + 10 x road hinge (margin 0.3), lib/agent_hinge.py,
  lib/drivable_hinge.py, rows where S is active), and the cosine between the two where both are non-zero.

## Verdict

Feasible = L1, L2, L3 all pass on C2. Any failure is reported with the per-type table; a failure of L3 with L1 / L2 passing means the
critic predicts consequences from vision and plan shape without reading the route (it could not replace the route hinge).
What would not be learned here: whether a critic gradient trains a better policy (no policy training in this pilot).
