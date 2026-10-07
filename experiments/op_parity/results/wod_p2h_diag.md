# Why P2H loses on WOD-E2E val: input interventions, frame source, recipe arms, plan decomposition

Written 2026-10-07. Inference only (no training). Question: decision 155 / [wod_p2h.md](wod_p2h.md): P2H10-F (s0 / s1) scores WOD val RFS 7.708 vs shipped
8.005 (-0.30 [-0.50, -0.09]), ADE@3s +0.28 m; with the adapter bias zeroed it is neutral (-0.02), so the loss is what the ego-feature bias does on
WOD. This file attributes the loss to causes. The design (arms, attribution rule) below was written and committed before any arm was served or
scored; results are appended under "Result".

Code: `scripts/pp_wod_diag.py` (`bias`: one `intent_bias` file per arm and variant; `report`: tables), `scripts/wod_zeroshot_openpilot.py`
(the decision 155 harness, now with several `--tag` / `--bias` pairs sharing one warm-up and the last step re-run per bias from a state snapshot,
and `--frames warp`). Predictions `preds/op_cinque_dx-<arm>_<var>[_W]`, tables under `results/wod_p2h_diag/`.

Settled, not re-tested (decision 155 / wod_p2h.md): command zeroed (no effect, re-run here only as part of the seed-1 set), acceleration
zero / position-derived (ADE moves, RFS stays near -0.27). Decision 108: WOD camera geometry (horizon, field of view) is right; the camera height
(1.81 vs 1.87 m) and the coordinate origin act through the shared vision / output path, identical for shipped and P2H, so they cannot produce a
P2H-minus-shipped difference by themselves and get no arm.

## Arms

All on the 479 rater frames (RFS) and 1 437 frames with futures (ADE), the decision 155 harness, real 10 Hz frames fed twice, 10 s warm-up. Unless
stated, each ego variant edits one group of the 20 features of the main mapping (pp_wod.wod_ego) and keeps the rest; run for both P2H seeds,
reported as the per-frame seed mean.

| arm | what changes | question |
|:--|:--|:--|
| main | none (decision 155 mapping) | reproduces 7.708; the multi-bias path vs the stored runs is the equivalence check |
| zero | bias = 0 | reproduces "weights neutral" |
| cmd0 | command one-hot = 0 | settled; seed 1 completes it |
| acc0 | ax = ay = 0 | settled for s0; seed 1 completes it |
| accnav | ax, ay z-scored onto the navtest moments (spread x2.8 / x6.6) | ego-feature distribution shift of the acceleration |
| velgiven | vx, vy = WOD's given velocity vector (instead of the position-derived speed, vy 0) | speed derivation |
| vx110 | vx x 1.1 | dose: how much plan speed / RFS the speed input moves |
| yaw0 | the 4 pose yaws = 0 | chord-heading yaw mapping |
| yawvel | pose yaws from the given velocity direction (op_interp.track_wod rule) | alternative yaw mapping |
| posecv | poses = straight constant-speed history at the input speed (x = v t, y = yaw = 0) | past-pose mapping (timing, alignment, origin, curvature) as a whole |
| cv | posecv + vy = ax = ay = 0 (the ego carries only speed + command) | all history beyond speed |
| biasmean | every frame gets the WOD-mean of the main bias | the frame-independent part of the bias |
| biasresid | main bias minus its WOD mean | the ego-dependent part |
| biasnav | every frame gets the mean bias over the 12 146 navtest ego rows | the training-distribution constant |
| P2-F main / zero (s0, s1) | P2 without the hinge, its own bias / bias 0 | does the hinge carry the loss |
| P1-F zero (s0, s1) | P1 (plan fine-tune, no ego inputs) | does the fine-tune without inputs lose on WOD |
| W: shipped, P2H main / zero (s0, s1) | frames -1.5 ... 0 s re-synthesised from the four 2 Hz keys by the ego-motion warp (protocol W of training: op_interp warp, track from past_states, front camera position); earlier warm-up frames real | frame source (real vs warp) |
| O1 / O2 (offline) | O1: P2H's path retimed to shipped's arc length per waypoint; O2: shipped's path at P2H's arc length | longitudinal (speed profile) vs lateral (path) |
| strata (offline) | speed bins, stopped / launch, intent turn vs straight, shipped lead_prob > 0.5 (following), scenario cluster | where the RFS is lost |

Not run: P3 on WOD (needs side-camera tokens; WOD slim shards hold only the front three cameras), a warp of the whole 10 s warm-up (the
past_states cover 4 s; the training protocol W only synthesises the 0.2 s lattice around the 2 Hz keys).

## Attribution rule (fixed before scoring)

- d(arm) = RFS(arm) - RFS(shipped); recovery r = (RFS(arm) - RFS(main)) / (RFS(shipped) - RFS(main)), the fraction of the main loss removed;
  the same for ADE@3s. CIs: paired bootstrap over sequences (rater sequences for RFS, all sequences for ADE), B 2 000, seed mean.
- An ego-channel or bias-part arm **carries** the RFS loss if r >= 0.5 and the CI of RFS(arm) - RFS(main) excludes 0; **part** if
  0.25 <= r < 0.5 with that CI excluding 0; **not** otherwise. The same labels for ADE@3s.
- **Frame source** carries it if (P2H - shipped)@W has a CI that includes 0 or lies above 0 and the interaction
  I = (P2H - shipped)@W - (P2H - shipped)@real has a CI excluding 0 with I >= 0.15 (half the main loss).
- **Hinge** carries it if P2-F's d has a CI that includes 0 and P2H's does not; **fine-tune without inputs** carries it if P1-F's d CI excludes 0
  below 0.
- **Longitudinal** carries it if O1 recovers r >= 0.5; **lateral** if O2 (shipped path, P2H speed) keeps >= 0.5 of the loss is the
  complementary check (O2 ~ P2H means speed carries it, O2 ~ shipped means path).
- Many arms (about 20 contrasts): a single CI excluding 0 at the margin is read as weak; the answer rests on arms that meet the rule for
  both seeds' mean and agree in direction for each seed.
