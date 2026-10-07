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

## Result

### Answer

**The loss is the frame-independent part of the adapter bias, and it is a speed-profile effect.** The P2H bias splits into a constant
(its mean over the WOD frames, rms 1.03 of the bias's 1.13; the mean bias over the navtest ego rows, rms 1.06, acts the same way) and an ego-dependent
remainder (rms 0.45). The remainder alone beats shipped (RFS 8.102, +0.097 [+0.009, +0.194]; ADE@3s -0.08 m), the constant alone is far worse
(-1.11), the full bias is -0.30, no bias is -0.02. So what P2H reads from the WOD ego state helps; what hurts is the offset the adapter applies
regardless of input. Every input-mapping intervention leaves RFS where it was (acceleration zeroed or rescaled to NAVSIM's spread, given
instead of position-derived velocity, yaw zeroed or from the velocity direction, the pose history replaced by a straight constant-speed one,
all history beyond speed removed: each within 0.07 of the main arm, recovery <= 0.23): the WOD feature mapping and the ego-feature
distribution shift are not the cause. The hinge is not either (P2 without hinge -0.33), nor the fine-tune itself (P1 without inputs +0.01).
The plans lose only through their speed: P2H's own path re-timed to shipped's speed profile scores +0.06 vs shipped (recovers 121 %), while
shipped's path at P2H's speed scores -0.46. The offset makes the plan roll away from standstill (stopped frames, 25 % of the rater set: +0.94 m
further at 3 s, RFS -0.69; -1.72 on the 25 where the car really stays stopped) and drive shorter at speed (>= 12 m/s: -1.5 m at 3 s, -0.67;
behind a lead: -0.4 m, -0.41). It is tied to the frame protocol it was trained under: on WOD frames re-synthesised by the training protocol's
warp, P2H and shipped are level (+0.05 [-0.20, +0.29]; interaction +0.35 [+0.13, +0.57]), mostly because the at-speed shortfall turns into a
gain there; the standstill creep stays (-0.48 under warp). A fix that needs no WOD data: subtract the navtest-mean bias at serving time
(post hoc arm `biasdenav`): RFS 8.096, +0.091 [+0.000, +0.188] vs shipped, ADE@3s -0.07 m (both seeds +0.09).

### Interventions (seed mean of P2H10-F-s0 / s1; RFS over 479 rater frames, ADE over 1 437 frames; shipped RFS 8.005, ADE@3s 1.041 m)

| arm | RFS | d vs shipped [95% CI] | vs main [95% CI] | recovery RFS | d ADE@3s (m) | vs main ADE@3s [95% CI] | recovery ADE | label (RFS / ADE) |
|:--|--:|:--|:--|--:|--:|:--|--:|:--|
| main | 7.708 | -0.297 [-0.499, -0.093] |  |  | +0.284 |  |  |  |
| zero | 7.985 | -0.020 [-0.055, +0.016] | +0.278 [+0.074, +0.485] | +0.93 | +0.057 | -0.226 [-0.280, -0.172] | +0.80 | carries / carries |
| cmd0 | 7.619 | -0.386 [-0.607, -0.184] | -0.089 [-0.207, +0.019] | -0.30 | +0.275 | -0.008 [-0.026, +0.011] | +0.03 | not / not |
| acc0 | 7.707 | -0.298 [-0.513, -0.094] | -0.001 [-0.056, +0.057] | -0.00 | +0.397 | +0.113 [+0.100, +0.127] | -0.40 | not / worse |
| accnav | 7.741 | -0.264 [-0.462, -0.085] | +0.033 [-0.060, +0.118] | +0.11 | +0.139 | -0.145 [-0.169, -0.123] | +0.51 | not / carries |
| velgiven | 7.707 | -0.298 [-0.498, -0.093] | -0.001 [-0.004, +0.003] | -0.00 | +0.283 | -0.001 [-0.001, -0.000] | +0.00 | not / not |
| vx110 | 7.774 | -0.230 [-0.429, -0.022] | +0.067 [+0.034, +0.100] | +0.23 | +0.225 | -0.059 [-0.068, -0.049] | +0.21 | not / not |
| yaw0 | 7.708 | -0.297 [-0.498, -0.094] | +0.001 [-0.006, +0.008] | +0.00 | +0.283 | -0.001 [-0.003, +0.001] | +0.00 | not / not |
| yawvel | 7.713 | -0.292 [-0.496, -0.091] | +0.005 [-0.004, +0.018] | +0.02 | +0.283 | -0.000 [-0.002, +0.001] | +0.00 | not / not |
| posecv | 7.711 | -0.294 [-0.495, -0.092] | +0.004 [-0.004, +0.012] | +0.01 | +0.279 | -0.004 [-0.006, -0.002] | +0.01 | not / not |
| cv | 7.714 | -0.290 [-0.506, -0.086] | +0.007 [-0.047, +0.064] | +0.02 | +0.393 | +0.109 [+0.096, +0.123] | -0.39 | not / worse |
| biasmean | 6.895 | -1.110 [-1.372, -0.851] | -0.813 [-0.987, -0.640] | -2.73 | +0.842 | +0.558 [+0.516, +0.602] | -1.97 | worse / worse |
| biasresid | 8.102 | +0.097 [+0.009, +0.194] | +0.395 [+0.192, +0.608] | +1.33 | -0.083 | -0.367 [-0.433, -0.305] | +1.29 | carries / carries |
| biasnav | 6.799 | -1.206 [-1.473, -0.935] | -0.908 [-1.092, -0.720] | -3.05 | +0.890 | +0.607 [+0.561, +0.652] | -2.14 | worse / worse |
| biasdenav (post hoc) | 8.096 | +0.091 [+0.000, +0.188] | +0.388 [+0.188, +0.593] | +1.31 | -0.066 | -0.350 [-0.416, -0.287] | +1.23 | carries / carries |
| O1 (P2H path, shipped speed; offline) | 8.067 | +0.062 [-0.021, +0.156] | +0.359 [+0.184, +0.542] | +1.21 | +0.009 | -0.274 [-0.332, -0.218] | +0.97 | carries / carries |
| O2 (shipped path, P2H speed; offline) | 7.541 | -0.464 [-0.671, -0.279] | -0.166 [-0.290, -0.058] | -0.56 | +0.277 | -0.006 [-0.010, -0.002] | +0.02 | worse / not |
| P2-F main (no hinge) | 7.680 | -0.325 [-0.530, -0.124] |  |  | +0.308 |  |  |  |
| P2-F zero | 7.989 | -0.016 [-0.050, +0.021] | +0.309 [+0.108, +0.514] | +0.95 | +0.057 | -0.251 [-0.305, -0.197] | +0.82 | carries / carries |
| P1-F (no inputs) | 8.019 | +0.014 [-0.023, +0.051] |  |  | +0.055 |  |  |  |
| shipped, warp frames | 7.507 | -0.498 [-0.734, -0.275] |  |  | +0.345 |  |  |  |
| main, warp frames | 7.558 | -0.446 [-0.666, -0.232] |  |  | +0.336 |  |  |  |
| zero, warp frames | 7.516 | -0.488 [-0.719, -0.261] | -0.042 [-0.275, +0.199] | -0.09 | +0.391 | +0.056 [-0.007, +0.120] | -0.17 | not / not |

"vs main" is against the main arm of the same frame protocol (for the warp rows: P2H main under warp). Labels follow the rule above ("worse": the CI excludes 0 on the wrong side). Tables: `wod_p2h_diag/arms.csv` (all columns), `bias_stats.json` (bias rms per arm), `meta.json` (equivalence, frame-source interaction per seed).

Per seed (d RFS vs shipped, s0 / s1): main -0.288 / -0.307; zero -0.024 / -0.016; biasresid +0.091 / +0.104; biasmean -1.051 / -1.170; biasnav
-1.180 / -1.232; accnav -0.279 / -0.249; vx110 -0.219 / -0.242; cmd0 -0.380 / -0.392; P2-F -0.282 / -0.368; P1-F +0.016 / +0.012; O1 +0.058 / +0.066.
Both seeds agree in sign and size on every arm.

Readings of the rule:
- **Bias parts: the constant carries the loss** (biasresid r = 1.33 with the vs-main CI excluding 0, both seeds; biasmean / biasnav make it
  three to four times worse). The parts are not additive (the model is nonlinear: constant alone -1.11, with the remainder -0.30); the
  decomposition is by arm, not an additive model.
- **Ego channels: none carries the RFS loss.** Acceleration rescaled to NAVSIM moments (`accnav`) carries half of the ADE loss (r 0.51; zeroing
  it doubles it), which is decision 155's "acceleration moves ADE, not RFS" again. The speed dose (`vx110`, +10 % vx) moves RFS +0.067 and the
  plan ~4 % further: a percent-level speed-mapping error cannot produce -0.30, and the two speed sources agree to 0.001.
- **Hinge: not** (P2-F d -0.325 [-0.530, -0.124], CI excludes 0 like P2H's). **Fine-tune without inputs: not** (P1-F +0.014 [-0.023, +0.051]).
- **Longitudinal: carries it** (O1 r 1.21, ADE r 0.97); O2 (shipped path, P2H speed) is worse than P2H (-0.46), the path of P2H is if
  anything better than shipped's.
- **Frame source: carries it by the rule** (P2H - shipped under warp +0.052 [-0.204, +0.289], CI includes 0; interaction I +0.349 [+0.130,
  +0.567] >= 0.15; s0 +0.072 / I +0.361, s1 +0.031 / I +0.338; ADE@3s under warp -0.009 [-0.077, +0.053]). Under warp, shipped itself drops
  0.50 RFS (7.507), and P2H with the bias zeroed is level with it (7.516): the bias is neutral on warp frames and harmful on real ones.

### Where the RFS is lost (seed-mean P2H vs shipped; d RFS per stratum, cluster-mean aggregation, so strata do not add)

| stratum | n | d RFS [95% CI] | O1 | zero | biasresid | biasmean | under warp (vs shipped@W) | lon 3 s (m) | lon 5 s (m) | plan / log 5 s, P2H / shipped |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| all | 479 | -0.297 [-0.499, -0.093] | +0.062 | -0.020 | +0.097 | -1.110 | +0.052 | +0.00 | -0.07 | 0.96 / 0.97 |
| stopped v<0.5 | 120 | -0.695 [-1.158, -0.254] | -0.077 | +0.011 | -0.033 | -1.843 | -0.481 | +0.94 | +3.13 | 1.00 / 0.62 |
| stopped, log stays (<1 m @5s) | 25 | -1.721 [-2.761, -0.595] | -0.033 | -0.014 | -0.144 | -2.416 | -1.231 | +1.03 | +4.04 |  |
| stopped, log moves (>=1 m @5s) | 95 | -0.494 [-0.973, +0.026] | -0.123 | +0.019 | -0.027 | -1.581 | -0.291 | +0.91 | +2.89 | 1.00 / 0.62 |
| launch v<2 & log5s>5m | 111 | +0.029 [-0.396, +0.419] | -0.012 | +0.002 | -0.141 | -0.563 | -0.042 | +0.65 | +1.57 | 0.93 / 0.73 |
| slow 0.5-5 | 179 | -0.131 [-0.451, +0.193] | +0.178 | +0.014 | +0.015 | -0.433 | -0.056 | +0.36 | +0.01 | 1.08 / 1.05 |
| mid 5-12 | 142 | -0.063 [-0.345, +0.223] | +0.001 | -0.047 | +0.176 | -0.772 | +0.518 | -0.82 | -2.15 | 0.91 / 0.99 |
| fast >=12 | 38 | -0.668 [-1.037, +0.049] | +0.007 | -0.085 | +0.347 | -3.045 | +0.366 | -1.54 | -2.81 | 0.90 / 0.96 |
| turn intent L/R | 52 | -0.553 [-1.115, +0.199] | +0.075 | -0.051 | +0.214 | -1.491 | -0.720 | +0.43 | +0.98 | 0.90 / 0.87 |
| straight intent | 427 | -0.261 [-0.462, -0.064] | +0.035 | -0.019 | +0.085 | -1.063 | +0.125 | -0.05 | -0.20 | 0.97 / 0.98 |
| lead_prob>0.5 (shipped) | 165 | -0.414 [-0.676, -0.172] | -0.018 | -0.032 | +0.069 | -1.310 | +0.137 | -0.41 | -0.82 | 0.97 / 1.00 |
| lead_prob<=0.5 | 314 | -0.215 [-0.499, +0.062] | +0.122 | -0.024 | +0.169 | -1.042 | -0.207 | +0.22 | +0.33 | 0.96 / 0.96 |
| night (luma < 50) | 133 | -0.208 [-0.735, +0.310] | +0.032 | -0.040 | +0.032 | -0.814 | +0.289 | +0.65 | +1.40 | 1.00 / 0.93 |
| day (luma >= 120) | 325 | -0.360 [-0.599, -0.115] | +0.065 | -0.014 | +0.119 | -1.230 | -0.041 | -0.29 | -0.71 | 0.95 / 0.99 |
| cluster Pedestrian | 52 | -0.784 [-1.319, -0.185] | +0.175 | +0.001 | +0.055 | -1.721 | -0.718 | -0.33 | -0.93 | 0.90 / 0.98 |
| cluster Multi-Lane Maneuvers | 42 | -0.810 [-1.538, -0.183] | +0.031 | -0.012 | +0.132 | -1.606 | -0.532 | -0.15 | +0.51 | 0.92 / 0.97 |

lon = mean longitudinal offset of the P2H plan from shipped's in shipped's heading; plan / log = median 5 s plan displacement over logged
displacement (frames with logged displacement > 2 m). Full table with lateral offsets and clusters: `wod_p2h_diag/strata.csv`.

- Standstill: P2H plans 0.9 m (3 s) / 3.1 m (5 s) of motion from rest where shipped plans almost none. On stopped frames whose log moves,
  P2H's 5 s displacement matches the log (ratio 1.00 vs shipped 0.62) and still scores lower (-0.49): the raters' trust region at low speed is
  half-size and their trajectories start slower than the log. Where the car stays stopped (25 frames) the creep costs -1.72.
- At speed (>= 12 m/s, and behind a lead) P2H is shorter (-1.5 m / -0.4 m at 3 s); under warp frames these strata turn positive (+0.37 mid,
  +0.52 fast) while the standstill loss remains (-0.48). The offset thus has a frame-protocol part (the speed read from warp vs real pairs, as
  Stage A: warp makes shipped's plan 5 % faster on NAVSIM) and a part that is not (the launch from rest).
- Clusters: Pedestrian -0.78 [-1.32, -0.19] and Multi-Lane Maneuvers -0.81 [-1.54, -0.18] are the distinguishable losses; Foreign Object Debris
  +0.28 and Cyclist +0.03 are not losses. Day -0.36, night -0.21 as in decision 155; the remainder (`biasresid`) is >= -0.14 in every
  stratum and the zero-bias arm within +-0.12.

### What a fix would be, and whether the WOD training lane covers it

- **Serving-time fix without WOD data:** feed the adapter's ego-dependent part only, i.e. subtract the navtest-mean bias (`biasdenav`):
  RFS +0.091 [+0.000, +0.188] vs shipped on WOD (recovery 1.31, ADE@3s -0.07 m), the same as `biasresid`. Untested on NAVSIM: the constant was learned there and probably carries part of P2H's navtest / navhard gain (under the
  warp frames it was trained on), so this fix is only usable if NAVSIM keeps its score with it, or as a per-board switch (real-frame boards
  without the constant). That NAVSIM check (P2H with `biasdenav` on navtest, inference only) is the next cheap step.
- **Training fix:** train the adapter under real-frame pairs (protocol R / real 10 Hz frames) so no warp correction is folded into the
  offset, or regularise the bias's mean towards zero.
- **The WOD training lane** (`plans/2026-10-07-wod-parity-prereg.md`: P2 recipe on WOD r2-train, real frames in all 9 slots, no hinge)
  refits the adapter including its constant on real WOD frames and WOD targets, so it covers the WOD board by construction, and its lack of
  a hinge does not matter (P2-F loses like P2H). It does not answer whether one navtrain-trained adapter can serve both boards; its prereg's
  motivating hypothesis (the acceleration spread) is ruled out here (`accnav` RFS +0.03, n.s.).

### Checks

- Multi-bias serving path (one warm-up, last step re-run per bias from a state snapshot) vs the stored decision 155 runs, all 1 437 frames:
  plan xy mean 0.0098 m, p99 0.060 m, max 0.16 m (s0; s1 0.0099 / 0.058 / 0.14), TensorRT fp16 level; zero / command-zero arms likewise match
  the stored ones on the 12-frame smoke (max 0.02-0.09 m against 0.4-2 m between arms).
- Warp frames: the four keys are bit-identical to the real frames; synthesised slots PSNR 16-31 dB against the real 10 Hz frames on three
  smoke targets (op_interp's WOD warp level). 1 of 1 437 targets has < 16 contiguous history frames and stays real.
- The retime operator applied to a plan with its own profile returns it exactly (max 0.0 m).

### Caveats

- Open loop, WOD val only. About 20 contrasts; the conclusions rest on arms that meet the rule in both seeds with large margins (the bias
  parts, O1, the warp interaction); single marginal CIs are not used.
- The warp arm re-synthesises only the last 1.5 s from the 2 Hz keys (the frames the plan reads at the 0.2 s lattice); earlier warm-up
  frames stay real. It reproduces training protocol W's source, not its 4-slot queue.
- The "launch prior" reading of the standstill creep (navtrain targets often launch from rest while the vision sees a static pair) is an
  interpretation, not tested here.
- `biasdenav` was added after the first table was read (post hoc); it is the deployable variant of the pre-registered `biasresid`.
- Strata use shipped's lead_prob as the "following" label and the night_gap luma labels; RFS per stratum is the cluster-mean aggregation.
