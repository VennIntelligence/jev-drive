# op_parity: Cinque with WA-JEPA's inputs, fine-tuned on navtrain

status: live
decisions: (pending) (inputs: 92, 96, 104, 111, 118, 128, 135, 136, 137, 138, 139)
index: Cinque + ego / pose / cmd / side cams vs WA-JEPA, equal inputs and data
key: experiments/op_parity/plans/2026-10-06-parity-prereg.md, lib/parity_adapter.py, experiments/op_parity/scripts/pp_prep.py, experiments/op_parity/scripts/pp_train.py, experiments/op_parity/scripts/pp_eval.py, experiments/op_parity/scripts/pp_unfreeze.py, experiments/op_parity/scripts/pp_navhard.py, jevdrive/navsim_zs.py, experiments/hugsim/results/wajepa_ref.md, lib/agent_hinge.py

**Question.** With no input disadvantage relative to WA-JEPA (command, ego velocity / acceleration, 4-pose history, side and rear cameras)
and the same fine-tuning data (navtrain), how much of WA-JEPA's lead over openpilot Cinque (HUGSIM 0.451 vs 0.278, navtest EPDMS 91.71)
remains: comma pretraining vs V-JEPA 2 + nuPlan video pretraining.

**Design.** Cinque's vision encoder stays frozen (its hidden tokens are cached once); the new inputs enter through a zero-initialised bias on
the 9 policy context frames (lib/parity_adapter.py: ego MLP tokens + side / rear camera tokens from Cinque's own encoder, read by 32 slot
queries); the plan pathway is fine-tuned on navtrain with an anchor (same navtrain frames, inputs zeroed, distilled to shipped). Arms P0
shipped, P1 inputs zeroed, P2 + ego / pose / command, P3 + side / rear cameras. Readouts: NAVSIM navtest EPDMS on the devkit that
reproduced WA-JEPA's 91.71, HUGSIM 64 (spins, launch stalls). Pre-registration: [plans/2026-10-06-parity-prereg.md](plans/2026-10-06-parity-prereg.md).

**Pilot (2026-10-06).** navtest EPDMS (devkit that reproduces WA-JEPA 91.71): P0 81.11, P1 81.73, P2 86.57 (+4.84 [+4.11, +5.55] vs P1),
P3 86.10, WA-JEPA 91.71 (P2 - WA-JEPA -5.14 [-6.02, -4.27]); side cameras add nothing; HUGSIM spin10 spins P1 8 / P2 7 / P3 8, launch-spin
onset unchanged. Gate holds: [results/pilot.md](results/pilot.md). Frame-protocol Stage A (no training): the image-pair gap sets the plan
speed (0.5 s pairs: x1.93), GIMM vs real frames -0.38 EPDMS: [results/stageA.md](results/stageA.md).

**Stage B (2026-10-06).** Train protocol x arm x 2 seeds on the pilot set: P2 navtest EPDMS G 86.63, W 85.73, N 82.63; the inputs help
less without synthesis (interaction N vs G -2.08 [-2.63, -1.52]); pre-registered rule picks **W** (G - W 0.90, guards pass), N fails (4.0):
[results/stageB.md](results/stageB.md). HUGSIM under `spec` (29 stuck / spinner scenarios): P2 ends every stuck run (24 -> 0) but fg
collisions 1 -> 10, HD 0.453 vs P0 0.306, P2 - P1 +0.104 [-0.042, +0.248]: [results/hugsim_spec.md](results/hugsim_spec.md).

**Full run (2026-10-06, protocol W, 103 k navtrain tokens, 2 seeds).** navtest EPDMS P0 80.51, P1 81.98, P2 88.21, P3 88.16, WA-JEPA 91.71:
P3 - P1 +6.18 [+5.40, +7.00], P3 - WA-JEPA -3.55 [-4.27, -2.84] (gap 11.2 -> 3.5; rest is DAC / TTC / NC). HUGSIM 64 HD exam / spec: P0 0.263 /
0.294, P2 0.396 / 0.393, WA-JEPA 0.451 (P2 - WA-JEPA -0.055 [-0.135, +0.025]); stuck runs 16 / 24 -> 0. Side cameras add nothing.
[results/full.md](results/full.md), [results/hugsim_full.md](results/hugsim_full.md).

**navhard two-stage (2026-10-06, protocol W, same devkit).** Combined P0 28.05, P1 27.60, P2 29.31, P3 28.56, WA-JEPA 35.41 (its released
checkpoint in our harness; request path checked on 1 021 navtest tokens, -0.45 vs its stored scores); references shipped GIMM 33.33,
factor_wm S3 35.77. P2 - P1 stage 1 +8.85 [+5.11, +12.65] but stage 2 -2.82; P2 - WA-JEPA -6.10 [-10.04, -2.30]; W frames cost the
shipped model 5.3 points here: [results/navhard.md](results/navhard.md).

**Vision unfreeze pilot (2026-10-06, pre-registered, 17 k navtrain tokens, 3 000 steps, 2 seeds).** navtest EPDMS F (frozen) 87.39, U1
stage 4 87.40, U1L LoRA 87.40, U2 whole encoder 87.51 (+0.11 [+0.01, +0.22]), V 1.40 m virtual camera 87.56 (+0.17 n.s.); no arm reaches
the +0.5 gate, the line stops (no full run). The remaining gap to WA-JEPA is not reachable by adapting the front encoder on navtrain:
[results/unfreeze_pilot.md](results/unfreeze_pilot.md), plan [plans/2026-10-06-unfreeze-prereg.md](plans/2026-10-06-unfreeze-prereg.md).

**Hinge lane (2026-10-06, pre-registered).** P2 recipe + footprint drivable-SDF hinge (lambda 10), 2 seeds: navtest EPDMS 88.67 vs P2 88.21 (+0.46 [+0.33, +0.61]), DAC failures 3.83% vs 4.33% (-0.50 pp [-0.64, -0.36]; predicted -0.9), navhard G +1.49, HUGSIM unchanged: [results/hinge.md](results/hinge.md), plan [plans/2026-10-06-hinge-prereg.md](plans/2026-10-06-hinge-prereg.md).

**Gap page (2026-10-06).** Where P2 still loses to WA-JEPA, per sub-metric (Shapley split of the EPDMS gap), navtest / navhard / HUGSIM 64, with GIF cases: open [results/gap/index.html](results/gap/index.html) (generators `scripts/pp_gap_*.py`).

**Joint action head (2026-10-07, pre-registered, small read stopped).** Action head (on-policy pathway) trained jointly with the P2 + hinge plan on
own-plan / logged curvature labels: offline gain on logged curvature 0.68 -> 0.87-0.99 and agreement with the plan 0.99 above 3 m/s, but HUGSIM turn23
`spec` HD JC 0.178 / JL 0.213 / JW 0.279 vs HP 0.251 (none >= +0.03): at 1-3 m/s the a_lat / v^2 command is noise (abs kappa 0.03-0.04 vs plan 0.015).
[results/joint_action.md](results/joint_action.md), plan [plans/2026-10-07-joint-action-prereg.md](plans/2026-10-07-joint-action-prereg.md).

**B2D `spec_plan_smooth` (2026-10-07, small read, stopped).** Smoothed plan curvature as the lateral source on 3 B2D routes (4 turns), P2 / P2H10 / shipped:
junction turns 0 / 3 under both `action` and `plan_smooth`; the failure moves from "not chosen / late" to "requested in time and at 0.9-2 x the needed curvature, but the
car is stopped in the junction" (the plan asks for a near stop): [results/b2d_plan_smooth.md](results/b2d_plan_smooth.md).

**B2D P2 cache (2026-10-07, built and validated, nothing trained).** The collected PDM-Lite dataset (decision 152) as an op_parity training cache: 383 089 rows at native 0.2 s
frame pairs (per-tick token store, 26 GB), teacher, MKZ-footprint hinge labels, raw distance to the next turn, route split b2d/b2dc-v2-train / -val, collision rule. P2-F-s0 on the cached tokens reproduces
the collection check (turn-direction 0.865 vs 0.863): [results/b2d_cache.md](results/b2d_cache.md). Draft pre-registration with the open decisions at the top:
[plans/2026-10-07-b2d-p2-prereg.md](plans/2026-10-07-b2d-p2-prereg.md). Scripts: `scripts/b2d_prep.py`, `b2d_split.py`, `b2d_validate.py`, `b2d_open_read.py`, `b2d_trainer_check.py`.

**Turn training pilot (2026-10-07, pre-registered).** Turn-balanced sampling (T1), + anchor off on turns (T2), + late lateral weight (T3) on the pilot P2H recipe:
closure of the > 20 deg navtest gap to WA-JEPA 0.01-0.02 (T1 > 20 deg +0.13 [-0.52, +0.76]; > 45 deg +0.97 but DAC failures unchanged), T2 / T3 fail the -0.3 guard;
gate not passed, no HUGSIM: [results/turn_train.md](results/turn_train.md), plan [plans/2026-10-06-turn-train-prereg.md](plans/2026-10-06-turn-train-prereg.md).

**Four directions (2026-10-07, CPU diagnosis of P2H, `spec_plan_smooth` on HUGSIM).** Sharp turns, wide-turn edge grazes, vehicle contacts and night, each sized
(replacement oracles, Shapley) on navtest / navhard / HUGSIM 64 with mechanism shares: collisions are the HUGSIM board (42.8 HD x 100 of headroom, mostly scripted
oncoming actors WA-JEPA hits too; reachable part the stopped lead, 3.8), navtest turn failures are inside corner cuts and replay-only grazes (sharp 1.04, wide 1.00,
~half decided at the frozen encoder), HUGSIM sharp-turn failures are 3-5x too fast junction entries; no night units on this tier. Ranked fixes with draft
pre-registrations (agent hinge, replay hinge, lead margin): [results/four_dirs.md](results/four_dirs.md).

**Lead standstill margin (2026-10-07, pre-registered, stopped at the offline gate).** Execution-layer rule (`jevdrive/openpilot/lead_margin.py`, HUGSIM opt
`lead_margin`, NAVSIM bench option `:lm`): stored D3b HUGSIM plans converted to avoiding in 1 / 10 scenarios (gate 7; the lead head reads +4.4 m too far in
the last 0.5 s, or tracks another object), navtest -0.75 [-0.96, -0.55] with NC + TTC failures 531 -> 565; no closed-loop run:
[results/lead_margin.md](results/lead_margin.md), plan [plans/2026-10-07-lead-margin-prereg.md](plans/2026-10-07-lead-margin-prereg.md), `scripts/lm_offline.py`.

**WOD-E2E val, P2H vs shipped (2026-10-07, measurement only).** Same harness as decision 34 (shipped RFS 8.005 reproduced), real 10 Hz frames, intent -> command, past states -> ego features: P2H10-F seed mean RFS 7.708 vs 8.005 (-0.297 [-0.501, -0.094]; s0 -0.287, s1 -0.306), ADE@3s +0.284 m, ADE@5s +0.551 m; day RFS -0.36 [-0.61, -0.12], night -0.21 [-0.72, +0.33] (n.s.). With the bias zeroed P2H weights read -0.02: the loss is the ego-state bias, not the weights: [results/wod_p2h.md](results/wod_p2h.md), `scripts/pp_wod.py`.

**WOD-E2E val, P2 recipe trained on WOD train (2026-10-07, pre-registered).** WOD's own inputs (decision 155 mapping), 179 k r2-train rows from the cached
Cinque trunk, 10 000 steps x 2 seeds: WP2 RFS 8.111 vs shipped 8.005 (+0.106 [-0.060, +0.279], n.s.; seeds +0.109 / +0.104), vs P2H +0.40 [+0.21, +0.59];
ADE@3s 0.574 vs 1.041 m (-0.47); the inputs-zeroed control WP1 is shipped (+0.007), so the gain is the WOD inputs, not the fine-tune; night RFS gap 0.49 ->
0.23 (dd +0.27 [-0.16, +0.74], n.s.), night ADE@3s -0.81 m; RFS bounded by the log's own 8.13: [results/wod_parity.md](results/wod_parity.md), plan
[plans/2026-10-07-wod-parity-prereg.md](plans/2026-10-07-wod-parity-prereg.md), `scripts/wod_parity.py`.

**WOD-E2E val, where WP2 loses RFS (2026-10-07, pre-registered, offline).** Gap to the top-rated rater trajectory 1.476 (log 1.456, shipped 1.582): 45 % of
frames lose nothing, 9 % are floored and carry 34 %; WP2's path at the top-rated speed profile recovers +0.72 [+0.52, +0.93] (49 %), the top-rated path at
WP2's speed +0.30 (20 %). WP2 equals the log only in total: -0.76 [-1.21, -0.34] at standstill, +0.47 [+0.13, +0.82] at 5-12 m/s. Where the log is rated
low the raters' first choice goes further (52 % of the log's gap), not slower. Nothing is recoverable offline: seed ensemble -0.009, arc-length scaling
out-of-fold +0.014 (global) / +0.018 / +0.072 (stratified; in-sample +0.11 / +0.20), all CIs contain 0:
[results/wod_gap.md](results/wod_gap.md), plan [plans/2026-10-07-wod-gap-prereg.md](plans/2026-10-07-wod-gap-prereg.md), `scripts/wod_gap.py`.
- **WOD standstill launch** (2026-10-08): WP2's 0.76 below the log at standstill is half the base model's late launch (all arms, mostly stop-sign junctions) and half the ego-only adapter acting blind at rest; gating the adapter off below 0.5 m/s (WLG, `pp_train --stop-gate 0.5`) gives stopped +0.39 [+0.10, +0.75] vs WP2, moving -0.02, all frames 8.187 (+0.18 [+0.05, +0.33] vs shipped); traffic lights cost no points: [results/wod_launch.md](results/wod_launch.md), plan [plans/2026-10-08-wod-launch-prereg.md](plans/2026-10-08-wod-launch-prereg.md), `scripts/wod_launch*.py`.
- **WOD rater preference into System 1** (2026-10-08): WLG fine-tuned on the val rater trajectories by sequence 5-fold; out-of-fold `top` +0.091 [+0.012, +0.170] and `f20` +0.089 [+0.018, +0.156] vs WLG (in-sample +0.35 / +0.28), `rank` and metric `hinge` null, standstill unmoved, 8-9 % of the selector ceiling: [results/wod_pref.md](results/wod_pref.md), plan [plans/2026-10-08-wod-pref-prereg.md](plans/2026-10-08-wod-pref-prereg.md), `scripts/wod_pref.py`.
- **WOD slot protocol** (2026-10-08): serving WOD val with the oldest policy slot zeroed (the navtrain training protocol, 8 real + 1 zero) instead of 9 real does not recover the P2H10 loss, it deepens it: P2H10 -0.208 [-0.328, -0.085], SH30 -0.182 [-0.309, -0.057], shipped -0.086 [-0.170, -0.002], WP2 +0.028 [-0.060, +0.116] RFS; P2H10 vs shipped goes -0.296 -> -0.505: [results/wod_slot.md](results/wod_slot.md), plan [plans/2026-10-08-wod-slot-prereg.md](plans/2026-10-08-wod-slot-prereg.md), `scripts/wod_slot.py`.

**Why P2H loses on WOD (2026-10-07, inference only).** The input-independent part of the adapter bias carries it: P2H with the bias minus its mean scores +0.097 [+0.009, +0.194] vs shipped (main -0.297), the constant alone -1.11; no ego-mapping intervention moves RFS (acceleration, speed source, yaw, pose history, all within 0.07); hinge irrelevant (P2-F -0.33), P1 +0.01. The loss is the speed profile (P2H path at shipped speed +0.06): rolling from standstill (stopped frames -0.69, +0.94 m at 3 s) and shorter at speed; under warp-synthesised WOD frames P2H is level with shipped (+0.05, interaction +0.35 [+0.13, +0.57]). Serving fix without WOD data: subtract the navtest-mean bias, +0.091 [+0.000, +0.188] (NAVSIM effect untested): [results/wod_p2h_diag.md](results/wod_p2h_diag.md), `scripts/pp_wod_diag.py`.

**Replay hinge (2026-10-07, pre-registered, all gates passed).** Drivable hinge on a differentiable torch proxy of the devkit's LQR replay (`lib/lqr_proxy.py`,
1 cm p95 / 100% DAC agreement with the devkit) + 0.5 m front-corner margin on turns: thin-decoder gate turning DAC failures -4.6 pp, pilot RMP-F-s0 vs HP-F-s0
EPDMS +0.43 / DAC failures -0.40 pp, full RMH10-F x 2 seeds navtest **89.19 vs P2H 88.67 (+0.52 [+0.38, +0.67])**, navhard +0.25 (n.s.), HUGSIM 64
`spec_plan_smooth` +0.002 (n.s.). The gain is all replay-only departures; raw-plan departures rise (the plan pre-compensates the devkit tracker):
[results/replay_hinge.md](results/replay_hinge.md), plan [plans/2026-10-07-replay-hinge-prereg.md](plans/2026-10-07-replay-hinge-prereg.md), `scripts/rh.py`, `scripts/rh_chain.sh`.

**Representation fix design (2026-10-07, design only, not launched).** Fix 4 of four_dirs compared (encoder unfreeze + dense SDF head, JEPA front tokens as adapter
memory, distillation, Cinque pre-head features); recommendation: WA-Cf (ceiling, already cached) and frozen V-JEPA 2.1 front tokens as P2H adapter memory, offline decoder
gate on the 3 154 navtest > 20 deg tokens first: [plans/2026-10-07-representation-design.md](plans/2026-10-07-representation-design.md).

**Ego-history probe (2026-10-07, inference only).** P2H / WA-JEPA plans on navtest with the ego history (velocity, acceleration, poses) replaced by a constant-velocity one, on sharp-turn approach tokens (T 368), pre-flip tokens (P 129) and matched straights: planned 4 s distance on T +0.13 m [-0.35, +0.56] (P2H), +0.04 [-0.23, +0.28] (WA-JEPA); the effect rides on the t0 acceleration input, P2H is 1.4-2.1 x as history-sensitive as WA-JEPA but equally on straights: the "slows before turns because the history shows it" hypothesis is not supported: [results/ego_history_probe.md](results/ego_history_probe.md).

**HUGSIM ax probe (2026-10-07).** P2H10-F-s0 / s1 `spec_plan_smooth` with the sim acceleration fed as ax (unmodified) vs ax = ay = 0 at the model input (`parity.zero_acc`, default off), 5 sharp-turn + 5 fast straight scenarios by a pre-written rule: entry speed at the turn changes by a median -1.09 m/s (3 / 5 scenarios below -1; stays 6-10 m/s, every sharp scenario still fails the same way), the straights slow by 1.0-1.6 m/s too (one collapses at 4.4 m/s): the ax feedback is not the main cause of the fast entries: [results/hugsim_ax_probe.md](results/hugsim_ax_probe.md).

**Agent hinge pilot (2026-10-07, pre-registered, gate stop).** Pilot P2H recipe + hinge of the plan footprint against the logged boxes of the agents
ahead (lambda_a 10, margin 0.5 m; K 32 and no side margin outside the lateral corridor after the pre-training geometry check, declared before training):
navtest NC + TTC failures 2.44% -> 2.27% (-0.16 pp [-0.26, -0.08], gate -0.3), EPDMS +0.18 [+0.05, +0.31] (gate +0.2), EP -0.10; gains on stopped-lead and
over-speed tokens (EPDMS +2.3), but 73% of the remaining NC failures still overlap a labelled box. Full stage not run:
[results/agent_hinge.md](results/agent_hinge.md), plan [plans/2026-10-07-agent-hinge-prereg.md](plans/2026-10-07-agent-hinge-prereg.md).

**Next.** Decision entries by main (navhard, unfreeze, turn training, four directions); review of the three 2026-10-07 drafts.

**HUGSIM.** Serving path and equivalence tests: [results/hugsim_harness.md](results/hugsim_harness.md). Pilot arms on the 10 spinner
scenarios: [results/hugsim_spin10.md](results/hugsim_spin10.md) (spins P1 8, P2 7, P3 8 of 10; the launch spin onset is unchanged).
Under the `spec` preset (decision 118's lateral path) on the 10 + 19 decision-118 stuck scenarios: [results/hugsim_spec.md](results/hugsim_spec.md)
(P2 stuck 0 of 29 vs 24, HD 0.453 vs 0.306, fg collisions 10 vs 1).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.

**Representation fix (2026-10-07, pre-registered, stage 0 + stage 1; stage 2 needs review).** Offline thin decoders on all 3 154 navtest turns > 20 deg:
WA-JEPA's NAVSIM-fine-tuned front tokens (WA-Cf, a diagnostic upper bound, not a method) added to Cinque's cut DAC failures 10.59% -> 6.37%; the
original frozen V-JEPA 2.1 ViT-L closes only 0.35 of that (gate 0.5, fails; alone it is no better than Cinque), Cinque's own pre-head map 0.14.
Policy pilot (s2-s4, 2 seeds): P2H + WA-Cf as adapter memory passes every criterion: navtest EPDMS +0.95 [+0.47, +1.37], turn DAC failures
-1.33 pp, NC + TTC -0.63 pp. Reading: the gain needs a NAVSIM-supervised encoder; next is distillation into Cinque's whole encoder with WA-Cf as the
teacher. [results/representation.md](results/representation.md), plan [plans/2026-10-07-representation-design.md](plans/2026-10-07-representation-design.md).

**Turn probe (2026-10-07, pre-registered, read-out probes only).** Drivable-boundary read-out from frozen tokens, navtest > 45 deg: Cinque 1.21 m vs WA-Cf 0.86 m (+0.35 [0.32, 0.39]), turn-vs-straight degradation larger for Cinque by 0.11 [0.07, 0.15]: rule verdict "representation lacks it (turn-specific)", but two thirds of the gap is already there on straight tokens and frozen V-JEPA 2.1 reads the boundary as well as WA-Cf without the driving gain of decision 160. [results/turn_probe.md](results/turn_probe.md), plan [plans/2026-10-07-turn-probe-prereg.md](plans/2026-10-07-turn-probe-prereg.md), `scripts/turn_probe.py`.

**Turn oracle (2026-10-08, pre-registered, privileged input as a probe only).** True drivable SDF as 32 adapter memory tokens vs the same tokens shuffled across logs, pilot scale, seed 0: > 45 deg inside-cut rate 4.55% vs 4.22% (closure -0.08 [-0.19, +0.02]), rule verdict "plan head", but the pathway never read the tokens (memory masked = same score). Step 2: a fresh thin head with the true SDF cuts > 20 deg DAC failures by 3.1 pp yet closes only 0.21 [0.03, 0.37] of the > 45 deg inside cut; 9 000 steps or hinge lambda 30 / 0.5 m do not make the P2 pathway use it (closure -0.05 / +0.03, no pilot). The inside cut is not a missing-geometry problem; the stronger hinge itself gives EPDMS +0.68. [results/turn_oracle.md](results/turn_oracle.md), plan [plans/2026-10-08-turn-oracle-prereg.md](plans/2026-10-08-turn-oracle-prereg.md), `scripts/turn_oracle.py`.

**Stop gate across boards (2026-10-08, pre-registered, serving side only).** WOD's standstill rule (adapter bias = 0 when fed speed < 0.5 m/s) on the stored P2H10 weights: navtest EPDMS -1.50 [-1.85, -1.19] (standstill tokens -19), HUGSIM 64 HD -0.084 [-0.141, -0.035] (stuck 0 -> 11.5; the gate latches, 0 oscillating scenarios); `biasdenav` -6.71 [-7.59, -5.90] / -0.169 [-0.253, -0.091]: the navtrain-trained bias is what launches the car, the WOD fix would have to be trained in: [results/stop_gate_xboard.md](results/stop_gate_xboard.md), `scripts/sg_xboard.py`.

**S2 go / hold and selector (2026-10-08, pre-registered, privileged oracles as ceilings only, no training).** System 1 + System 2 late fusion on WOD val: a perfect selector among 20 path x speed candidates of WP2 is worth +1.07 [+0.89, +1.26] RFS (72 % of the gap to the top-rated trajectory), 80 % of it from the speed choice alone and 4 % needing path and speed together; a binary go / hold symbol decoded with a fixed train launch profile is worth +0.21 [-0.08, +0.51] at standstill (5 buckets +0.44 [+0.03, +0.88], continuous +1.07). Zero-shot Qwen3-VL-4B delivers none: go / hold +0.12 [-0.27, +0.52] at standstill and -0.23 [-0.37, -0.10] on moving frames, joint selection -0.74 [-1.00, -0.49]; it answers the rule state (holds at every red light and stop sign) where the raters' first choice moves within 5 s. [results/s2_gohold.md](results/s2_gohold.md), plan [plans/2026-10-08-s2-gohold-prereg.md](plans/2026-10-08-s2-gohold-prereg.md), `scripts/s2_gohold.py`.

**S2 thin head (2026-10-08, pre-registered, rater labels only out of fold by sequence).** A head that picks one of WP2's 20 path x speed candidates (ceiling +1.07): imitating the log's choice is worth -0.06 [-0.25, +0.13] (hindsight labels cannot help on all frames; +0.51 at standstill). A ridge head on rater scores gains +0.148 [+0.062, +0.248] out of fold from ego state + command + the plan alone, all of it on the 52 turn-intent frames (+1.12; 0.000 on the 427 straight frames). No visual stream adds: Cinque -0.046 [-0.084, -0.013], frozen Qwen3 one frame -0.036 [-0.078, +0.005], 4 frames x 0.2 s -0.054, x 0.5 s -0.082 [-0.141, -0.027] against that floor; Qwen - Cinque +0.011 [-0.027, +0.048]; multi-frame - single-frame -0.019 / -0.047. A LoRA fine-tune pilot (decoder layers 15-18, multi-frame + ego streams, hindsight pre-training then rater k-fold) is +0.005 [-0.080, +0.086] against the frozen feature. F30, path-only and a margin rule do not improve on F20 / argmax; on shipped the head gives +0.05 to +0.07. Qwen3 feature path 0.32 s (one frame) / 0.83 s (four) per decision, the head 0.014 ms. [results/s2_thinhead.md](results/s2_thinhead.md), [plans/2026-10-08-s2-thinhead-prereg.md](plans/2026-10-08-s2-thinhead-prereg.md).

**Strong hinge (2026-10-08, pre-registered).** P2H10 recipe with the drivable hinge at lambda 30 / margin 0.5 m (no memory tokens), 2 seeds: navtest EPDMS 89.55 vs P2H10 88.67 (+0.87 [+0.64, +1.15]), vs RMH10 +0.35; raw-plan out-of-bounds -1.14 pp (> 20 deg -2.5 pp) with replay-only unchanged, so a real path gain unlike the replay hinge; > 45 deg inside-cut -0.49 pp, EP and straight unchanged; navhard +1.83 [+0.42, +3.37], HUGSIM flat (turn23 -0.023): [results/strong_hinge.md](results/strong_hinge.md), plan [plans/2026-10-08-strong-hinge-prereg.md](plans/2026-10-08-strong-hinge-prereg.md).

**Hinge scan (2026-10-08, pre-registered).** Pilot-scale lambda {10, 30, 100} x margin {0.25, 0.5, 1.0} around SHP (30 / 0.5): no point wins by the rule (best lambda 30 / margin 1.0 EPDMS +0.32 [-0.01, +0.68] with straight -0.31 and EP -0.10); strength is one trade-off axis, SH30 stays, no full run. SH30 on WOD val: RFS 7.734 vs P2H10 7.708 (+0.026 [-0.027, +0.078]), vs shipped -0.271; turn endpoints slightly less inward, no cross-domain gain: [results/hinge_scan.md](results/hinge_scan.md), plan [plans/2026-10-08-hinge-scan-prereg.md](plans/2026-10-08-hinge-scan-prereg.md).

**Standstill by the scene, per board (2026-10-08, pre-registered; question reframed before scoring).** P2H10 with the stop gate trained in and no anchors on gated rows (P2HGA): navtest 88.46 vs 88.67 (-0.21 [-0.34, -0.08]), HUGSIM 64 HD 0.411 vs 0.431 (-0.020 [-0.043, -0.001]), WOD +0.159 vs P2H10: no collapse (the serving gate cost -1.50 / -0.084) but no gain, the launch prior moves from the adapter constant into the plan weights; with anchors kept the trained gate fails like the serving one (pilot -1.37). One adapter on navtrain + WOD (MX): navtest 88.79 (+0.12 [-0.09, +0.32]), WOD 8.103 (= WP2), HUGSIM 0.335 (-0.096 [-0.158, -0.040]); its constant is as large and as load-bearing as the single-domain ones: [results/mixed_domain.md](results/mixed_domain.md), plan [plans/2026-10-08-mixed-domain-prereg.md](plans/2026-10-08-mixed-domain-prereg.md), `scripts/mixed_domain.py`, `scripts/mixed_domain_chain.sh`.

**Turn prior (2026-10-08, pre-registered).** What the decision-173 floor head does on the 52 turn-intent frames: it cuts the 5 s travel (hold / creep on 70 % of them, median 0.44 of the plan) and moves the path 2-4 m outward; the standstill-start part (+1.80 on WP2) is what WLG already fixed (head on WLG +0.28 [-0.10, +0.86]); the 0.5-3 m/s speed part stays (+0.81 [+0.31, +1.40]). Out of fold the head adds +0.054 [-0.023, +0.136] on WLG and +0.008 / +0.004 on the preference plans (nested); no fixed rule reproduces it (log-derived alpha 1.02 = identity, -0.005; out-of-fold rule +0.030 [-0.052, +0.098]); training arm not triggered. Turn-gated head (post hoc) +0.075 [+0.017, +0.142]. [results/turn_prior.md](results/turn_prior.md), [plans/2026-10-08-turn-prior-prereg.md](plans/2026-10-08-turn-prior-prereg.md), figures [figs/turn_prior/](figs/turn_prior/).

**Hinge m10 (2026-10-08, pre-registered).** SH30 recipe at lambda 30 / margin 1.0, full scale, 2 seeds: navtest EPDMS 89.57 vs SH30 89.55 (+0.02 [-0.36, +0.41]), straight EPDMS -0.54 [-0.96, -0.18] (lane keeping 97.4 -> 95.9), EP flat, turn > 20 deg +1.32, raw-plan out-of-bounds -0.76 pp, navhard +0.74 [-0.78, +2.43], HUGSIM HD -0.039 [-0.103, +0.021]; fails the replace rule, SH30 stays: [results/hinge_m10.md](results/hinge_m10.md), plan [plans/2026-10-08-hinge-m10-prereg.md](plans/2026-10-08-hinge-m10-prereg.md).

**Turn ceiling (2026-10-08, pre-registered, privileged ceiling, CPU only).** Best-of-K over a small family around SH30's own plan (lateral offset +-0.5 m, curvature gain 0.85 / 1.15, speed 0.8 / 1.2) on the 3 154 navtest tokens > 20 deg: best-of-19 +12.20 EPDMS [+10.57, +13.76] (line +4.0; WA-JEPA - SH30 there +6.52), > 45 deg +14.24, 20-45 deg +10.31; one axis alone +6.0 (offset) / +7.8 (curvature) / +7.7 (speed), needing two axes at once only +1.42; no fixed transform helps (best -0.33, curvature x 1.15 -18.9), so there is no global under-turn: the gain is per-token repair, DAC failures 7.31% -> 0.10%. WA-JEPA's 7.92 lead on > 45 deg is 62% DAC. `score-poses` gained `--traffic non_reactive` (navtest's policy; the reactive default failed the identity gate on EP). navtrain has 28 323 tokens > 20 deg and no v2 metric cache: [results/turn_ceiling.md](results/turn_ceiling.md), plan [plans/2026-10-08-turn-ceiling-prereg.md](plans/2026-10-08-turn-ceiling-prereg.md), figures [figs/turn_ceiling/](figs/turn_ceiling/).
