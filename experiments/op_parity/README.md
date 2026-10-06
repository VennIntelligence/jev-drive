# op_parity: Cinque with WA-JEPA's inputs, fine-tuned on navtrain

status: live
decisions: (pending) (inputs: 92, 96, 104, 111, 118, 128, 135, 136, 137, 138, 139)
index: Cinque + ego / pose / cmd / side cams vs WA-JEPA, equal inputs and data
key: experiments/op_parity/plans/2026-10-06-parity-prereg.md, lib/parity_adapter.py, experiments/op_parity/scripts/pp_prep.py, experiments/op_parity/scripts/pp_train.py, experiments/op_parity/scripts/pp_eval.py, experiments/op_parity/scripts/pp_unfreeze.py, experiments/op_parity/scripts/pp_navhard.py, jevdrive/navsim_zs.py, experiments/hugsim/results/wajepa_ref.md

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

**Next.** Decision entries by main (navhard, unfreeze).

**HUGSIM.** Serving path and equivalence tests: [results/hugsim_harness.md](results/hugsim_harness.md). Pilot arms on the 10 spinner
scenarios: [results/hugsim_spin10.md](results/hugsim_spin10.md) (spins P1 8, P2 7, P3 8 of 10; the launch spin onset is unchanged).
Under the `spec` preset (decision 118's lateral path) on the 10 + 19 decision-118 stuck scenarios: [results/hugsim_spec.md](results/hugsim_spec.md)
(P2 stuck 0 of 29 vs 24, HD 0.453 vs 0.306, fg collisions 10 vs 1).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
