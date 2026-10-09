# alpasim: AlpaSim E2E Closed Loop Challenge feasibility

status: live
decisions: 184, 185, 188, 189, 199, 201, 202, 205, 209, 210, 211 (inputs: 142, 144, 104, 116, 133, 149, 169, 170, 174, 177)
index: 48 public scenes: WA-JEPA 0.978, SH30 0.947, AlpaSim-aligned AP2 0.932, LTF 0.874
key: docs/alpasim.md, experiments/alpasim/results/sh30_smoke.md, experiments/alpasim/lib/sh30_core.py, experiments/alpasim/lib/sh30_driver.py, experiments/alpasim/scripts/run_native.py, experiments/alpasim/scripts/run.sh, experiments/alpasim/scripts/driver_tap.py, docs/openpilot-interface.md, docs/zeroshot-adapters.md, scripts/op_lb.py, experiments/hugsim/lib/zs_agent.py

**Question.** Can SH30 (openpilot Cinque + adapter + drivable hinge, navtest EPDMS 89.55) enter the nuPlan track of the AlpaSim E2E
Closed Loop Challenge 2026 by 2026-10-31, and what does the public 1 485-scene closed-loop suite cost on our box.

**Conclusion so far.** AlpaSim's nuPlan track runs natively on the box at the deployed commit (no Docker): 2.96 s per scene at 8
concurrent rollouts on one card; the shipped LTF sample scores 0.8735 mean scene score on 48 public scenes (decision 184). A scene is
5.5 s, 10 `drive` calls at 2 Hz, with no history before t = 0. Team `lao-siji` was registered by the user on 2026-10-09 (pending organiser review); nothing has been submitted. Both are the user's actions.

**How to run.** [docs/alpasim.md](../../docs/alpasim.md#running-it-on-our-box-measured-2026-10-08-decision-184): `scripts/setup_env.sh`,
`fetch_data.sh`, `setup_ltf.sh`, then `run.sh` as a pool job.

**SH30 smoke (2026-10-08, decision 185).** SH30-F-s0 serves the driver API with real inference on all 480 decisions of 48 public
scenes: mean scene score 0.9465, two zeros (at-fault collisions); the LTF sample on the same scenes 0.8735, five zeros. Execution
evidence only: one run, one seed, 48 scenes of one drive. Cold start = constant-velocity back-extrapolation + back-warped first frame.
105 ms median per `drive` (target 0.1 s), 3.5 GiB. Table, mismatches and figures: [results/sh30_smoke.md](results/sh30_smoke.md)
(`figs/sh30_frames_right_turn.jpg`: the frames the model saw with its plan; `figs/sh30_bev_first8.jpg`: plans against the driven track).
Run: `run.sh <dir> sh30 ...` in place of `ltf`.

**WA-JEPA driver (2026-10-08, decision 188).** The released WA-JEPA through its shipped NAVSIM agent, unmodified, behind the driver API
(`lib/wajepa_core.py`, `lib/wajepa_driver.py`, `run.sh <dir> wajepa`): 480 / 480 real inferences, mean scene score 0.9777 fp32 / 0.9792 bf16, one
zero (offroad, cause open). Inference 1.3 s (fp32) / 0.37 s (bf16) per `drive` against the 0.1 s target; cold start (oldest frame repeated,
its own client's rule) shifts the first plan by 3.4 m offline. Execution evidence only: [results/wajepa_smoke.md](results/wajepa_smoke.md)
(`figs/wajepa_frames_right_turn.jpg`, `figs/wajepa_bev_first8.jpg`).

**AP2: openpilot aligned to AlpaSim's inputs (2026-10-08, decision 189, pre-registered).** SH30's recipe retrained on navtrain with rows built as
AlpaSim delivers a decision (1-4 keyframes with the served `backwarp` rule, AlpaSim's own route generator -> 4-way command, the simulator's ego
definitions): `AP2-AB-s0`. Offline on navtest under AlpaSim inputs vs SH30 as served: ADE at 1 keyframe 0.992 -> 0.666 m, EPDMS +3.19
[+2.07, +4.37]; with full history level (+0.14 [-0.38, +0.66]). Closed loop, 48 scenes: 0.9320 (41 at score 1, 3 at-fault collisions) vs SH30
0.9465 (35, 2): not separable. Inputs used / not usable / substituted, ruled-out options (zero-slot cold start, route waypoints into the
adapter) and mismatches: [results/ap2_smoke.md](results/ap2_smoke.md) (`figs/ap2_frames_right_turn.jpg`, `figs/ap2_bev_first8.jpg`), plan
[plans/2026-10-08-alpasim-aligned-prereg.md](plans/2026-10-08-alpasim-aligned-prereg.md). Run: `run.sh <dir> ap2`.

**C0: 400 landed public scenes (2026-10-09, decision 199).** SH30 0.9306, AP2-AB-s0 0.9335, 15 at-fault events each; a second AP2 run is scene-for-scene identical (deterministic sim); per-scene best-of-two 0.9627 (+0.0292 over the best single driver, C2 line met as an upper bound). Part007 landed later and is uncovered. [results/c0_public400.md](results/c0_public400.md).

**C1: zero-score review and arbitration (2026-10-09, decision 202).** No zero of SH30 (23) or AP2 (24) on the 400 scenes is a lead-vehicle collision: all 17 collisions are side / cut-in contacts after the ego left its lane. The main mechanism, shared by both drivers (12 / 13 zeros), is a 1-2 m sideways shift on a straight stretch away from the turn the route announces; an offline replay with the command forced rules out a late command, the cause is open. Plans leave on their own (tracking error <= 0.2 m), not a cold-start effect; slow scenes are a closed-loop drift to lower speed. Arbitration: disagreement gates are negative offline, the standstill gate reads +0.0024 [-0.0066, +0.0131] on 300 held-out scenes, line not met, dropped. [results/c1_zero_review.md](results/c1_zero_review.md) (case strips in `figs/c1/`), [results/c1_arbitration.md](results/c1_arbitration.md), plan [plans/2026-10-09-c1-arbitration-prereg.md](plans/2026-10-09-c1-arbitration-prereg.md).

**C0b: 700 landed public scenes, five drivers (2026-10-09, decision 201).** SH30 0.9140, AP2 0.9223, OT30-F-s0 / s1 0.9258 / 0.9335, WA-JEPA (reference, fp32) 0.9111; CIs (27 logs) about +-0.04. OT30 against the registered line: mean of the seeds minus SH30 +0.0156 [-0.0011, +0.0326] (lower bound misses 0 by 0.001), at-fault collision zeros 19 -> 11 / 11: not a candidate by the line; the gain is collisions, not offroad / corridor (31 vs 31 / 27). WA-JEPA's 48-scene lead was a part001 effect (part008 / 009: 0.83). SH30 + AP2 oracle 0.9501 (+0.0278). Results [results/c0b_public700.md](results/c0b_public700.md), per-scene table `results/c0b_per_scene.csv` / `.json`, plan [plans/2026-10-09-ot30-closedloop-prereg.md](plans/2026-10-09-ot30-closedloop-prereg.md). Run: `scripts/c0b_chain.py` (pool jobs + stall watchdog), `scripts/c0b_report.py`.

**OT2-C: adapter ensemble (2026-10-09, decision 211, pre-registered).** Averaging the plans of OT30-F-s0 + s1 on one shared encoder pass (`lib/ens_driver.py`, `run.sh <dir> ens`): 0.9312 on the 700 scenes, -0.0022 [-0.0097, +0.0045] against the better member (line +0.008: not met; stop rule applied). The seeds' 4 s endpoints differ by 0.05 m (median) on a shared state, fork decisions 0.4 %: the best-of-two ceiling (+0.0133) is the maximum of two noisy outcomes of one policy. One more member costs 20 ms (110 ms per step). The simulator is deterministic only for a fixed scene list. [results/ot2_c_ensemble.md](results/ot2_c_ensemble.md), plan [plans/2026-10-09-ot2-ensemble-prereg.md](plans/2026-10-09-ot2-ensemble-prereg.md).

**M1: cause and fix of the pre-turn shift (2026-10-09, decision 205, pre-registered).** The route does not bend in these scenes: the ego yaws about 5 deg on a straight road and the route's first waypoint, 42 m ahead, moves sideways in the rig frame. Trigger: the strong drivable hinge (lambda 30 / 0.5 m); amplifier: the plan continues the yaw rate shown in the synthesised slots (slope 0.93, as the logs do on navtest) and the MPC executes it, so heading error grows linearly. Serving the lambda-10 checkpoint `P2H10-F-s0` with the driver unchanged: held-out 300 scenes 0.9288 vs SH30 0.8919, +0.0369 [+0.0087, +0.0668], zeros 16 vs 27, M-class zeros 2 vs 8, all three registered lines met; 700 scenes 0.9484 vs 0.9140 (s1 0.9496). An input-side yaw damping (`SH30_MOTION`, off by default) removes the M zeros but breaks turns. Cost: the open-loop gain of the strong hinge (decision 170), a per-board recipe. [results/m1_preturn_shift.md](results/m1_preturn_shift.md) (before / after strips `figs/m1/pair_*.jpg`, `figs/m1/heading_growth.png`), plan [plans/2026-10-09-m1-yaw-damping-prereg.md](plans/2026-10-09-m1-yaw-damping-prereg.md). Run: `SH30_TAG=P2H10-F-s0 run.sh <dir> sh30`.

**OT2-A: off-track ladder (2026-10-09, decision 209).** The shipped plan does not return from a lateral offset in the off-track probe (response within 4 s -0.03 [-0.21, +0.11] on +-0.5 m rows, -0.15 [-0.28, -0.05] on +-1.5 m rows); adapters add 0.24-0.33, off-track rows 0.73-1.00. "Adaptation erodes recovery" is dropped. [../op_parity/results/ot_ladder.md](../op_parity/results/ot_ladder.md).

**OT2-B: off-track rows under the AlpaSim input standard (2026-10-09, decision 210, pre-registered, trimmed after decision 205).** `APO-a05m10` (AP2 + 10 % of +-0.5 m rows, `scripts/ap2_ot.py`) 0.9409 on the 700 scenes, +0.0177 [+0.0077, +0.0277] over AP2 (two seeds each), at-fault events 32 -> 16.5: a candidate by the registered lines, no guardrail run, below the lambda-10 P2H10 (0.948 / 0.950). 25 % is worse than 10 % (-0.0286 [-0.0412, -0.0177]). The +-1.5 m trainings were cut; the `ot2` row caches are kept for lane OT3 ([results/ot2/ot2_cache_keys.md](results/ot2/ot2_cache_keys.md)). Second baseline seeds: SH30-F-s1 0.9219, AP2-AB-s1 0.9240. [results/ot2_b_dose.md](results/ot2_b_dose.md), plan [plans/2026-10-09-ot2-dose-prereg.md](plans/2026-10-09-ot2-dose-prereg.md).

**Next.** A lambda-10 checkpoint with off-track rows, and closed-loop recovery rows with a yaw rate (decision 205); P2H10's remaining zeros (turns, 7 of 16 held out); the at-fault collisions of SH30 and AP2 case by case (zeros decide this board); WA-JEPA latency on an idle card; the full public suite once the remaining 14 asset shards are on disk (`fetch_data.sh all`, running 2026-10-08); the two
collisions and the 11 slow scenes; step latency under 0.1 s (warp on the GPU or cross-session batching); a Docker host for the
submission image and a read-only-root test; seed 1.

**Read more.** [docs/alpasim.md](../../docs/alpasim.md).

<!-- files:begin -->
<!-- files:end -->
