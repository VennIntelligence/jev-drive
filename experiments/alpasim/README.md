# alpasim: AlpaSim E2E Closed Loop Challenge feasibility

status: live
decisions: 184, 185, 188, 189, 199 (inputs: 142, 144, 104, 116, 133, 149, 169, 170, 174, 177)
index: 48 public scenes: WA-JEPA 0.978, SH30 0.947, AlpaSim-aligned AP2 0.932, LTF 0.874
key: docs/alpasim.md, experiments/alpasim/results/sh30_smoke.md, experiments/alpasim/lib/sh30_core.py, experiments/alpasim/lib/sh30_driver.py, experiments/alpasim/scripts/run_native.py, experiments/alpasim/scripts/run.sh, experiments/alpasim/scripts/driver_tap.py, docs/openpilot-interface.md, docs/zeroshot-adapters.md, scripts/op_lb.py, experiments/hugsim/lib/zs_agent.py

**Question.** Can SH30 (openpilot Cinque + adapter + drivable hinge, navtest EPDMS 89.55) enter the nuPlan track of the AlpaSim E2E
Closed Loop Challenge 2026 by 2026-10-31, and what does the public 1 485-scene closed-loop suite cost on our box.

**Conclusion so far.** AlpaSim's nuPlan track runs natively on the box at the deployed commit (no Docker): 2.96 s per scene at 8
concurrent rollouts on one card; the shipped LTF sample scores 0.8735 mean scene score on 48 public scenes (decision 184). A scene is
5.5 s, 10 `drive` calls at 2 Hz, with no history before t = 0. Nothing has been registered or submitted; both are the user's actions.

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

**Next.** The at-fault collisions of SH30 and AP2 case by case (zeros decide this board); WA-JEPA latency on an idle card; the full public suite once the remaining 14 asset shards are on disk (`fetch_data.sh all`, running 2026-10-08); the two
collisions and the 11 slow scenes; step latency under 0.1 s (warp on the GPU or cross-session batching); a Docker host for the
submission image and a read-only-root test; seed 1.

**Read more.** [docs/alpasim.md](../../docs/alpasim.md).

<!-- files:begin -->
<!-- files:end -->
