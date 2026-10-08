# alpasim: AlpaSim E2E Closed Loop Challenge feasibility

status: live
decisions: 184, 185 (inputs: 104, 116, 133, 149, 169, 170, 174, 177)
index: SH30 driver runs: 0.9465 on 48 public scenes (LTF 0.8735), 105 ms / step
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

**Next.** The full public suite once the remaining 14 asset shards are on disk (`fetch_data.sh all`, running 2026-10-08); the two
collisions and the 11 slow scenes; step latency under 0.1 s (warp on the GPU or cross-session batching); a Docker host for the
submission image and a read-only-root test; seed 1.

**Read more.** [docs/alpasim.md](../../docs/alpasim.md).

<!-- files:begin -->
<!-- files:end -->
