# alpasim: AlpaSim E2E Closed Loop Challenge feasibility

status: live
decisions: 184 (inputs: 104, 116, 133, 149, 169, 170, 174, 177)
index: runs natively on the box, 2.96 s / scene; LTF sample 0.8735 on 48 scenes
key: docs/alpasim.md, experiments/alpasim/scripts/run_native.py, experiments/alpasim/scripts/run.sh, experiments/alpasim/scripts/driver_tap.py, docs/openpilot-interface.md, docs/zeroshot-adapters.md, scripts/op_lb.py, experiments/hugsim/lib/zs_agent.py

**Question.** Can SH30 (openpilot Cinque + adapter + drivable hinge, navtest EPDMS 89.55) enter the nuPlan track of the AlpaSim E2E
Closed Loop Challenge 2026 by 2026-10-31, and what does the public 1 485-scene closed-loop suite cost on our box.

**Conclusion so far.** AlpaSim's nuPlan track runs natively on the box at the deployed commit (no Docker): 2.96 s per scene at 8
concurrent rollouts on one card; the shipped LTF sample scores 0.8735 mean scene score on 48 public scenes (decision 184). A scene is
5.5 s, 10 `drive` calls at 2 Hz, with no history before t = 0. Nothing has been registered or submitted; both are the user's actions.

**How to run.** [docs/alpasim.md](../../docs/alpasim.md#running-it-on-our-box-measured-2026-10-08-decision-184): `scripts/setup_env.sh`,
`fetch_data.sh`, `setup_ltf.sh`, then `run.sh` as a pool job.

**Next.** SH30 as a driver: causal W-protocol frames from 2 Hz CAM_F0, cold-start rule for the first 1.5 s, command from the route,
3-scene smoke; a Docker host for building the submission image; the remaining 14 asset shards need a disk decision.

**Read more.** [docs/alpasim.md](../../docs/alpasim.md).

<!-- files:begin -->
<!-- files:end -->
