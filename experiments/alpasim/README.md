# alpasim: AlpaSim E2E Closed Loop Challenge feasibility

status: live
decisions: (pending) (inputs: 104, 116, 133, 149, 169, 170, 174, 177)
index: nuPlan track, 2 Hz cameras, NMPC tracks our plan; off-road is a hard zero
key: docs/alpasim.md, docs/openpilot-interface.md, docs/zeroshot-adapters.md, scripts/op_lb.py, experiments/hugsim/lib/zs_agent.py

**Question.** Can SH30 (openpilot Cinque + adapter + drivable hinge, navtest EPDMS 89.55) enter the nuPlan track of the AlpaSim E2E
Closed Loop Challenge 2026 by 2026-10-31, and what does the public 1 485-scene closed-loop suite cost on our box.

**Status (2026-10-08).** Rules, container contract, scoring and the leaderboard are read: [docs/alpasim.md](../../docs/alpasim.md).
Local run and the SH30 driver estimate are in progress. Nothing has been registered or submitted; both are the user's actions.

**Next.** AlpaSim on the box without Docker (shipped starter driver, then the shipped LTF sample on a handful of navtest scenes),
measured cost, then the SH30 driver estimate and a 3-scene smoke.

**Read more.** [docs/alpasim.md](../../docs/alpasim.md).

<!-- files:begin -->
<!-- files:end -->
