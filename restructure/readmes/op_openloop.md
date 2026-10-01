# op_openloop: openpilot on open-loop boards (2 Hz, NAVSIM nav, WOD)

status: concluded
decisions: 34, 36, 37, 39, 66, 73
headline: openpilot NAVSIM 47-52 PDMS is mostly input protocol: frame interpolation lifts Cinque navtest 52.1 to 84.2

**Question.** How does unmodified openpilot score on open-loop boards (WOD-E2E RFS, NAVSIM, nuScenes), and how much of its low NAVSIM score is the 2 Hz input contract rather than the model?

**Conclusion.** WOD-E2E val RFS: Cinque 8.005 vs cv 7.10 (decisions 34); nuScenes L2 worse than constant velocity, collision better (decisions 39). The NAVSIM 47-52 PDMS is mostly the input protocol: GIMM-VFI frame interpolation inside the contract lifts Cinque native navtest PDMS 52.1 -> 84.2 and navhard EPDMS 9.3 -> 33.3 (corrected in place, decisions 37; rig/time-axis sensitivity in decisions 36). Navigation arm: laneChange desire at t0 - 1.0 s gives navtest PDMS +0.72 [+0.48, +0.95] (turns +2.14), turn desires never help, navhard no gain (decisions 66). OpenBLAS audit: no NAVSIM number needed changing (decisions 73).

**Read more.** research/openpilot-openloop-standing.md, research/openpilot-openloop-integration.md, research/navsim-openblas-audit.md, research/navhard-deficit-breakdown.md, docs/navsim.md; pre-registrations: `git show bcbdde4:todos/2026-09-25-openpilot-openloop-comparison.md`, `git show bcbdde4:todos/2026-09-29-op-leaderboard.md`

<!-- files:begin -->
<!-- files:end -->
