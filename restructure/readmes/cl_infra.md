# cl_infra: Closed-loop harness cost, CARLA scaling, worker profile

status: concluded
decisions: 16, 17, 83
headline: CARLA 0.9.15 renders on the box; 220 routes take a measured 3.11 h (209 complete); reduced profile: 3.3x fewer threads

**Question.** Can CARLA closed-loop evaluation run on the headless box, what does a full Bench2Drive round cost, and which thread/worker profile should be the default.

**Conclusion.** CARLA 0.9.15 renders on the box once `libegl1` is installed (decisions 16). Cost is hours, not days, but the first extrapolation (about 1.1 h for 220 routes) was corrected in place to a measured 3.11 h, 209/220 routes complete, with Town12/Town13 (not "Large Map") being the expensive and crashing maps (decisions 17). Default worker profile is `reduced` (CARLA thread pools 4 each, client 8, numeric threads 2): throughput 59.1 vs 59.8 tick/s for stock, same-route DS identical for 95.1% vs 94.5% stock-stock, 3.3x fewer threads; camera-less agents scale to 12 workers per card, camera agents stay at 6 (decisions 83). PDM-Lite only; the W = 12 stock-faster reading is unrepeated.

**Read more.** docs/closed-loop-runbook.md, docs/closed-loop-acceptance.md, docs/bench2drive-cost.md, docs/carla.md, research/carla-efficiency.md, `git show bcbdde4:todos/2026-10-01-cl-lib.md`, `git show bcbdde4:todos/2026-09-25-closed-loop-infra-acceptance.md`, `git show bcbdde4:todos/2026-09-22-b2d-large-maps.md`.

<!-- files:begin -->
<!-- files:end -->
