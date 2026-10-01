# cl_infra: Closed-loop harness cost and worker profile

status: concluded
decisions: 16, 17, 83
index: 220 routes take a measured 3.11 h; reduced profile
key: scripts/carla_parallel.sh, scripts/carla_threads.py, scripts/carla_bench.py, scripts/b2d_sweep.py, scripts/b2d_scale.py, scripts/infra_scale.sh, scripts/infra_verify.sh, scripts/lanes/cl_worker_profile.py, jevdrive/cl_profile_report.py, scripts/make_infra_scale_figs.py

**Question.** Can CARLA closed loop run on the box, at what cost, with which worker profile?

**Conclusion.** Needs libegl1 (decisions 16); 220 routes take 3.11 h, 209 complete (17). Default `reduced`: 59.1 vs 59.8 tick/s, 3.3x fewer threads (83).

**Read more.** research/carla-efficiency.md, docs/closed-loop-runbook.md, docs/bench2drive-cost.md

<!-- files:begin -->
<!-- files:end -->
