# cl_infra: Closed-loop harness cost and worker profile

status: concluded
decisions: 16, 17, 83, 222
index: 220 routes take a measured 3.11 h; reduced profile

**Question.** Can CARLA closed loop run on the box, at what cost, with which worker profile?

**Conclusion.** Needs libegl1 (decisions 16); 220 routes take 3.11 h, 209 complete (17). Default `reduced`: 59.1 vs 59.8 tick/s, 3.3x fewer threads (83).

**Zero-copy pool merges (2026-10-20).** `rf_pait.py merge` now indexes the source tokens (`front.parts.json`, `jevdrive.tokens`)
instead of copying them. On on1_JTPH it is bitwise identical through every reader path, gives an identical 60-step training,
and takes 85 s / 35 GB instead of about 10 min / 173 GB: [results/zero-copy](results/zero-copy/README.md).

**Read summaries from a facts cache (2026-10-22).** Each AlpaSim read chunk writes its per-window facts at the end of
the read (`jevdrive.facts`, ~9 s); summaries join them. nuplan_dag Part 1: 23 s on 4 cores instead of 179 s on 32,
outputs identical: [results/read_summaries.md](results/read_summaries.md).

**Read more.** research/b2d-closed-loop/index.html, docs/closed-loop-runbook.md, docs/bench2drive-cost.md

<!-- files:begin -->
## Files

- [`rs_bench.sh`](scripts/rs_bench.sh) (scripts): Read-summary timings, old path vs the facts cache, and the identity check; with `rs_check.py` ([results](results/read_summaries.md))
- [`sigkill_trigger.py`](scripts/sigkill_trigger.py) (scripts): The SIGKILLs of one night against the box sampler; with `memwatch.py` (1 s sampler), `canary.py`, `canary_fork.py` ([results](results/sigkill-trigger/README.md), decision 222)
- [`zc_merge.sh`](scripts/zc_merge.sh) (scripts): Zero-copy pool merge check lane; with `zc_check.py` (bitwise rows / pinned gather) ([results](results/zero-copy/README.md))
- [`pool_usage.py`](scripts/pool_usage.py) (scripts): One day of the GPU pool: idle card-hours by cause, why jobs waited, old vs new accounting replay ([results](results/pool-fix/README.md), decision 181)
- `carla_parallel.sh` (archive): How many headless CARLA servers fit on …
- `carla_threads.py` (archive): CARLA server thread census and a …
- `carla_bench.py` (archive): Headless CARLA smoke test and …
- `b2d_sweep.py` (archive): Run one route under a list of harness …
- `b2d_scale.py` (archive): Scaling ladder for the Bench2Drive …
- `infra_scale.sh` (archive): CARLA harness scaling ladders of the …
- `infra_verify.sh` (archive): Behaviour equivalence of the harness …
- [`cl_worker_profile.py`](https://github.com/VennIntelligence/jev-drive/blob/9e7ec92abe143091022e2b04e04d0e8a58e36c90/experiments/cl_infra/archive/cl_worker_profile.py) (archive): PDM-Lite on 40 fixed Bench2Drive …
- `cl_profile_report.py` (archive): Worker-profile experiment readout
- `make_infra_scale_figs.py` (archive): Figures and the flat result table of …

[archive/](archive/) 22 one-off code · [results/](results/) 38 result files · [figs/](figs/) 5 figures
<!-- files:end -->
