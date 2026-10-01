## Before a long run (moved from CLAUDE.md)

Applies to OUR code only. Third-party libraries and other people's reproduction or baseline code are run
as they ship: measure them, do not rewrite them.
- Estimate wall time first. Anything above ~3 h gets a profiling pass before it starts: measure on a
  subset, find the actual bottleneck (disk, RAM, CPU, GPU compute, GPU memory, network), then rewrite the
  hot path as an expert would - parallel, vectorised, overlapping I/O with compute - until it is gone.
- Tune the defaults for our box, not for a generic machine: batch size, DataLoader workers, prefetch,
  dtype, chunk sizes. Derive limits at runtime (`jevdrive.common.n_cpus()`, free VRAM, free disk) so the
  code still runs elsewhere, but leave OUR best values as the defaults.
- Verify the optimized code gives the same results (numerical equivalence on a subset), then record the
  before/after numbers and the bottleneck in the experiment's README or plan.
- Staged launch for every job of more than ~1 h (closed-loop B2D, CARLA generation, long training, big feature runs):
  run 1 unit, inspect it; then ~10 units (routes, worlds, folds, shards), inspect them against a written sanity
  checklist (completion / blocked / crash rates, value ranges against a known reference, outputs non-degenerate);
  only then the full batch. A pilot that fails the checklist stops the batch; never find out after the full run.
- While a long job runs cleanly, report every 3-5 hours, not per file or step. Report at once only for an error,
  a stall, a decision, or completion.
- The box is elastic: cards, cgroup CPU quota, RAM and pids.max change between instances (7 cards / 175 cores /
  644 GiB on 2026-09-28, 3 cards / 75 cores / 276 GiB on 2026-10-01; RTX 6000D, 83.6 GiB each; the host always shows
  208 CPUs). Never hardcode them: read `python -m jevdrive.cl probe`, size pools with `jevdrive.common.n_cpus()`.
  Run independent work in parallel across cores (one job per file/archive/shard), keep hot data in RAM, batch on
  the GPU and overlap I/O with compute.
