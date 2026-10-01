# carla_rewind: In-place CARLA rewind vs from-scratch fork generation

status: concluded
decisions: 65
headline: CARLA in-place rewind is not equivalent to from-scratch generation (ego speed p95 0.305 vs 0.3); speedup only about 3.0x

**Question.** Can rewinding the world to a fork tick inside one Bench2Drive route replace per-branch from-scratch runs for WL fork data, and can maps be reused across runs.

**Conclusion.** Not equivalent (decisions 65): best method `tree+w40f` restored the hazard pedestrian (position p95 0.036 m, starts walking 97.1%) but missed ego speed p95 (0.305 vs 0.3), collision agreement, cut-in car (0.39 m) and openpilot `temporal` cosine (0.87 vs 0.99). WL-2 generates from scratch; zygote (pre-warmed route process) is equivalent and saves about 12 s per run; map reuse is rejected. Rewind speedup is about 3.0x (the proposal's 6.7x does not hold); rewind stays as an off-by-default option in `scripts/wl_fork_agent.py`.

**Read more.** research/carla-rewind-branching.md, research/wl2-results.md, `git show bcbdde4:todos/2026-09-29-carla-rewind.md`, `git show bcbdde4:tmp/2026-09-29-carla-rewind-state.md`.

<!-- files:begin -->
<!-- files:end -->
