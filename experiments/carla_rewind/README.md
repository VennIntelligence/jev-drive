# carla_rewind: In-place CARLA rewind vs from-scratch forks

status: concluded
decisions: 65
index: rewind not equivalent (ego speed p95 0.305 vs 0.3); only 3.0x faster

**Question.** Can in-place rewind replace per-branch from-scratch runs for WL fork data?

**Conclusion.** Not equivalent: best `tree+w40f` missed ego speed p95 (0.305 vs 0.3) and openpilot `temporal` cosine (0.87 vs 0.99); speedup 3.0x, not 6.7x (decisions 65). WL-2 generates from scratch.

**Read more.** research/carla-rewind-branching.md, research/wl2-results.md, `git show bcbdde4:todos/2026-09-29-carla-rewind.md`

<!-- files:begin -->
## Files

- `carla_rewind.py` (lib): In-place rewind of a running …
- `rewind_eval.py` (archive): CARLA rewind fidelity check
- `rewind_gen.sh` (archive): CARLA rewind fidelity runs
- `poc_carla_rewind.py` (archive): POC CARLA Rewind & Branching Simulation …
- `rewind_physics_probe.py` (archive): Vehicle-physics probe for the CARLA …
- `rewind_probe.sh` (archive): One CARLA server on the carla-rewind …

[archive/](archive/) 5 one-off code · [results/](results/) 4 result files · [lib/](lib/) 1 library
<!-- files:end -->
