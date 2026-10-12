# rule_oracle: rule-based planner as AlpaSim nuPlan driver

status: closed
decisions: -
index: closed 2026-10-12 (user): scoped only, not run; zero-shot form killed by d269 (heads unusable)

**Question.** Can a mature rule-based planner (the PDM-Closed idea) drive the local AlpaSim nuPlan 18-decision loop (held 460 scenes),
as a privileged oracle on the true map and replayed traffic (ceiling probe only, never an image) and zero-shot on the frozen openpilot
model's lane / edge / lead heads, and is there a ceiling above the learned line (`YS-L18`)?

**Conclusion.** Closed by the user on 2026-10-12 without running: a 450-line planner is not a credible ceiling on this road geometry, and the zero-shot form lost its input when decision 269 found the openpilot lane / edge / lead heads unusable on nuPlan city scenes. Scoped (2026-10-12): both forms are feasible; the cheaper path is a ~450-line numpy PDM-Closed re-implementation
in AlpaSim's own venv that reads map, replayed traffic and logged path through the runtime's scene loader by the `debug_info.scene_id` the
driver receives locally (same objects and frame as the simulator and evaluator); four registered arms (log replay L0, log route A1, honest
route A2, lead-only A2L) cost about 2.5 card-hours.

**Next.** Implement S0 (CPU checks) and S1 smoke, then the held read against the registered lines of the plan; form 2 only if R1 holds and
R4 does not.

**Read more.** [plans/2026-10-12-rule-oracle-scoping.md](plans/2026-10-12-rule-oracle-scoping.md) (Chinese; interface, code sketch,
cost table, registered lines, leak guards).

<!-- files:begin -->
## Files

- `2026-10-12-rule-oracle-scoping.md` (plans): RULE-ORACLE 范围界定：规则规划器（PDM-Closed 思路）作 …
- `README.md` (experiments/alpasim_long): full-length nuPlan rollouts in local …
- `sh30_driver.py` (experiments/alpasim/lib): SH30 as an AlpaSim nuPlan-track driver
- `run.sh` (experiments/alpasim/scripts): One AlpaSim closed-loop run as plain …
- `driver_tap.py` (experiments/alpasim/scripts): Logging gRPC tap between the AlpaSim …
- `ao_e1.py` (experiments/anchor_offset/scripts): does the served plan's lateral offset …
- `zm_read.py` (experiments/zeros_mech/scripts): what the zeros of the on-policy S-hinge …
- `long_read.py` (experiments/alpasim_long/scripts): per-scene and per-decision metrics of …
- `alpasim.md` (docs): AlpaSim E2E Closed Loop Challenge 2026

[plans/](plans/) 1 live plans
<!-- files:end -->
