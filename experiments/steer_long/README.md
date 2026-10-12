# steer_long: sharp steering and slowness in the plan-to-actuator chain

status: live
decisions: pending
index: pre-registered forensic read on stored rollouts; no numbers yet
key: research/next-round/steer-long.md, experiments/steer_long/scripts/sl_pull.py, experiments/steer_long/scripts/sl_read.py

**Question.** In stored AlpaSim rollouts (nuPlan held 452 at 18 decisions, PAI 281), does the large steering before off-road / spin /
collision zeros start in the plan (model) or in the MPC's steering, is the model's yaw-history loop visible, and what makes the car slow and
stop without a reason; which driver-side rule (curvature / curvature-rate cap), official controller gain or serving switch follows.

**Conclusion.** Pending.

**Read more.** [research/next-round/steer-long.md](../../research/next-round/steer-long.md) (Chinese: evidence, hypotheses,
pre-registration in section 3, reads, ranked proposals).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py --topic <name>` (your topic only; a bare run or `--all` would rewrite
other lanes' READMEs, so `--all` is a dry run unless `--write`; `--check` is read-only).
