# steer_long: sharp steering and slowness in the plan-to-actuator chain

status: live
decisions: pending
index: PAI high-speed spins = force-GT handoff (9-13 zeros/label); slowness = speed-history loop (CF recovers 83 %)
key: research/next-round/steer-long.md, experiments/steer_long/results/sl_report.md, experiments/steer_long/results/sl_long.md, experiments/steer_long/results/sl_mpc.md, experiments/steer_long/results/sl_cf.md, experiments/steer_long/scripts/sl_read.py, experiments/steer_long/scripts/sl_long.py, experiments/steer_long/scripts/sl_handoff.py, experiments/steer_long/scripts/sl_cf.py

**Question.** In stored AlpaSim rollouts (nuPlan held 452 at 18 decisions, PAI 281), does the large steering before off-road / spin /
collision zeros start in the plan (model) or in the MPC's steering, is the model's yaw-history loop visible, and what makes the car slow and
stop without a reason; which driver-side rule (curvature / curvature-rate cap), official controller gain or serving switch follows.

**Conclusion.** The sudden large steering at speed is not a plan sweep: on PAI it happens at the force-GT handoff (1.7 s), 28 / 75
rollouts at >= 25 m/s lose yaw control within 1.5 s (9-13 zeros per label); an offline replay of AlpaSim's own MPC ties it to the plan being
2-10 m short of the ego's speed at 2 s and a weak steering-rate penalty at speed, and `rel_front_steering_angle_weight` 20 / 50 suppresses it
(7 / 8, 8 / 8). Slowness is the model (101-107 % of the speed deficit is in the plan), a ratchet: 85-95 % of unexplained slowdowns start in the
plan and almost never recover, and feeding the policy the log's speed history recovers 83 % of the plan's speed. No curvature cap passes the
collateral rule. Decision entry pending (main).

**Next.** Ranked proposals with pilots in the note, section 5: PAI MPC gain 20 (Tokyo, two stages, ~6 h); speed-history hold switch (nuPlan held, ~3 card-h).

**Read more.** [research/next-round/steer-long.md](../../research/next-round/steer-long.md) (Chinese: evidence, hypotheses,
pre-registration in section 3, reads, ranked proposals).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py --topic <name>` (your topic only; a bare run or `--all` would rewrite
other lanes' READMEs, so `--all` is a dry run unless `--write`; `--check` is read-only).
