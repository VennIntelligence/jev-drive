# DAgger rows of on1_NDP: teacher labels, T2, T4 (nd_teach.py label)

Rows 104 (6 scenes); policy decisions (k >= 2) 92, fed >= 2 m/s 63, teacher label 63 (no label {}). Frame check (k = 0 anchor vs the scene loader's logged pose, m, p50 / p90 / p99): [0.0, 0.0, 0.0].

**T2** (scorer's drivable raster, ego box outline; rows whose own state is inside and >= 6 / 8 label poses on the 56 m canvas, n = 63): teacher label inside 0.984 (served plan 1.000); line >= 0.90 -> **PASS**. States already off the drivable area: 0, teacher deeper inside than the served plan in nan of them.

**T4** (drop infeasible labels): lateral acceleration (position estimator, p50 / p90 / p99) teacher [0.189, 3.008, 3.802] vs served [0.096, 1.96, 2.784]; > 4.0 m/s^2 1, label leaves the drivable area from an inside state 1; dropped 2 (3.2%; line <= 10 %) -> **PASS**.

Drawable rows 61 (8 in the recipe's held-out pool logs: weight 0). |d0| p50 / p90 / p99 [0.201, 2.627, 2.979], signed median 0.17097215354442596; |d0| > 0.5 m left / right 16 / 0, > 1 m 13 / 0. |label - served| lateral at 4 s p50 / p90 / p99 [0.213, 2.242, 3.458] m.
