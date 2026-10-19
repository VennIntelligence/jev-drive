# LAT-ZERO pre-registration (2026-10-20, written before any per-window class count)

Question: B2 (`STANDFMXH-L18-s1`, s0 as replicate) keeps ~20 easy lateral zeros (decision 393: turn offroad / corridor 8, moving corridor 7,
straight offroad 5 on s1). Is each one (a) overspeed at curvature, (b) a path error at normal speed, or (c) something else?
CPU only, stored reads: the per-decision tables `ext_full_H{,_s1}` (B2 + JTN-L18-s1 natively on the same windows), their per-step scorer
metrics, the human tracks (`hum.pkl` of `stand_b/ext_full_H_s1`), garage's per-step metrics (`tmp/easy_items/steps_extra.parquet`).

## Windows

Every easy item (garage = 1) that decision 393's classifier puts in `zero: turn window, offroad / corridor`, `zero: corridor, moving non-turn`
or `zero: offroad, non-turn`, separately for B2 s1 and B2 s0 (`results/easy_items/misses_STANDFMXH-s{0,1}.csv.gz`).

## Quantities (per read and window)

- exit time t_ev: first scored step (eval_relevant) with offroad, left_corridor_laterally, any collision or dist_to_gt_trajectory >= 4 m.
- pre-exit decisions: valid decisions with k >= 2 and t0 in [t_ev - 3 s, t_ev).
- r3: median over pre-exit decisions of v / vh (ego speed against the human's speed at the ego's station, d384's convention).
- human slow-down: 1 - min(human speed over stations [s_a, s_exit + 5 m]) / human speed at s_a, s_a = the ego's station at the first pre-exit
  decision, s_exit = the ego's station at the last one.
- human curvature: mean |d heading / d s| of the human track over the same station span (rad/m).
- exit offset: lateral_dist_to_gt_trajectory at the exit step.
- lead: min lead_gap over pre-exit decisions.
- descriptors only (not in the rule): plan curvature / human curvature over the plan's first 2 s of arc at equal arc, plan lateral departure
  from the human track at 2 s minus the ego's current offset (signed toward the exit side), lateral-acceleration ratio v^2 k_plan / vh^2 k_h,
  driving command, whether JTN-L18-s1 and the other B2 seed pass the window, garage's speed ratio and max lateral offset.

## Rule (first match; thresholds fixed here, not changed after the counts)

1. c-early: t_ev <= 1.517 s (the policy has executed at most one decision) or no pre-exit decision with k >= 3.
2. c-nonlateral: the scorer type is not offroad / corridor / offroad+corridor (e.g. a no-launch progress zero).
3. c-human-marginal: exit offset < 0.5 m (the ego is on the human track when it is scored off the road / corridor).
4. c-lead: min lead gap over pre-exit decisions < 8 m.
5. a-overspeed: r3 >= 1.05 and (human slow-down >= 10 % or human curvature >= 0.02 rad/m).
6. b-path: everything else (normal speed, or overspeed on a straight, non-slowing stretch). Sub-labels for b (descriptive): under-turn
   (human curvature >= 0.02, plan / human curvature < 0.8), over-turn (> 1.2), drift (plan lateral departure at 2 s >= 0.3 m toward the exit side),
   else execution / unclear.

## Lever and value

- The lever is chosen after the counts, priced by setting its target windows to 1 in B2 s1's (and s0's) response vector on the frozen
  board-mix pool (`slow_rescore.Frozen`, board-mix and g15, log-clustered bootstrap), as decision 390 / 393 price classes.
- Collateral (easy misses added elsewhere) is estimated offline from where the lever would act on windows B2 currently passes.
- Training-data counts for the failing situation are reported as found, no threshold.
