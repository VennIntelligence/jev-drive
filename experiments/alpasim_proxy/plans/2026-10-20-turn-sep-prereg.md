# TURN-SEP pre-registration (2026-10-20)

Written before any feature value, AUC or garage trace was computed. Diagnostic only: CPU, stored reads, no training, no rollouts.

**Question.** Decision 396 found that the speed ratio cannot tell B2's failing sharp turns from the ones it passes. Is there a
decision-time state, computable from a training row (plan, the human's logged path / future, ego speed; map-free), that does?

**Population.** d395's sharp-turn held windows (`tmp/lat_zero/sharp_windows.csv`, human path turns >= 45 deg within 20 m).
- Reads: B2 s1 `STANDFMXH-L18-s1` (primary) and B2 s0 (replicate). JTN-L18-s1 / s0 and RSG-L18-s0 are reported as description only.
- Window label per read: **fail** = d395's lateral zero (`latzero`); **pass** = score >= 1 - 1e-4. Partial scores and non-lateral
  zeros are left out.
- The 9 a2 target windows of d395 (s1) are reported separately.

**Decisions.** k >= 2, ego >= 2 m/s, the human path turns >= 15 deg over the next 20 m from the ego's station (`turn20`) = the
"turn domain".
- Fail windows: decisions in **[t_ev - 3 s, t_ev - 1 s]** (primary: at least 1 s before the scored exit, so that the ego is not
  already out; t_ev = d395's exit time). [t_ev - 3, t_ev) is secondary.
- Pass windows: every turn-domain decision of the window.
- Window score = the max of the feature over its eligible decisions. This favours pass windows, which have more decisions, so it is
  conservative.

**Candidates (sign fixed in advance: larger = riskier).** Human path = the logged track; sigma = the turn's sign over the next 20 m;
"outward" = toward the outside of the turn, relative to the ego's current offset from the path.
1. `ratio`: ego speed / human speed at the ego's station (the d396 baseline).
2. `arc`: plan 4 s arc / the human's 4 s arc from the time the human was at the ego's station.
3. `out4`, `out8`: plan outward offset from the human path at an equal 4 m / 8 m of plan arc.
4. `out_end`: plan outward offset at the plan's 4 s point.
5. `kdef`: curvature deficit over the first 8 m of arc, sigma x (human heading change - plan heading change) (rad).
6. `alat`: v_ego^2 x the human path's max curvature over the next 20 m (m/s^2): lateral acceleration the turn asks at the current speed.
7. `alat_plan`: max over the plan of v_plan^2 x kappa_plan (the plan's own lateral acceleration).
8. `lag`: turn-in lag (m), the plan arc at which the plan has turned half of the human's heading change over the next 15 m, minus
   the human's arc for the same; capped at the plan's arc when the plan never turns that far.
9. `lat0`: the ego's current outward offset from the human path (m).
10. Turn geometry (window-level, not a policy state): `rmin` (negated min radius over the next 20 m), `turn_deg`, `right` (1 = right).
11. Oracle only (map): `edge` = minus the scorer's `min_distance_to_lane_boundary_m` at the decision step. A probe of whether the
    map carries the information, never a term input.
12. `combo`: logistic regression on 1-9 (standardised), leave-one-window-out within a read; also fitted on s1 and applied to s0.

**Success line.** A candidate passes if, on **both** B2 s1 and s0:
- (i) window-level AUC (fail vs pass sharp windows, primary span) >= **0.85**; and
- (ii) at the threshold where it fires on <= **1 %** of the turn-domain-or-not decisions (k >= 2) of every held window B2 s1 scores 1
  (d396 G3's denominator; decisions outside the turn domain count as no fire), it fires on >= 6 / 9 a2 targets (s1) and on >= 50 %
  of the failing sharp windows of each seed (s0 uses s1's threshold).

An oracle candidate that passes says the map has the signal; it does not count as a training term. If no map-free candidate passes,
the turn lever is recommended closed for this board.

**Garage.** Garage 0014 has no plan dump; its executed path is in the read's controller traces (10 Hz). On the a2 windows and the
sharp set, compare garage with B2 s1 / s0 executed paths: speed at turn entry and at the apex (station of max human curvature),
turn-in station (where the executed heading has turned half of the human's), outward offset at the apex, and the max outward offset.
Descriptive, no line.
