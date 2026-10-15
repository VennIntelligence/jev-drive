# Pre-registration: PLAN SWITCHING across consecutive decisions on stored closed-loop logs (no training, no rollouts, CPU only), 2026-10-15

**Why.** Decision 307 M4: within failing PAI turns, 0-10 m before the entry, 31 % of the decisions ask for a straight or partial turn and
69 % for the full turn (r_arc). If the plan switches between a full-turn mode and a straight / partial mode from one decision to the next,
the controller (MPC tracking 1-2 s, re-solved every 0.1 s on PAI, 0.5 s on nuPlan) executes a time average of the sequence: a wider,
later line than any single plan. Completed turns would show a consistent sequence. Context that stands: the onset is placed in the world
(d306), the plan follows the road at its own covered distance in late-wide turns (d307), the command names the turn (d307 M2), the entry
state alone is not enough (d303), laterally noisy plans are catastrophic in closed loop (d307 flow head). Written and committed before
any statistic below is computed. Amendments are appended, never edited in place.

## Data (stored reads, read-only)

The RECEDING extracts (d306; `scripts/rf_receding.py`), which carry per rollout the route, the controller trace (10 Hz), the human log,
the zero time t_f, the turn row, and every served decision's t0 / anchor / `poses`:
- PAI 281 gains 20: Tokyo `/data/runs/route_night/receding/pai.pkl` (built from `/data/runs/route_night/r281/rsg` and
  `/data/runs/route_night/v2pai/{RV2a-L18-s0,RV2b-L18-s1,APP1X-L18-s0}`), decisions every 0.1 s.
- held 460 @18: box `$DATA_DIR/runs/route_follow/receding/held.pkl` (built from `runs/route_mem/read/<tag>/c*/run`), decisions every 0.5 s.
- Mac working copies `tmp/receding/{pai,held}.pkl` (md5 identical to the host copies, checked before this prereg).
Turns, entry / exit, completion and failure classes exactly as RECEDING's `read_turn` (classes: completed, late_wide, straight, over_turn,
other, no_zero; taxonomies as in `scripts/rf_receding.sh` read; the d299 over-turn zeros `x_*` as class over_turn).

| population | role | turns |
|:--|:--|:--|
| pai_RSG-L18-s0 | primary (S1, S2) | strict 36 |
| pai_RV2a-L18-s0, pai_RV2b-L18-s1, pai_APP1X-L18-s0 | replications, own verdicts | strict 36 each |
| held_RSG (RSG-L18-s0 + s1 pooled, clusters = log) | replication, own verdict | 24 + 24 |
| held_RSG-L18-s0, held_RSG-L18-s1, held_RV2a-*, held_RV2b-* | descriptive | 24 each |
| over-turn zeros (d299; x_pai_RSG, x_held_RSG-*) | descriptive class `over_turn` | 2 + 5 + 4 |

## Window and per-decision quantities

Window of a turn: served decisions with the ego station s_e in [entry - 20, entry + 20] m and t0 < t_f (to the failure), in t0 order.
Consecutive pair: two window decisions adjacent in t0 order whose spacing equals the population's decision period (0.1 s PAI, 0.5 s held,
+-20 ms); other pairs are skipped (counted). Sign sigma = +1 left turn, -1 right; every lateral quantity is sigma x (left-positive offset),
i.e. positive = toward the inside of the turn.

- **r(k)** (d307 Amendment 3): the plan's 4 s heading change toward the turn dpsi(k) (RECEDING `onset`) divided by the route's heading
  change over the plan's own covered arc, route_arc(k) = sigma x (h(s_e + a4) - h(s_e)) (h = rf_pai_follow.heading, deg); defined when
  route_arc >= 15 deg, else NaN.
- **j(k)** (plan-to-plan lateral jump, m): the previous decision's plan (origin + 8 poses) is carried to world with its anchor and into the
  current anchor frame (anchors are the measured ego poses, so this is "carried forward by the measured ego motion"); j = sigma x the signed
  perpendicular offset of the current plan's 2.0 s point (pose 4 of 8) from that carried polyline (last segment extended if the projection
  falls past its end). The offset is measured to the path, not to the previous plan's time-matched point, so a pure speed change along the
  same path gives j = 0. Defined on consecutive pairs.
- **e(k)** (tracking error, m): the plan served 1.0 s earlier (the decision at t0 - 1.0 s, +-20 ms; PAI 10 decisions back, held 2) carried
  to world; e = sigma x the signed perpendicular offset of the car's position at t0 (the decision anchor, rear axle; RECEDING G-frame
  0.000 m) from that path. Negative = the car is outside (wide of) the plan it was given. Defined when that decision exists (it may lie
  before the window).
- **Human log as a plan** (sanity): RECEDING's `Log.plan_at_station(s_e)` (the log's 4 s from where it reaches the same route station),
  its r_h (dpsi over the route over the human's own arc) and j_h (the same perpendicular offset, carried through the human poses' own
  world origins). No tracking error for the human.

## Per-turn statistics

- **switching rate** = share of consecutive pairs with both r defined where one r >= 0.6 and the other <= 0.3 (either direction); defined
  when the turn has >= 3 such pairs (else NaN, counted).
- **flip rate** (descriptive, comparable across 10 Hz and 2 Hz): the r sequence reduced to states F (r >= 0.6) / S (r <= 0.3), partial
  values dropped; F <-> S changes per second of window time with r defined.
- **|j|** = median |j(k)| over the turn's pairs (>= 3 pairs); also the median signed j and the p90 |j|.
- **|e|** = median |e(k)| over the window decisions (>= 3); also the median signed e.
- **time-average of r** = mean of r over the window decisions with r defined (decisions are equally spaced, so the mean is the time
  average); **max r** = the 90th percentile of r over the same decisions (robust maximum); both need >= 3 decisions.
- **executed ratio** R_exec = sigma x (ego yaw at the last window decision - at the first) / sigma x (route heading at the last window
  decision's station - at the first); defined when that route change >= 15 deg.

## Claims and statistics

Groups: failed (every non-completed turn) vs completed. Bootstrap B = 10 000, seed 0, percentile 95 %, clusters = scenes (held: the log
id, both seeds together); a replicate resamples clusters and recomputes both group statistics; replicates with an empty group are
dropped (counted); a replicate whose denominator is 0 gives a ratio of +inf when the numerator is > 0 (kept) and is dropped when both
are 0 (counted); percentiles by the inverted CDF.

- **S1** (primary pai_RSG-L18-s0): ratio = mean switching rate (failed) / mean switching rate (completed). Means, not medians: a per-turn
  rate is often 0 and the median of such a rate is degenerate. Supported: ratio >= 2 and CI lower > 1; refuted: CI upper < 2, or ratio
  <= 1; else inconclusive; < 3 turns with a rate in either group: inconclusive.
- **S2** (same population): ratio = median of per-turn |j| (failed) / median (completed); the same rule.
- Replications with their own verdicts under the same rules: pai_RV2a, pai_RV2b, pai_APP1X, held_RSG (pooled seeds). The other held sets
  and each held seed alone: descriptive.
- **S3** (descriptive): |e| failed vs completed (medians, difference with its CI) and signed e; and, across the turns of a population,
  R_exec against the time-average of r and against max r: OLS slope / intercept / R^2 and the median |R_exec - x| (identity). Fixed
  reading: **the controller averages the sequence** if the time-average's median |R_exec - x| is the smaller of the two and its OLS
  slope lies in [0.6, 1.4]; **the controller follows the committed plans** if the same holds for max r; else **neither**. Read on
  pai_RSG-L18-s0, PAI pooled (4 checkpoints, clusters = scene) and held pooled; failed and completed turns together and failed only.
- **By class** (descriptive): switching rate, flip rate, |j|, |e| per class (late_wide, straight, over_turn, no_zero, other) vs completed,
  ratio with its CI, no verdict.
- **Gates** (before S1 / S2 are read):
  - G-frame: median |anchor - controller position| at t0 <= 0.3 m and median |anchor yaw - controller yaw| <= 1 deg per population;
  - G-human: on pai_RSG-L18-s0 the human "plan"'s mean switching rate <= 0.02 and median per-turn |j_h| <= 0.10 m. Failing it, S1 / S2
    read inconclusive (the measure makes switching out of a consistent path) and only descriptives are reported.

## What the outcomes mean

- S1 and S2 supported: failing turns are fed an inconsistent plan sequence; a design that holds the plan across decisions (commitment /
  hysteresis on the mode, or a consistency term on on-policy rows) is the next candidate, and S3 says whether the cost comes from the
  controller averaging.
- Refuted: failing turns' plans are as consistent as completed ones; M4's 31 % is a between-turn mixture (some turns are straight
  throughout), not switching within a turn, and the lead closes.
- Inconclusive: no design decision rests on it; counts are reported.

## Outputs

`scripts/rf_switching.py` (read on the Mac from the RECEDING extracts), `results/switching/` (turns.csv per turn, decisions.csv.gz,
switching.json, switching.md with the CIs and verdicts), `figs/plan_switching.png` (r(k), j(k), e(k) against the distance to the entry for
4 failed and 4 completed pai_RSG-L18-s0 turns: completed = the first 4 by scene id with a switching rate; failed = the first 2 late_wide
and the first 2 straight by scene id with a switching rate), one README paragraph. research/decisions.md is not edited by this lane.

## Amendments (appended after the prereg commit; each says whether it came before or after a statistic was seen)

1. (After the first read; S1 / S2 / S3 and the class table were seen.) Switching pairs are absent: 0 of 1596 r-defined consecutive
   pairs on pai_RSG-L18-s0, 1 of 1580 on pai_APP1X (a174faed, 5.6 m before the entry), 0 on every held set. On pai_APP1X the registered
   S1 rule returns "supported" mechanically (failed mean 0.001 / completed 0, ratio +inf); this is reported as registered and marked
   degenerate, and the substantive reading is "no switching". The |j| per pair is confounded by speed (failed turns approach slower:
   PAI RSG median 4.0 vs 6.4 m/s, so each 0.1 s step covers less road). Added, descriptive, no verdict:
   (a) **plan creep**: j per metre of ego travel between the two decisions (pairs with ds > 0.05 m; per-turn median) and the cumulative
       signed j over the window (sum, m; negative = each new plan places its 2 s point outside the previous plan's path), failed vs
       completed with the bootstrap CI, per class; control = the same on decisions 40-80 m before the entry (signed by the coming
       turn's side; ~0 expected) where the rollout reaches back that far;
   (b) **M4 composition**: per failed turn, the set of states (d307 M4: S r < 0.25, P, F r >= 0.6) present 0-10 m before the entry, and
       the share of the variance of r there that lies between turns;
   (c) **executed vs planned heading**: per decision, the plan's heading toward the turn at 1 s and 2 s against the car's yaw change
       over [t0, t0 + 1 s] / [t0, t0 + 2 s] (controller trace; decisions with t0 + T < t_f); per turn the median executed - planned
       (deg), failed vs completed;
   (d) the figure gains a fourth row, the cumulative signed j; legends move below the panels.
   Nothing registered changes.
