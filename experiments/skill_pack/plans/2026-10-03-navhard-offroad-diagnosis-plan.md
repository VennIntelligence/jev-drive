# navhard stage 2: why does the open-loop model leave the drivable area (offline diagnosis plan)

Written 2026-10-03, before any full-set number of this diagnosis was read. What had been read before: the aggregate tables of
`research/openpilot-diagnosis/index.html`, the seven example scenes of `tmp/2026-10-02-openloop-review.html`, and the code path
(`scripts/op_lb.py`, `jevdrive/op_interp.py`, `jevdrive/openpilot/model.py`, the devkit scorer). No retraining, no submission.
Models: native = frozen openpilot Cinque, GIMM frames, desire schedule `none` (`runs/op_lb/lb_navhard/plans/gimm@cinque.npz`,
pose file `preds/gimm-cinque__base.npz`); N4 = `runs/skill_pack/raise/n4/navhard_n4.npz`.

## Conventions

- Ego frame = rear axle at t0, x forward, y left. "Raw plan" = the 8 poses (0.5 ... 4.0 s) of the pose files (what the scorer
  receives); "dense raw plan" = native `plan_pos` / `plan_yaw` on openpilot's 33 time knots converted with
  `jevdrive.op_interp.to_rear`. "Simulated" = official LQR + bicycle states (`PDMSimulator`, 0.1 s, 41 states).
- PDM reference = `metric_cache.trajectory` (PDM-Closed), in the ego frame via `transform_trajectory`.
- Drivable-area failure = official DAC = 0 (any of the four corners outside ROADBLOCK / INTERSECTION / DRIVABLE_AREA / CARPARK
  at any of the 41 states, t = 0 included). The replay must reproduce the official per-token DAC for every token or the
  script aborts.
- Outside distance = distance of the farthest-outside corner to the union of those polygons, per state (cm).
- First-departure time = first state index (x 0.1 s) with a corner outside.

## Q1 definitions

- Opposite side: both plan and PDM reference are at |y(4 s)| > 1 m, with opposite signs, and |y_plan(4 s) - y_ref(4 s)| > 2 m.
  Reported by command at t0 (left / straight / right), map and stage, for native and N4, with the share that also fails DAC.
  A convention error would appear as (i) a rate near 100 % on a command, or (ii) a mirror-asymmetric pattern (left-reference to
  right-plan much more frequent than the reverse), or (iii) a map-specific pattern (sg-one-north is left-hand traffic and the
  only map with `traffic_convention = (0, 1)`). Also checked: sign agreement between raw-plan heading at 4 s and PDM heading.
- Case 01 perturbations on the native model (frozen, GPU 2, same GIMM frames and step schedule as the cached run; the
  unperturbed rerun must reproduce the cached plan to < 1 cm or the comparison is void): desire none / turnRight / turnLeft with
  pulse onset at -1.5 / -0.5 / 0 s, single-step (modeld's rising edge) and a pulse repeated on every step from onset; history
  replaced by the t0 key frame (all 31 steps); horizontally mirrored frames with the output mirrored back (y -> -y,
  yaw -> -yaw), with traffic convention as is and swapped; traffic convention forced (1, 0). "Drives the right turn" = the
  perturbation that moves the raw plan 4 s lateral end by more than 3 m toward the PDM side.

## Q2 definitions

- Failure set: stage-2 tokens with DAC = 0, per model. Statistics: first-departure time, outside distance at first departure and
  at the worst state, lateral offset (simulated centre, ego frame) relative to PDM reference at 0.5 / 1 / 2 / 4 s, start-state
  lateral offset and heading error w.r.t. the route centreline (`metric_cache.centerline`, nearest point, signed, left +).
- Classes, mutually exclusive, applied in this order:
  1. `start_outside`: the t = 0 footprint is already outside the drivable area (unavoidable, DAC = 0 for any plan);
  2. `wrong_direction`: "opposite side" as defined above (stage-2 end of the plan on the other side of the reference);
  3. `early_clip`: first departure <= 1.5 s and the start is off-centre (|lateral offset| >= 0.5 m) or mis-headed
     (|heading error| >= 0.1 rad) w.r.t. the route centreline;
  4. `junction`: the first-outside corner or the ego centre at departure is within 3 m of an INTERSECTION polygon, or the
     nearest route lane there is a lane connector;
  5. `late_undershoot`: first departure > 1.5 s, departure side opposite to the turn direction of the PDM reference (reference
     heading change at 4 s >= 0.3 rad), and |y_plan| < |y_ref| at departure;
  6. `other`.
  Examples per class: the three failures with median outside distance, static trajectory plots.
- Counterfactuals use the official scorer on the cached scenes (own in-process harness with the devkit's `pdm_score`,
  `SceneAggregator`, `calculate_individual_mapping_scores`; it must reproduce the cached per-token scores and the official
  combined EPDMS of both models to 1e-6 / 0.01). Stage 1 is never modified; the stage-2 plan is changed for all stage-2 tokens
  ("all") or only for tokens that failed DAC at baseline ("failures only", privileged gate). Variants of the dense plan (0.1 s):
  - `lag-dt` (dt = 0.3, 0.5, 1.0 s): y'(t) = y(t + dt), yaw'(t) = yaw(t + dt) (held at the 4 s value beyond the horizon), x(t)
    unchanged ("same lateral profile earlier");
  - `scale-k` (k = 1.25, 1.5): y'(t) = k * y(t) where y(t) has the sign of the PDM reference end lateral (|y_ref(4 s)| > 0.5 m),
    unchanged otherwise; yaw' = yaw + atan2(dy' - dy, dx) correction so the heading follows the new path;
  - `pdm-1s`: poses at t <= 1 s are the PDM reference's; the model plan after 1 s is rigidly moved so that its pose at 1 s
    coincides with the PDM pose at 1 s.
  All are diagnostic upper bounds using privileged information (the PDM reference or the failure label) and are labelled so.
  Reported: stage-2 DAC compliance (mean over the 5 462 stage-2 tokens), stage-2 EPDMS, official-aggregation combined EPDMS,
  number of baseline failures fixed and number of baseline passes broken.
- Lag attribution: raw-plan heading change / curvature over the first 1 s (psi(1 s) / 1 s, and yaw rate at 0.5 s) of the native
  plan in the stage-2 DAC failures vs normal navtest scenes (cached navtest plans, moving > 3 m/s) and vs the ego's own history
  curvature at t0; plus the conversion's own contribution: lateral difference at 0.5 / 1 / 2 s between the dense raw plan, the
  8-pose file, and the simulated state.

## Q3 definitions

- openpilot outputs (cached `heads` of the native plan file): road edges (2, 33, 2) at x grid X_IDXS, lane lines (4, 33, 2).
  Edge points into the ego frame: x_ego = x_op + 1.66 (camera lever arm), y_ego = -y_op. Map boundary along the first 30 m: at
  x = 5, 10, 15, 20, 25, 30 m, the nearest drivable-area boundary left and right of the centre line (ray in the ego frame).
- "Edge right": model edge within 1.0 m of the map boundary on the side the plan crosses. "Plan crosses own edge": any raw-plan
  point (x in 2 ... 30 m) beyond the model's edge on that side, with centre test and with a 1.15 m half-width footprint test.
- Split of DAC failures at stage 2 that have an in-horizon plan departure side: (a) edge right and plan crosses it = plan
  inconsistent with perception; (b) edge wrong (> 1.0 m off the map boundary) in the synthesized view; (c) edge right and plan
  does not cross it by the test (footprint / map mismatch), (d) edge not defined (map boundary beyond 30 m or ego outside).

## Q4

Per-sub-metric score lost by N4 on navhard by stage from its per-token CSV (loss of each term = how much combined score is gained
if that term is set to its best value, 1 for multiplicative, 1 for weighted, same method as `navhard_deficit.py` marginal
analysis, uniform stage-2 weighting and the official combined), with traffic-light compliance and the progress loss at
slow / not-started tokens (v0 < 1 m/s, EP shortfall) split out; existing numbers quoted from the deficit note where they exist and
marked "recomputed" otherwise.

## Addendum (written after the full-set stage-2 numbers, before the gain was scored)

Added on request, definitions fixed before the corresponding scores were read:

- Q3 refinement: the edge error is measured at the departing corner (signed model edge minus the corner's lateral position on the departure side;
  right = within 1 m), because the first design (5 m cross-sections, `q3_edge_error_grid.csv`) did not separate failures from passes.
- Camera audit: calibration (yaw / pitch / roll, intrinsics, distortion, image size, coverage) of CAM_F0 for stage 1 vs stage 2; model inputs
  shown next to the raw frame; case-01 input variants (wide blanked, road blanked, wide copied from road, road copied from wide, both inputs cropped
  to the central 50 % / 25 % columns, virtual-camera yaw -2 ... +2 deg and pitch +-1 deg with the warp history); branch context flags (an INTERSECTION
  polygon in the forward fan x in [3, 40] m, +-35 deg; drivable width at x = 20 m) vs the opposite-side rate.
- Amplitude: on navtrain (3 000-token subset with plans) and navtest (12 146) against the human future, per horizon 1-4 s, signed ratio plan / human
  (|y_h| > 0.5 m or |heading_h| > 0.05 rad) and slope through the origin, by human curvature bin (|heading change at 4 s| < 0.1 / 0.1-0.4 / > 0.4 rad),
  speed and command, and the near-term arc extrapolation (curvature = heading at 1 s over arc length, circular arc over the plan's own arc length).
- Gain: lateral gain g fitted on navtrain only (least squares of human y on plan y through the origin; `gain_single` one value over 1-4 s, `gain_horizon`
  one value per horizon 1 / 2 / 3 / 4 s, linear in between), applied to every token (no reference or label), heading corrected by the change of the path
  tangent; scored with the official v1 scorer on navtest (paired token bootstrap) and the devkit two-stage scorer on navhard (group bootstrap over the
  225 mapping groups). Not a privileged counterfactual.
