# Offline analyses of the bypass: misfire attribution and a time-based "static" trigger (2026-10-02)

Two analyses on the finished closed-loop logs of the diagnostic batch (decision 84). No CARLA, no GPU, no new driving. Results:
[results/bypass_misfire.md](../results/bypass_misfire.md), [results/bypass_timedet.md](../results/bypass_timedet.md).

## Data

- `$DATA_DIR/runs/vlm_arb/arms/<unit>/attempts/<route>/<k>/` of the units listed in `results/routes.csv` for arms `drive` (38 runs) and
  `pbyp` (38 runs); 19 routes x 2 traffic seeds; target (static-obstacle) routes 24497, 2520, 19324, 19832, the other 15 are non-target.
- Files read: `plans.jsonl` (plan step at 20 Hz: ego speed, `ctx` ground truth = ego light `tl` 0 green / 1 yellow / 2 red and `tl_dist`, `lead_gap`,
  `lead_v` of the nearest vehicle in lane; openpilot lead head `lead[0] = [x, y, v, a]` and `lead_prob` `lp`; `pc.bypass`), `privileged.jsonl`
  (0.2 s snapshots: ego, actors within 70 m after the extract, `pc.obstacles`, `pc.bypass_state`), `contacts.jsonl`, `results.json`,
  `vlm_decisions.jsonl` (old format in `eval-drive-s0`: one line per plan step with the latest answer; new format `k = a / s` lines).
- `scripts/bypass_extract.py` (stdlib only) turns them into one compact gzip JSON per run on the box (`$DATA_DIR/tmp_bypass/`, 6.5 MB in total);
  `bypass_misfire.py` and `bypass_timedet.py` run on the Mac (Python 3.9 + numpy) on a copy of that directory.
- Gaps: the 13 seed-0 `drive` runs of `eval-drive-s0` have no `privileged.jsonl` (only ego-side signals and junction distance); `pbyp` runs have no
  `vlm_decisions.jsonl`; junction exits are known only to the ego's sampling step.

## Definitions: analysis 1

- **Activation**: first snapshot at which the privileged geometry holds a bypass state (`pc.bypass_state`) whose (start_s / 5, ids) differs from the
  previous snapshot's. `t0` is that snapshot; the state ends when the ego is 23 m past the obstacle (`t_end`; the log end if that never happens).
- **Blocker**: the nearest (by route arc) of the state's `ids`. `s` is its projection on the route polyline (as in `b2d_privileged_geometry.py`, clamped to
  [0, L]); `ext` is the signed distance beyond the last route point (+) or behind the first (-) for clamped projections.
- **Class** (first match wins; `RESUME` = the blocker exceeds 0.5 m/s for three consecutive snapshots after `t0`):
  1. `scenario_obstacle`: target route and the blocker never exceeds 0.5 m/s in the log. These are the correct activations (8: 4 routes x 2 seeds).
  2. `behind_route_start`: `s <= 0.05` (the projection is clamped to the first route point; the blocker is about 10 m behind the ego's spawn).
  3. `beyond_route_end`: `s >= L - 0.05`.
  4. `parked_or_broken`: never exceeds 0.5 m/s, not a target route (none occurs).
  5. `red_light_queue`: a light stop line reconstructed from the ego's own `ctx` (arc position `ego_s + tl_dist`, clustered over all runs of the route;
     timers agree across runs to 0.1 s) lies 0 to 45 m ahead of the blocker and that light was red or yellow at any time between 2 s before `t0` and the resume.
  6. `junction_queue`: blocker within 35 m before a junction entrance, inside a junction, or within 15 m after an exit (entrances from the drive runs' `junc_dist`).
  7. `moving_traffic_pause`: any other blocker that resumes.
  `scenario actor of the route's own scenario type` could not be identified from the logs (actors are not tagged), so it is not a class; no observed
  blocker needed it.
- **Outcome within 15 s**: official events (collisions with first-contact time from `contacts.jsonl`; red-light and blocked events located on the ego
  trajectory; `outside_route_lanes` has no time or position) in `[t0, t0 + 15 s]`. **Episode events**: events between `t0` and the next activation or
  `t_end + 2 s`. An event is "also in drive" when the paired drive run has an event of the same kind within 10 m.
- **Replay** of a suppression rule: for an activation, step through the 0.2 s snapshots from `t0` to `t_end`; the activation fires at the first snapshot at which
  the blocker is static for at least T s, is not resuming, and the ego-side terms pass; it is prevented if the blocker resumes (three snapshots above 0.5 m/s)
  or the terms never pass. Ego-side signals after `t0` (ego arc position, light state and distance, junction distance) come from the paired drive run of the
  same route and seed, i.e. what the ego sees when it does not bypass. Terms: X = ego light red or yellow closer than X m; N = ego within N m before a
  junction entrance or inside one; Nb = same for the blocker; T = blocker static for at least T s (original 2 s); valid = blocker projection strictly inside
  the route; G = ego light red or yellow within 50 m at any time in the last G s; tmin = no activation before tmin s. This is a log replay: the closed loop
  after the replayed time is not simulated.

## Definitions: analysis 2

- **Stop episode** (drive runs): maximal run of plan steps with ego speed below the bound v (0.3, 0.5, 1.0 m/s), gaps up to 0.6 s bridged, at least 1 s
  long, counted from 6.5 s (the ego is held until about 6 s). Episodes ending with the run are marked "ran to end".
- **Ground-truth cause** (first match): `obstacle` = target route and the scenario obstacle (start of the first bypass state beyond 20 m in the paired `pbyp`
  run) 0 to 45 m ahead of the ego's rear axle at the episode start; `red_light` = ego light red or yellow closer than 50 m in at least half of the steps;
  `junction` = junction entrance within 15 m (or inside one) at the start; `lead` = a vehicle within 40 m ahead per `ctx.lead_gap`; `other`.
- **Detector** (strict, per episode): fires at the first plan step with episode age >= T at which (a) a lead is reported, (b) the light suppression does not
  hold, (c) the junction suppression does not hold. Lead = openpilot lead head (`lp[0] > 0.5` and `0 < x < 40 m`; `head_static` adds `|lead[0].v| < 0.5`)
  or the logged openjev Q_block answer (moving_lead or static_block). Light suppression `truth` = `ctx.tl` in {1, 2} and `ctx.tl_dist < X` (privileged),
  `vlm` = latest logged Q_light answer at most 2.5 s old is red_or_yellow_for_ego, `none`. Junction suppression = `junc_dist <= N` from the route map.
- **Scores**: per episode (fired or not), obstacle recall among episodes with duration >= T and among all, false triggers by cause, per obstacle run
  (4 routes x 2 seeds) whether any obstacle episode fired and the time from the first obstacle stop to the first trigger. Raw counts everywhere; route
  clusters listed next to the false triggers.
- **Creep-tolerant variant**: mean ego speed over the last T s below a bound, scored per fire (one fire per 10 s), a fire is false when no obstacle is within 45 m.
- **VLM behaviour**: logged answers whose frame time lies inside an obstacle stop (v < 0.3 m/s, obstacle within 45 m), by obstacle type and distance bucket;
  truth for Q_side is `left_free` (negative path offset in CARLA's left-handed frame; the seed-0 old-format `gt_side` says right_free and was not used).

## Choices not dictated by the task

6.5 s as the start of counting stops; 45 m as "obstacle ahead"; 15 m / 35 m for junction proximity in the cause and class labels; 0.5 m/s resume threshold; the
grids (X 15, 30, 50; N 10, 25, 40; T 2 to 20; G 5, 10; v 0.3, 0.5, 1.0). The rule choices are fits on the same 38 runs and are not confirmations.
