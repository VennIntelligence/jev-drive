# Offline analyses on existing vlm_arb logs: definitions (release replay, route-point error, pull-out collisions)

Three offline reads on the logs of the `vred*`, `pbyp*` and `drive` arms; no CARLA, no GPU, no driving run, no change to driving code. Results: [release_replay](../results/release_replay.md), [route_junction_error](../results/route_junction_error.md), [pullout_collisions](../results/pullout_collisions.md). All code is under `scripts/` as new files; [bypass_extract.py](../scripts/bypass_extract.py) and [bypass_common.py](../scripts/bypass_common.py) are reused unchanged.

## Data path (reproduce)

1. Run lists come from the `*_runs.csv` files in `results/` (arm, seed, route, attempt directory); the chase-camera reruns are the `v2-gif*` units.
2. On the box (CPU, stdlib): `python3 release_extract.py OUT < runs.csv` (A and B; one gzip JSON per attempt: arbitration steps, light answers with probabilities and ground truth, truth lights, plan-step context, ego track, the logged route) and `python3 bypass_extract.py OUT < runs.csv` (C; privileged snapshots at 0.2 s, contacts, bypass state, route). Both read the attempt directories only.
3. The extracts are copied to `tmp/vlm_arb_offline/` on the Mac (gitignored); the analysis scripts run there with `/usr/bin/python3` (numpy): `release_replay.py` then `release_replay_report.py`, `route_junction_error.py`, `pullout_collisions.py`.

## A. Release policy replay

- **Hold episode**: consecutive arbitration steps with rule R2 active; segments less than 6 s apart are one episode (a K=2 release followed by a re-hold stays inside it). The window of answers starts 3 s before the hold (history for rules that need it; no release is counted before the hold started) and ends 3 s after a logged green release (1 s after any other end).
- **Governing light**: the light whose stop line is the last one within [entrance - 15 m, entrance + 1 m] of the junction of the hold (the agent's own rule). Truth state of that light over time, per run: plan-step context (`ctx.tl`, `tl_id`, `tl_dist`) plus the states of all lights within 40 m logged with each answer (`gt.lights`). Light timelines are not shared between arms (the scenario sets the ego light when the car arrives).
- **Position** at an answer, from the logged route progress at the query time: A = front bumper before the stop line; B = past the stop line, rear axle before the junction entrance; C = rear axle past the entrance (up to 6 m past, as the scorer's tail window). "Past the line or under the light" = B + C.
- **Policies**: K consecutive green answers (K = 1 to 4); single answer with p_green >= threshold (0.5 to 0.99); K = 2 with p_green >= 0.9 / 0.99 on both; cumulative evidence S = max(0, S + clip(ln((p_green + 0.001) / (p_red + 0.001)), -7, 7)), reset to 0 on p_red >= 0.9, release at S >= 3, 5, 7, 10 (p_green / p_red = the option probabilities of `green_for_ego` / `red_or_yellow_for_ego`); "transition required" = a red answer (p_red >= 0.9, 0.7 or 0.5) directly before a run of K green answers with p_green >= 0.9; position-gated = K=2 before the stop line, K=4 with p_green >= 0.99 once the front bumper is past it.
- **Metrics**: false first release = first release of the episode with the truth light red or yellow at the release time (the arrival time of the answer); false events = all false releases with the rule re-armed after each release and 3 s between counted events; delay = release time minus the green onset, with a fresh rule state at the onset, missed = no release before the light left green or the window ended (counted above 1 s and 2 s); green window = onset to the next non-green sample.
- **Infraction re-read**: official `red_light` messages of the vred family; the infraction time is not logged (the location is the light's), so the time the tail (rear axle + 1.06 m) passes the stop line / the junction entrance, taking the first at which the light was red, stands in for it. Category = what ended the last hold before that crossing (K=2 green release that was false at its time / R5 25 s fallback / no hold / green release followed by red again).
- **Avoidability margin**: same motion after the release, release moved to the green onset; margin = latest release time that clears before red minus the green onset (negative: impossible).

## B. Route-point error

- **Official input**: the dense plan as logged by the agent (`route.json`: `xy`, `cmd` of `global_plan_world_coord`) and the plan downsampled by `route_manipulation.downsample_route(plan, 50)` (branch order re-implemented; 2-D distances).
- **Command change** = first point of a maximal run with command LEFT, RIGHT or STRAIGHT (a lane-change run is not a junction command). A junction is matched to the nearest such run starting within [entrance - 5 m, exit + 5 m].
- **Truth**: junction entrance = first map-flagged route point (exact arc from the logged `ego_s + junc_dist` while `junc_dist > 1 m`, keyed by junction id); stop line = `ego_s + REAR_TO_BUMPER + tl_dist` of the plan-step context, median per light; arcs are along the dense route polyline.
- **Error** = command-change arc minus truth, signed; for the downsampled plan the arc along the downsampled polyline, and the remaining distance from the car's projection when it stands 5 to 50 m before the entrance.

## C. Pull-out collisions

- Route frame: arc position and signed lateral offset on the route polyline; the adjacent lane is the one on the side of the bypass offset (lateral offset about -3.25 / -3.5 m, i.e. right).
- **Pull-out start** = last time before the lane entry with the ego centre within 0.15 m of its lane centre. **Lane entry** = outermost body corner on the lane side crosses half the lane separation, coming from inside the own lane for at least 1 s (corner = centre offset + 0.92 m x cos(yaw error) + 2.45 m x sin(yaw error) toward the lane).
- **Other vehicle**: moving (>= 3 m/s at 0.2 s before the contact), heading within 45 degrees of the route, lateral offset within 1.75 m of the adjacent lane centre. Gap = bumper to bumper along the route; time gap = gap / its speed; ttc = gap / closing speed. **Braked** = acceleration over 0.6 s windows <= -3 m/s2 between pull-out start and 0.5 s before the contact.
- **Clock**: contact times are on the world clock, 1.0 to 1.4 s ahead of the scenario clock of the privileged log (per route: 19324 1.1 s, 19832 1.0 s, 24497 1.35 s, 2520 1.4 s, from the frame numbers of both logs); all reported times are on the scenario clock.
- **Headways**: crossings of the obstacle start line by vehicles of the adjacent lane, one run (at least 100 s observed) per route and traffic seed.

## Status labels used in the results

*Verified* = read directly from a log field; *inferred* = computed from several logs under a stated assumption; *not in the logs* = said so, and the nearest quantity used. Sample sizes are small throughout (19 routes, 2 traffic seeds, one obstacle or one light per route): every number is a diagnostic read.
