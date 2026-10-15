# Pre-registration: LC-HYSTERESIS: does the route command flip L/R -> S mid lane change? (reading, then a serve-side pilot), 2026-10-15

**Label.** The latch pilot is a diagnostic / exam technique, not adopted without the user.

**Observation (user).** At required lane changes on a clear road the trajectory first drifts toward the target lane, then snaps straight a
moment later. **Hypothesis (main session).** The 3-way command flips L -> S once the ego has moved part of the way, and the lane-centring
prior pulls the car back. The served rule (submission image `navsim_transfuser_challenge/navigation.py`, `command_from_route`, read
2026-10-15): the first route waypoint >= 5 m from the rig in the latest route message (AlpaSim places the first waypoint ~42 m ahead) gives
LEFT if its y > 2 m, RIGHT if y < -2 m, else STRAIGHT; y is in the rig frame of the message time, so it shrinks with the ego's lateral
move AND with its yaw toward the target lane (3 deg of yaw is ~2.2 m at 42 m). Written and committed before any statistic below.

## Reading (CPU; stored gains-20 PAI 281 reads, no rollout)

Runs: RSG-L18-s0 `/data/runs/route_night/r281/rsg`, RV2a-L18-s0 and RV2b-L18-s1 `/data/runs/route_night/v2pai/<tag>` (all gains 20,
VCONT 1.0, LEAD 1). Held 460 (nuPlan, box) is not read: its route is the nuPlan route at 2 Hz with a different command source; out of scope.
Sites (rf_laneoff `lc_map`, d299 Amendment 3, per run): **validated** = the 6 map-validated sites of d299 (route leaves its lane chain for
an adjacent lane, gap <= 6 m, the human moved 2-6 m to that side); **proxy** = every map site with gap <= 6 m (23 per run; the map check is
the cleaning; the human gate dropped). Reached = the ego crossed the site station s before its zero event.

Per decision k with the ego station in [s - 40, s + 60] m (`rf_lchyst.py`): served cmd (drive.jsonl `cmd`: 0 L, 1 S, 2 R), the cmd's own
input (y of the first waypoint >= 5 m of the latest route message with ts <= t0), the ego's move toward the target side since s - 40 (tangent
frame at s), d_target = the ego's remaining lateral distance to the route point at s + 40 (tangent frame at s), plan lateral at 2 s toward
the side (sign x poses[3].y).

- Flip at a site: the first decision k_f with cmd = S after at least one decision with cmd = the site's side, inside the window.
- **C1 (drift-then-snap)** at a site: a flip exists with d_target(k_f) >= 1.5 m, plan 2 s lateral toward the side at k_f - 1 >= 0.3 m, and
  min over k_f .. k_f + 2 of it <= 50 % of the k_f - 1 value. Counted on RSG-L18-s0's reached sites (validated + proxy, deduplicated by
  scene); arms reported alongside.
- **C2 (threshold)**: at the C1 flips, d_target(k_f) in [1.5, 2.5] m (and the move so far, the cmd input y at the flip, reported).
- Output: `results/lc_hyst/reading.md` (per site timeline table) and `figs/lc_hyst_timeline.png` (cmd, d_target, plan 2 s lateral vs k).

## Pilot (GPU, only if C1 holds on >= 3 RSG sites)

`JEV_CMD_LATCH=<m>` in pai_driver.py (unset = bit-identical serving, shown on 2 scenes against r281/rsg): once the served rule has emitted L
or R while the route is straight (|heading change of the message's waypoints| < 15 deg) with |y| >= 2 m, hold that command until the ego's
lateral offset to the latched target (the latch-time waypoint carried into the current frame) drops below <m> (default 0.5) or 8 s pass.
RSG-L18-s0 at gains 20 on the reached sites' scenes (validated + proxy, deduplicated); lines: at the validated sites, shifted >= 2 m AND in
the route's lane after (RSG today 1 / 6, of 4 reached); no new zero on these scenes vs r281/rsg. Report both tables.
