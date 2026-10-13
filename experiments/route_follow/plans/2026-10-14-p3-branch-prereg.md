# Pre-registration: turning school P3, branch ceiling from the policy's own output, 2026-10-14

Context: [P2](2026-10-13-p2-branch-prereg.md) and its read [results/p2_branch](../results/p2_branch/README.md): the human log brings the ego
to the junction and RSG-L18-s0 takes over; 4 / 12 PAI turn scenes complete from some handover, 8 fail from every one. P1: the open-loop plan
at human states is close to the human's. Written, committed and pushed before any rollout. Amendments are appended, never edited in place.

**Question.** Is the policy's own output close to a working turn, so that a push of the right shape for a few seconds is enough and the
policy then continues the turn itself? If the ceiling over a family of simple transforms of the policy's own plan is high, demonstration rows
on the policy's own states (filtered self-demonstration / DAgger with the simulator as expert) are the next training line. If it stays at
P2's level, the policy cannot produce the turn and a teacher is needed.

## Instrument

- Stack exactly as P2: `experiments/route_follow/scripts/rf_branch.sh` -> `onp_pass2.sh` (MODE=driver) -> `pai_run.sh`, `onp_log_driver.py`
  with `ONP_SWITCH`, checkpoint **RSG-L18-s0**, serving `JEV_VCONT=1.0 JEV_LEAD=1`, wizard `controller.gains.rel_front_steering_angle_weight=20`,
  image `jev-alpasim:p2h10-f-s0-6f3d05d1`, HARMONIZER 0, full 20 s rollouts, Tokyo card 1, CONC 4, compose prefix p3-. Runner:
  `experiments/route_follow/scripts/rf_p3.sh`.
- Same 12 scenes and the same handover decisions as P2 (P2's `plan.json`, reused unchanged): handovers **d10** and **d0** (k_eff as there).
- New, `onp_log_driver.py` `ONP_BRANCH=g<gain>s<scale>[t<s>]` (unset = off: the driver is the P2 driver): for the decisions k_eff .. k_eff + 29
  (window T_b = 3.0 s at 10 Hz) the served plan's 8 poses (ego frame, after the VCONT / LEAD serving change, i.e. what the MPC would get) are
  transformed before `make_cached_plan`; before and after the window the raw policy plan is served. Transform `branch_poses`:
  - curvature gain g: each pose's yaw and each segment's course (from the previous pose, the ego first) about the ego heading are multiplied
    by g; segment lengths are kept; the path is re-integrated (same arc length, g times the heading change);
  - speed scale s: the (transformed) path is re-sampled at s times each pose's arc length (same path, shorter: s times the speed);
  - g = s = 1 returns the plan untouched.
  Records gain `branch` and, inside the window, `poses_branch` (served); `poses` stays the raw plan.
- Cells (transforms), in run order: identity (1, 1); (1.5, 1); (1.25, 1); (1, 0.8); (1, 0.6); (1.5, 0.8); (2.0, 1); (1.5, 0.6). Each at d0 then
  d10: 8 x 2 x 12 = 192 rollouts, about 3.5 h. If time is short, (2.0, 1) and (1.5, 0.6) are dropped first (they run last).

## Smoke gates (before the transform arms; the runner stops on a failure)

- **S1**: the identity arms (d0@g1s1, d10@g1s1, 24 rollouts) against P2's stored d0 / d10 runs: the served plan of every decision bit-identical
  in all 24 scenes (the d241 spread is accepted only if an amendment explains a difference). These identity runs are the identity cells.
- **S2**: f9e361aa at d0@g1.5s1 (one rollout) against the identity run: the transform is on at exactly 30 decisions starting at k_eff, the
  raw and transformed 8 poses at k_eff are logged and visibly differ, no record outside the window has `poses_branch`, and the first decision
  after the window serves the raw plan. Before k_eff the plans equal the identity run's.

## Measures

As P2, `rf_branch.py read` unchanged in its definitions: entry / exit / target on the log, **turn completed** (station at or past the target
within 2.5 m of the route, before the zero event and the rollout end), zero type, score. New (secondary), at the window end decision
k_eff + 30: offset from the log (`log_now`, total and lateral), distance to the route, fed speed.

## Lines

- **Ceiling** = scenes (of 12) with at least one completing cell over all (handover, transform) cells, identity included.
- **Pass: ceiling >= 8 / 12.** Transforms of the policy's own plan contain the turn; the self-demonstration line is supported.
- **Fail: ceiling <= 5 / 12** (P2's identity ceiling at d0 / d10 is 4). The policy's own plan does not contain the turn under these
  transforms; the school needs an external teacher.
- **6 - 7 / 12: inconclusive**; report per scene which transform completes and the failure type of the rest.
- Secondary (no line, reported): for each completing scene whether curvature, speed or both were needed (the simplest completing cell);
  window-end offset / route distance / speed, completing vs failing cells; whether the raw policy keeps the turn after the window (the
  completions themselves show it, since the policy drives alone from k_eff + 30; an amendment may re-run the completing cells with a 1.5 s
  window if time permits).

## Not claimed

12 scenes, one seed, one checkpoint; the handover is still from the log near the entry (route C). A completion means "a simple push of the
policy's own plan, for 3 s, from a human entry state, is enough", not that self-demonstration rows will train it. A transform that completes
by slowing the car may be cut short by the log's car behind (rear-end); such rollouts are reported as not completed, as in P2.
