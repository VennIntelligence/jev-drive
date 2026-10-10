# HLEAD: openpilot's lead path in the HUGSIM closed loop removes the front-vehicle collisions (9 / 9 no longer end on the lead) and leaves HUGSIM 64 HD where it was (-0.005 [-0.068, +0.049])

Written 2026-10-10. Pre-registration: [plans/2026-10-10-hlead-prereg.md](../plans/2026-10-10-hlead-prereg.md) (pushed before any run with the
switch). One seed (`SH30-F-s0`), one preset (`spec_plan_smooth`), all runs through `python -m jevdrive.bench` on the GPU pool, 1.6
card-hours. Script [`scripts/lbd_hlead.py`](../scripts/lbd_hlead.py); tables [hlead/tables.md](hlead/tables.md), one row per scenario in
[hlead/paired_units.csv](hlead/paired_units.csv), the bench report in [hlead/bench/](hlead/bench/). No training, no parameter was tuned.

## What was run

| arm | run dir under `$DATA_DIR/runs/bench/hugsim/` | what |
|---|---|---|
| A0 | `SH30-F-s0_spec_plan_smooth` | the stored run decision 237 classified (HD 0.4432) |
| A0r | `SH30-F-s0_spec_plan_smooth-rhlead-off` | the same configuration after the code change, switch off (repeat `hlead-off`) |
| A1 | `SH30-F-s0_spec_plan_smooth-ca9065d178c07` | `--opts '{"op_lead": {}}'` |

The switch is the AlpaSim one (`JEV_LEAD`, decision 226), same code: radard's lead probability filter and vision lead, the lead MPC, the
planner tick moved unchanged from `experiments/alpasim/lib/serve_fix.py` to `jevdrive/openpilot/lead_long.py` (serve_fix re-exports
every name; its served poses are bit-identical before and after the move on 1 200 random decisions). The HUGSIM agent option `op_lead`
pulls the plan's points back along the plan's own path to the pointwise minimum of the plan's speed and the lead MPC's speed solution,
after `forward_only` and before `straight_stop`; the curvature that steers is untouched. The planner runs in the model's dilated clock
(ego speed x 1.25, 4 ticks per simulator step); camera to ego-box front 1.5 m. Stage 1 (the 9 L1 scenarios + the 5 completed scenarios
with the lead head on most) passed its gate (L1 fixed 5 / 9, 4 of 5 clean scenarios still complete), then all 64 ran.

## HD on HUGSIM 64 (paired per scenario, bootstrap over scenarios, B 10 000)

| arm | HD | RC | NC | TTC | complete | fg collision | bg collision | off route | max_steps (standing) | launch stall |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 (stored) | 0.443 | 0.560 | 0.700 | 0.631 | 26 | 28 | 9 | 1 | 0 | 1 |
| A0r (switch off, re-run) | 0.444 | 0.561 | 0.702 | 0.633 | 26 | 28 | 9 | 1 | 0 | 1 |
| A1 (lead path) | 0.438 | 0.524 | 0.749 | 0.717 | 25 | 24 | 10 | 1 | 4 | 14 |

| difference | HD [95% CI] |
|---|---|
| A1 - A0 | **-0.005 [-0.068, +0.049]** |
| A1 - A0r | -0.006 [-0.068, +0.047] |
| A0r - A0 (same configuration twice) | +0.001 [-0.007, +0.012] |
| base seed spread, for scale | SH30-F `spec_plan_smooth` s0 0.4432 / s1 0.4338 (0.009); P2H10-F `spec` 0.388 / 0.402 (0.014) |

By the registered reading: no discernible difference. The lead path trades progress (RC -0.036) for fewer contacts (NC +0.049, TTC
+0.086) and the two cancel in HD. The interval is wide because the result is a few scenarios of +-0.6 to +-0.96 each.

By the class decision 237 gave each scenario:

| class in A0 | n | HD A0 | HD A1 | A1 - A0 | what happened in A1 |
|---|---|---|---|---|---|
| longitudinal, L1 (lead in the driven band) | 9 | 0.132 | 0.287 | +0.155 [+0.036, +0.304] | below |
| longitudinal, L2 (fast turn entry) | 4 | 0.198 | 0.221 | +0.023 [+0.011, +0.036] | 3 unchanged, 1 off_route -> bg_collision; none fixed |
| clearance (C1 7, C2 5) | 12 | 0.101 | 0.188 | +0.087 [+0.001, +0.218] | 1 C1 complete (+0.69, an AttackPlanner actor missed), 1 C2 complete that A0r also completes (repeat noise), 6 C1 hit by the same actor, 2 C2 unchanged, 2 C2 now end on an oncoming actor |
| other (O2b 11, O2c 1) | 12 | 0.075 | 0.061 | -0.014 [-0.025, -0.003] | 12 / 12 hit by the same actor; slightly less distance covered before it |
| route | 1 | 0.636 | 0.620 | -0.016 | unchanged |
| complete | 26 | 0.909 | 0.806 | -0.103 [-0.227, -0.001] | 22 still complete (mean +0.008, worst -0.042, 1.11x the steps), 4 lost, below |

## The 9 L1 units

Registered labels: fixed = ends complete or max_steps; unchanged = still fg_collision; new failure = another end. "struck in A1" is finer
than the label: no L1 unit ends on the actor it struck in A0.

| scenario | A0 | A1 | HD A0 -> A1 | registered label | what the run shows |
|---|---|---|---|---|---|
| scene-032-medium-00 | fg | complete | 0.343 -> 0.984 | fixed (complete) | slows for the standing car, passes it at 2.9 m lateral, finishes |
| scene-0254-hard-00 | fg | max_steps | 0.057 -> 0.094 | fixed (standing) | stands 4.0 m behind the standing car for 85 s |
| scene-0411-medium-00 | fg | max_steps | 0.154 -> 0.205 | fixed (standing) | stands 4.4 m behind for 74 s |
| scene-0528-medium-00 | fg | max_steps | 0.134 -> 0.188 | fixed (standing) | stands 4.7 m behind for 83 s |
| scene-2510_2710-hard-00 | fg | max_steps | 0.000 -> 0.002 | fixed (standing) | stands 3.8 m behind for 81 s |
| scene-032-medium-02 | fg | bg_collision | 0.257 -> 0.615 | new failure | follows the 3 m/s lead 17.7 m behind for 13.5 s, then brushes a background object at arc 35.9 m (A0 struck the lead at 34.9 m) |
| scene-1290_1490-medium-01 | fg | bg_collision | 0.146 -> 0.387 | new failure | follows the 2 m/s lead, 10 m further than A0; once the lead is gone the plan turns left and hits background at 6 m/s |
| scene-0138-extreme-00 | fg | fg_collision | 0.071 -> 0.070 | unchanged | creeping at 0.7 m/s 10 m (centre distance) behind the standing car when a second, oncoming actor (7.5 m/s, head-on) hits the front |
| scene-0254-extreme-00 | fg | fg_collision | 0.028 -> 0.042 | unchanged | same: 0.5 m/s, 10 m (centre distance) behind the standing car, struck head-on by an oncoming actor (6.2 m/s) |

- Registered count: fixed 5 / 9 (1 complete, 4 standing), unchanged 2, new failure 2: at the registered line (>= 5) for "the lead path
  removes the front-vehicle collision". Collision with the A0 lead itself: 0 / 9.
- The HD gain of the class (+0.155) comes from three units (+0.64, +0.36, +0.24). The four that now stand behind a parked car gain
  0.002-0.05: HUGSIM puts a standing ConstantPlanner car in the lane and scores route completion, and openpilot's lead path has no
  way around it.
- Decision 157's cautions, checked: at standstill the true bumper gap is 3.8-4.7 m and the lead head reads 0.7-1.1 m too far there
  (smaller than the 4.4 m decision 157 saw at 1.9 m; the 6 m stop distance of the lead MPC absorbs it). No unit hits a side-offset
  lead. Decision 140's case (rule releases the car onto a target the lead head does not see) does not occur in these 9; the limit has
  no release.

## The 4 fast-turn-entry (L2) units

None fixed. 0041-medium-00, 570_770-medium-00 and 5980_6180-easy-00 end as before (bg_collision); 570_770-easy-00 changes from off_route
to bg_collision. HD +0.006 to +0.042 each. The lead head is on in 4-19% of their steps; the lead path does not look at curvature.

## Scenarios that got worse (A1 - A0 <= -0.05, A0r within 0.05 of A0): 3

| scenario | A0 -> A1 | HD | cause |
|---|---|---|---|
| scene-090-hard-01 | complete -> fg_collision (front) | 1.000 -> 0.036 | too slow at launch: an oncoming AttackPlanner actor is read as a lead 14 m ahead (the lead head has no negative lead speed, decision 153), the launch is held to 2.2 m/s at 2.25 s where A0 has 4.1 m/s, and the actor strikes the front at 4.75 s; A0 was past it |
| scene-124-hard-01 | complete -> fg_collision (front) | 0.992 -> 0.074 | the same sequence, struck at 5.25 s |
| scene-053-medium-02 | complete -> off_route | 0.975 -> 0.150 | too slow at launch behind a departing lead (cut 1.2-1.7 m of plan in the first 1.5 s): the lead gets 20 m away and leaves the lead head (prob 0.99 -> 0.02), the plan then swings 8 m left and the car leaves the route; in A0 the car stays 13-19 m (centre distance) behind the lead with prob 1.0 for the whole run |

No stall among scenarios A0 completed (the 4 standing runs are all former L1 collisions), no new rear-end contact (the one rear contact,
095-extreme-01, is the same actor as in A0). A fourth completed scenario, 2800_3000-easy-00, ends in bg_collision in A1 and in A0r:
repeat noise, not counted. Better by >= 0.05: 9 scenarios (5 L1, 1 C1, 1 C2 by 0.09 without changing its end, 2 completed ones by
0.05-0.08). Contacts with AttackPlanner actors move both ways with the timing: 1 gained (034-hard-01), 2 lost.

## Limiter activity

The limit changes the plan in 79% of all steps and pulls the plan end back by >= 0.5 m in 52% (42% with a lead present, 10% without:
the lead MPC's no-lead solution accelerates at no more than 2.0 m/s^2 in the model's units, 1.28 m/s^2 on the simulator's clock, and
the adapter's launches ask for more). A lead is present (filtered probability > 0.5) in 61% of all steps, 22% in the scenarios A0
completes. The runner's launch-stall count (peak speed over the first 40 steps < 1.6 m/s) goes from 1 to 14: 5 L1, 5 O2b, 3 clearance
and 1 completed scenario (which still completes).

## Switch off = before

The option adds one request flag and one stored tuple when on, nothing when off. A0r against A0: every step's sent plan and ego
position identical in 18 of 64 scenarios (HD identical in 36), same end in 62; where the plans differ, the first difference is a
median 0.007 m at a median step 2, the size two serving engines differ by. Two existing repeats of an untouched configuration
(`P2H10-F-s0_spec_plan_smooth-rr1` / `-rr2`) are identical in 56 of 64 with a largest HD difference of 0.026, so the closed loop was
not bit-reproducible before this change either; here 2 scenarios flip their end between A0 and A0r (164701907483-easy-00 bg -> complete,
2800_3000-easy-00 complete -> bg; largest |HD difference| 0.26).

## What was not determined

- Seed spread of the effect: one seed, one rollout per scenario. The mean is set by about eight scenarios of +-0.24 to +-0.96.
- Whether the three losses repeat: all three follow from the slower launch, two of them through an AttackPlanner actor that re-plans
  against the ego; another seed or repeat can move them.
- The speed-continuity switch (`JEV_VCONT`) was not run: its 1.0 s is derived from AlpaSim's MPC and has no counterpart for HUGSIM's
  tracker.
- Whether the launch slow-down is the lead reading or the 2.0 m/s^2 cap in each lost scenario (both act in the first 2 s).
- The lead MPC is solved with 8 SLSQP iterations per tick (decision 226's implementation) and is not converged: the same toy stop
  ends at a 4.8 m gap on the Mac and 5.3-5.6 m on the box depending on the BLAS build. Runs on one machine are comparable; the
  AlpaSim numbers of decision 226 carry the same sensitivity between box and image.

## Reproduce

```bash
B=".venv/bin/python -m jevdrive.bench"
$B run --model SH30-F-s0 --bench hugsim --preset spec_plan_smooth --opts '{"op_lead": {}}' --scenarios all64
$B run --model SH30-F-s0 --bench hugsim --preset spec_plan_smooth --repeat hlead-off --scenarios all64
.venv/bin/python experiments/lowboard_diag/scripts/lbd_hlead.py extract && .venv/bin/python experiments/lowboard_diag/scripts/lbd_hlead.py report
$B report --bench hugsim --preset spec_plan_smooth --arms lead=run:SH30-F-s0_spec_plan_smooth-ca9065d178c07 \
   off_rerun=run:SH30-F-s0_spec_plan_smooth-rhlead-off --vs SH30-F-s0 --out experiments/lowboard_diag/results/hlead/bench
```
