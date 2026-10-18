# PAI-HOLD-DIAG: would the nuPlan static hold (leash3 / stop3) fix the PAI ACC-GAP zeros? (2026-10-18)

**Answer: no.** The scorer rule is the same on PAI. But on PAI 281 the hold's handoff latch reaches 2 of the 76 ACC-GAP zeros. It
would also latch 34 scoring scenes where the human launches. Offline net: leash3 -0.076, stop3 -0.090 per scene. The zero-only upper
bound is about +0.007. Do not port it, and do not spend a 281-scene read on it.

CPU only, stored rollouts only (decision 330 stage 2, `$DATA_DIR/runs/route_follow/acc_gap/read/s2/runs/rsg-l18-s0_c00{0..7}/sim`).
Code: `scripts/pai_hold_diag.py` (`extract`, `report`). Per-rollout facts: `$DATA_DIR/runs/route_follow/acc_gap/hold_diag/facts.csv.gz`
(box). Full tables, including every corridor / offroad / other zero and every latched rollout:
[pai_hold_diag_tables.md](pai_hold_diag_tables.md). The scorer rows reuse the PAI-CORRIDOR extract
(`experiments/alpasim_proxy/results/pai_corridor.md`, which ran the leash3-only version of this question on four reads). That extract
agrees with this one: 34 latched, 32 launches, leash3 -0.076 on ACC-GAP.

**Counts.** On 281 rollouts ACC-GAP has 76 zeros: corridor 53, offroad 14, at-fault collision 9. Decision 330's "75" is the 280
scenes both arms scored. The extra zero is 054b5901, a corridor zero, so d330's ACC-GAP count is corridor 52. The brief's
"corridor 50 / offroad 16" are the ACC arm's numbers in d330.

## 1. Scorer rule on PAI: same code, same disc for a still human

The PAI stack's scorer is `third_party/alpasim` 0bb4c4b on the box, `src/eval/src/eval/scorers/ground_truth.py`. Its corridor setting
is `max_dist_to_gt_trajectory: 4.0` in `src/wizard/configs/e2e_challenge/ec2.yaml:67` and in each stored read's `sim/eval-config.yaml`.

```python
def _end_direction(gt_linestring) -> np.ndarray | None:
    """Unit heading of the final ground truth segment, or None if degenerate."""
    ...
    direction = coords[-1] - coords[-2]
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 0 else None

def _lateral_distance_to_gt(gt_linestring, point, gt_length_m, end_direction):
    if end_direction is None or gt_length_m <= 0:
        return float(gt_linestring.distance(point))
    if gt_linestring.project(point) < gt_length_m - 1e-9:
        return float(gt_linestring.distance(point))
    end_point = np.asarray(gt_linestring.coords, dtype=np.float64)[-1, :2]
    offset = np.array([point.x, point.y], dtype=np.float64) - end_point
    return float(abs(offset[0] * end_direction[1] - offset[1] * end_direction[0]))
...
            lateral_distance_to_gt_trajectory.append(_lateral_distance_to_gt(full_gt_linestring, ego_polygon.centroid,
                                                                             full_gt_distance_traveled_m, gt_end_direction))
...
                name="left_corridor_laterally",
                ... float(value >= corridor_m) for value in lateral_distance_to_gt_trajectory
```

Two more pieces of the aggregation complete the rule:
- `aggregation/main.py:156` cuts every step after the first one with `dist_to_gt_trajectory >= 4.0` (`RemoveTimestepsAfterEvent`).
- `left_corridor_laterally` then zeroes the scene.

**Still human, PAI and nuPlan alike.** If the human stands still at the end of the track, the last GT segment is jitter (< 1 mm) or
exactly 0. With exactly 0, `None` gives plain distance. With jitter, the projection of a point ahead lands inside the tangle at the
rest point, not on the end. Either way lateral = distance, so the corridor is a 4 m disc around the rest point.

**The PAI difference is the GT itself.** The GT is the human's whole 20 s rig track, median 164 m. The disc only appears when the
human ends the 20 s still, which is 41 of 281 scenes. Mostly that human was moving at the handoff and stopped mid-scene. On ACC-GAP,
19 of 24 tripped still-end scenes are corridor zeros. With a moving end, 2 of 80 trips past the GT end are zeros
(pai_corridor report table 2).

## 2. The 53 corridor zeros (and the other 23 zeros) by mechanism

**Inputs.** "At handoff" means the hold's own inputs at the PAI handoff call: 1.6 s after the first call (`serve_fix.ACC_T_HAND`),
and the call 0.1 s before it. Force-GT makes the ego's measured speed the human's.

**Latch rules** (static_hold.py constants):
- leash3 latches if v < 0.3 m/s.
- stop3 also latches if a < -0.5 m/s² and v / -a <= 3 s. Its cap is v² / (-2a) + 3 m.

**Trip.** The trip is the first metric step with d >= 4 m.

| class | n | what happens | hold reach |
|---|--:|---|---|
| A human still from handoff to end (leash3 latch) | **1** | 3fe2bac8: human never moves (GT 0.19 m), ego creeps 18.7 m at 3.8 m/s, trips at 5.3 s | leash3 fixes (+1.0) |
| B human braking to a stop at handoff (stop3 latch) | **1** | 2bbf1efc: 3.7 m/s, -1.3 m/s² at handoff, cap 8.2 m, human rests 5.8 m on from 4.6 s; ego creeps to 10 m and trips at 18.2 s at 0.15 m/s | stop3 fixes (upper ~0.77) |
| C human stops mid-scene (still end), not latchable | **18** | human moving at handoff (median 10.8 m/s), stops at 6.7-19.3 s. In 15 the ego drives through the rest point (trip 2.7-5.0 m past it at 3.8-15.7 m/s; ego speed when the human came to rest median 12 m/s, 0-71 m behind). In 3 the ego trips before the rest point, in a turn (2466266e, 62cfe9d3, 9b6c68b4) | none: nothing at serve time knows the rest point |
| D turn / turn exit (human path turns >= 30° within ±3 s of the trip point) | **20** | ego at 3.5-20 m/s, tens of metres before the GT end. Includes abd45a30, e0f6ebe1, a4d2f7cf (d328 / d330's turn losses) | none. 2 are leash3 latches where the human launches (de9db5dc, fd01e9ab) and 1 a stop3 latch (e37aa159). A hold would only turn them into near-zero progress |
| E high-speed lateral drift (straight, ego >= 20 m/s) | **4** | 0e05cb0c, 0feb3afa, 32be03e4, 4cb05d65: 27-39 m/s, lateral exit 11.7-15.0 s in, the d328 / d330 drift class | none |
| F other (moving end, straight, < 20 m/s) | **9** | 2 are stop-like with a creeping end (2d10a08d, 35e082a4: human stops at 16.0 / 6.6 s, ego passes 4.6-4.7 m beyond at 7-16 m/s); the rest are mid-path lateral exits. 1 stop3 latch (b40f4729) where the human launches | none |

**Offroad (14).** None latches under either rule. All 14 have the human moving at the handoff (1.5-26 m/s), and in 13 the human never
stops. 7381c9fa ends still, but in an 82° turn.

**At-fault collision (9).** One is a leash3 latch with a launching human (22fa5c92). A hold would trade the collision for near-zero
progress, an upper bound of about +0.07, not modelled further.

## 3. Upper bound and launch risk

**Offline model** (`report`). A latched ego stays within the cap of its handoff point.
- **Fix.** The human is still at the end and rests within cap + 4 m. The held score is 1 if GT < 5 m, else min(1, cap / (0.8 GT)).
- **Launch.** The human later goes farther than the cap. The held score is <= min(1, cap / (0.8 GT)).
- Collisions inside the leash are not modelled, so the estimate is optimistic for the hold.

| rule | latched | human still to end | zeros fixed | gain | launches | launches scoring > 0 now | cost | net / 281 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| leash3 | 34 | 4 | 1 (3fe2bac8) | +1.19 | 32 | **29** | -22.52 | **-0.076** |
| stop3, extra latches | 9 | 2 | 1 (2bbf1efc) | +0.88 | 7 | **5** | -4.91 | -0.014 |
| stop3 total | 43 | 6 | 2 | +2.08 | 39 | **34** | -27.43 | **-0.090** |

- **Upper bound on zeros fixed: 2 of 76.** That is +1.8 / 281 ≈ +0.007 with an oracle release that never costs a launch, below
  any +0.01 bar.
- **Launch risk: 34 scoring scenes.** In 29, the ego is < 0.3 m/s at the handoff and the human later moves away. In 5 more, stop3's
  braking rule fires and the human then drives on. Most are ACC's good launches: 23 of the 29 leash3 launches score >= 0.82 today.
- **Why PAI and nuPlan differ.** On nuPlan the window ends 5.5 s in. Holding a launch there costs little (-0.014), and 40 % of
  windows are human-still. On PAI the human drives on for up to 18 s, and progress over the whole track is the score.
- **A release rule cannot rescue it.** The nuPlan release study (d351 diag, d354) found no serve-time signal that separates "the
  human will launch" (plan-length / lead AUC 0.59-0.85, best release +0.003).

## 4. pai_driver.py wiring and what a port would take

`pai_driver.py` does not wire `JEV_HOLD`: there is no import of `static_hold` and no hook. The hold exists only as `lib/sh30_hold.py`,
which monkey-patches `sh30_driver.make_cached_plan`. It refuses `JEV_TRACK` because the feasible reference bypasses
`make_cached_plan`.

The PAI image serves `JEV_TRACK=gate` (`TR.build` -> `TR.cached_plan`), so the wrapper approach does not apply. A port would be a
direct hook in `Driver.drive`:
1. Keep a per-session `HD.new()`.
2. After `FX.apply` (ACC / GAP have re-timed `o["poses"]`) and before `TR.build`, set
   `o["poses"], hi = HD.step(s.hold, o["poses"], v, p0[:2], t0, o)`.
3. Log `hi`.

`static_hold` also hard-codes the nuPlan clock:
- `T_HAND = 0.5` would need to be `serve_fix.ACC_T_HAND = 1.6`.
- stop3's deceleration would come from consecutive 0.1 s calls instead of 0.5 s calls.

About 20-30 lines plus an off-is-bit-identical parity check, a new PAI image, a smoke and an audit. Given section 3 it is not worth
doing.

**Where the PAI mass actually is.** The real stop-class mass is class C plus the 2 stop-like F zeros: 17 zeros where the ego drives
through the rest point of a human who stopped mid-scene. Fixing them needs the driver to stop where the human stops (stop line /
stopped lead / red light; d315 / d328). That is a different line from a handoff latch.
