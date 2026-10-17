# PAI corridor: is the nuPlan still-end lever there on PAI 281? (2026-10-18)

**Answer: no.** The scorer rule is the same, but PAI scenes almost never start with a human who then stays still, so a handoff latch
(leash3, decision 350) has almost nothing to gain on PAI. It would cost 0.04-0.08 on the scenes where the human launches. The still-end
corridor zeros that do exist on PAI come from humans who stop mid-scene; a handoff latch cannot reach them.

Code `scripts/pai_corridor.py` (`human`, `steps`, `report`), reusing `corridor_rule.py extract` and its end-state classes. Full tables:
[pai_corridor/report.md](pai_corridor/report.md). Per-rollout CSVs on the box under `$DATA_DIR/runs/alpasim_proxy/pai_corridor/`. CPU only,
stored reads only, no closed-loop run.

## 1. Scorer settings: same scorer, different scene shape

Same scorer code and aggregation on both tracks (`third_party/alpasim` 0bb4c4b, `src/eval/.../scorers/ground_truth.py`,
`aggregation/scene_score.py`):
- `max_dist_to_gt_trajectory: 4.0`
- `progress_saturation_threshold: 0.8`, `min_gt_distance_for_full_score_m: 5.0`
- the same lateral-past-the-end rule

Sources: `src/wizard/configs/e2e_challenge/ec2.yaml` (the PAI EC2 preset), `base_config.yaml`, and each stored read's
`sim/eval-config.yaml` / `wizard-config.yaml`.

| | nuPlan arena (`e2e_challenge_nuplan_common/base.yaml`) | PAI 281 (`base_config.yaml` + `e2e_challenge/ec2.yaml`; stored reads) |
|---|---|---|
| control step, steps | 0.5 s, window 5.5 s | 0.1 s, 200 steps = 20 s |
| force-GT (handoff) | 0.5 s | 1.7 s |
| driven decisions | 10 | ~183 (every 0.1 s after 1.7 s) |
| GT polyline | the human's 5.5 s window | the human's whole 20 s rig track (matches the scorer's `gt_dist_traveled_m` to < 0.5 m) |
| GT length | static stratum median 0.003 m | median 164 m; < 5 m in 2 of 281 scenes |

On PAI the rule fires exactly as on nuPlan.

| read | still-end, tripped | still-end corridor zeros | moving-end, trip past the GT end | corridor zeros among those |
|---|--:|--:|--:|--:|
| g10 | 17 | 17 | 58 | 3 |

With a still end, every trip past the GT end is a zero, with lateral = d (the 4 m disc). With a moving end, overshoot is free (lateral / d
0.11). The other reads behave the same way. Trips past the GT end are zeros in 0.84-1.00 of still-end scenes against 0.03-0.05 of
moving-end ones (report table 2).

## 2. Still-end corridor zeros per stored read

Still end = the human's last 10 Hz GT segment is < 1 mm (corridor_rule.py's class), which holds for 41 of the 281 scenes. A corridor zero
is a score of 0 with `left_corridor_laterally` set.

| read (decision) | mean | zeros | corridor zeros | still-end corridor zeros | trip past the GT end | latched at the handoff | human moving at the handoff |
|---|--:|--:|--:|--:|--:|--:|--:|
| g10 (d327), 280 | 0.5713 | 75 | 60 | 17 | 14 | **1** | 16 |
| g10 + holdlead (d327) | 0.6124 | 76 | 61 | 18 | 15 | **1** | 17 |
| g10 + ACC (d328), 280 | 0.6760 | 75 | 57 | 19 | 16 | **1** | 18 |
| g10 + ACC + GAP (d330) | 0.6909 | 76 | 60 | 21 | 16 | **1** | 20 |

- **Who stands still at the handoff.** 34 scenes have the human < 0.3 m/s at the 1.7 s handoff, so leash3 would latch on them. Only 2 of
  the 34 have the human still to the end:
  - 5fcb2ba7: 1.0 in every read.
  - 3fe2bac8: 0 in every read; the ego rolls 22-34 m.

  The other 32 are launches: the human is more than 3 m from the handoff point 2.2-14.3 s later, with GT lengths of 12-236 m.
- **The rest of the still-end zeros.** In 16-20 per read the human is moving at the handoff (median 10.8 m/s), stops, and stands for
  the last ~6 s (median). The ego then overshoots the rest point by ≥ 4 m: progress at the trip is ≥ 0.99 in 14-16 of them. These are the
  decision 328 stop-class zeros ("the human stopped, the car did not"). ACC adds to them: 17 → 19 → 21 still-end corridor zeros for
  g10 → ACC → ACC+GAP. It keeps the cruise speed through the stop.

## 3. Offline leash3 estimate

The latch is static_hold.py's rule: ego speed < 0.3 m/s at the handoff (force-GT makes it the human's speed). The ego is then kept within
3 m of the latch point for the session. That means d ≤ 3 m < 4 m (no corridor zero), and the progress score is 1 if GT < 5 m, else at most
min(1, 3 / (0.8 L)). Collisions inside the leash are not modelled, so the estimate is optimistic.

| read | latched | gain (3fe2bac8) | cost on 32 launch scenes | net per scene |
|---|--:|--:|--:|--:|
| g10 | 34 | +1.00 | -12.46 | **-0.041** |
| g10 + holdlead | 34 | +1.00 | -12.48 | **-0.041** |
| g10 + ACC | 34 | +1.00 | -17.87 | **-0.060** |
| g10 + ACC + GAP | 34 | +1.00 | -22.32 | **-0.076** |

- **Ceiling.** The ceiling for any handoff hold, even one with an oracle release that never costs a launch, is the one scene:
  **+1 / 281 = +0.0036**, below the +0.01 bar.
- **Why the launch cost is high here.** On nuPlan the launch cost is small (-0.014) because the window ends 5.5 s in. On PAI the human
  drives on for up to 18 s and the score is progress over the whole track.
- **ACC.** ACC makes the trade worse. ACC and ACC+GAP score higher on the launch scenes (they launch with the human and keep speed), so
  holding them costs more. Under ACC+GAP the ego already creeps 2.9 m in 5fcb2ba7, which is still inside 4 m and still 1.0.

## Recommendation for the PAI probe

- **Do not add a static hold to the PAI image.** Net -0.04 to -0.08, ceiling +0.004. No hook is proposed.
- **The PAI-side still-end mass sits elsewhere: 14-16 zeros per read, ≈ +0.05 if all were recovered.** These are scenes where the human
  stops mid-scene and the ego drives ≥ 4 m past the rest point.
  - Recovering them needs the driver to stop where the human stops: a stop line, a stopped lead, or a red light (decisions 315, 328).
  - A leash cannot do that, because nothing in the driver knows the rest point.
  - It is a separate line and is not estimated here.
