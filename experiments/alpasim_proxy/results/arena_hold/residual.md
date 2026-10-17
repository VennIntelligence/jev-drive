# ARENA-RESID: where RSG-L18-s0 + stop3 still loses points on the held arena windows (2026-10-18)

CPU only, stored reads, no closed-loop read. Code `scripts/arena_resid.py` (`extract`, `offroad`, `report`). Full tables are in
`$DATA_DIR/runs/alpasim_proxy/arena/resid/report.md` on the box, with the per-step / per-call parquets next to it. The reads are those
of [hold2.md](hold2.md): stop3, leash3, the stored arena reads RSG / NR1F / NR1HV2 / YR, and the JT reads JTN (off, stop3) and
JTB (stop3).

**Result: there is no serve-side lever worth +0.005 in what is left. Of the 125 points that separate stop3 from 1.0, 26 cannot be
recovered by any policy, because the human's own track is offroad. Most of the remaining 99 sit in classes with no run-time signal:
launches, a lead the model misreads, and turns (a closed line). So no prereg supplement, no new switch and no read.**

## Scorer facts used (third_party alpasim `src/eval`)

- **Score.** 0 on an at-fault (front or lateral) collision, offroad, or a lateral corridor exit. Otherwise
  `min(clamp(progress_rel_to_total, 0, 1) / 0.8, 1)`, or 1 when the human travels < 5 m.
- **Progress.** The ego centroid's normalised projection onto the human's window track, scaled to the reachable part. It is taken at
  the last kept metric step.
- **Kept steps.** From the first policy-relevant step (1.017 s; the force-GT steps are dropped, `eval_relevant`) up to and including
  the first step with offroad, any collision, or d >= 4 m.
  - A rear collision is not at fault, but it still freezes progress where it happens.
  - Offroad is the ego footprint not covered by a lane polygon, by the lane union, or by the road-area union.
- **Check.** Recomputing the score from the stored per-step metrics reproduces every non-zero score exactly (max error 1e-16).

## 1. Point accounting

stop3 has a mean of 0.9438 on 2 228 windows, so the gap to 1.0 is **125.2 points (0.0562)**. Points = summed (1 - score).

| mechanism | class | points | windows | per window | share |
|---|---|--:|--:|--:|--:|
| **launch hold cost** (static latch, human launches) | creep 19.0, slow 11.3, rear-hit 9.9 | **40.2** | 80 | 0.0181 | 0.32 |
| **human track offroad** (any policy scores 0) | human pose offroad at engagement 18, human track offroad later in the window 8 | **26.0** | 26 | 0.0117 | 0.21 |
| policy offroad | turn / junction 11, straight near the edge 4, held static roll 2 | 17.0 | 17 | 0.0076 | 0.14 |
| free-session partials | slow 13.1 (155 windows), rear-hit 1.8 | 14.8 | 163 | 0.0067 | 0.12 |
| at-fault collisions | AF-front 12 (11 static-latched, a stopped lead at 2.36-2.88 m), AF-side 1 | 13.0 | 13 | 0.0058 | 0.10 |
| corridor | free sessions, human brakes to a stop after the handoff | 8.0 | 8 | 0.0036 | 0.06 |
| stop-latch partials | the stop leash cuts a human who does not fully stop | 6.2 | 26 | 0.0028 | 0.05 |
| **all** | | **125.2** | 333 | 0.0562 | 1.00 |

By stratum: launch 44.5, turn 24.0, other 23.5, static 16.0, braking 14.9, cruise 2.2.

Excluding the 26 human-offroad windows (2 202 windows), stop3 is 0.9550 and RSG 0.7950. The windows are offroad for every model, so
pairings and rankings do not move. The arena prereg (amendment 1) already drops windows where a human replay scores 0, but only when a
replay read exists. None was run on the arena. The classification here is geometric (section 2), not a replay read.

## 2. Offroad zeros (43)

Method:
- **Human footprint.** For each window, the human's logged track, with the Pacifica footprint at every 0.1 s pose, is checked against
  the drivable surface. The surface is the union of lane and road-area polygons, built with the scorer's own polygon builders
  (route_follow's `rf_offroad.Road`).
- **Ego footprint.** The same check at the first relevant offroad step, from the ego anchors.
- **Validation.** This approximation agrees with the scorer: the ego footprint is uncovered there in 41 of 43 windows.

| group | windows | first offroad (s) | ego to the human track (m) | ego vs human heading (deg) | human footprint min margin (m) | also offroad in RSG / NR1F / NR1HV2 / YR / JTN off / JTN stop3 / JTB stop3 |
|---|--:|--:|--:|--:|--:|---|
| human pose offroad at engagement | 18 | 1.02 | 0.02 | 0.2 | -0.25 | 18 / 18 / 18 / 18 / 17 / 17 / 18 |
| human track offroad later in the window | 8 | 2.02 | 0.05 | 1.3 | -0.04 | 8 / 7 / 6 / 8 / 6 / 6 / 6 |
| policy, turn / junction | 11 | 3.52 | 1.63 | 12.9 | +0.43 | 11 / 7 / 4 / 9 / 7 / 7 / 7 |
| policy, straight near the edge | 4 | 3.27 | 0.48 | 7.0 | +0.54 | 4 / 2 / 2 / 3 / 3 / 3 / 2 |
| held static roll | 2 | 4.27 | 2.30 | 1.7 | +0.17 | 2 / 2 / 0 / 2 / 0 / 0 / 2 |

Notes (columns 3-5 are medians):
- **Artefacts (26).** In 18 windows the human pose is already offroad during force-GT, and the ego, still on that pose at 1.02 s,
  is zeroed at the first scored step. In 8 more, the ego is on the human's path (median 0.05 m) where the human's own footprint leaves
  the surface. These are map / scorer artefacts. They sit in a few logs: 10 of all 43 offroad zeros come from one log
  (`2021.10.06.08.16.17_veh-52_00181_00574`), and 5 each from two others.
- **Policy (15).** These are 0.4-4.7 m off the human's path with the heading off by a median 7-13 deg, mostly at junctions and curves. They are
  shared by 2-7 of the 7 reads. This is the turn weakness that turn night (d299-d314) closed at serve time: turn-gain, receding turn
  and intent hold were all refuted.
- **Held static roll (2).** One log; the human is parked 0.17 m from the road edge. leash3 / stop3 roll 2.1-2.5 m off the edge, while
  `full` scores 1 on both. This belongs to the leash's cost, like the 11 AF-front zeros.

For the brief's question "serve rule with run-time information": the 26 artefacts cannot be fixed by any policy, the 15 policy offroads
are a closed line, and the 2 static rolls are the leash trade-off, which is already balanced (diag.md (b)).

## 3. Slow partials

The 185 "slow" partials of hold2.md split three ways:
- **Free sessions (128).** 12.8 points.
- **Static-latched launches (34).** 11.3 points, the hold's cost.
- **Stop-latched (23).** 5.4 points.

Free-session partials (163 windows, including 27 at 0.98-1 and 8 rear-hits), by the nearest obstacle over the kept steps
(scorer `min_distance_to_obstacle_m`):

| nearest obstacle | windows | points | median ego / human distance | median progress / 0.8 |
|---|--:|--:|--:|--:|
| 0 (contact: the 8 rear-hits) | 8 | 1.8 | 0.63 | 0.85 |
| < 5 m | 97 | 8.4 | 0.68 | 0.93 |
| 5-10 m | 20 | 1.6 | 0.69 | 0.93 |
| 10-20 m | 11 | 1.1 | 0.72 | 0.95 |
| > 20 m or none | 27 | 2.0 | 0.69 | 0.94 |

A typical slow window loses about 0.1: the car does 0.68 of the human's distance, and the score forgives up to 0.8.

**Speed lever.** If every free slow window reached 0.8x with no new collision, the gain would be +12.8 points (+0.0057). That ceiling
has no safety condition: 117 of the 155 non-contact windows had an obstacle within 10 m. With the condition the brief suggests (no lead within
X m), only the windows with no obstacle within 10-20 m qualify: 3.1 points (+0.0014). Even that assumes the policy can see the gap,
and its lead head cannot (diag.md (b): it reads 21 m for a lead at 2.4 m). d329 and d353 both found that faster costs front collisions.

**Not pursued:** the ceiling is below +0.005 even before collision risk.

## 4. Launch cost (40.2 points)

diag.md showed this cost is inseparable using RSG's plan and lead outputs. Two new observables were checked:

- **JTN's held plan.** JTN was trained without the arc term and stands still natively (no-hold +0.089, d353), so its plan while held
  could have been a cleaner "about to launch" signal. It is not: AUC 0.84 against RSG's 0.76 (82 launches with the human >= 5 m
  vs 807 still-end windows). Releasing at a 2 s arc >= 2 m frees 96 % of launches but 32 % of still-end windows; at >= 3 m, 93 % / 28 %.
  Released still-end windows are mostly zeros: unheld JTN scores 0.37 on the 261 that an arc >= 2 m would release (63 % zeros),
  and 0.996 on the rest.
- **Leash length.** Still-end latched windows never get farther than 3.02 m from the human's rest point under the 3 m leash, so the
  corridor allows up to ~3.9 m. Offline, using diag.md's estimator (calibrated at 3 m: 91.9 estimated vs 99.7 read on launch windows):

  | leash (m) | launch progress | new front hits (stopped lead in (3.05, leash] m) | net | per window |
  |--:|--:|--:|--:|--:|
  | 3.3 | +2.7 | 2 (-0.8) | +1.9 | +0.0008 |
  | 3.5 | +4.4 | 3 (-1.3) | +3.1 | +0.0014 |
  | 3.7 | +6.2 | 4 (-2.3) | +3.8 | +0.0017 |
  | 3.9 | +8.0 | 10 (-7.7) | +0.2 | +0.0001 |

  At most +0.0017, with the corridor margin cut from 1 m to 0.3 m. Below the line.

- **Rear-hits (9.9 points).** In 15 held launch windows the logged follower hits the held car, which freezes progress. The policy
  sees no rear traffic, so there is nothing to act on.

## 5. Levers and recommendation

| lever | offline estimate | collision risk | verdict |
|---|--:|---|---|
| leash 3.7 m for static latches | +3.8 (+0.0017) | 4 new front hits counted; corridor margin 0.3 m | below +0.005 |
| speed-up on clear road, no obstacle within 10 m | <= +3.1 (+0.0014) | no run-time gap signal | below |
| speed-up on every free slow window (no safety condition) | <= +12.8 (+0.0057) | 117 / 155 non-contact windows have an obstacle within 10 m | ceiling only, unsafe |
| release a launch from the held plan (RSG / JTN) | <= +6.8 in-sample (diag.md) | still-end windows released 28-53 % | below |
| stop-rule thresholds | <= +0.5 (diag.md sensitivity) | none | below |
| turn offroad, serve time | <= 11 | closed line (d299-d314) | not reopened |

**No lever reaches +0.005, so per the brief: no prereg supplement, no new `JEV_HOLD`, no closed-loop read.**

RSG-L18-s0 + stop3 stays the candidate. Correcting the ruler is free: on the held arena, the ceiling for any policy is 0.988, not 1.0.

## What remains (all need the model, not the server)

1. **Launch: 40 points, +0.018 ceiling.** This is the largest single residual. It needs a model whose plan, while held, separates
   "the human is about to go" from "the human stays". Neither RSG (AUC 0.76) nor JTN (0.84) does. The cue the human uses (the lead
   moving, a light) is what the lead head misreads.
2. **Policy offroad in turns: 15 points.** These turns are shared across the L18 family.
3. **The lead gap at 2.4-2.9 m: 11 AF-front, 9 of them from one log.** The lead head reads about 21 m there.
4. **Free slow: 13 points.** About 0.1 per window, mostly with an obstacle near. Any speed gain has to come with a gap signal the
   model can see.
