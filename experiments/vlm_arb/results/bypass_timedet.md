# bypass_timedet: can "static" be decided by time instead of by a single-frame model?

Offline analysis of the 38 `drive` runs (19 routes x 2 traffic seeds; the reference arm that stops in front of obstacles and queues alike).
No CARLA, no GPU, no new driving. Definitions (stop episode, causes, detector) are in
[plans/2026-10-02-bypass-offline.md](../plans/2026-10-02-bypass-offline.md); code `scripts/bypass_timedet.py` (+ `bypass_common.py`,
`bypass_extract.py`); the full 540-setting grid is `bypass_timedet_grid.csv`. Verified = computed from the logs; inferred is stated in the text.

## 1. Stop episodes in the 38 drive runs (ego v < 0.3 m/s for at least 1 s, counted from 6.5 s)

| cause (ground truth) | episodes | >= 3 s | >= 5 s | >= 8 s | median [s] | max [s] | ran to end of run | routes |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| obstacle | 30 | 17 | 17 | 10 | 5.4 | 173.5 | 8 | 4 |
| red_light | 25 | 8 | 8 | 0 | 2.0 | 5.8 | 0 | 5 |
| junction | 21 | 7 | 7 | 1 | 1.9 | 59.9 | 1 | 7 |
| lead | 0 | 0 | 0 | 0 | - | - | 0 | 0 |
| other | 35 | 4 | 4 | 1 | 2.0 | 60.0 | 1 | 11 |
| all | 111 | 36 | 36 | 12 | 2.0 | 173.5 | 10 | 16 |

Obstacle episodes by route (stop episodes with the scenario obstacle within 45 m ahead; durations in s, `->` = never left):

| route | seed | obstacle | episodes | durations | obstacle distance at start [m] |
|:--|--:|:--|--:|:--|:--|
| 19324 | 0 | Accident | 5 | 2, 2, 2, 2, 168-> | 31, 13, 11, 9, 7 |
| 19324 | 1 | Accident | 5 | 6, 2, 2, 1, 165-> | 32, 18, 13, 8, 7 |
| 19832 | 0 | ParkedObstacle | 3 | 2, 2, 174-> | 44, 20, 7 |
| 19832 | 1 | ParkedObstacle | 3 | 2, 2, 173-> | 44, 19, 8 |
| 24497 | 0 | ConstructionObstacle | 4 | 6, 2, 29, 141-> | 23, 10, 8, 4 |
| 24497 | 1 | ConstructionObstacle | 4 | 6, 2, 18, 151-> | 24, 10, 8, 4 |
| 2520 | 0 | ConstructionObstacle | 3 | 5, 5, 166-> | 28, 9, 3 |
| 2520 | 1 | ConstructionObstacle | 3 | 5, 5, 166-> | 28, 9, 3 |

How to read: 30 of the 111 stop episodes have the scenario obstacle ahead, but only 17 last 3 s or longer. The drive ego approaches an obstacle in stop-and-go steps (a 2 s stop, a creep at 2 to 4 m/s, the next stop), then sits behind it until the 200 s run limit. Red-light stops (25) and junction stops (21) dominate the rest; 35 stops have no cause in the logs (mostly 2 s stalls).

## 2. Time-based detector, per stop episode

Detector: ego below the speed bound v for T s, a lead reported ahead, and neither suppression. Lead: openpilot lead head (`plans.jsonl` fields `lead[0] = [x, y, v, a]` and `lp[0]`; reported when `lp[0] > 0.5` and `0 < x < 40 m`; `head_static` also needs the head's own speed estimate below 0.5 m/s; `vlm` uses the logged openjev Q_block answer being moving_lead or static_block). Light suppression: `truth` = `ctx.tl` red or yellow and `ctx.tl_dist` below X (privileged), `vlm` = latest logged openjev Q_light answer (at most 2.5 s old) is red_or_yellow_for_ego, `none`. Junction suppression N: route-map distance to the next junction entrance at most N m (or inside one), from the logged `junc_dist` (not privileged). Counts are raw; a false trigger is a stop episode that is not an obstacle stop in which the detector fires.

| stop bound | T [s] | light | N [m] | lead | obstacle episodes >= T fired | all obstacle episodes fired | obstacle runs hit | delay first obstacle stop to first trigger, median [s] | max [s] | false triggers: red / junction / lead / other | non-obstacle episodes >= T | false-trigger routes (count) |
|:--|--:|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| v<0.5 | 3 | none | off | head | 15 / 17 | 15 / 29 | 8 / 8 | 3.0 | 21.1 | 1 / 3 / 0 / 1 | 19 | 15612 1, 27043 1, 27870 1, 9196 2 |
| v<0.5 | 5 | none | off | head | 15 / 17 | 15 / 29 | 8 / 8 | 5.0 | 23.1 | 1 / 2 / 0 / 1 | 19 | 15612 1, 27870 1, 9196 2 |
| v<0.5 | 5 | truth | 15 | head | 15 / 17 | 15 / 29 | 8 / 8 | 5.0 | 23.1 | 0 / 0 / 0 / 1 | 19 | 9196 1 |
| v<0.5 | 5 | vlm | 15 | head | 15 / 17 | 15 / 29 | 8 / 8 | 5.0 | 23.1 | 0 / 0 / 0 / 1 | 19 | 9196 1 |
| v<0.5 | 5 | vlm | off | head | 15 / 17 | 15 / 29 | 8 / 8 | 5.0 | 23.1 | 0 / 1 / 0 / 1 | 19 | 15612 1, 9196 1 |
| v<0.5 | 5 | truth | off | head | 15 / 17 | 15 / 29 | 8 / 8 | 5.0 | 23.1 | 0 / 2 / 0 / 1 | 19 | 15612 1, 27870 1, 9196 1 |
| v<0.5 | 5 | vlm | 15 | head_static | 15 / 17 | 15 / 29 | 8 / 8 | 5.0 | 23.1 | 0 / 0 / 0 / 1 | 19 | 9196 1 |
| v<0.5 | 5 | vlm | 15 | vlm | 17 / 17 | 17 / 29 | 8 / 8 | 5.0 | 23.1 | 0 / 0 / 0 / 3 | 19 | 22535 1, 37969 1, 9196 1 |
| v<0.5 | 3 | vlm | 15 | head | 15 / 17 | 15 / 29 | 8 / 8 | 3.0 | 21.1 | 0 / 0 / 0 / 1 | 19 | 9196 1 |
| v<0.5 | 8 | vlm | 15 | head | 8 / 10 | 8 / 29 | 8 / 8 | 25.9 | 30.3 | 0 / 0 / 0 / 1 | 2 | 9196 1 |
| v<0.5 | 12 | vlm | 15 | head | 8 / 10 | 8 / 29 | 8 / 8 | 29.9 | 34.1 | 0 / 0 / 0 / 1 | 2 | 9196 1 |
| v<0.5 | 2 | vlm | 15 | head | 25 / 27 | 25 / 29 | 8 / 8 | 2.0 | 2.0 | 1 / 0 / 0 / 12 | 60 | 15483 3, 15612 2, 16529 2, 22535 3, 24944 1, 27297 1, 9196 1 |
| v<0.3 | 5 | vlm | 15 | head | 15 / 17 | 15 / 30 | 8 / 8 | 5.0 | 25.4 | 0 / 0 / 0 / 1 | 19 | 9196 1 |
| v<1.0 | 5 | vlm | 15 | head | 15 / 17 | 15 / 28 | 8 / 8 | 5.0 | 22.7 | 0 / 1 / 0 / 1 | 19 | 22535 1, 9196 1 |
| v<0.5 | 5 | truth | 30 | head | 15 / 17 | 15 / 29 | 8 / 8 | 5.0 | 23.1 | 0 / 0 / 0 / 1 | 19 | 9196 1 |

How to read: with T of 3 to 5 s every obstacle stop that lasts that long fires (15 of 17 at 5 s) in all 8 obstacle runs, and the false triggers are few because the lead head rarely reports a lead at a red light or in a junction; the suppression terms remove the rest (red 1 -> 0, junction 2 -> 0 at T = 5 s, N = 15 m). Ground-truth light and the logged openjev light answer give the same counts here. The delay from the first obstacle stop is the T itself when the first stop is a long one (5 of 8 runs) and 20 to 25 s when the ego first takes 2 s stalls on the way in. The trigger cannot fire inside a 2 s stall: 13 of the 30 obstacle episodes (v < 0.3) are shorter than 3 s by construction. The one false trigger left at T = 5 s with both suppressions is 9196 seed 1: the ego stands for 60 s right after its two official collisions with the fire truck (a collision-induced standstill, so arguably a real blocker, but not a scenario obstacle). Taking the lead from the VLM answer instead of the lead head fires on 17 of 17 but adds false triggers (3 at T = 5 s).

Non-dominated settings over the full grid (540 settings, `bypass_timedet_grid.csv`) among those that hit all 8 obstacle runs, ordered by false triggers, then delay:

| stop bound | T [s] | N [m] | lead | obstacle episodes >= T fired | median delay [s] | max [s] | false triggers | light settings with the same result |
|:--|--:|--:|:--|--:|--:|--:|--:|:--|
| v<0.3 | 3 | 30 | head | 15 / 17 | 3.0 | 23.4 | 1 | none, truth, vlm |
| v<0.3 | 3 | 30 | head_static | 15 / 17 | 3.0 | 23.4 | 1 | none, truth, vlm |
| v<0.3 | 3 | 15 | head | 15 / 17 | 3.0 | 23.4 | 1 | truth, vlm |
| v<0.3 | 3 | 15 | head_static | 15 / 17 | 3.0 | 23.4 | 1 | truth, vlm |
| v<0.5 | 3 | 30 | head | 15 / 17 | 3.0 | 21.1 | 1 | none, truth, vlm |
| v<0.5 | 3 | 30 | head_static | 15 / 17 | 3.0 | 21.1 | 1 | none, truth, vlm |
| v<0.5 | 3 | 15 | head | 15 / 17 | 3.0 | 21.1 | 1 | truth, vlm |
| v<0.5 | 3 | 15 | head_static | 15 / 17 | 3.0 | 21.1 | 1 | truth, vlm |
| v<1.0 | 3 | 30 | head_static | 15 / 17 | 3.0 | 20.7 | 1 | none, truth, vlm |
| v<1.0 | 3 | 15 | head_static | 15 / 17 | 3.0 | 20.7 | 1 | truth, vlm |

Creep-tolerant variant (mean ego speed over the last T s below a bound instead of a continuous stop; at most one fire per 10 s; a fire is false when no obstacle is within 45 m ahead; scored per fire, not per episode):

| rule | T [s] | light | N [m] | lead | obstacle runs hit | median delay [s] | max [s] | false fires | by cause |
|:--|--:|:--|--:|:--|--:|--:|--:|--:|:--|
| mean v < 0.5 | 5 | vlm | 15 | head_static | 8 / 8 | 4.0 | 21.1 | 6 | other 6 |
| mean v < 1.0 | 5 | vlm | 15 | head_static | 8 / 8 | 3.4 | 20.1 | 8 | other 7, red_light 1 |
| mean v < 1.5 | 5 | vlm | 15 | head_static | 8 / 8 | 2.9 | 14.6 | 10 | other 8, red_light 2 |
| mean v < 1.0 | 8 | vlm | 15 | head_static | 8 / 8 | 17.0 | 22.1 | 10 | other 8, red_light 2 |
| mean v < 1.0 | 5 | vlm | 15 | head | 8 / 8 | 3.4 | 20.1 | 11 | other 10, red_light 1 |

How to read: averaging over the creep removes the 20 s delays only by loosening the stop test, and then slow following of a moving lead fires (cause `other`); the strict per-episode detector is the better trade on these logs.

## 3. What the single-frame model answers once the ego is stopped behind the obstacle

Logged openjev answers (the only model in these logs), frame time inside an obstacle stop (ego v < 0.3 m/s, scenario obstacle within 45 m ahead). Truth for Q_block is static_block; truth for Q_side is left_free on all four routes (the path offset is negative, which is the left lane in CARLA's left-handed frame, and the new-format logs say left_free; the old-format `gt_side` of the seed-0 logs says right_free, an inconsistent label that was not used). Answers in one stopped scene are near-duplicates, so the number of runs is the effective sample.

| obstacle | distance [m] | answers | routes | Q_block static_block | moving_lead | clear | Q_side left_free (correct) | none_free | right_free | left_free among static_block answers |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ConstructionObstacle | 0-5 | 1240 | 2 | 1224 (99%) | 16 | 0 | 163 (13%) | 1076 | 1 | 162 / 1224 |
| ConstructionObstacle | 5-10 | 109 | 2 | 108 (99%) | 1 | 0 | 8 (7%) | 100 | 1 | 8 / 108 |
| ConstructionObstacle | 10-20 | 7 | 1 | 7 (100%) | 0 | 0 | 0 (0%) | 7 | 0 | 0 / 7 |
| ConstructionObstacle | 20-45 | 46 | 2 | 17 (37%) | 29 | 0 | 0 (0%) | 46 | 0 | 0 / 17 |
| Accident | 5-10 | 665 | 1 | 0 (0%) | 665 | 0 | 0 (0%) | 665 | 0 | 0 / 0 |
| Accident | 10-20 | 17 | 1 | 0 (0%) | 17 | 0 | 0 (0%) | 17 | 0 | 0 / 0 |
| Accident | 20-45 | 16 | 1 | 0 (0%) | 16 | 0 | 0 (0%) | 16 | 0 | 0 / 0 |
| ParkedObstacle | 5-10 | 686 | 1 | 0 (0%) | 686 | 0 | 0 (0%) | 686 | 0 | 0 / 0 |
| ParkedObstacle | 10-20 | 4 | 1 | 0 (0%) | 4 | 0 | 0 (0%) | 4 | 0 | 0 / 0 |
| ParkedObstacle | 20-45 | 13 | 1 | 0 (0%) | 13 | 0 | 0 (0%) | 13 | 0 | 0 / 0 |

Q_block during other stops (ego v < 0.3 m/s inside a stop episode of at least 2 s without the obstacle):

| stop cause | answers | static_block | moving_lead | clear |
|:--|--:|--:|--:|--:|
| red_light | 133 | 4 (3%) | 66 | 63 |
| junction | 195 | 0 (0%) | 148 | 47 |
| other | 204 | 0 (0%) | 162 | 42 |

While the ego is still moving towards the obstacle (v >= 0.3 m/s, obstacle within 45 m): Accident 68 answers, static_block 0; ConstructionObstacle 130 answers, static_block 84; ParkedObstacle 59 answers, static_block 0.

How to read: for the cone-and-barrier construction zones the model names the obstacle (static_block) in 99% of the stopped frames within 10 m and in 37% of those 20 to 45 m away; for the accident vehicles and the parked vehicle it answers moving_lead in 100% of the stopped frames at every distance, for stops of up to about 170 s. The free side is almost never named: Q_side is none_free in 87% to 100% of the construction frames, left_free (correct) in 13% at 0 to 5 m, 7% at 5 to 10 m and never beyond, and none_free for every vehicle-obstacle frame. So a stopped ego does not make the single frame readable; the time-based trigger has to supply the static decision, and the side has to come from elsewhere (inferred: these are openjev answers; Qwen3-VL-4B was not run in closed loop).
