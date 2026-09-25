# Adapter acceptance checklist for external driving models in CARLA

Read this when you are about to score a model we did not train (openpilot, Alpamayo, UniAD, VAD, ...) in closed
loop, smoke or full; run the checks before the scored run. Every item is an adapter check, not a model check: it must pass, or its failure must be explained by the
model's own output, before a Driving Score is reported. The openpilot exam lost a whole smoke round to two adapter
bugs and a standstill-start gap that this list would have caught
([openpilot-migration.md](../todos/2026-09-24-zeroshot-exam/openpilot-migration.md), sections A and D1).

Run the checks on one or two short routes with frame dumps on (`dump_every`), in shadow mode where it says so
(`"drive": "oracle"` in `scripts/b2d_zeroshot_agent.py`: the route oracle drives, the model only plans), and write
the outcome into the run's todo.

| # | Check | How | Pass |
|---|---|---|---|
| 1 | Coordinates and heading | Log the model's trajectory next to the truth track in shadow mode (`scripts/zeroshot_b2d_openpilot_diag.py`); project the plan into the model's input image | 1 s / 2 s errors of the order of the model's open-loop error on real data; left/right, forward sign and yaw sign agree; the plan lies on the road in the image |
| 2 | Reference point | Feed a zero-motion ("stay") output through the controller at standstill | The controller holds the brake. A trajectory whose t = 0 point is not the controller's reference point (camera vs rear axle) shows up as throttle while the model says stop |
| 3 | Sensor timing | Log per-camera frame numbers per planning step | Every camera of a step comes from the same simulator frame; context spacing is exactly what the model was trained with (no sensor_tick jitter) |
| 4 | Warm-up | Plot the first second of outputs | Recurrent / temporal models have filled their history before their output reaches the controller |
| 5 | Start from standstill | Closed loop, empty road ahead | The ego leaves the start within ~10 s; if the model never asks to move, decide and document the engagement rule (e.g. engage while rolling) before scoring |
| 6 | Lane keeping | Closed loop on a straight and a curved segment | No lane invasion over 200 m |
| 7 | Junction turns, left and right | Closed loop on one left- and one right-turn junction route | The ego follows the route through both; if the model has no route input, the turn-guidance rule (desire, controller blending, handover) is chosen and documented before scoring |
| 8 | Red light and stop sign | Closed loop on a signalized and a stop-sign route | The ego stops; note whether the model or a rule does it |
| 9 | Resume after stop | Same routes | The ego resumes after the light turns green / the stop is served |
| 10 | Controller identity | Diff the controller source against the shipped one | Verbatim, or the deviation is written into the pre-registration |

## Routes and tooling

A route set that covers the list in five Bench2Drive routes, none of them in the openpilot or Alpamayo smokes:
2086 (NonSignalizedJunctionLeftTurn), 2903 (NonSignalizedJunctionRightTurn), 3144 (VanillaSignalizedTurnEncounterRedLight),
2416 (VanillaNonSignalizedTurnEncounterStopsign), 3540 (HardBreakRoute: the lead brakes hard, then the ego resumes).
Lane keeping, plan frame and heading are checked on the stretches between the events.

For openpilot, `scripts/zeroshot_b2d_op.sh accept <gpu>` runs the set and `scripts/zeroshot_b2d_op_accept.py` scores
items 1-10 automatically from `plans.jsonl`, `ticks.jsonl` and the official `results.json` (thresholds in its
docstring, e.g. items 2, 5, 9: throttle share <= 5 % while a stay plan is in force, move within 20 s, no stop >= 45 s).
The same logs exist for every model run through `scripts/b2d_zeroshot_agent.py`, so the script is the template for
the next model.

## Gaps that need a rule, not a fix

Items 5 and 7 are about the interface between a model and the benchmark, not only about bugs: openpilot, for
example, does not start from a full stop without a driver and has no navigation input. Such gaps are resolved by a
documented rule before the scored run, and the score is reported with that rule's name.

The rule used for openpilot is shared control with a learned partner (`scripts/b2d_partner.py`): the official
Bench2DriveZoo TCP agent runs every tick and drives only while the ego is at standstill (latched after 0.5 s below
0.1 m/s, released after 1 s at or above 1 m/s) or inside a junction-turn zone (15 m before to 5 m after a LEFT / RIGHT
route command), the model drives otherwise, and a change of driver is blended over 0.5 s. The handover reads only
observable state (speed, time, the route the benchmark hands every agent and the ego's progress along it), never an
outcome. The driver is logged per tick; report the distance and time share of each driver and which driver held the
car at each infraction. A privileged route oracle is not a partner: it knows the map, so it is not used to start or
turn a scored model.

Last verified: 2026-09-25
