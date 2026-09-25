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

Items 5 and 7 are about the interface between a model and the benchmark, not only about bugs: openpilot, for
example, does not start from a full stop without a driver and has no navigation input. Such gaps are resolved by a
documented rule before the scored run, and the score is reported with that rule's name.

Last verified: 2026-09-25
