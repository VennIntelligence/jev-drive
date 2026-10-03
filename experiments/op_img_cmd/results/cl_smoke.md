# op_img_cmd closed-loop smoke (sky arrow fine-tune vs shipped drive), diagnostic run, 2026-10-04

Pre-registration: addendum "closed-loop smoke as a diagnostic" in [../plans/2026-10-04-img-cmd-ft2-prereg.md](../plans/2026-10-04-img-cmd-ft2-prereg.md).
The open-loop gate was not passed (NA failed the missing-exit control on dev), so this is a diagnostic, not a pass. Main's reasons: the
failing control is an arrow to a non-existent exit, which cannot occur when the official route command is always feasible, and decision
103 puts the command lever mainly at B2D junction turns.

Setup: `q3NA-s0` (primary) and `q3SA-s0` (secondary), each vs shipped `drive` (route desire on). Routes 27297, 27043, 9196, 24944 (the four
turn_agree-failure routes), 2 traffic seeds, so n = 8 runs per arm. The sky arrow is drawn by the server on every frame from the official
route only (next LEFT / RIGHT / STRAIGHT command and distance to it; no map). Turn agreement uses the op_l_b2d_report definition
(LEFT / RIGHT command within 15 m, moving, plan-to-route divergence <= 1.0 m), pooled over turn-window steps.
Lane: `scripts/img_cl_lane.py`, readout `scripts/img_cl_report.py`, per-run rows in [cl_smoke.csv](cl_smoke.csv).

**Pre-registered line:** turn_agree >= drive + 0.10 (pooled) and DS not more than 5 points below drive (descriptive, n too small for significance).

## Result

| arm | runs | DS mean | RC | turn_agree (pooled) | vs drive | turns ok (runs) | veh collisions | red lights | blocked |
|:--|--:|--:|--:|--:|--:|:--|--:|--:|--:|
| drive | 8 | 60.1 | 94.8 | 0.440 (583/1325) | - | 6 / 8 | 6 | 3 | 1 |
| skyNA (primary) | 8 | 69.6 | 100 | 0.487 (714/1467) | **+0.047** | 7 / 8 | 4 | 3 | 0 |
| skySA (secondary) | 8 | 80.2 | 94.8 | 0.526 (767/1459) | **+0.086** | 7 / 8 | 3 | 0 | 1 |

Per route and seed (DS; turn-agree steps / turn-window steps):

| route, seed | drive | skyNA | skySA |
|:--|:--|:--|:--|
| 24944, 0 | 70; 119/189 | 70; 123/241 | 100; 152/232 |
| 24944, 1 | 100; 140/251 | 100; 101/240 | 100; 136/219 |
| 27043, 0 | 60; 67/138 | 60; 70/160 | 60; 78/159 |
| 27043, 1 | 60; 63/146 | 60; 66/162 | 60; 65/152 |
| 27297, 0 | 70; 79/167 | 70; 69/145 | 100; 84/153 |
| 27297, 1 | 70; 31/141 | 100; 66/150 | 100; 90/174 |
| 9196, 0 | 19.6; 20/149 | 42.0; 121/194 | 21.2; 72/180 |
| 9196, 1 | 31.3; 64/144 | 54.6; 98/175 | 100; 90/190 |

## Reading

- **Line not met on turn agreement:** NA +0.047 and SA +0.086, both under +0.10. DS is not lower than drive for either (higher, +9.5 / +20.1), so
  the second condition holds; the first fails, so there is no signal by the pre-registered line (SA is close).
- **The gain is concentrated in route 9196.** There turn_agree goes 0.29 (drive, 84/293) to 0.59 (NA, 219/369) and 0.44 (SA, 162/370).
  Without 9196 the other three routes read drive 0.484, NA 0.451, SA 0.556 (NA below drive, SA +0.07). 9196 is also where DS differs
  most (drive 19.6 / 31.3). With n = 2 seeds per route this is a single-route effect, not a general one.
- **No obvious harm from the arrow:** no extra red-light or timeout infractions; the two sky arms have fewer vehicle collisions (4 and 3 vs 6),
  but with 8 runs this is within noise.
- The missing-exit failure that stopped the open-loop gate did not show up here (the route command is always a real exit), as argued in the
  addendum. This run does not test it; it only shows that the model with the true command is no worse than drive.
- Not verified: a no-arrow control of the fine-tuned model (to separate the arrow from the fine-tune itself), more routes, more seeds.

## Arrow drawing check

Stage 1 (27043, seed 0) and the full runs dump every 200th overlaid frame (`$DATA_DIR/runs/op_img_cmd/cl/dump/<arm>/`). Review sheet:
[../figs/cl_smoke_sheet.png](../figs/cl_smoke_sheet.png) (road view on top, wide view below; skyNA right turn at 13.6 m, skyNA left at 22 m at
night, skyNA straight at 60 m, skySA left at 13.2 m). Look at: the arrow shape matches the route command, it sits above the horizon in the
road view rows 1-29 and top-left in the wide view, it does not cover vehicles, and it is large when the junction is close and small at far
distance (straight = small arrow). All four are drawn correctly. Stage 1 also dumped right-turn frames (17 m, 8 m) as expected.

## Run notes

Card 1, lease `img-cl`. First stage-1 attempt lost a CARLA server (rc 139) and was retried by the lane (attempt 2 finished, 561 ticks). All 24 routes
of the full run finished (6 units, 4 routes each), 12:51 to 13:40 JST. ONNX built with `op_l_onnx.py build --no-adapter` in `envs/op-train`.
