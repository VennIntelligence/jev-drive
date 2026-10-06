# b2d_collect: smoke and 10-route stage (2026-10-06)

Data: `$DATA_DIR/runs/b2d_collect/data/{smoke,ten}` (route set v1, ids 900000+). Lane: `scripts/b2dc_lane.py`; gates in its docstring.

## What was fixed on the way (each a real failure, not a guess)

| Stage | Failure | Fix |
|---|---|---|
| smoke | leaderboard rejects the chase camera (3 m mounting radius) | chase view spawned as a plain CARLA camera on the hero, outside the agent's sensor list |
| smoke | the frame-lag ADE test and a clip-wide picture-motion correlation peak off 0 | both confounded by scene motion and the model's speed bias; time alignment is now checked structurally (camera frame id = state frame id, every tick) and empirically at the launch from the spawn (ground starts moving in the picture within 1 tick of the log) |
| smoke | shipped Cinque / P2 plan straight through the left turn | the model's open problem (decision 121), not a data error: reported, not gated; the gate uses heading sign on lane following |
| ten | route 900239 planned 111 m, driven 3431 m | keypoints inside junctions let the leaderboard's planner snap to another connector; route set v2 has no keypoints inside junctions and keeps only routes whose re-planned path matches (122 of the drawn candidates rejected) |
| ten | routes 900239 / 900524 sat 350-620 s waiting | agent ends the route after 45 s standing or 240 s of simulation (`timing.stop` in meta.json) |
| ten | route commands: a 108 deg T-junction turn labelled STRAIGHT, a 148 deg left turn RIGHT | the agent plan's RoadOptions are unreliable in junctions; turns are labelled from geometry (heading 5 m after the junction minus 5 m before, > 30 deg) with junction ids from the map |
| ten | check OOM on a 12 k-tick clip | samples per clip capped, encoded in chunks |

## Gates and readings

| Reading | smoke (1 clip) | ten (10 clips) |
|---|---:|---:|
| completed / routes | 1 / 1 | 10 / 10 |
| picture index = tick, consecutive frames, dt error | yes, yes, 7e-10 s | yes, yes, 7e-10 s |
| camera frame id = state frame id (all ticks) | yes | yes |
| launch from spawn: picture lag within 1 tick | 1 / 1 (lag 0) | 3 / 3 |
| logged future footprint inside drivable SDF (>= -0.3 m) | 0.993 | 0.939 (ParkingExit 0.47: starts off the lane graph) |
| turn-command precision (driven turn of that side follows) | 1.00 | 0.91 |
| heading sign on lane following, P0 / P2 | 1.0 / 1.0 (2 samples) | 0.91 / 0.89 (44) |
| heading sign at junction turns, P0 / P2 (report) | 0.67 / 0.89 (9) | 0.77 / 0.88 (48) |
| 4 s path length ratio, P0 / P2 (report) | 0.38 / 0.75 | 0.82 / 0.86 |
| expert DS (PDM-Lite) | 100 | mean 90.6, 2 collisions |
| body pitch p95 / road-frame horizon shift p95 | 0.89 deg / 14 rows | 3.8 deg / 61 rows (hilly Town15 / Town11) |

## Throughput and storage (measured)

- 1 worker (card shared with HUGSIM jobs): real-time factor 0.41; agent per tick: PDM-Lite 21 ms, frame packing 7.5 ms, logging 5 ms,
  H.264 pipe 0.8 ms; ~50 s per route for map load and setup; 7.2 GB VRAM, 2.2 cores.
- 5 workers in one job: 31.6 GB VRAM (6.3 per worker), 11.4 cores (2.3 per worker); in-route RTF 0.33-0.64 per worker; normal routes
  (14-25 s simulated) took 32-137 s wall each.
- Storage: lossless model frames 88 KB per tick (512 x 512 yuv420p, H.264 qp 0, 4.5x below raw); 1000 clips of ~500 ticks ~ 45-60 GB,
  labels + SDF + actors a few GB. Free: 587 GB.
- Full run (v2, 1000 routes): 3 jobs x 6 workers (one per card), ~650 routes / h at steady state -> ~1.5-2 h.

GIFs (chase view | road | wide model frames as Cinque gets them; dashed = nominal horizon row, green = expert's logged 4 s future,
red = P2 plan): [smoke 900025](../figs/smoke_900025.gif) (fog, blocked intersection, left), [900467](../figs/ten_900467.gif) (night,
signalised right turn), [900431](../figs/ten_900431.gif) (night, T junction, stop sign).
