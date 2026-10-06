# b2d_collect: full collection (route set v2, 2026-10-07)

Data: `$DATA_DIR/runs/b2d_collect/data/all/` (attempts/<route>/<k>/clip/: frames.mp4, chase.mp4, ego / route / actors / lights, labels.npz,
sdf.npz); `index.csv` (one row per clip: type, town, turn side, leaderboard result and infractions, ticks, RTF), `check/` (validation).
Split `b2d/b2dc-train@v2` (1000 routes). Lane gates (`scripts/b2dc_lane.py`): all pass.

| | |
|---|---:|
| clips (complete) | 998 of 1000 (b2d_run lists 7 routes as never finished; 6 of them still left a complete clip, 920798 none) |
| ticks (20 Hz) / simulated hours / moving hours | 902 178 / 12.5 / 6.7 |
| ticks with a full 4 s future | 822 338 |
| clip length: median / mean / p90 ticks | 381 / 904 / 3570 (cap 4800 = 240 s; long LB scenarios and stops at lights) |
| junction-turn clips: left / right / none | 301 / 296 / 401 |
| towns | Town12 537, Town13 243, Town11 69, Town15 27, Town07 21, Town03 19, Town04 / 05 / 06 18 each, Town01 12, Town02 9, Town10HD 7 |
| early end: stuck 45 s / max 240 s | reported per clip in meta.json `timing.stop` |
| PDM-Lite DS (mean) / clips at 100 / clips with a collision / with a red light | 95.6 / 87.4 % / 33 / 5 |
| model frames on disk | 102 GB (113 KB per tick, lossless); data dir 101 GB in all |
| wall | collect 3.3 h (00:04-03:22, 3 jobs x 6 workers, one per card; median 0.33 s per tick, 48 server restarts); labels 2 min; check 68 min |

Validation (`check/check.json`, 20 948 replay samples; gates in bold):

| Reading | Value |
|---|---:|
| **picture index = tick, consecutive frames, camera frame = state frame (every tick of every clip)** | yes / yes / yes |
| **launch from the spawn visible in the picture within 1 tick** (598 clips with one) | 0.967 (lag 0: 0.831) |
| **logged future footprint inside the drivable SDF (>= -0.3 m)** | 0.988 |
| **turn commands followed by a driven turn of that side** | 0.938 |
| **heading sign on lane following, shipped Cinque (830 samples)** | 0.824 (P2 0.845) |
| heading sign at junction turns (2 621 samples), P0 / P2 | 0.69 / 0.86 |
| ADE 8 poses to the log, P0 / P2; 4 s path-length ratio, P0 / P2 | 4.67 / 4.51 m; 0.82 / 0.85 |
| straight command before a > 45 deg heading change within 60 m (report; includes curved roads and the 30-60 m before a turn) | 0.33 |
| body pitch p95 (max over clips) / road-frame horizon shift | 11.8 deg / 191 rows (Town11 / Town15 hills: the camera is fixed to the body) |

Reading: the stored frames, states and labels line up (structurally on every tick, and in the picture at launches); the footprint and the
command labels agree with what the expert drove. Shipped Cinque reads 0.69 of the junction-turn heading signs, P2 0.86. That is the model's
open problem (decision 121), not a data error. ADE above 4 m comes from stop-and-go starts the models cannot time, not from misalignment.

GIFs (chase view | road | wide model frames; dashed = nominal horizon row, green = logged 4 s future, red = P2 plan; Chinese captions):
[920025](../figs/all_920025.gif) (night, Town12, blocked intersection, left), [920024](../figs/all_920024.gif) (fog, Town15, right).
Smoke and 10-route stage: [stages.md](stages.md).
