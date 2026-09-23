# Captured compass failure: same-input motion replay

The new bounded compass prediction reproduces the entire finite prefix exactly, then returns a finite degraded estimate at the recorded failure frame. This is a deterministic open-loop replay of existing sensor inputs, not a new simulator run or proof that the route will subsequently finish.

| Check against recorded output | Frozen a1b50ed adapter | Candidate adapter |
|---|---:|---:|
| Compared estimated poses and controls | 181 | 181 |
| Maximum absolute pose-coordinate / yaw difference | 0 | 0 |
| Maximum absolute throttle / steer / brake difference | 0 | 0 |
| Compared generated 20×2 trajectories | 46 | 46 |
| Maximum absolute trajectory-coordinate difference, m | 0 | 0 |
| Final frame 502 | `Nonfinite motion sensor` exception | finite pose and control |

The recorded controller is retained in both replays; only the adapter changes. Route projection, replanning every four sensor ticks, signed-speed normalization, trajectory generation, and controller step timing are reproduced. The route's saved GPS/world pairs reconstruct its projector and the saved settings reconstruct its rear-axle pose filter. Recorded actor truth is not used for state estimation, trajectory generation, controls, or comparison: only recorded estimated pose/control/trajectory fields are compared.

## Exact capture boundary

The captured attempt is `/data/runs/b2d/controller/smoke-v4-capture/pursuit-seed0/attempts/2390/1`. Its campaign manifest names commit `a1b50ed4b385b338a134a51b2f23b806adbc0c0a`; the frozen replay imports source bytes from that campaign's provenance snapshot, not from git history or the current runtime.

- Motion records: 182; recorded control records: 181.
- Frames: 321 through 502, consecutive, with all three motion sensor frame IDs matching each row.
- Simulation timestamps: 0.05000000074505806 through 9.100000135600567 s; first-to-last span 9.050000134855509 s.
- The first 181 sensor records are entirely finite.
- The only nonfinite field in the entire capture is final-row `sensors.IMU.data[6]`, the compass, encoded as the string `"nan"`. GPS latitude, longitude and altitude; all other six IMU values; and speed remain finite.
- The last finite compass was frame 501 at 9.050000134855509 s. At the failing observation, its age is 0.05000000074505806 s.

**The actual dropout duration is unknown.** The capture ends on its first nonfinite compass observation; there are no recorded subsequent bad or recovered compass samples. It is inaccurate to call this a proven one-tick dropout or to infer the root cause of the NaN from this trace alone.

At frame 502 the candidate reports `compass_dropout_prediction`, `degraded=true`, `compass_valid=false`, and the 50 ms age above. The finite world rear-axle pose is `[2385.5188681267064, 2334.529240067929]` m, yaw `1.571098431778414` rad. Control is throttle `0.18336438047442183`, CARLA steer `0.005286244687455283`, brake `0`. The previous trajectory is 50 ms old; normal four-tick replanning is preserved. This row has no recorded control counterpart because the original process failed before producing it.

The candidate's 0.2 s prediction limit, prolonged-dropout brake/reset integration and fresh-sensor recovery are tested separately by the runtime owners. This captured-data helper does not synthesize future sensor records or claim to validate closed-loop recovery. No gain, trajectory-speed, steering or official scoring change is made here.

## Preserved files and verification

[artifacts-v2/summary.json](artifacts-v2/summary.json) is the current replay result. `artifacts-v1` is preserved as the initial edition; v2 snapshots the final adapter after its owner removed an unsupported cause hypothesis from a comment. The numerical replay results are identical. Neither edition was overwritten.

Each edition contains:

- `inputs/`: exact captured motion, controls, trajectories, route reference, agent configuration and archived controller configuration.
- `source/`: the replay helper, frozen adapter, frozen controller, and exact candidate adapter bytes used for that edition.
- `frozen_a1b50ed.jsonl` and `candidate_compass_prediction.jsonl`: every replayed frame, generated trajectories, diagnostics, comparisons and the final exception/output.
- The per-frame `*.jsonl` files (both outputs above and `inputs/{control,motion,trajectories}.jsonl`) are moved out of git; GPU box: `$DATA_DIR/runs/b2d/controller/git-offload-v1/todos/2026-09-22-b2d-controller/results/smoke-v4-motion-replay/artifacts-v{1,2}/`.
- `failed-run-inventories.json`: every file's path, size and SHA256 in both stopped failed campaigns, including the earlier `/data/runs/b2d/controller/smoke-v4` run. The earlier run has no captured motion input and is indexed, not replayed.
- `manifest.json`: original-to-archived input/source identities, output sizes/hashes and Python/NumPy versions.

Before indexing, both campaign event streams had final `end` records. All four recorded server/route PIDs were absent from `/proc` (190809, 191033, 193276, 193522). The helper refuses indexing if a recorded PID still exists or a campaign lacks an end event; it also checks each file remains unchanged during hashing. An outer campaign `completed` event denotes harness termination, not successful route completion.

## Reproduce

Python 3.8 and NumPy 1.23.5 are available in `/data/envs/carla`. No simulator process, git command or network access is used.

From `/data/worktrees/jev-drive-controller-v2`, use the preserved candidate source to reproduce the same comparison:

```bash
/data/envs/carla/bin/python \
  todos/2026-09-22-b2d-controller/results/smoke-v4-motion-replay/replay.py \
  --capture /data/runs/b2d/controller/smoke-v4-capture \
  --failed /data/runs/b2d/controller/smoke-v4 \
  --adapter-source todos/2026-09-22-b2d-controller/results/smoke-v4-motion-replay/artifacts-v2/source/candidate_adapter.py \
  --out /data/runs/b2d/controller/smoke-v4-motion-replay-reproduction-v1
```

The output must not already exist. Nonfinite input strings are decoded into numeric NaN solely to replay the original observation; output JSON uses `allow_nan=False`. Per-frame output and exact maximum errors make discrepancies independently inspectable. This evidence changes no runtime source, configuration, official result, or acceptance gate.
