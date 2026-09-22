# Reusable driving runtime

Read this when connecting another driving model or reusing the frame/preview helpers outside this repo.

## Keep three layers separate

```text
Bench2Drive harness: b2d_run.py / b2d_route.py / b2d_report.py
    server lifecycle, route isolation, recovery, timing, reports
Reusable helpers: scripts/drive_runtime/
    sensors.py  exact-frame collection, caller-supplied preprocessing, optional display feeds
    preview.py  RGB shared memory + JSON telemetry; no model or simulator connection
TCP adapter: b2d_tcp_visual_agent.py / b2d_tcp_preprocess.py
    TCP camera names, JPEG/crop/resize, weights, controls, prediction coordinates
```

`drive_runtime.sensors` uses only the standard library. `drive_runtime.preview` additionally
uses NumPy and Linux `/dev/shm`. Both support Python 3.8+. Neither imports CARLA, Torch,
Bench2Drive or Pygame. Do not install the root training environment just to use these helpers.
Put this repo's `scripts` directory on `PYTHONPATH`, or vendor the small `drive_runtime` folder.

```python
from concurrent.futures import ThreadPoolExecutor
from drive_runtime.sensors import TimedSensorQueue, collect_frame

with ThreadPoolExecutor(max_workers=3) as workers:
    queue = TimedSensorQueue(
        workers,
        transforms={"front": lambda value: value * 2},  # replace with model preprocessing
        combine=lambda futures: futures["front"].result(),
        optional_tags=("display",),
    )
    queue.begin(12)
    queue.put(("front", 12, 3))  # in production, called by your existing sensor callback
    inputs = collect_frame(queue, ["front"], 12, timeout=1)
    model_input = queue.combined_future.result()  # 6; never a different frame
    display = queue.optional_frame("display", 12)  # None is allowed
```

The adapter owns the executor and awaits results before issuing control. Worker exceptions
propagate. Required inputs must all have the requested frame; optional display sensors may lag
and are not preprocessing dependencies. The optional path is not for dropping model inputs.
With no optional sensors, our TCP adapter still uses the original official collector.

For preview, call `LivePreview(directory).publish(state, images)`, where `state` is JSON-compatible
and `images` maps arbitrary names to uint8 RGB arrays. Call `close()` at shutdown. `PreviewReader`
reads the newest complete packet; slow viewers do not hold up control. This transport is generic;
`b2d_viewer.py` is the supplied TCP-oriented frontend, not a universal six-camera dashboard.

## Connecting a new model

1. Implement the official agent's `setup`, `sensors`, `run_step` and `destroy` contract.
2. Keep model-specific preprocessing and outputs in that adapter. Supply transforms/combine
   callbacks to the generic queue if overlap is useful; do not copy TCP's JPEG/cropping blindly.
3. Publish display data explicitly. For lagging images, match trajectory metadata by frame ID;
   omit the overlay when no matching prediction exists.
4. Pass `--agent`, `--agent-config` and `--python` to the existing harness. Model weights alone
   are not enough: input conventions and the control/output adapter must match.

## Two independent switches

- `B2D_TCP_OPTIMIZE=1`: same-frame CPU overlap/copy optimization, unchanged model input pixels.
- `B2D_ASYNC_DISPLAY=1`: experimental removal of TCP's existing `bev` debug camera from the
  control barrier. Model sensors remain exact-frame; debug display may lag. Default is 0.

The second changes **display synchronization**, not renderer quality or driving camera settings.
TCP never feeds `bev` to its network. Its legacy tick still receives a display image for compatibility;
before the first debug frame, that display-only image is blank. The wrapper retains eight frames
of prediction metadata and the viewer refuses mismatched image/trajectory overlays.

Neither switch claims formal online Leaderboard acceptance. Keep benchmark protocol settings
fixed for comparable scores. [Leaderboard 2.0's startup example](https://leaderboard.carla.org/get_started_v2_0/)
uses Epic; these experiments keep Epic, all four cameras, original resolutions and timestep.

Profiling evidence: `/data/runs/b2d/reuse-opt/` (Town12 route 1711, two strict/async pairs,
350 ticks each, drop 20 warmup ticks). These capped runs are not route-completion scores.

| Mode | Repeat 1 ms/tick | Repeat 2 ms/tick |
|---|---:|---:|
| Strict synchronized display | 86.61 | 86.76 |
| Optional asynchronous display | 84.76 | 84.35 |

Approximately 2.5% higher throughput in this local comparison. Sensor waiting dropped from
~61 ms to ~47.6 ms, but ~9 ms of preprocessing became exposed instead of hidden under the
display-camera wait. This is not evidence that all camera waiting is Python overhead.
The modest additional gain is why asynchronous display remains opt-in. All four capped runs
ended normally with no restarts; 18 tests and 35 full preprocessing equality checks passed.
External stdlib-only execution of the frame helper and simulator/model-free import of preview
were verified separately. Testing servers and the viewer were stopped after measurement.

Last verified: 2026-09-22
