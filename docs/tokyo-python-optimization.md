# TCP: Python-only optimization on tokyo-devbox

The reusable helper extraction and subsequent optional-display experiment are documented in
[driving-runtime.md](driving-runtime.md). TCP-specific preprocessing stays in the adapter.

2026-09-22. No CARLA renderer, vendor checkout, model weights, policy camera settings,
control rules, or simulation timestep changes. The visible chase camera and Pygame
dashboard remain enabled. This is short-run profiling, not a driving-score evaluation.

## Implementation

Enable `B2D_TCP_OPTIMIZE=1` with `scripts/b2d_tcp_visual_agent.py`. The following individual
environment flags override the umbrella switch; set all to 0 for the previous baseline:

| Flag | Change |
|---|---|
| `B2D_TCP_PIPELINE` | Start each camera's JPEG conversion as its requested frame is dequeued |
| `B2D_TCP_FAST_COLOR` | Direct BGRA-to-RGB conversion, avoiding implicit strided-input copy |
| `B2D_TCP_DEBUG_VIEWS` | Keep debug RGB arrays as views instead of converting/copying again |
| `B2D_TCP_EARLY_RGB` | Assemble/resize model RGB in the worker pool while remaining sensors arrive |

`TimedSensorQueue` observes the original `SensorInterface.get_data` loop. Its exact-frame
filter and requirement to wait for all sensors remain untouched. Stale/future frames are
not submitted for processing. The final RGB future is awaited before official TCP inference;
there is no old-image reuse, control delay, frame decimation, or asynchronous simulation step.
One three-worker pool is owned by each agent and shut down at destruction. Exceptions propagate.

Byte-identical output checks cover direct color conversion, parallel/early assembly, immutable
input buffers, stale/future-frame exclusion and worker errors. The separate benchmark extracts
the installed official `tick` and wrapper `tick`, comparing every return field for 35 synthetic
full-resolution input cases with varying compass (including NaN). It passed. These checks prove
equivalence for supplied input data, not identical trajectories across nondeterministic runs.

## Measurements and interpretation

Final steady-tick means: baseline 101.383 / 101.819 / 105.959 ms (rounded source values
are in the report); final optimization 92.576 / 91.17 ms. Approximately 103.1 -> 91.9 ms,
or 9.7 -> 10.9 ticks/s: about 12% higher throughput in these short runs. Startup/map loading
is excluded from this comparison. Main-thread exposed preprocessing falls from ~19.2 ms
to ~0.27 ms because work is overlapped, not because JPEG computation disappears.
All eight 350-tick profiling runs ended normally with exit 0; 15 tests passed. Servers
and viewer were stopped after measurement.

Evidence: `/data/runs/b2d/python-opt/report.md` and `measurements.json`. Same Town12 route 1711,
350 ticks per run, first 20 excluded. Runs 0/3/5 are baselines; 1/2 and 4 are intermediate
variants; 6/7 are the final optimization repeated. New server/port block for each run; each
At the time of those original runs, startup verified adapter 0 -> physical GPU 1 UUID
`GPU-b90dd90e-394b-7800-f23f-5892a8e3d0f1`. The current host exposes that one RTX 3090 as CUDA
index 0. PyTorch sees it directly; do not set `CUDA_VISIBLE_DEVICES`.

Baseline observations: camera data spent approximately 0.02–0.04 ms in the Python queue,
and image callback copies totaled approximately 0.8 ms per tick. Those are not the dominant
costs. The debug chase camera (`bev` sensor ID) was last in all 330 measured frames of each
baseline. Incoming image times include rendering, transport and callback work; these probes
cannot attribute all that delay to the renderer alone.

The final optimization hides CPU work under those sensor waits. The displayed "preprocess"
time is **exposed main-thread latency after all sensors arrived**, not total CPU computation.
Some CPU/GPU timing and scene evolution vary between runs; do not extrapolate a short local
improvement to all 220 routes. The baseline already includes the earlier parallel-JPEG change,
so this measures incremental improvement, not improvement over an entirely stock agent.

## Reproduce

Run long commands in tmux session `jev`, with the desktop `DISPLAY=:0`. Use the existing
TCP setup documented in [tokyo-box.md](tokyo-box.md), adding `B2D_TCP_OPTIMIZE=1` to the
environment and `--route-ids 1711 --max-ticks 350` to the runner. Use a fresh output directory
and unused `--server-index`; retain `--windowed --gpu-rank 0` and verify the GPU UUID after
startup. Keep the viewer enabled for a like-for-like comparison.

```bash
DATA_DIR=/data OMP_NUM_THREADS=4 /data/envs/b2d-tcp/bin/python \
  -m unittest discover -s scripts -p 'test_b2d_*.py'

B2D_TCP_FAST_COLOR=1 B2D_TCP_DEBUG_VIEWS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /data/envs/b2d-tcp/bin/python scripts/benchmark_b2d_preprocess.py

/data/envs/carla/bin/python scripts/b2d_optimization_report.py \
  /data/runs/b2d/python-opt \
  --json /data/runs/b2d/python-opt/measurements.json \
  --markdown /data/runs/b2d/python-opt/report.md
```

The optional viewer `--follow-runs /data/runs/b2d/python-opt` switches to the latest child
run without reconnecting to CARLA or introducing work into the control process.
