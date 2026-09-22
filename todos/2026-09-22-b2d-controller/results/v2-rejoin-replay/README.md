# V2 rejoin adapter replay evidence

This is an offline replay of 15 baseline sensor-derived pose traces (four G2 routes and supplemental S route17563, all three presets). It cannot establish closed-loop tracking quality or dynamic feasibility.

Immutable full evidence: `/data/runs/b2d/controller/v2/rejoin-replay-v1/`. The directory contains minimal per-tick pose inputs, original route references, exact helper/adapter/test source snapshots with SHA256 hashes, every emitted 20x2 trajectory and diagnostics in `rejoin_samples.jsonl`, and all5607 timing repetitions in `latency_samples.csv`. This folder mirrors the small summary/manifest for source control.

Invocation from the isolated v2 worktree:

```bash
/data/envs/carla/bin/python scripts/b2d_controller_rejoin_replay.py \
  --source /data/runs/b2d/controller/development3 \
  --source /data/runs/b2d/controller/development-s-v1 \
  --out /data/runs/b2d/controller/v2/rejoin-replay-v1 \
  --cruise-mps 6 --repeats 3
```

The output path is exclusive; use a new suffix to rerun.

No construction errors across15cases. Maximum timed chord speed6.000000000003322m/s, i.e. floating-point rounding around the configured6m/s bound. Active trajectory generation latency: median0.2734ms, p95=.3532ms, p99=.3964ms, maximum.7983ms. These are wall measurements under uncontrolled host load, not simulator throughput. All trajectory and terminal-hold timings are separately reported to avoid hiding active work behind cheap parked updates.

Curvature concerns remain explicitly logged, particularly near parking. The reference curvature bound is geometric; it does not prove tire-force or controller tracking feasibility. Live G2 remains required.
