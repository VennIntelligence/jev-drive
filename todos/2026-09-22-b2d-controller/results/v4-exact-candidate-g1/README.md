# Exact frozen candidate: supplementary synthetic G1

**16/16 cases pass the existing helper gates.** This evidence was collected **after formal-v4 started**, to close the exact-configuration coverage gap in the earlier additive-lookahead PI G1 artifact. It does not retrospectively establish pre-run validation, replace formal G4, or qualify a new default.

Every constructed controller is asserted to use pursuit, max lookahead, PI Kp=0.5/Ki=0.25, the near speed window, and the exact archived candidate physical configuration. The archived controller source is byte-identical to the formal-v4 provenance copy. The process-local factory injects that configuration into the existing analytic helpers; production files and the archived helper source remain unmodified.

| Existing helper cases | Count | Observed result |
| --- | ---: | --- |
| R20 at 6 m/s, both signs, trajectory delays 0/0.1/0.3 s | 6 | Maximum steady lateral RMS 0.00931 m; speed RMS 0.09376 m/s |
| R20 with offset ±0.5 m and heading ±5° | 4 | Maximum steady lateral RMS 0.00989 m |
| Analytic S, trajectory delay 0/0.3 s | 2 | Lateral RMS 0.003853/0.009982 m; speed RMS 0.02393/0.03170 m/s |
| 8 m/s deceleration and stop | 1 | Deceleration speed RMS 0.11222 m/s; stop error −0.02833 m; 5 s hold, zero drift |
| 6→8→0→6 m/s, actuator delay 0/0.1/0.2 s | 3 | Maximum steady speed RMS 0.00726 m/s; zero parked drift |

The circle gate is steady lateral RMS <0.2 m and reference-speed RMS <0.5 m/s; the S gate is lateral RMS ≤0.3 m and speed RMS <0.5 m/s. Stop and speed-step gates are preserved in the archived helper functions and in each result. For either circle direction, 0.3 s trajectory delay changes lateral RMS by −0.005711 m relative to zero delay, also satisfying the original ≤0.3 m delayed RMS / ≤0.1 m increment criterion. These synthetic delay results do not establish that real-world latency improves tracking.

This is an independent analytic-path / synthetic bicycle experiment, with synthetic acceleration, drag, brake friction and instantaneous tire response. The plant retains the existing rounded measured steering-curve constants; the controller uses the exact archived measured constants. There are no gears, model predictions, sensors, CARLA interactions, or measured slope physics. The 20 future points are generated directly from analytic trajectories, never padded from native TCP's shorter prediction horizon.

- [Summary and effective configuration per case](summary.json)
- [Complete tick traces and case events](/data/runs/b2d/controller/v4-exact-candidate-g1/traces/)
- [Archived controller, helper sources, config and runner](source/)
- [SHA-256 inventory, including summary and all traces](hashes.json)

The complete traces occupy 9.9 MB and are preserved outside the worktree at `/data/runs/b2d/controller/v4-exact-candidate-g1/traces/`; their original relative identities and hashes remain in the inventory. No raw evidence was discarded.

Reproduce into a fresh output directory on this workspace:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python todos/2026-09-22-b2d-controller/results/v4-exact-candidate-g1/run.py --out /tmp/b2d-exact-candidate-g1-replay
```

The runner reads the documented frozen source/config paths and asserts their runtime equality before execution; a later changed runtime intentionally aborts that check. Numerical results are deterministic for these fixed analytic inputs; per-step wall-clock timings and collection timestamps are not expected to reproduce byte-for-byte.
