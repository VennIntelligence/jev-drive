# Fixed PI candidate: pre-live G1 evidence

Single candidate: `longitudinal_mode="pi"`, proportional gain1.0 per(m/s), integral gain.25 per(m/s*s). `vendor` remains the default; CARLA/TCP scalar semantics and default control outputs are unchanged. Conditional anti-windup, actual valid tick elapsed time, PI-only motion-gap safety above.2s, and safe/stop/reset integral clearing are explicit.

Why test this: matched straight and S routes at6m/s show similar speed oscillations while trajectory target remains6m/s; applied gear cycles coincide with brake/full-throttle cycling. The old CARLA longitudinal proportional term has effective SI gain3.6, while its finite.5s integral window cannot retain a persistent zero-error cruising effort. A true integral can retain that effort while the lower proportional gain reduces relay-like saturation. This is a hypothesis, not proof that PI eliminates real transmission dynamics.

`/data/runs/b2d/controller/v2/pi-g1-v1` preserves all14case raw per-tick traces, cases.jsonl, SHA256/source manifests, exact source snapshots and summary. No prior run was overwritten. Reproduce with a new output path:

```bash
/data/envs/carla/bin/python scripts/b2d_controller_pi_selftest.py \
  --out /data/runs/b2d/controller/v2/pi-g1-v2
```

All14syntheticcases pass: mirrored circles and delayed references, offset/heading recovery, S trajectory, acceleration/speed steps0→6→8→0→6 with0/.1/.2s actuator delay, and three lateral presets executing the same longitudinal stop test. The8m/s-to-zero3s ramp has speed RMS.04098m/s, final position error−.03187m, and5s zero-displacement/zero-speed hold. Step steady-state RMS at the three delays is.01443/.01743/.02053m/s.

Tests:15existing controller tests,5new PI tests and4independent review tests pass. Independent review replays default vendor controllers against frozen pre-PI source, checks elapsed-time invariance, long saturation/unwind, custom limits and safe resets.

The synthetic plant uses3*throttle−.08*speed−8*brake with static friction. It has no gear model or identified real actuator lag. These results authorize a controlled live comparison, not a default upgrade or score claim.
