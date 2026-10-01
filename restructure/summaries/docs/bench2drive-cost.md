**Summary.** Measurements of what a Bench2Drive closed-loop run costs; sizing and launching now live in
docs/closed-loop-runbook.md (current default), this doc is the evidence. GPU-box numbers are from earlier instances
(RTX PRO 6000 96 GB); whole-box totals were not rechecked on the 7x RTX 6000D box (per-server costs and the
6-servers-per-card knee were). Full 220 routes, 8 workers, Qwen3-VL features + stand-in driver: 11,196 s = 3.11 h,
209 finished, 11 never did; Town12+Town13 are 69% of routes and 83% of wall (ms/tick 106.2 / 156.8 vs ~64 small towns).
Layout (2026-09-25): six servers per GPU, 2.5 cores per worker, `--client-threads 8`; pids.max is 20480. The 3.1 h
reflects a driver that never arrives (0 successes, mean RC 10.9%), not the simulator floor. Tokyo sections: single
RTX 3090, policy=none Dev10 ~17.9 min / 37.46 min for 3 configs x 2 seeds; real TCP six-case run 21.34 min.

**Sections.**
- Harness cost and layout on the five-GPU box (2026-09-25, previous instance) - per-server cost, knee, layout
- Tokyo controller diagnostic: complete Dev10 seed0, 2026-09-22 - policy=none three-preset cost table
- How these numbers were made - measurement method
- The profile: where a tick goes - one-route phase breakdown
- Cameras cost per sensor, not per pixel - sensor count matters, resolution barely
- What paid, what did not - decimation, render size, compiled helper rejected
- With a real policy in the loop - cost with model inference
- Parallelism: the ceiling is a property of the configuration, not of the box - Town12 ladder, ports
- What a 220-route round costs, measured - 3.11 h breakdown per town, what moves it
- Reliability, which is the part that decides whether any of this matters - watchdog, resume, failures
- What these numbers do not cover - caveats: stand-in driver, one map, shared load
- Tokyo frozen v4 controller comparison (2026-09-23) - six Dev10 configs, 60 official records
- Tokyo actual TCP paired comparison (2026-09-23) - real TCP checkpoint, six cases, 1280.562 s
