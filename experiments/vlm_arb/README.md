# vlm_arb: VLM slow channel and arbitration table

status: live
decisions: 84, 85, 86, 87
index: Zero-shot Qwen3-VL-4B light reading: red-light infractions 13 to 6 (privileged 5), DS +5.0; fixed bypass pbyp2, stop-line R2 vred2, yellow rule vred3

**Question.** Can a frozen vision-language model answering four discrete categorical decision questions about camera frames replace privileged simulator state in the op-drive arbitration layer without catastrophic latency or rule-interaction penalties?

**Conclusion.** Diagnostic batch (19 routes x 2 seeds): junction slow-down +5.4 DS [+1.6, +10.1] vs the speed-matched control; privileged red light +4.3 [-3.0, +11.2]; privileged bypass -28.8 on non-target routes. openjev fails the light and block lines (ego red recall 57.8%), so no VLM arm ran. Offline, Qwen3-VL-4B reads ego red 93-94% zero-shot and reaches 68-125 ms with one forward pass at reduced resolution (decisions 84, 85).

**vred (2026-10-02).** Zero-shot Qwen3-VL-4B (one pass, 1153 tokens, per-card servers, L = 0.35 s measured under load) driving R2 + R5: red-light infractions 13 -> 6 (`pred` 5), DS +5.0 [+0.7, +10.6] vs `drive`, no harm on routes without a light; diagnostic read on 19 routes, confirmation lines not evaluated ([results/vred.md](results/vred.md), [plans/2026-10-02-vlm-vred.md](plans/2026-10-02-vlm-vred.md)). Remaining infractions: yellow-onset late stop, stop target beyond the light's stop line, light turning red after the crossing.

**Bypass, offline (2026-10-02).** 22 of 26 privileged-bypass misfires on non-target routes are a route-projection clamp bug (blocker behind the route start or beyond its end), not queues; a time-based static detector fires on 8/8 obstacle runs with 1 false trigger: [results/bypass_misfire.md](results/bypass_misfire.md), [results/bypass_timedet.md](results/bypass_timedet.md), definitions in [plans/2026-10-02-bypass-offline.md](plans/2026-10-02-bypass-offline.md).

**pbyp2 / vred2 / vred3 (2026-10-02, diagnostic).** `pbyp2` (bypass with the projection fix, static >= 5 s, light memory, same-direction gap check): 0 misfire activations on the 15 other routes (`pbyp`: 26), DS -3.4 [-13.0, +6.4] against `drive` there (`pbyp`: -28.8); obstacle routes DS 42.1 [26.0, 69.2] (`drive` 27.8) because the frozen gap rule stalls the car on 3 of 4 under continuous traffic; with no gap check (`pbyp2ng`, 8 runs) 74.1 [48.2, 100.0] ([results/pbyp2.md](results/pbyp2.md)). `vred2` (R2 at the light's stop line): median stop 0.16 m short of it (`vred`: 13 of 23 stops beyond), ego-green recall while holding 28.6% -> 94.9%, but red-light infractions 7 (`vred` 6) and DS 66.5 ([results/vred2.md](results/vred2.md)). `vred3` (+ approach-only slow-down, yellow rule, commit): 7 infractions, DS 67.6 ([results/vred3.md](results/vred3.md)). Plan, parameters, deviations D25-D32, inputs by source: [plans/2026-10-02-pbyp2-vred2.md](plans/2026-10-02-pbyp2-vred2.md).

**pbyp2 / vred2 / vred3 (2026-10-02).** Bypass misfires gone after the projection fix (0 vs 26), but the same-direction gap check blocks 3 of 4 obstacle routes; without it obstacle-route DS 27.8 -> 74.1 with collisions against same-direction traffic. Moving R2's stop target to the stop line and the yellow rule did not reduce red-light infractions (7 vs 6): [results/pbyp2.md](results/pbyp2.md), [results/vred2.md](results/vred2.md), [results/vred3.md](results/vred3.md), plan [plans/2026-10-02-pbyp2-vred2.md](plans/2026-10-02-pbyp2-vred2.md) (decision 87).

**Offline reads on the logs (2026-10-02).** Release-policy replay on the logged red-light answers (K=2 releases falsely in 8 of 62 hold episodes, 3 of the 6 original `vred` infractions followed a false release; a cumulative-evidence rule halves the release delay at fewer false releases): [results/release_replay.md](results/release_replay.md). The official route's command change is within 3.1 m of the true junction entrance on all 13 junctions but 2.5 to 10 m after the light's stop line: [results/route_junction_error.md](results/route_junction_error.md). The vehicles that hit the pulling-out car braked in 8 of 19 contacts and never enough; the adjacent lane carries a vehicle every 3.2 s: [results/pullout_collisions.md](results/pullout_collisions.md). Definitions: [plans/2026-10-02-offline-analyses-definitions.md](plans/2026-10-02-offline-analyses-definitions.md).

**Next.** Replace the CARLA-map inputs (junction entrance, stop line) by the official route commands; shorten the green-release delay; a gap policy for the bypass.

**Read more.** [plans/2026-10-02-vlm-arb.md](plans/2026-10-02-vlm-arb.md), [results/report.md](results/report.md), [results/phase_a.md](results/phase_a.md), [results/lightsweep.md](results/lightsweep.md), [plans/2026-10-02-vlm-thin.md](plans/2026-10-02-vlm-thin.md) + [results/vlm_thin.md](results/vlm_thin.md) (offline: one-pass scoring, resolution and a cut language model with a thin head for Qwen3-VL-4B's light reading; speed and accuracy), [research/decisions/082.md](../../research/decisions/082.md).

<!-- files:begin -->
## Files


[archive/](archive/) 3 one-off code · [results/](results/) 59 result files · [plans/](plans/) 5 live plans · [scripts/](scripts/) 25 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
