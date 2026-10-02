# vlm_arb: VLM slow channel and arbitration table

status: live
decisions: 84, 85, 86
index: Zero-shot Qwen3-VL-4B light reading: red-light infractions 13 to 6 (privileged 5), DS +5.0

**Question.** Can a frozen vision-language model answering four discrete categorical decision questions about camera frames replace privileged simulator state in the op-drive arbitration layer without catastrophic latency or rule-interaction penalties?

**Conclusion.** Diagnostic batch (19 routes x 2 seeds): junction slow-down +5.4 DS [+1.6, +10.1] vs the speed-matched control; privileged red light +4.3 [-3.0, +11.2]; privileged bypass -28.8 on non-target routes. openjev fails the light and block lines (ego red recall 57.8%), so no VLM arm ran. Offline, Qwen3-VL-4B reads ego red 93-94% zero-shot and reaches 68-125 ms with one forward pass at reduced resolution (decisions 84, 85).

**vred (2026-10-02).** Zero-shot Qwen3-VL-4B (one pass, 1153 tokens, per-card servers, L = 0.35 s measured under load) driving R2 + R5: red-light infractions 13 -> 6 (`pred` 5), DS +5.0 [+0.7, +10.6] vs `drive`, no harm on routes without a light; diagnostic read on 19 routes, confirmation lines not evaluated ([results/vred.md](results/vred.md), [plans/2026-10-02-vlm-vred.md](plans/2026-10-02-vlm-vred.md)). Remaining infractions: yellow-onset late stop, stop target beyond the light's stop line, light turning red after the crossing.

**Bypass, offline (2026-10-02).** 22 of 26 privileged-bypass misfires on non-target routes are a route-projection clamp bug (blocker behind the route start or beyond its end), not queues; a time-based static detector fires on 8/8 obstacle runs with 1 false trigger: [results/bypass_misfire.md](results/bypass_misfire.md), [results/bypass_timedet.md](results/bypass_timedet.md), definitions in [plans/2026-10-02-bypass-offline.md](plans/2026-10-02-bypass-offline.md).

**Next.** Move R2's stop target to the light's stop line; fix the bypass projection clamp, add a same-direction gap check, rerun `pbyp`.

**Read more.** [plans/2026-10-02-vlm-arb.md](plans/2026-10-02-vlm-arb.md), [results/report.md](results/report.md), [results/phase_a.md](results/phase_a.md), [results/lightsweep.md](results/lightsweep.md), [plans/2026-10-02-vlm-thin.md](plans/2026-10-02-vlm-thin.md) + [results/vlm_thin.md](results/vlm_thin.md) (offline: one-pass scoring, resolution and a cut language model with a thin head for Qwen3-VL-4B's light reading; speed and accuracy), [research/decisions/082.md](../../research/decisions/082.md).

<!-- files:begin -->
## Files


[archive/](archive/) 3 one-off code · [results/](results/) 21 result files · [plans/](plans/) 2 live plans · [scripts/](scripts/) 14 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
