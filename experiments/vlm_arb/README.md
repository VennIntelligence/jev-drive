# vlm_arb: VLM slow channel and arbitration table

status: live
decisions: 84, 85
index: Junction slow-down +5.4 DS vs speed-matched; Qwen3-VL-4B reads ego red 93-94%, 68-125 ms

**Question.** Can a frozen vision-language model answering four discrete categorical decision questions about camera frames replace privileged simulator state in the op-drive arbitration layer without catastrophic latency or rule-interaction penalties?

**Conclusion.** Diagnostic batch (19 routes x 2 seeds): junction slow-down +5.4 DS [+1.6, +10.1] vs the speed-matched control; privileged red light +4.3 [-3.0, +11.2]; privileged bypass -28.8 on non-target routes. openjev fails the light and block lines (ego red recall 57.8%), so no VLM arm ran. Offline, Qwen3-VL-4B reads ego red 93-94% zero-shot and reaches 68-125 ms with one forward pass at reduced resolution (decisions 84, 85).

**Next.** Run `vred` with zero-shot Qwen3-VL-4B (one pass, 559-1153 tokens); attribute the bypass misfires event by event before any perception-driven bypass.

**Read more.** [plans/2026-10-02-vlm-arb.md](plans/2026-10-02-vlm-arb.md), [results/report.md](results/report.md), [results/phase_a.md](results/phase_a.md), [results/lightsweep.md](results/lightsweep.md), [plans/2026-10-02-vlm-thin.md](plans/2026-10-02-vlm-thin.md) + [results/vlm_thin.md](results/vlm_thin.md) (offline: one-pass scoring, resolution and a cut language model with a thin head for Qwen3-VL-4B's light reading; speed and accuracy), [research/decisions/082.md](../../research/decisions/082.md).

<!-- files:begin -->
## Files


[archive/](archive/) 3 one-off code · [results/](results/) 21 result files · [plans/](plans/) 2 live plans · [scripts/](scripts/) 14 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
