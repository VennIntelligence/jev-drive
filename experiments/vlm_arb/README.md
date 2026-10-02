# vlm_arb: VLM slow channel and arbitration table

status: live
decisions: 84
index: VLM slow-channel arbitration for traffic lights and obstacles

**Question.** Can a frozen vision-language model answering four discrete categorical decision questions about camera frames replace privileged simulator state in the op-drive arbitration layer without catastrophic latency or rule-interaction penalties?

**Conclusion.** Live experiment. Evaluating OpenJev (DiffusionGemma-26B NVFP4) and candidate VLMs across 58 Bench2Drive evaluation routes.

**Next.** Complete Phase A shadow reading across candidate models, verify single-unit integration checklist, and run Phase B closed-loop evaluation arms (`drive`, `dslow`, `jslow`, `vred`, `vbyp`, `vall`).

**Read more.** [plans/2026-10-02-vlm-arb.md](plans/2026-10-02-vlm-arb.md), [results/report.md](results/report.md), [results/phase_a.md](results/phase_a.md), [results/lightsweep.md](results/lightsweep.md), [research/decisions/082.md](../../research/decisions/082.md).

<!-- files:begin -->
## Files


[archive/](archive/) 3 one-off code · [results/](results/) 14 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 9 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
