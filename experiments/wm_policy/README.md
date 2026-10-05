# wm_policy: comma worldmodel-4B as a history-only driving predictor

status: concluded
decisions: (pending)
index: Stopped at step 0: no in-distribution history-only mode (anchors fixed)
key: experiments/wm_policy/results/step0.md, experiments/reactivity/archive/i4_worldmodel_probe.py

**Question.** Run without the ego's future, is comma's worldmodel-4B (plan head) itself a usable driving predictor,
compared with Cinque on held-out comma1M / WOD clips?

**Conclusion.** Not testable in distribution: the shipped model always prefills five clean *future anchor* frames
(logged future video, fidx 10..49 = 0.2-9.8 s ahead) and its cfg null branch drops only the pose, not the anchors;
the paper calls the anchored model "offline only" and trains no anchor-free variant. Stopped before any scored run,
as the brief allows ([results/step0.md](results/step0.md)).

**Next.** None unless the main session approves an explicitly out-of-distribution arm (see step0.md, "Options").

**Read more.** [results/step0.md](results/step0.md); probe that runs the model as shipped:
`experiments/reactivity/archive/i4_worldmodel_probe.py`.

<!-- files:begin -->
<!-- files:end -->

Layout: `results/` small result files. No code, no GPU time used.
