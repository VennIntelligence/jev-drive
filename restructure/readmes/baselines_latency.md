# baselines_latency: latency of released baselines and backbone load check

status: concluded
decisions: 11, 15, 18
headline: Batch 1: Qwen-Drive-4B 702 ms, AutoVLA 1362 ms, openjev 471 ms, our head under 0.1 ms; latency is no contribution

**Question.** What do released baselines (AutoVLA, openjev, Qwen-Drive) cost per decision on our card, does AutoVLA's adaptive think gate transfer to Waymo, and do our chosen backbones load?

**Conclusion.** Same RTX PRO 6000, batch 1: Qwen-Drive-4B 702 ms (1258 ms with reasoning), AutoVLA 1362 ms, openjev 471 ms (333 ms with placeholder ego, input-dependent re-reads); our head is < 0.1 ms and end-to-end about 33 ms, as released numbers (decisions 11). AutoVLA never thinks on WOD-E2E val (think rate 0/150, identical think blocks, flat 1.4 s), so no slower tier exists (decisions 15). Decisions 18 limits the latency story: published fast B2D methods (FIVE-VLA 33 ms at 90.95 DS) sit in our latency cell, and the 1-4B VLA numbers elsewhere are 150-300 ms, so AutoVLA's 1362 ms is an eager-HF engineering artifact; low latency and closed-loop latency injection are not a contribution. Note on provenance: entry 18's literature rows are citations, not our measurements.

**Read more.** docs/baselines.md, research/frozen-vlm-planner.md

<!-- files:begin -->
<!-- files:end -->
