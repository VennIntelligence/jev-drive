# baselines_latency: Latency of released baselines

status: concluded
decisions: 11, 15, 18
index: batch 1: Qwen-Drive-4B 702 ms, AutoVLA 1362 ms; our head <0.1 ms
key: scripts/bench_baselines/_bench.py, scripts/bench_baselines/autovla.py, scripts/bench_baselines/openjev.sh, scripts/bench_baselines/qwen-drive.py, scripts/bench_baselines/load_backbones.py, scripts/bench_baselines/autovla_waymo.py

**Question.** What do released baselines cost per decision, and does AutoVLA's think gate transfer to Waymo?

**Conclusion.** Batch 1: Qwen-Drive-4B 702 ms, AutoVLA 1362 ms, openjev 471 ms, our head < 0.1 ms (decisions 11); AutoVLA think rate 0/150 (decisions 15); latency is no contribution (decisions 18).

**Read more.** docs/baselines.md, research/frozen-vlm-planner.md

<!-- files:begin -->
<!-- files:end -->
