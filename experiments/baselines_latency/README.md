# baselines_latency: Latency of released baselines

status: concluded
decisions: 11, 15, 18
index: batch 1: Qwen-Drive-4B 702 ms, AutoVLA 1362 ms; our head <0.1 ms

**Question.** What do released baselines cost per decision, and does AutoVLA's think gate transfer to Waymo?

**Conclusion.** Batch 1: Qwen-Drive-4B 702 ms, AutoVLA 1362 ms, openjev 471 ms, our head < 0.1 ms (decisions 11); AutoVLA think rate 0/150 (decisions 15); latency is no contribution (decisions 18).

**Read more.** docs/baselines.md, experiments/probe_planner_v0/README.md

<!-- files:begin -->
## Files

- `_bench.py` (archive): Shared latency harness for the baseline …
- `autovla.py` (archive): Latency of AutoVLA as released, batch …
- `openjev.sh` (archive): Latency of openjev as released
- `qwen-drive.py` (archive): Latency of Qwen-Drive-1.0-4B as …
- `load_backbones.py` (archive): Load check for our feature backbones in …
- `autovla_waymo.py` (archive): how often does the adaptive model …

[archive/](archive/) 11 one-off code
<!-- files:end -->
