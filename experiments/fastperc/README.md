# fastperc: Fast-channel perception, SAM vs YOLO

status: concluded
decisions: 45
index: YOLO26x-seg 20 ms p95 for 3 cameras, 1/25 of SAM 3.1, equal recall

**Question.** Which detector fits a 20 Hz fast channel with pedestrian recall near SAM 3.1?

**Conclusion.** YOLO26x-seg runs 3 cameras at 20 ms p95 (SAM 3.1: 540 ms) with recall +0.005 [-0.010, +0.019] (decisions 45). The gap is BEV placement: UniDepth lifts 20-40 m recall 0.22 -> 0.57 (nuScenes).

**Read more.** experiments/fastperc/results/, research/decisions.md (45), `git show bcbdde4:todos/2026-09-26-fast-perception.md`

<!-- files:begin -->
## Files

- `fastperc.py` (jevdrive): SAM 3.1 speed-ups and faster detectors …
- `fastperc_figs.py` (archive): Figure for the fast-perception …
- `fastperc.sh` (archive): Fast-perception runs on GPU 0
- `fastperc_setup.sh` (archive): Environments + weights for the …

[archive/](archive/) 3 one-off code · [results/](results/) 2 result files · [figs/](figs/) 2 figures
<!-- files:end -->
