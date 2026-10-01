# fastperc: Fast-channel perception, SAM vs YOLO

status: concluded
decisions: 45
index: YOLO26x-seg 20 ms p95 for 3 cameras, 1/25 of SAM 3.1, equal recall
key: jevdrive/fastperc.py, jevdrive/fastperc_figs.py, scripts/fastperc.sh, scripts/fastperc_setup.sh

**Question.** Which detector fits a 20 Hz fast channel with pedestrian recall near SAM 3.1?

**Conclusion.** YOLO26x-seg runs 3 cameras at 20 ms p95 (SAM 3.1: 540 ms) with recall +0.005 [-0.010, +0.019] (decisions 45). The gap is BEV placement: UniDepth lifts 20-40 m recall 0.22 -> 0.57 (nuScenes).

**Read more.** research/results/fast-perception/, research/decisions.md (45), `git show bcbdde4:todos/2026-09-26-fast-perception.md`

<!-- files:begin -->
<!-- files:end -->
