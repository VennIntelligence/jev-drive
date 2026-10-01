# fastperc: fast-channel perception, SAM 3.1 vs YOLO-class detectors

status: concluded
decisions: 45
headline: YOLO26x-seg runs 3 cameras at 20 ms p95 (1/25 of SAM 3.1), equal pedestrian recall; gap is BEV placement, not detector

**Question.** Which detector fits a 20 Hz fast channel (p95 <= 50 ms for 3 cameras) with pedestrian recall within 0.03 of SAM 3.1?

**Conclusion.** YOLO26x-seg 640 fp16 runs 3 cameras at 20 ms p95 (about 1/25 of SAM 3.1's 540 ms) with pedestrian recall +0.005 [-0.010, +0.019] vs SAM 3.1; the SAM 3 family floor is the grounding head (36 ms per prompt) and the distilled encoder does not help (decisions 45). The recall gap is BEV placement, not the detector: with monocular metric depth (UniDepth v2) 20-40 m pedestrian BEV recall goes 0.22 -> 0.57 on nuScenes and 0.13 -> 0.397 on P5. Rule-gate rerun on YOLO state was not done.

**Read more.** research/decisions.md (45, N5 addendum), `git show bcbdde4:todos/2026-09-26-fast-perception.md`

<!-- files:begin -->
<!-- files:end -->
