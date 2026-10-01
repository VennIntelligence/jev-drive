# zeroshot_openloop: zero-shot open-loop exams of Alpamayo 1.5 and openpilot

status: concluded
decisions: 34, 37, 39
headline: WOD RFS: Cinque 8.005, Alpamayo 8.034, Lebowski 7.886, all above cv 7.103; NAVSIM openpilot rows reflect input protocol

**Question.** How do Alpamayo 1.5 and openpilot (Cinque v3, Lebowski, small), run untrained behind per-dataset adapters, score on WOD-E2E val, NAVSIM navtest/navhard, nuScenes val and PhysicalAI-AV?

**Conclusion.** WOD-E2E val (479 rater frames): Cinque RFS 8.005 [7.79, 8.22], Alpamayo medoid-of-6 8.034, Lebowski 7.886, all above cv 7.103 and above our Waymo-trained heads (7.31); Cinque vs logged future -0.13 [-0.35, +0.10] (decisions 34). NAVSIM: Alpamayo 44.3 PDMS / 43.2 EPDMS, openpilot 47-52 PDMS, cv 20.7 / 25.9; navhard EPDMS 9-11, no better than cv (decisions 37). The openpilot native-plan rows are an input-protocol reading, not model ability: corrected in place 2026-09-26 (2 Hz input drops WOD RFS to 4.9-5.2) and 2026-09-28/29 (frame interpolation lifts Cinque to 84.2 PDMS on navtest, 33.3 EPDMS on navhard); the literature TransFuser EPDMS column was also corrected to 84.0 (decisions 37). nuScenes (quarter, n = 1159): L2 worse than cv by +0.14 to +0.36 m, collision lower; on PhysicalAI-AV openpilot loses to Alpamayo (ADE@6.4 s 2.35-2.86 vs 1.77 m) (decisions 39). Nav text did not help in any exam.

**Read more.** research/openpilot-openloop-standing.md, research/openpilot-openloop-integration.md, docs/navsim.md, docs/zeroshot-adapters.md, `git show bcbdde4:todos/2026-09-24-zeroshot-exam/wod-e2e.md` (also `navsim.md`, `nuscenes-physicalai.md` in the same directory)

<!-- files:begin -->
<!-- files:end -->
