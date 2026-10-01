# top10: top-10 intersection across boards, T1-T3 exams

status: concluded
decisions: 46, 58
headline: No top-10 family intersects the three boards except SparseDrive (B2D #10); no model memorises positions, 0.2% is shift

**Question.** Does any method family rank top 10 on both real-data open-loop boards and CARLA closed-loop boards, and when we run the leaderboard models on our own pair exams (T1 SparseDriveV2 + ZTRS, T2 DrivoR + WA-JEPA, T3 BridgeDrive + BLUE), do their scores carry reaction ability?

**Conclusion.** No family intersects, except SparseDrive at B2D #10 (reads CARLA ground-truth pose); intersection is set by training-data ecosystem and input contract, not capability (decisions 46). Exams: NAVSIM scores reproduced (SparseDriveV2 PDMS 92.22, DrivoR 93.69); zero-shot open-loop loses to cv on WOD for T1 / DrivoR; I3 real-appearance vehicle pairs flip 23.6 / 45.5% (T1), 33.7 / 66.1% (DrivoR / WA-JEPA) against null 5-6%, about half of openpilot `ridge_late` selectivity; BridgeDrive route + speed channel 0.2%, BLUE 26.5% versus SimLingo 34.6% (BLUE - SimLingo -8.1 pp [-12.4, -4.5]). Ghost-test and shift/swap perturbations (decisions 58) show none of TFv6, BridgeDrive, BLUE, SimLingo memorises positions or collapses (BLUE swap +18 pp [+6, +30] is below the 10 pp gate on a lone seed), so the 0.2% is open-loop distribution shift. Entry statuses are still marked pending.

**Read more.** research/leaderboard-vs-ability.md (sections 8, 9), `git show bcbdde4:todos/2026-09-26-top10-intersection.md`, `git show bcbdde4:todos/2026-09-26-night-queue-4.md` (G section for entry 58); intersection board tables are kept in [results/top10-intersection/](results/top10-intersection/)

<!-- files:begin -->
<!-- files:end -->
