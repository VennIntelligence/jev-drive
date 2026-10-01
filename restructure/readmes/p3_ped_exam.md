# p3_ped_exam: real-appearance pedestrian exam via OmniRe 3DGS edits

status: concluded
decisions: 44
headline: 3DGS pedestrian insertion/deletion stopped: donors slide, deletions smear; only about 10 WOD segments qualify

**Question.** Can OmniRe/3DGS deletion and insertion of pedestrians in real WOD clips give a real-appearance pedestrian exam (P3), later with a distance x lateral x state dose-response grid?

**Conclusion.** The 3DGS insertion/deletion route was stopped on 2026-09-28 after the user reviewed 38 items: inserted donors slide instead of walk and carry background halos, and deleted vehicles leave smears (decisions 44, corrected in place; the earlier "gate passed" reading was superseded). Deletion null flip was 10/230 = 4.35% [1.30, 7.39] on 10 scenes, but under the validity filter 0 of 230 scored frames are real questions, and only about 10 WOD v2 segments qualify, so a >= 60-scene deletion exam is unreachable. What survives is the dose grid design and the threat-label rule (`jevdrive.ped_dose.threat_labels`); generation moved to CARLA (see cosmos).

**Read more.** research/p3-exam-filter.md, research/p3-review-sheet.md, research/insertion-options.md, `git show bcbdde4:todos/2026-09-28-ped-dose-response.md`, `git show bcbdde4:tmp/2026-09-28-p3-filter.md`

<!-- files:begin -->
<!-- files:end -->
