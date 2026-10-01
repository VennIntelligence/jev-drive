# real_transfer: real-data transfer gates G0-G3 for the reaction head

status: concluded
decisions: 44
headline: No transfer to real data: all 12 student zero-shot cells harmful (NAVSIM PDMS -1.1 to -2.7); gating only shrinks harm

**Question.** After E1-E3 failed, can a student, real-trained gates, HUGSIM 3DGS vehicle pairs or better edit pairs bring the CARLA reaction to real data?

**Conclusion.** No (decisions 44, corrected in place at G2: "two paths" became "three paths"). G0: all 12 student zero-shot cells harmful (NAVSIM PDMS -1.1 to -2.7). G1: gates only shrink the harm by shrinking the whole delta (WOD gone, NAVSIM main arm still -1.61 [-2.14, -1.08]); the only positive is a lead-TTC gate times a constant brake (+0.53 PDMS), independent of the pairs. G2: retraining on HUGSIM I3 vehicle pairs removes most of the harm on I3 itself but no head exceeds the prior, and the linear head is worse on WOD/NAVSIM. G3: edit quality is not the bottleneck; pedestrian events in WOD train are only about 230.

**Read more.** research/results/real-data-transfer/, research/decisions.md (44), `git show bcbdde4:todos/2026-09-26-real-data-transfer.md`

<!-- files:begin -->
<!-- files:end -->
