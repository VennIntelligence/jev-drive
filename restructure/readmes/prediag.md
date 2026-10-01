# prediag: pre-diagnostic round on Waymo E2E (L0, P0-P4)

status: concluded
decisions: 3d, 20, 21, 22, 23, 24
headline: Pre-onset vision delta is a null (CI [-0.062, +0.030]); frozen-feature readout is exhausted

**Question.** Where does vision help beyond the ego prior at critical (pre-onset, high-surprise) moments on Waymo, and does a different readout or backbone change that? Steps: L0 surprise weighting, P0 train-split recheck, P1 judge, P2 readout ladder, P3 backbone ladder, P3e multimodal heads, P4 CARLA feature gap.

**Conclusion.** Train split (decisions 3d): pre-onset delta CI [-0.062, +0.030], DiD +0.111, so the ego-prior-increment framing is falsified. L0 lands in branch 2, representation not head: linear readout of the frozen features is exhausted, reweighting only moves error between subsets (decisions 20). Judge fixed (decisions 22): primary ADE on s_ego deciles 1-9, RFS alongside, top decile reported with RFS and ADE vs rater_best only (decile 10 is multimodal). P2 (b)(c)(d)(e): no readout (temporal input, spatial tokens, gated residual) buys pre-onset (decisions 23). P3: Qwen3-VL-32B, Wan2.2 and V-JEPA 2 are null (V-JEPA train split -0.030, CI crosses zero); only Qwen native video is nonzero at the -0.05 m threshold edge (decisions 24). P4 (todo only, no entry): Waymo-trained heads are unusable on CARLA frames (domain AUC about 1.0, vocabulary uncoverable +15.3 pp). Overtaken for backbones by driving-trained features (see driving_backbones).

**Read more.** research/prediag-2026-09/README.md, research/p3-exam-filter.md, `git show bcbdde4:todos/2026-09-22-p1-judge.md`, `git show bcbdde4:todos/2026-09-22-p3-backbone-ladder.md`, `git show bcbdde4:todos/2026-09-23-p4-carla-feature-gap.md` (also p0-train-split-recheck, p2-readout-ladder, p3e-multimodal-heads of the same dates)

<!-- files:begin -->
<!-- files:end -->
