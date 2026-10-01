# tfv6_rules: TFv6 rules x interface, B2D by hazard family

status: concluded
decisions: 38, 31
headline: Only public per-route data done: single-eval DS SD 0.80, top-5 within noise; rules x interface runs not done

**Question.** Do TFv6 rule heuristics on/off x control interface A/B change hazard reaction (experiment 1), and how much of the public Bench2Drive ranking is evaluation noise (experiment 6).

**Conclusion.** Only experiment 6 (public per-route data, 22 entries, 209 routes) has results: single-evaluation DS SD 0.80, adjacent top ranks (TFv6 89.6, BLUE 90.6, SparseDriveV2 89.1, SimLingo 87.0, R2SE 86.3) differ within noise, and sudden-hazard families are near saturation while the gap sits in planning/yielding families (decisions 38, status pending; the BLUE sudden-hazard advantage was downgraded in place). Experiment 1 (rules x interface closed-loop pairing) and our own TFv6 repeats have no results: the run stopped on 2026-09-25 14:57 and nothing was written back.

**Read more.** research/leaderboard-vs-ability.md, `git show bcbdde4:todos/2026-09-25-tfv6-rules-interface/README.md`.

<!-- files:begin -->
<!-- files:end -->
