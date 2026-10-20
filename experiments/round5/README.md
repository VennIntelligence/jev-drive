# round5: joint PAI recipe from three data streams

status: live
decisions: none yet
index: D3: 42 % of train scenes informative; faster wins launches +0.124

**Question.** Can one PAI recipe that mixes human demonstrations (S1), the policy's own states (S2) and branch outcomes on its own states (S3),
with class-balanced shares, beat NTV0-L18-s0 + ACC + GAP on PAI 281?

**Conclusion.** Not trained yet. Diagnostics so far:
- D3, outcome variance for S3 ([results/d3_outcome_variance.md](results/d3_outcome_variance.md)): NTV0 + ACC + GAP rolled 4 ways (as served, ACC
  targets x1.15 / x0.85, route-seeking TTS choice) on 149 curated train scenes.
  - 63 scenes (42 %) are informative (best - worst >= 0.2).
  - The one consistent winner is faster on partials: 11 up / 0 down; launches +0.124 [+0.062, +0.198].
  - Zeros are fixed scene by scene (14 / 43) by arms that also break full scores, so no class rule (leave-one-out -0.016).
  - Stops are unreachable by these perturbations.
  - 11 456 winner decisions; cost 1.49 card-h per 100 scenes x 4 arms.

**Next.** The framework draft's slots get filled from 3b and D1-D3. The S3 sketch is [plans/d3_awr_sketch.md](plans/d3_awr_sketch.md).

**Read more.** [plans/2026-10-21-round5-framework-draft.md](plans/2026-10-21-round5-framework-draft.md) (Chinese working note).

Box: D3 run dir `$DATA_DIR/runs/round5/d3/` (jobs.txt, stages.txt, checks/, report/, DONE).

<!-- files:begin -->
## Files

- `2026-10-21-round5-framework-draft.md` (plans): 第 5 轮联合训练框架草稿（10-21 夜，主会话，未登记）
- `d3_outcome_variance.md` (results): is there outcome signal for closed-loop …
- `d3_awr_sketch.md` (plans): S3（后果行）怎么进配方：一页草图（10-21 夜，D3 之后，未登记）
- `d3.sh` (scripts): is there outcome signal for closed-loop …
- `d3.py` (scripts): is there outcome signal to learn from …
- `d3_scenes.tsv` (results): 
- `tables.md` (results): D3 tables
- `d1_budget.md` (results): where the remaining PAI points are, and …

[results/](results/) 13 result files · [plans/](plans/) 3 live plans · [scripts/](scripts/) 7 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `results/` small result files, `plans/` live plan notes.
