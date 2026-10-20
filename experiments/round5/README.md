# round5: joint PAI recipe from three data streams

status: live
decisions: none yet
index: D3: 42 % of train scenes informative; faster wins launches +0.124; S3 rows on1_PAIO built (51 443, gates R1-R3 pass)

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

- S3 outcome rows, built ([plans/2026-10-21-s3-rows.md](plans/2026-10-21-s3-rows.md); [results/s3/tables.md](results/s3/tables.md),
  [results/s3/rows.md](results/s3/rows.md), checks in [results/s3/checks/](results/s3/checks/); code [scripts/s3.sh](scripts/s3.sh) / [scripts/s3.py](scripts/s3.py)):
  D3's 4 arms on all 719 non-held eligible train scenes (712 scored; D3's roll reused for 149).
  - Informative: 281 / 712 (39 %). By class: stop 23 %, turn 58 %, launch 62 %, other 33 %. F - A on launches +0.089 [+0.050, +0.136] (12 up / 0 down).
  - Row scenes (winner = best non-A arm with best - A >= 0.1, else A): partial-fixed 61 (F 56), zero-rescued 75 (L 43, S 17, F 15), A-wins 148.
  - Pool `on1_PAIO`: 51 443 rows (10 Hz, 284 scenes), label = the winner's served plan. Rows by type: partial-fixed 11 414, zero-rescued 13 398,
    A-wins 26 631. Tags: tab `otype` / `ocls` / `osub` / `oarm`; `outcome.npz` holds post-hand, 2 Hz slot, divergence k and A's served plan at the same k.
  - Merged pool `on1_JTPHO` = on1_JTPH + PAIO (zero-copy tokens), with draw file `R2JFO` (PAIO at weight 0).
  - Gates:
    - R1: 284 / 284 re-rolls reproduce score, served poses and anchors bit for bit.
    - R2: label = served plan, max |diff| 7.6e-6 m. Input path: 5.6 % over (line 10 %), wrong checkpoint 99.6 %.
    - R3: 60-step R2J smoke with PAIO at weight 0 is bit-identical.
  - Cost: 9.7 card-h of rolls. Disk: about 126 GB new.

- R5-R, v1 recipe stage 1 ([plans/2026-10-21-r5-prereg.md](plans/2026-10-21-r5-prereg.md); [results/gate_r5r.md](results/gate_r5r.md),
  [results/paifix_r5r_vl_vs_R3MV.md](results/paifix_r5r_vl_vs_R3MV.md)): R3MV-L18-s0's recorded argv + `--on-road-pai road_on1_JTPH.npz` (rule d,
  W 1 = the S rows' road hinge weight, road_lam 10 x ho_w 3); 89 % of the drawn PAI rows hinged (smoke 0.896), W 0 bit-identical to R3MV's argv.
  **Fails**: G-LAUNCH 0.570 vs R2C 0.865 (launched 3 / 6, rear-hit 3, the same as R3MV); T281 route-turn zeros 5 / 12 (ok); TARGET 0.514 (ok);
  class guard fails on offroad (7 vs R2C 3; R3MV 4). Against R3MV (same seed): TARGET -0.060 [-0.140, +0.011], zeros +4 (new 5, fixed 1), offroad
  4 -> 7 (new 4: curve inside / outside, straight, junction inside); the road hinge did not cut offroad zeros. Inertia stop / turn 0.744 / 0.945
  (R3MV 0.744 / 0.948). No PAI 281 read (stop rule). About 1 card-h.

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
