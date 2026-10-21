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
- R5-A, stage 2 (prereg addendum 1: no road hinge; [results/gate_r5a.md](results/gate_r5a.md),
  [results/paifix_r5a_vl_vs_R3MV.md](results/paifix_r5a_vl_vs_R3MV.md)): R3MV's argv + 9 extra S3 outcome rows a batch (on1_JTPHO, draw R5A:
  partial-fixed 4.5 / zero-rescued 2.25 / A-wins 2.25; --on-k 73 --batch 137 --yr-mass 0.095 keeps 51 / 13 / 64), seeds 0 and 1 from NTV0-L18-s0;
  smoke and the weight-0 identity pass. **Both seeds fail, launch worse than MV**: G-LAUNCH 0.328 / 0.319 vs R2C 0.865 (launched 2 / 6, rear-hit
  5 / 5; R3MV 0.569, 3 / 6); TARGET 0.546 / 0.466 (s1 fails the line); T281 route-turn zeros 2 / 2 (ok); class guard ok. S3 effect, s0 vs R3MV
  (same seed): G-LAUNCH -0.241 [-0.527, -0.065], TARGET -0.028 [-0.105, +0.042], T281 +0.020; s1 vs R3MV-s0 (not a same-seed pair): G-LAUNCH
  -0.250, TARGET -0.108 [-0.197, -0.031]. Offline (after the read, not registered): the S3 rows' labels are the served plans, and at standstill
  (fed < 0.5 m/s, 4 182 rows, 0.83 a batch) 44 % stand (4 s path < 0.5 m) and the movers cover 1.95 m median in 4 s (navtrain launch labels
  8.3 m, PAI human 4.2 m): D3's "faster wins" came from the ACC targets, not from a faster plan, so imitating the winner's served plan teaches
  the model's own slow launch. Inertia stop / turn 0.758 / 0.934, 0.768 / 0.926. No PAI 281 read. About 1.7 card-h.
- R5-B, stage 3 (prereg addendum 2; s3-rows plan addendum 1; [results/s3/relabel.md](results/s3/relabel.md)): S3 labels relabelled with the winner
  rollout's realised 4 s rig poses (log_fut's transform on the rollout's own 10 Hz anchors; tokens unchanged). **Stopped at L0, nothing trained**:
  partial-fixed launch rows (fed < 0.5 m/s) realised 4 s path median 3.59 m vs the 4.2 m line (served plan on the same rows 2.39 m, stand share
  44 % -> 13 %; PAI human 4.16 m, navtrain 8.28 m). Even the F winners' realised launches are 4.01 m, launch-class scenes 2.97 m: the winners
  launch only slightly faster than A, so their realised futures cannot carry a human-like launch. L1: the controller runs about 2x the plan's
  path at standstill launches, 1.00x when moving; 4 s position error median 1.89 m. 11 360 / 51 443 rows (22 %) lack a complete 4 s future (all
  at the rollout end). Pre-flight only (CPU, minutes); no on1_PAIOR / on1_JTPHOR built.
- R5-N, stage 4 (prereg addendum 3; [results/lead_rows.md](results/lead_rows.md), [results/lead_281.md](results/lead_281.md),
  [results/gate_r5n.md](results/gate_r5n.md), [results/paifix_r5n_vl_vs_R3MV.md](results/paifix_r5n_vl_vs_R3MV.md)): R3MV's argv with draw R5N =
  every PAI human row whose state has a lead at weight 0 (r5.py LEAD rule: in-path object with L >= 1.5 m, within hw + W/2 + 0.3 m of the human's 4 s
  path, heading < 60 deg, gap in [-1, max(40 m, 4 s x v0)]; stretched rows take their plain row's flag), parts renormalised (32 PAI rows a batch).
  Offline: 39.8 / 38.5 / 41.2 % of MV's rows removed (mostly cruise; stop-approach 1.63 -> 1.54 a batch); PAI 281 launches lead 26 / 37, R2J loses
  8 lead / 4 no-lead launches (not lead-specific); G-LAUNCH 6 / 6 lead; overshoot zeros 3 / 23 lead. **Fails**: G-LAUNCH 0.670 vs R2C 0.865
  (launched 4 / 6, rear-hit 2; vs R3MV +0.101 [-0.032, +0.335], one scene fixed); class guard fails on offroad (7 vs 3); TARGET 0.532 (ok; vs R3MV
  -0.043, TURN -0.095 [-0.204, -0.005], 4 new TURN zeros); T281 route-turn 3. Inertia 0.753 / 0.960. No s1, no PAI 281 read. About 1.9 GPU job-h.

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
