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
- R5-G / R5-G2 and the LAUNCH-37 exam, stage 5 (prereg addenda 4 / 4b; [results/gonow_rows.md](results/gonow_rows.md),
  [results/gate_r5g2.md](results/gate_r5g2.md) (both arms; R5-G alone: [results/gate_r5g.md](results/gate_r5g.md)),
  [results/launch37_g2.md](results/launch37_g2.md)). Arms: R3MV's argv with one draw-file change each.
  - R5-G: MV + PAI GO-NOW rows kept (fed < 0.5 m/s, label path >= 0.5 m by 1.0 s) + R3MVL's navtrain launch rows x 3. GO-NOW is only 0.31 rows
    a batch (4 s 10.75 m, the same as navtrain GO-NOW 10.71 m; 45 % behind a lead).
  - R5-G2: the GO-NOW rows of ph / ph15 x 14.05 = 4 rows a batch; ph05's 225 are dropped.
  - **Neither passes**:
    - R5-G is the first PAI-row arm over the G-LAUNCH line: 0.822, 5 / 6. It fails TARGET (0.493 vs the 0.503 line; TURN 0.552, 12 zeros)
      and the offroad class guard (8 vs R2C 3).
    - R5-G2 launches worse: G-LAUNCH 0.513 (3 / 6), TARGET 0.464, overshoot zeros 14.
  - LAUNCH-37 = the 37 PAI 281 launch scenes, ACC + GAP, 4 chunks, about 0.13 card-h an arm. Stored NTV0 / R2C / R2J are reused.

    | read | launched / 37 | rear-hit | mean |
    |:--|--:|--:|--:|
    | NTV0 | 33 | 9 | 0.774 |
    | R2C | 33 | 11 | 0.734 |
    | R2J | 21 | 19 | 0.528 |
    | R3MV | 35 | 10 | 0.822 |
    | R3MVL | 35 | 9 | 0.815 |
    | R5-N | 32 | 11 | 0.729 |
    | R5-G (36 scored) | 35 | 9 | 0.862 |
    | R5-G2 | 29 | 12 | 0.677 |

    Under ACC + GAP the MV-family arms launch as well as NTV0. The G-LAUNCH loss (native serve) does not show here.
  - Registered comparisons, both separated:
    - R5-G vs R5-N: LAUNCH-37 +0.113 [+0.037, +0.206] (launched +3), TARGET -0.039 [-0.130, +0.051].
    - R5-G2 vs R5-N: LAUNCH-37 -0.052 [-0.143, +0.035], TARGET -0.067 [-0.138, -0.003].
  - R5-G vs R3MVL is a null on LAUNCH-37: +0.024 [-0.007, +0.076], launched 0 / 0. So R5-G's G-LAUNCH 0.822 against MVL's 0.690 is one scene,
    within noise. The separation R5-G > R5-N is R5-N losing launches (vs R3MV -0.093 [-0.174, -0.025]).
  - More GO-NOW rows hurt: R5-G2 - R5-G -0.166 [-0.280, -0.063], launched -6.
  - No PAI 281 read (rule: gate + LAUNCH-37 >= 30). About 1.7 card-h (VRAM-booked share; 6.1 pool job-hours).

- MV-LC (R5-LC), stage 6 (prereg addenda 5 / 5a; [results/lc_rows.md](results/lc_rows.md), [results/pai281_R5LC_ag.md](results/pai281_R5LC_ag.md),
  [results/pcs_r5l.md](results/pcs_r5l.md)). R3MV's argv; PAI human rows whose 4 s label is a lane change off the route lane (D1's rule: v0 > 5,
  |y4| > 2.5 m, |yaw4| < 12 deg, route polyline not within 1.2 m of the 4 s point) at weight 0, parts renormalised. Offline: 10 986 rows (ph 4 219,
  ph05 365, ph15 6 402; nearly all cruise, median 26-36 m/s); lane-change share of drawn moving PAI rows 12.0 -> 6.2 % (navtrain 8.0 %).
  Arms MV-LC s0 / s1 + R3MV s1; exam PAI 281 ACC + GAP (275 common scenes).

    | read | mean | zeros | route-turn | offroad | zero-other | AF-front | rear-hit | slow | launched / 37 | ruler PCS (vs NTV0) |
    |:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
    | NTV0 | 0.745 | 58 | 15 | 12 | 6 | 3 | 25 | 11 | 33 | 2195 |
    | R2C | 0.742 | 59 | 15 | 11 | 7 | 4 | 24 | 10 | 33 | -29 [-95, +19] |
    | R3MV s0 | 0.741 | 61 | 2 | 20 | 14 | 3 | 24 | 5 | 35 | -276 [-375, -217] |
    | R3MV s1 | 0.712 | 70 | 3 | 21 | 21 | 3 | 21 | 4 | 34 | -282 [-384, -225] |
    | MV-LC s0 | 0.746 | 61 | 3 | 21 | 10 | 6 | 18 | 7 | 34 | -266 [-366, -189] |
    | MV-LC s1 | 0.724 | 67 | 4 | 24 | 11 | 6 | 22 | 6 | 35 | -285 [-394, -208] |

  - **Not an adopt-candidate** (s1 below NTV0; offroad + zero-other 31 / 35 vs the cap 21) and both seeds fail the ruler rule.
  - The filter removes drift / lane-change zeros (vs same-seed R3MV: zero-other -4 [-10, +1] s0, -10 [-17, -4] s1), but offroad stays
    +9 / +12 over NTV0 and AF-front +3; mean vs R3MV +0.005 / +0.012 (n.s.); ruler vs R3MV +17 [-25, +62] / -5 [-50, +36].
  - Seed noise on 281: R3MV s1 - s0 -0.029 [-0.062, +0.003] (zero-other +7). New zeros / partials on the 181 scenes NTV0 scores 1.0:
    R2C 6 / 2, R3MV 21 / 9 and 28 / 8, MV-LC 19 / 8 and 23 / 10 (offroad now the largest class).
  - About 10.6 GPU pool job-hours (trainings 1.18, 281 chunks 9.32, smoke / identity 0.12); ruler about 20 CPU jobs at PRIO 15.

- OFFROAD-MV, diagnosis of the MV family's extra offroad zeros on PAI 281 ACC + GAP ([results/offroad_mv.md](results/offroad_mv.md);
  code [scripts/offroad_mv.py](scripts/offroad_mv.py) / [scripts/offroad_mv.sh](scripts/offroad_mv.sh)). R3MV s0 / s1 and MV-LC s0 / s1: 51 zeros on
  21 scenes, 9 scenes in >= 3 of the 4 reads. Not lane changes (MV-LC keeps all 9).
  - Mechanisms: corner cutting on curves and turns (23 events), unforced slow-down then drift to the right edge (13), over-speed on a highway
    (4), and a steady lateral offset on straights (9).
  - Read-wide, every PAI-row arm makes about 2.3 x NTV0's / R2C's unforced slow plans.
  - On curves, the f = 1.5 stretch rows put 60 % of labels above 3 m/s^2 lateral (navtrain 0.3 %). The human paths themselves are not tight.
  - Two draw-file fixes ranked: F1 stretch rows with a_lat > 3 m/s^2 or v0 > 30 m/s; F2 PAI slow-downs with no LEAD and no stop.

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
