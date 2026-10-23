# round9: adaptive S pressure and speed-scaled route window

status: live
decisions: 436, 441, 443
index: draft only; S adaptive pressure + speed-scaled route window, no reads

**Question.** Does a clean S pool with adaptive S pressure (S / imitation gradient ratio held at a target) beat R6DAG on PAI 281, and is a speed-scaled route look-ahead worth a base retrain?

**Conclusion.** None yet; draft plan only.

**Next.** S-MIX offline value and route CPU checks (dispatched 10-22), then pre-register the S joint arm.

**Read more.** Plan (Chinese, unregistered): [plans/2026-10-22-round9-draft.md](plans/2026-10-22-round9-draft.md).
Route-branch CPU check (2026-10-22): [results/route_check/README.md](results/route_check/README.md). DAG-LANE (2026-10-22, decision 450: not adopted; (OFF + ON) / 2 0.780 / 0.783 vs R6DAG 0.789, ON offroad up; user choice after route_check: no route-input base retrain): one DAgger iteration on R6DAG's recipe with the heading command and the teacher's human-lane rule, pre-registered in [plans/2026-10-22-dag-lane.md](plans/2026-10-22-dag-lane.md) (Chinese), code `scripts/dl.py` / `scripts/dl.sh`, results [results/dag_lane/](results/dag_lane/). PBT fitness noise (CPU, from stored 281 reads): [results/fitness_noise/README.md](results/fitness_noise/README.md). PAI-S gate (CPU, 2026-10-22): S hinges on the DAgger states fail the gate (learnable-active 6 %, 1 row a batch at dose 16, 23 % above 13 m/s), not folded into the DAgger arm: [results/pai_s_gate/README.md](results/pai_s_gate/README.md). HINGE-AUDIT (CPU, 2026-10-22): unconditionally the human satisfies agent / side / onc (0.1-1.4 %); road (12-16 %), navtrain drivable (14 %) and soft (6 %) are ill-posed (margin vs human clearance, raster window border clamp = 58 % of the PAID plan road mass); proposed expert-calibrated hinges (spec: learnable mass 25 -> 87 % S, 16 -> 99.6 % PAID, LA rows +33 %), pre-registered in [plans/2026-10-22-hinge-audit.md](plans/2026-10-22-hinge-audit.md): [results/hinge_audit/README.md](results/hinge_audit/README.md). PAI-BASE (2026-10-22, decision 449, closed): a base that imitates cleaned PAI human rows from the start (RB0-P, 30 / 50 / 70 % of the imitation rows) loses to RB0-F on PAI 281 (-0.045 to -0.085; route-turn zeros fall, offroad and drift zeros rise more), so the registered gate stopped the line before the L18 / DAgger stages: [results/pai_base/README.md](results/pai_base/README.md), plan [plans/2026-10-22-pai-base.md](plans/2026-10-22-pai-base.md). NU-PAIBASE (2026-10-22, decision 451, closed): the same bases (RB0-P70 / P30) carried through the nuPlan candidate B2's recipe with only the init changed (s0 / s1) do not beat B2: frozen PCS 1349-1358 vs 1369 / 1398, held 460 zeros +2 to +8, and the base-stage corridor gain does not survive the S stage: [results/nu_paibase/README.md](results/nu_paibase/README.md), plan [plans/2026-10-22-nu-paibase.md](plans/2026-10-22-nu-paibase.md). STOP-GAP (CPU, 2026-10-22): under ACC + GAP we rest ~1 m closer to a stopped lead than the human (3.8 vs 4.5 m) and creep ~0.4 m before the lead moves (lead MPC on a standstill reading that is flat at 2.7-3.9 m for true 0-6 m), but it costs ~0.2 points per seed at most (0 rest rear-hits, overshoot zeros never stop first); keep JEV_GAP 0.7,2.7: [results/stop_gap/README.md](results/stop_gap/README.md). NTV0-NOS (2026-10-23, decision 456): NTV0-L18-s0's argv without the S rows (`--on-k 0`) reads 0.624 on PAI 281 ACC + GAP, -0.119 [-0.160, -0.081] vs NTV0-s0 and at RB0-F level, on both scorers: the continuation alone gives nothing, NTV0's gain is the S rows (collisions, offroad, drift): [results/ntv0_nos/README.md](results/ntv0_nos/README.md), plan [plans/2026-10-23-ntv0-nos.md](plans/2026-10-23-ntv0-nos.md).

<!-- files:begin -->
## Files

- `2026-10-22-round9-draft.md` (plans): 第 9 轮草稿（10-22 晚，主会话与用户讨论，未登记）
- `s-rows-and-paper.md` (research): S 行：现在的分布、做过和没做过的、能写成什么论文（2026-10-21 …
- `loss_map.md` (experiments/round7/results): where R6DAG still loses points on PAI …

[plans/](plans/) 1 live plans
<!-- files:end -->

Layout: `scripts/` entry points (live), `results/` small result files, `plans/` live plan notes.
