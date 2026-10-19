# practice_nuplan: new nuPlan start states for on-policy rows

status: closed
decisions: 368, 369, 377, 385
index: new start-state S rows; static rows cause the right bias (d377); full-length arms without them not adopted (d385)

**Question.** Do S rows from new start states on the 28 train logs (arena train-split windows >= 1.5 s from every public start point,
rolled by JTN-L18-s0 with the default serve) leave held-460 zeros no worse than on1_JT alone in a 1 000-step JTN pilot (zeros <= control + 2)?

**Conclusion.** Gate passed: 41 786 new S rows (4 253 windows, shards 0-5; PN6 failed R1 (a) 0.9898) merged into on1_JTX; JTN 1 000-step pilot held-460 zeros 37 vs control 42, paired +0.0024 [-0.0156, +0.0213] (decision 368).

**Joint arm (supplement 1, decision 369).** NTVX-L18-s0 = NTV0-L18-s0's command on on1_JTX / JTX00S (pool the only variable): fails both boards. Arena stop3 vs RSG + stop3 -0.0042 [-0.0112, +0.0015], held 460 zeros 45 (line <= 37; NTV0 32); PAI 281 ACC+GAP vs NTV0 -0.0157 [-0.038, +0.006], zeros 64 vs 59 ([results/NTVX-L18-s0/adoption.md](results/NTVX-L18-s0/adoption.md)).

**Lateral diagnosis (supplement 2, decision 377).** NTVX's held lateral zeros come from the new start-state rows taken at standstill
(16 732 rows, 16.6 % of the S draws). There the trainee's plan creeps about 4 m into a stopped lead, and the shape-only agent hinge
(active on 33 % of them) can only push the plan sideways, on average to the right.
- Result: a constant right plan bias (3.2 cm at 4 s, re-centring unchanged) that drifts 0.11 m in the 18-decision closed loop.
- Confirmed: from NTV0, 1 000 steps on NTVX's draw give -3.3 cm [-4.3, -2.6]; the same draw without the new static rows gives -0.2 cm
  [-1.2, +0.6].
- Labels and rasters are not the cause ([results/lateral_diag.md](results/lateral_diag.md)).

**Full-length arms without the static rows (supplement 3, decision 385).** On1_JTX with the 16 732 new standing rows dropped, one change
against each parent ([results/adoption.md](results/adoption.md), PCS secondary [results/pcs.md](results/pcs.md)).
- NTVXS-L18-s0 (NTV0 command, draw `JTX00SNS`): native arena held vs NTV0 +0.0103 [+0.0034, +0.0188], held zeros 33 vs 32, drift +0.047 m
  (the right bias is gone); PAI 281 ACC+GAP vs NTV0 -0.0221 [-0.0424, -0.0034], zeros 66 vs 59. Passes the nuPlan line only against the weak
  parent (vs JTN-L18-s1 native -0.111).
- JTXS-L18-s1 (JTN-L18-s1 command, draw `JTX00NS` = JTX00 without the static rows): native vs JTN-L18-s1 +0.0016 [-0.0130, +0.0200], held
  zeros 38 vs 36, drift +0.045 m; PAI vs NTV0 -0.0393 [-0.0625, -0.0187].
- Frozen board-mix PCS: NTVXS - NTV0 +5 [-2, +14], JTXS - JTN-L18-s1 -9 [-33, +10].

**Status.** Not adopted: the new moving start states give no usable gain at full length. nuPlan native stays JTN-L18-s1, the PAI
candidate stays NTV0 + ACC + GAP; the on1_JTX line is closed.

**Read more.** [plans/2026-10-19-pn-prereg.md](plans/2026-10-19-pn-prereg.md) (prereg, Chinese working note).

<!-- files:begin -->
## Files

- `2026-10-19-pn-prereg.md` (plans): PRACTICE-NUPLAN 预登记：同 28 个 log …
- `pn.sh` (scripts): Box, repo root
- `pn.py` (scripts): new start states on the 28 train logs = …
- `on_rows.py` (experiments/onpol_nuplan/scripts): finished on-policy rollouts -> one …
- `jt.sh` (experiments/defect_fix/scripts): JT, NR1 prereg supplement 9
- `practice-plan.md` (research/next-round): 练习计划：按「可归因的试次」改训练管线（2026-10-19，工作方案）
- `onpol-scaleup.md` (research): 在线池扩容盘点：要不要「多训一小时」、能不能「多加 on-policy …

[results/](results/) 16 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 6 entry points
<!-- files:end -->
