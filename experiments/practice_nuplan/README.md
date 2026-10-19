# practice_nuplan: new nuPlan start states for on-policy rows

status: live
decisions: 368, 369, 377
index: new start-state S rows; NTVX's held lateral loss = static rows' sideways hinge push (-3.3 cm)

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

**Next.** Use on1_JTX only with the new static rows dropped (draw `JTX00SNS`), or with the agent hinge on standing rows replaced. Whether
the new moving start states help needs one full-length NTV0-recipe arm on JTX00SNS (not run).

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
