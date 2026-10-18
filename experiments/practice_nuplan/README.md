# practice_nuplan: new nuPlan start states for on-policy rows

status: live
decisions: pending
index: arena train windows away from public starts, rolled by JTN, merged into on1_JT; 1 000-step pilot gate

**Question.** Do S rows from new start states on the 28 train logs (arena train-split windows >= 1.5 s from every public start point,
rolled by JTN-L18-s0 with the default serve) leave held-460 zeros no worse than on1_JT alone in a 1 000-step JTN pilot (zeros <= control + 2)?

**Conclusion.** Pending.

**Next.** Smoke (10 windows) -> stage 2 (first chunk of each shard, in-queue checklist) -> all chunks -> per-shard pools PN0..6 -> on1_JTX ->
pilot JTX1K-L18-s0 vs control JTNC1K-L18-s0 on held 460 @18 -> `pn.py gate` (results/pilot.md).

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

[plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
