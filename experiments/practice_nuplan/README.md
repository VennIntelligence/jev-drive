# practice_nuplan: new nuPlan start states for on-policy rows

status: live
decisions: 368, 369
index: arena train windows away from public starts, rolled by JTN, merged into on1_JT; 1 000-step pilot gate

**Question.** Do S rows from new start states on the 28 train logs (arena train-split windows >= 1.5 s from every public start point,
rolled by JTN-L18-s0 with the default serve) leave held-460 zeros no worse than on1_JT alone in a 1 000-step JTN pilot (zeros <= control + 2)?

**Conclusion.** Gate passed: 41 786 new S rows (4 253 windows, shards 0-5; PN6 failed R1 (a) 0.9898) merged into on1_JTX; JTN 1 000-step pilot held-460 zeros 37 vs control 42, paired +0.0024 [-0.0156, +0.0213] (decision 368).

**Joint arm (supplement 1, decision 369).** NTVX-L18-s0 = NTV0-L18-s0's command on on1_JTX / JTX00S (pool the only variable): fails both boards. Arena stop3 vs RSG + stop3 -0.0042 [-0.0112, +0.0015], held 460 zeros 45 (line <= 37; NTV0 32); PAI 281 ACC+GAP vs NTV0 -0.0157 [-0.038, +0.006], zeros 64 vs 59 ([results/NTVX-L18-s0/adoption.md](results/NTVX-L18-s0/adoption.md)).

**Next.** on1_JTX does not go into another full-length arm until the held lateral / offroad zeros it adds (20 -> 32 / 15 -> 23 vs NTV0) are explained.

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

[results/](results/) 5 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
