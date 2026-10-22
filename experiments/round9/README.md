# round9: adaptive S pressure and speed-scaled route window

status: live
decisions: 436, 441, 443
index: draft only; S adaptive pressure + speed-scaled route window, no reads

**Question.** Does a clean S pool with adaptive S pressure (S / imitation gradient ratio held at a target) beat R6DAG on PAI 281, and is a speed-scaled route look-ahead worth a base retrain?

**Conclusion.** None yet; draft plan only.

**Next.** S-MIX offline value and route CPU checks (dispatched 10-22), then pre-register the S joint arm.

**Read more.** Plan (Chinese, unregistered): [plans/2026-10-22-round9-draft.md](plans/2026-10-22-round9-draft.md).

<!-- files:begin -->
## Files

- `2026-10-22-round9-draft.md` (plans): 第 9 轮草稿（10-22 晚，主会话与用户讨论，未登记）
- `s-rows-and-paper.md` (research): S 行：现在的分布、做过和没做过的、能写成什么论文（2026-10-21 …
- `loss_map.md` (experiments/round7/results): where R6DAG still loses points on PAI …

[plans/](plans/) 1 live plans
<!-- files:end -->

Layout: `scripts/` entry points (live), `results/` small result files, `plans/` live plan notes.
