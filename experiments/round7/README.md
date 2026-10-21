# round7: PAI recipe after DAgger, Harmonizer risk

status: live
decisions: 435, 436
index: loss map: R6DAG loses 52 / 63 pts on 281; Harmonizer turns plans tighter

**Question.** Which recipe, built on the round-6 DAgger map teacher, beats NTV0 on PAI 281 (ACC + GAP) on both seeds and on the PCS ruler, and does the Harmonizer (on in the PAI finals) change our plans?

**Conclusion.** Loss map (decision 436): R6DAG loses 51.6 (s1) / 62.8 (s0) points; the most recoverable are turn-inside offroad and unforced slow-down partials (which carry the whole ruler gap to NTV0); stop-line overshoot is the largest block but never solved under ACC + GAP. Harmonizer open-loop (decision 435): lateral within the line, longitudinal over it, junction plans 0.3-0.5 m tighter at 4 s from the frozen encoder. Harmonizer closed loop, PARTIAL ([results/harmonizer_closedloop.md](results/harmonizer_closedloop.md)): NTV0 full 281 ON - OFF +0.002 [-0.032, +0.035] (route-turn zeros -6, offroad +5); R6DAG-s1 on 108 scenes -0.014, DiD +0.005 [-0.064, +0.077]; R6DAG-s1 c003-c007 paused behind DAG2 reads.

**Next.** Fill the recipe from DAG2 / REC results; resume the Harmonizer closed-loop read of R6DAG-s1 (`harmcl.sh resume`) when the main session frees cards, then the decisions row.

**Read more.** [plans/2026-10-21-round7-framework-draft.md](plans/2026-10-21-round7-framework-draft.md) (Chinese working draft), the review page research/round7/loss_diag/index.html.

<!-- files:begin -->
## Files

- `2026-10-21-round7-framework-draft.md` (plans): 第 7 轮框架草稿（10-21 下午，主会话，未登记）
- `loss_map.md` (results): where R6DAG still loses points on PAI …
- `harmonizer_openloop.md` (results): Harmonizer open-loop check
- `index.html` (research/round7/loss_diag): rb{position:sticky;top:0;z-index:9;backg …
- `2026-10-21-harmonizer-check.md` (plans): Harmonizer 开环检查：登记线（2026-10-21，写于任何渲染 / …
- `harm.sh` (scripts): Harmonizer open-loop check

[results/](results/) 12 result files · [plans/](plans/) 2 live plans · [scripts/](scripts/) 5 entry points
<!-- files:end -->
