# round7: PAI recipe after DAgger, Harmonizer risk

status: live
decisions: 435, 436, 439, 441, 440
index: loss map: R6DAG loses 52 / 63 pts on 281; Harmonizer turns plans tighter, closed loop halves R6DAG lead (NTV0 flat)

**Question.** Which recipe, built on the round-6 DAgger map teacher, beats NTV0 on PAI 281 (ACC + GAP) on both seeds and on the PCS ruler, and does the Harmonizer (on in the PAI finals) change our plans?

**Conclusion.** Loss map (decision 436): R6DAG loses 51.6 (s1) / 62.8 (s0) points; the most recoverable are turn-inside offroad and unforced slow-down partials (which carry the whole ruler gap to NTV0); stop-line overshoot is the largest block but never solved under ACC + GAP. Harmonizer open-loop (decision 435): lateral within the line, longitudinal over it, junction plans 0.3-0.5 m tighter at 4 s from the frozen encoder. Harmonizer closed loop (decision 439, [results/harmonizer_closedloop.md](results/harmonizer_closedloop.md)): ON - OFF NTV0 +0.002 [-0.031, +0.035] (route-turn zeros -6, offroad +5, cancel), R6DAG-s1 -0.035 [-0.066, -0.005] (longitudinal: rear-hit +5, launches 35 -> 32); R6DAG - NTV0 +0.072 OFF -> +0.035 [-0.006, +0.078] ON, DiD -0.037 [-0.078, +0.005]: the Harmonizer halves R6DAG's lead, does not flip it. Joint arm R7J (decision 441, [results/joint/](results/joint/README.md)): F2 + keep-pace DAgger labels, S-MIX, route-heading command; not adopted: OFF 0.787 / 0.786 vs R6DAG 0.779 / 0.816, Harmonizer ON 0.774 / 0.765 vs R6DAG 0.796 / 0.780; rear-hit / slow partials and the ruler gap shrank (-153 / -145), s1 offroad +9.

**Next.** Fill the recipe from DAG2 / REC results; judge PAI candidates on Harmonizer-ON 281 reads (or both); check R6DAG's ON longitudinal shift open-loop (harm.py forward on the stored 300 states) before any ON-render augmentation.

**HARM-MIX lane (built 10-22, decision 440).** Twin store `on1_PAIHTW` (103 904 rows), gates A-D passed. Harmonizer-ON twins of ~34 % of the PAI human rows (246 scenes; PAIH / 05 / 15) for mixed OFF / ON training, and the trainer switch `rm_train --twin <store>:<p>` (off by default; lib/twin_tokens.py). Plan [plans/2026-10-22-harmonizer-mix.md](plans/2026-10-22-harmonizer-mix.md) (Chinese, written before any render), code scripts/hmix.sh / hmix.py, results [results/harm_mix/](results/harm_mix/). No training arm yet.

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
