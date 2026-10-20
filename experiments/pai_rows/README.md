# pai_rows: PAI human-state and frame-stretch training rows

status: live
decisions: none yet
index: building: 853 eligible NuRec scenes (half A 428); H / stretch / S row pools

**Question.** Build training pools from every training-eligible NuRec scene on the box: PAI human-state imitation rows (the ego follows the
human log, label = the human's logged future), their frame-stretched variants (slots re-picked at 0.5x / 1.5x spacing, ego row scaled to
match, label unchanged; against the frame-channel speed copy of decision 404), and on-policy S rows (the d402 pipeline) on the scenes the
d402 pool lacks; plus a class-stratified half split for the data-scaling arms of round 2.

**Conclusion.** Not yet: rows are being built (pools on1_PAIH, on1_PAIH05, on1_PAIH15, on1_CPAIX).

**Next.** Gates of the prereg (stage checks, R1 trainer forward, f = 1 bit identity), then the row / feasibility tables in results/.

**Read more.** [plans/2026-10-20-pai-rows-prereg.md](plans/2026-10-20-pai-rows-prereg.md) (scenes, recipe, gates, half split and passes per row, cost);
[results/scenes.md](results/scenes.md) (the 853 scenes by class and half).

Pools (box, `$DATA_DIR/runs/op_parity/cache/`): `on1_PAIH` (human-state rows, names `on:paih:<scene>:<k>`, fut = log_fut), `on1_PAIH05` /
`on1_PAIH15` (stretched, `on:paih05:..` / `on:paih15:..`, tab `f`, `base`), `on1_CPAIX` (S rows on the 288 new scenes). Splits:
`physical_ai_av/pai-rows@v1` (853), `physical_ai_av/pai-rows-half-a@v1` (428). Run dir `$DATA_DIR/runs/pai_rows/`.

<!-- files:begin -->
## Files

- `2026-10-20-pai-rows-prereg.md` (plans): PAI-ROWS 预登记：盘上全部可训练 NuRec 场景的 PAI …
- `pr.sh` (scripts): box, repo root
- `pr.py` (scripts): PAI human-state imitation rows on every …
- `scenes.md` (results): PAI-ROWS scene table
- `scenes.tsv` (results): 
- `onp_cache.py` (experiments/onpol_pai/scripts): the PAI row files -> what the trainer …
- `rf_pai_pool.py` (experiments/route_follow/scripts): Route-night section 9 step 5
- `onp_log_driver.py` (experiments/onpol_pai/lib): the simulated ego is steered along the …

[results/](results/) 3 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
