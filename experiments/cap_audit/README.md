# cap_audit: capability scorecard on every stored decision

status: live
decisions: 278
index: worse than every reference only on cruise lag: YS2G +0.59 m behind at 5-10 m/s

**Question.** On every stored decision (not only zeros), by capability and nuPlan scenario type, where does our AlpaSim driver fall short of
the human log and of the reference drivers rendered on the same scenes (WA-JEPA, LTF, DiffusionDrive, garage 0014)?

**Conclusion.** On the 10-decision public rollouts (1479 scenes, 136 logs) only one capability is worse than every reference by the registered
line: lag behind the human on cruise starts (5-10 m/s), YS2G +0.59 m [+0.22, +0.97], YR10m10-F-s0 +0.77 [+0.38, +1.19] beyond the slowest
reference; we cruise 0.87-0.92 m/s slower than the human there. Longitudinal, lever = new label course L. Lateral, legality and interaction
cells are at or better than the best reference almost everywhere; exploratory, across-traffic turns spend more time in the oncoming lane than
three of four references, more with the S rows (decision 278).

**Next.** Coordinator decides whether the longitudinal course L (research/self-consequence) is the next training lever; speed must be conditioned (d252).

**Read more.** [plans/2026-10-12-cap-audit-prereg.md](plans/2026-10-12-cap-audit-prereg.md) (Chinese, pushed before any read; amendment 1);
[results/cap_report.md](results/cap_report.md) (gates, flags, classification, exploratory); [results/cap_table.md](results/cap_table.md) (every cell);
human page [research/cap-audit/index.html](../../research/cap-audit/index.html). Figures: [figs/ca_heatmap.png](figs/ca_heatmap.png),
[figs/ca_bev.png](figs/ca_bev.png), explained on the HTML page. Box: `$DATA_DIR/runs/cap_audit/collect-{pub,held}/` (per-decision parquet, bev.pkl).
Per-scene tables are rebuilt by `ca_report.py tables` into `tmp/cap_audit/`.

<!-- files:begin -->
## Files

- `cap_report.md` (results): where our driver falls short, on every …
- `cap_table.md` (results): capability scorecard on every stored …
- `flags.csv` (results): 
- `2026-10-12-cap-audit-prereg.md` (plans): CAP-AUDIT 预登记：我们的 AlpaSim driver …
- `ca.py` (scripts): per-decision capability measures of our …
- `ca_report.py` (scripts): per-scene capability measures from …
- `278.md` (research/decisions): CAP-AUDIT：我们的 AlpaSim driver …

[results/](results/) 13 result files · [figs/](figs/) 4 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 5 entry points
<!-- files:end -->
