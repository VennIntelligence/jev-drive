# gt_decomp: leaderboard GT distance, along vs cross track

status: concluded
decisions: 280
index: GT is nearest-point: lag adds 0; YR 0.941 = 0.681 cross + 0.260 past path end

**Question.** Of the AlpaSim leaderboard "Distance to GT Trajectory", how much is along-track (ahead / behind the human at the same time) vs
cross-track (off the human path), for our driver vs WA-JEPA, LTF, DiffusionDrive and garage 0014, and does our excess sit in the along-track part?

**Conclusion.** The evaluator's GT distance is the distance to the nearest point of the whole logged path (scorers/ground_truth.py 106-108), so
cruise lag adds exactly 0; its only along-track part is driving past the end of the logged path (ahead of the human): YR10m10-F-s0 0.941 m =
0.681 cross + 0.260 past-end (12.6% of scenes, 65 of 85 clips >= 4 m); the human-timing counterfactual moves GT by -0.014 m outside those scenes.
A cruise speed-up course L would not move GT and can win back at most 0.022 score of progress, so it is not the next arm (decision 280).

**Read more.** [plans/2026-10-12-gt-decomp-prereg.md](plans/2026-10-12-gt-decomp-prereg.md) (Chinese, pushed before any read);
[results/gt_decomp.md](results/gt_decomp.md) (all tables, public and held); human page [research/gt-decomp/index.html](../../research/gt-decomp/index.html)
(shows and explains figs/gd_split.png and figs/gd_lag.png). Box: `$DATA_DIR/runs/gt_decomp/collect-{pub,held}/`.

<!-- files:begin -->
## Files

- `gt_decomp.md` (results): the leaderboard GT distance …
- `2026-10-12-gt-decomp-prereg.md` (plans): GT-DECOMP 预登记：榜单 "Distance to GT …
- `gd.py` (scripts): the leaderboard "Distance to GT …
- `gd_report.py` (scripts): tables and figures from gd.py collect
- `gd.json` (results): 
- `gd_split.png` (figs): 
- `gd_lag.png` (figs): 
- `280.md` (research/decisions): GT-DECOMP：榜单 "Distance to GT …

[results/](results/) 22 result files · [figs/](figs/) 4 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
