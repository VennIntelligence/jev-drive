# corridor: lane corridor as a turn label

status: concluded
decisions: 240
index: Corridor centreline as target: -9.9 EPDMS on turns; wrong exit 2.6% of DAC fails
key: experiments/corridor/results/corr0.md experiments/corridor/results/corr0/tables.md experiments/corridor/plans/2026-10-10-corr0-prereg.md experiments/corridor/lib/lanegraph.py experiments/corridor/scripts/corr_geom.py experiments/corridor/scripts/corr.py experiments/corridor/scripts/corr_report.py experiments/corridor/scripts/corr_chain.sh experiments/corridor/scripts/corr_bev.py

**Question.** Is the lane corridor of the commanded exit, taken from the nuPlan map, the right supervision label for fixing SH30's turn failures (measured without training)?

**Conclusion.** No as a lateral target: re-targeting SH30's own curve error onto the corridor centreline costs -9.89 [-11.95, -7.87] EPDMS on > 20 deg navtest tokens and the centreline itself is 3.46 below the stored plan, while wrong exit / lane holds only 2.6% of > 45 deg DAC failures (66.4% along-track). The corridor does carry the 4 s heading (map 4.51 deg RMS against SH30's 5.62; 7.25 against 11.04 on > 45 deg), and navtrain has 18 989 tokens in 821 logs with a >= 20 deg alternative exit (decisions 240).

**Read more.** [results/corr0.md](results/corr0.md); pre-registration with amendments A and B in [plans/](plans/2026-10-10-corr0-prereg.md). The map, the logged future and the driven lane sequence are privileged: analysis only.

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points, `lib/` code other topics import, `results/` small result files, `figs/` figures, `plans/` plan notes.
Refresh the file list and INDEX.md with `python tools/topic_index.py`.
