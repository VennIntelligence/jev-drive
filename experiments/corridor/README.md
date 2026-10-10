# corridor: lane corridor as a turn label

status: concluded
decisions: 240, 243
index: Corridor centreline as target: -9.9 EPDMS on turns; wrong exit 2.6% of DAC fails; HEAD1: frozen-token heading head 8.54 vs SH30 11.24 deg at > 45 deg, independence R2 0.513 < 0.538, stage 2 not run
key: experiments/corridor/results/head1.md experiments/corridor/results/head1/tables.md experiments/corridor/plans/2026-10-10-head1-prereg.md experiments/corridor/scripts/head1_labels.py experiments/corridor/scripts/head1_train.py experiments/corridor/scripts/head1_read.py experiments/corridor/results/corr0.md experiments/corridor/results/corr0/tables.md experiments/corridor/plans/2026-10-10-corr0-prereg.md experiments/corridor/lib/lanegraph.py experiments/corridor/scripts/corr_geom.py experiments/corridor/scripts/corr.py experiments/corridor/scripts/corr_report.py experiments/corridor/scripts/corr_chain.sh experiments/corridor/scripts/corr_bev.py

**Question.** Is the lane corridor of the commanded exit, taken from the nuPlan map, the right supervision label for fixing SH30's turn failures (measured without training)?

**Conclusion.** No as a lateral target: re-targeting SH30's own curve error onto the corridor centreline costs -9.89 [-11.95, -7.87] EPDMS on > 20 deg navtest tokens and the centreline itself is 3.46 below the stored plan, while wrong exit / lane holds only 2.6% of > 45 deg DAC failures (66.4% along-track). The corridor does carry the 4 s heading (map 4.51 deg RMS against SH30's 5.62; 7.25 against 11.04 on > 45 deg), and navtrain has 18 989 tokens in 821 logs with a >= 20 deg alternative exit (decisions 240).

**HEAD1 (decision 243): can a head on the frozen vision tokens describe the turn better than the policy's plan?** Yes on accuracy, not enough on independence. A 3.4 M-parameter head on the frozen Cinque tokens + ego + command, trained on navtrain with log-level folds to predict the heading against arc length of the logged path, is 8.54 deg RMS [7.47, 9.53] from the logged 4 s heading on > 45 deg navtest tokens (read at the plan's own 4 s arc length) against 11.24 for SH30-F's own plan (-2.70 [-3.57, -1.74]) and 15.13 for the same head without vision; the privileged map label is at 7.04. Its error is correlated with the policy's (0.53 against the pilot policy): the registered independence statistic is R2 0.513 [0.420, 0.586] against a line of 0.538 (predicted pilot gain +0.48 against +0.5), so stage 1 fails on independence in the marginal band and the memory-channel pilot was not run. Against SH30-F the statistic is 0.278 (predicted +0.26). The logged-path label beats the map label (8.54 / 10.07 deg) and the learning curve has not flattened at 834 logs (14.1 / 13.3 / 10.9 / 9.0 deg): [results/head1.md](results/head1.md), pre-registration [plans/2026-10-10-head1-prereg.md](plans/2026-10-10-head1-prereg.md), Chinese page [research/corridor/](../../research/corridor/index.html).

**Read more.** [results/corr0.md](results/corr0.md); pre-registration with amendments A and B in [plans/](plans/2026-10-10-corr0-prereg.md). The map, the logged future and the driven lane sequence are privileged: analysis only.

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points, `lib/` code other topics import, `results/` small result files, `figs/` figures, `plans/` plan notes.
Refresh the file list and INDEX.md with `python tools/topic_index.py`.
