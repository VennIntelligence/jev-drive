# simlingo_catalogue: SimLingo official vs bundled B2D evaluation

status: concluded
decisions: none
headline: No result recorded; batch of 220 routes x 2 arms has no results section, so SimLingo/BLUE B2D scores are not re-based

**Question.** simlingo#43 reports DS 75.50 with the official Bench2Drive and 86.53 with SimLingo's bundled copy at an identical SR of 67.12%. Does the same checkpoint reproduce a DS gap, and which code differences (4000-tick cut D1, completion threshold 99 vs 90 D2, crash handling in scoring D3) explain it.

**Conclusion.** No result was recorded. The code diff and pre-registered criteria were written and smoke runs passed, but the batch (design a': 220 routes x 2 arms x seed 1) has no results section and no decision entry, and no results directory exists. The SimLingo/BLUE B2D scores are therefore not re-based by this experiment (decisions 38 only cites it as an open check).

**Read more.** research/leaderboard-vs-ability.md, `git show bcbdde4:todos/2026-09-25-simlingo-catalogue/README.md`.

<!-- files:begin -->
<!-- files:end -->
