# leaderboard_audit: What leaderboard scores are made of

status: concluded
decisions: 35, 38
index: NAVSIM v2 +10.9 is the scorer; B2D DS SD 0.80

**Question.** Which parts of high leaderboard scores reflect driving ability, and how noisy is B2D?

**Conclusion.** 35 findings: nuScenes gains are the ego prior, NAVSIM v2 +10.9 is the scorer, B2D interface +14 DS vs rules about 1 (decisions 35). B2D single-eval DS SD 0.80 (decisions 38).

**Read more.** research/leaderboard-vs-ability.md, research/human-baselines-and-leaderboard-integrity.md, `git show bcbdde4:todos/2026-09-24-hack-audit/README.md`

**Loss budget (2026-10-04).** Oracle ceilings per failure class on navtest, navhard, HUGSIM 64 and B2D, shipped Cinque vs it_dw3 + selector: [results/loss_budget.md](results/loss_budget.md) (not achievable gains).

<!-- files:begin -->
## Files

- `make_leaderboard_ability_figs.py` (archive): Figures for the research article on …
- `synthesis_answers.md` (results): 对 `questions_for_synthesis.md` 45 项的逐条回应

[archive/](archive/) 1 one-off code · [results/](results/) 178 result files · [figs/](figs/) 12 figures
<!-- files:end -->
