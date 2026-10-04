# leaderboard_audit: What leaderboard scores are made of

status: concluded
decisions: 35, 38, 103, 108, 109, 112, 124, 125, 126
index: NAVSIM v2 +10.9 is the scorer; B2D DS SD 0.80

**Question.** Which parts of high leaderboard scores reflect driving ability, and how noisy is B2D?

**Conclusion.** 35 findings: nuScenes gains are the ego prior, NAVSIM v2 +10.9 is the scorer, B2D interface +14 DS vs rules about 1 (decisions 35). B2D single-eval DS SD 0.80 (decisions 38).

**Read more.** research/leaderboard-vs-ability.md, research/human-baselines-and-leaderboard-integrity.md, `git show bcbdde4:todos/2026-09-24-hack-audit/README.md`

**Loss budget (2026-10-04).** Oracle ceilings per failure class on navtest, navhard, HUGSIM 64 and B2D, shipped Cinque vs it_dw3 + selector: [results/loss_budget.md](results/loss_budget.md) (not achievable gains).

**Real vs render (2026-10-04).** Same-pose real vs 3DGS frames through openpilot (HUGSIM nuScenes 19 scenes, navhard near-pose pairs): renders move road
edges (1.8x the frame-to-frame floor) but not the plan; they raise the launch gain to a small history yaw by 43% in every scene; no image-side fix
helps: [results/real_vs_render.md](results/real_vs_render.md) (pre-registration [plans/2026-10-04-real-vs-render-prereg.md](plans/2026-10-04-real-vs-render-prereg.md)).

**navhard DAC attribution (2026-10-04).** The 1 314 / 1 249 DAC failures (shipped / it_dw3 + selector) split into causes, each with a replay check and an oracle ceiling. About half are on the scorer side: pavement outside the cached polygon (26-29%, about 5.2 points) and tracker lag, where the plan is inside but the replay is not (21%, 2.3-2.8). A fifth are displaced stage-2 starts, mostly with no feasible arc. Planning errors (wrong direction, under-turn, over-turn, corner cutting) are about 30%, 4.7-4.9 points:
[results/navhard_dac.md](results/navhard_dac.md) (pre-registration [plans/2026-10-04-navhard-dac-prereg.md](plans/2026-10-04-navhard-dac-prereg.md)).
Follow-ups: no map-free gate isolates the tracker-lag tokens (precision 7-14%), compensating only the scorer-identified ones would give about +2 EPDMS (oracle); an eye check puts about 3/4 of the M class on real pavement ([plans/2026-10-04-navhard-dac-gated-comp-prereg.md](plans/2026-10-04-navhard-dac-gated-comp-prereg.md)).

**Unified openpilot interface (2026-10-05).** One as-on-the-car spec for every board ([docs/openpilot-interface.md](../../docs/openpilot-interface.md));
small-step checks: HUGSIM keeps the static warm-up (cold start does not launch), B2D spec camera 1.22 m at the bumper line, nored inflated decision 102's
drive by 2.1 DS, virtual 1.22 m cameras cost NAVSIM / WOD open-loop score: [results/unified_interface.md](results/unified_interface.md).

<!-- files:begin -->
## Files

- `make_leaderboard_ability_figs.py` (archive): Figures for the research article on …
- `synthesis_answers.md` (results): 对 `questions_for_synthesis.md` 45 项的逐条回应

[archive/](archive/) 1 one-off code · [results/](results/) 178 result files · [figs/](figs/) 12 figures
<!-- files:end -->
