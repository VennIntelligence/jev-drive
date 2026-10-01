# skill_pack: NAVSIM skill pack N0-N4, native plan + scorer head

status: concluded
decisions: 64, 68, 69, 70, 71, 72, 73, 75
headline: Navtest PDMS climbs 84.2 native to 91.59 (N3), flat at N4 (-0.17 [-0.42, +0.09]); gains are metric alignment

**Question.** Can a scorer head that selects among Cinque's native plan and extra candidate slots raise NAVSIM PDMS without changing openpilot weights?

**Conclusion.** Navtest PDMS climbs 84.2 (native) -> N0 84.94 (+0.77 [+0.28, +1.27]) -> N1 87.32 (independent native slot) -> N2 90.60 (MLP head + 22 native family slots) -> N3 91.59 (6e4 rows), then N4 91.43 (-0.17 [-0.42, +0.09], flat; data saturates at 6e4 rows) (decisions 64, 68, 70, 72, 75). N1b and arm S/A were flat or small (decisions 69, 71). Gains are metric alignment, not capability (EP beats human on 38-44% of tokens); navhard stayed about 3 points below native for N1-N3 and only N4 is above (official 36.07 vs native 33.3, single reading) (decisions 68, 72, 75). Scores unaffected by the OpenBLAS bug (decisions 73).

**Read more.** research/leaderboard-skill-pack.md, research/navhard-deficit-breakdown.md, research/navsim-openblas-audit.md, docs/navsim.md; pre-registrations: `git show bcbdde4:todos/2026-09-29-skill-pack-n0.md`, `git show bcbdde4:todos/2026-09-29-n1-scorer.md`, `git show bcbdde4:todos/2026-09-30-navsim-raise.md`, `git show bcbdde4:todos/2026-09-29-navtrain-mcache.md`

<!-- files:begin -->
<!-- files:end -->
