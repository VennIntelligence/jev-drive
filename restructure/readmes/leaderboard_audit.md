# leaderboard_audit: what leaderboard scores are made of

status: concluded
decisions: 35, 38
headline: 35 findings: nuScenes gains are ego prior, NAVSIM v2 +10.9 is the scorer; rank agreement holds only within a code family

**Question.** Which parts of high leaderboard scores can be attributed to real driving ability (E layer) rather than recipe (R layer) or metric proxies, and how large is Bench2Drive evaluation noise?

**Conclusion.** Read-only audit (35 findings, 1,121 ablation rows, 204 cross-board scores): nuScenes gains are the ego-state prior, NAVSIM v2 gains are the EPDMS proxy scorer (+10.9 from the scorer alone), Bench2Drive mixes real ability with interface and rules (interface +14 DS, rules about 1); cross-board rank agreement only holds within a code family (B2D vs Longest6 n = 5 rho = 0.05, NAVSIM v1 vs v2 top n = 11 rho = 0.21); only four gains map to E layer, all closed-loop; the Give_Way +26.7 was downgraded in place to +2.72 DS / +4.85 SR (decisions 35). Bench2Drive: single-evaluation DS SD 0.80, adjacent ranks within noise (|dDS| < 2.3); the lead of BLUE on sudden hazards is of unknown origin (downgraded 2026-09-26); sudden-hazard families are near saturated, the gap is in planning / yielding routes (decisions 38).

**Read more.** research/leaderboard-vs-ability.md, research/capability-vs-leaderboard.md, research/results/leaderboard-text-analysis/synthesis_answers.md, `git show bcbdde4:todos/2026-09-24-leaderboard-synthesis.md`, `git show bcbdde4:todos/2026-09-24-hack-audit/README.md`, `git show bcbdde4:todos/2026-09-24-leaderboard-text-analysis/README.md`, `git show bcbdde4:todos/2026-09-25-tfv6-rules-interface/README.md` (entry 38 pre-registration)

<!-- files:begin -->
<!-- files:end -->
