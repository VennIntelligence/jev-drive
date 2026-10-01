# top10: Top-10 intersection across boards, T1-T3

status: concluded
decisions: 46, 58
index: no top-10 family on all boards except SparseDrive
key: scripts/top10_exam.sh, jevdrive/top10_exam.py, jevdrive/top10_t2.py, jevdrive/top10_t3.py, scripts/top10_t3_agent.py, scripts/top10_t3_batch.sh, scripts/make_top10_tables.py, scripts/make_top10_t3_figs.py

**Question.** Does any method family rank top 10 on real open-loop and CARLA closed-loop boards, and do scores carry reaction ability?

**Conclusion.** No family intersects except SparseDrive (B2D #10) (decisions 46). I3 flips 23.6-66.1% vs null 5-6%; BridgeDrive 0.2% is shift; none memorises positions (decisions 58).

**Read more.** research/leaderboard-vs-ability.md, `git show bcbdde4:todos/2026-09-26-top10-intersection.md`

<!-- files:begin -->
<!-- files:end -->
