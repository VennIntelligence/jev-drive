# top10: Top-10 intersection across boards, T1-T3

status: concluded
decisions: 46, 58
index: no top-10 family on all boards except SparseDrive

**Question.** Does any method family rank top 10 on real open-loop and CARLA closed-loop boards, and do scores carry reaction ability?

**Conclusion.** No family intersects except SparseDrive (B2D #10) (decisions 46). I3 flips 23.6-66.1% vs null 5-6%; BridgeDrive 0.2% is shift; none memorises positions (decisions 58).

**Read more.** research/leaderboard-vs-ability.md, `git show bcbdde4:todos/2026-09-26-top10-intersection.md`

<!-- files:begin -->
## Files

- `top10_exam.sh` (archive): Top-10 exam runner in a model's own env …
- `top10_exam.py` (archive): Top-10 intersection exams, executor T1
- `top10_t2.py` (lib): Top-10 intersection, executor T2
- `top10_t3.py` (archive): BridgeDrive and BLUE on the P5 v1 BA …
- `top10_t3_agent.py` (archive): the P5 v1 BA worlds again, with …
- `top10_t3_batch.sh` (archive): one chain per GPU over, all sharing
- `make_top10_tables.py` (archive): Render the per-board top-10 CSVs as …
- `make_top10_t3_figs.py` (archive): Figure for the top-10 exams, executor T3

[archive/](archive/) 19 one-off code · [results/](results/) 60 result files · [figs/](figs/) 4 figures · [lib/](lib/) 9 library
<!-- files:end -->
