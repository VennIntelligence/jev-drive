# night_queue_3: Examinees on P6, closed-loop heads, Q4-Q6

status: concluded
decisions: 35, 44, 47, 48, 52, 53
index: No public lateral-readout examinee bypasses; zero speed -22.1 PDMS

**Question.** Do public examinees and our heads bypass on P6; what do Q4-Q6 read?

**Conclusion.** Apart from PDM-Lite no lateral-readout examinee bypasses (decisions 47); Q3 misses gates (52). Zeroing NAVSIM speed drops DrivoR PDMS by 22.115, WA-JEPA 8.159: ego-prior reading (35, 44, 48, 53).

**Read more.** research/review-sheet-nq3.md, experiments/night_queue_3/results/, `git show bcbdde4:todos/2026-09-26-night-queue-3.md`

<!-- files:begin -->
## Files

- `nq3_p6.py` (jevdrive): the P6 v0 judge shared by Q1 and Q2
- `nq3_q1.py` (jevdrive): every examinee that needs no …
- `nq3_q2.py` (lib): eliciting bypass on frozen openpilot …
- `nq3_q4a.py` (archive): Night queue 3, lane D, Q4a
- `nq3_q5.py` (archive): hack audit of the top-10 families …
- `nq3_q6_table.py` (archive): the backbone x head x exam main table …
- `nq3_a.py` (archive): CARLA generation for P6
- `nq3_cl_report.py` (lib): closed-loop tables
- `nq3_a.sh` (archive): the whole lane as one unattended chain
- `nq3_c.sh` (archive): Night queue 3, lane C chain
- `nq3_d.sh` (archive): Q4a -> Q4b -> Q5 -> Q6 as one …

[archive/](archive/) 44 one-off code · [results/](results/) 77 result files · [figs/](figs/) 1 figures · [lib/](lib/) 6 library
<!-- files:end -->
