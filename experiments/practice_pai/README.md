# practice_pai: PAI practice pool, small-pool pilot

status: concluded
decisions: 367
index: gate FAIL: pilot - control -0.010 [-0.053, +0.033], zeros 45 vs 44

**Question.** Do S rows (consequence hinges only) built from RSG-L18-s0's own PAI rollouts under the PAI probe serve, mixed into RSG's pool for a
1 000-step continuation, leave the PAI 281 closed loop no worse than the same continuation without them (practice plan item 3)?

**Conclusion.** Gate failed (decision 367): on PAI 281 chunks c000 / c002 / c004 / c006 (144 scenes) the pilot (RSG pool + 25 634 PAI S rows from 160 scenes, half the draws) scores 0.661 against the control's 0.671, paired -0.010 [-0.053, +0.033], zeros 45 vs 44; against the stored RSG + ACC + GAP read -0.015 / -0.005. Weak evidence: 1 seed, 1 000 steps (~0.25 passes over the PAI rows), the old train160 scenes.

**Next.** Stopped as registered; no full roll (it is download-bound anyway: 1 761 scenes, 3.2 TB, none on disk). Results: [results/gate.md](results/gate.md), [results/pai281_4ch.md](results/pai281_4ch.md).

**Read more.** [plans/2026-10-19-pai-practice-prereg.md](plans/2026-10-19-pai-practice-prereg.md) (scene choice, recipe, gate, full-roll estimate).

<!-- files:begin -->
## Files

- `2026-10-19-pai-practice-prereg.md` (plans): PRACTICE-PAI 预登记：PAI 专用练习池，小池 pilot + …
- `pp.sh` (scripts): RSG-L18-s0 rolled on the practice-pai …
- `pp.py` (scripts): split registration, the pool's …
- `rf_pai_pool.py` (experiments/route_follow/scripts): Route-night section 9 step 5
- `rf_pait.py` (experiments/route_follow/scripts): Turning school, demonstration source 1
- `pai281.py` (experiments/defect_fix/scripts): Paired PAI 281 read-out of one …
- `practice-plan.md` (research/next-round): 练习计划：按「可归因的试次」改训练管线（2026-10-19，工作方案）

[results/](results/) 7 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
