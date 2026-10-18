# practice_pai: PAI practice pool, small-pool pilot

status: live
decisions: (pending)
index: pilot on practice-pai@v1 (= train160) queued; closed-loop gate on 4 PAI 281 chunks

**Question.** Do S rows (consequence hinges only) built from RSG-L18-s0's own PAI rollouts under the PAI probe serve, mixed into RSG's pool for a
1 000-step continuation, leave the PAI 281 closed loop no worse than the same continuation without them (practice plan item 3)?

**Conclusion.** Pending: the chain is queued (prereg's gate: zeros <= control + 2 and paired mean >= 0 on chunks c000 / c002 / c004 / c006).

**Next.** Gate PASS: the curated-train full roll (1 761 scenes, none on disk; download-bound, estimate in the prereg). FAIL: stop item 3.

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

[plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
