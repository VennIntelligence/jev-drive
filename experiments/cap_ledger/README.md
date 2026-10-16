# cap_ledger: which capabilities each training stage lost

status: live
decisions: 331
index: P2-F lost only see-and-brake: wait, stop behind car, yield

**Question.** Besides waiting (decision 326), which driving capabilities of the shipped Cinque (P0) did the first adapter stage P2-F, or a
later stage of our lineage, train away: stopping for a red light / stop line / stopped lead, passing on green, following, free-road cruise
speed, lane change, junction turns, curve keeping, yielding to pedestrians / crossing vehicles?

**Conclusion.** On 5 731 unfiltered test-log frames in 13 behaviour cells (plus decision 326's wait / go rows), the first adapter stage P2-F
loses only the longitudinal see-and-brake family: waiting (0.978 -> 0.012), stopping behind a stopped car with a flat history (0.845 -> 0.406)
and yielding to pedestrians / crossing vehicles with a flat history (0.22 / 0.34 -> 0.06 / 0.11); RB0-F loses the same, later stages neither
repair nor add. Following, cruise speed, turns and curve keeping are gained at P2-F, green pass and red-light slowing unchanged, lane change only
slightly eroded by YR / RBSG2. Registered verdict "patch plus one" (add yield rows to the wait patch), not a base rebuild (decision 331).

**Next.** Yield rows (flat history, arc-over-human loss) into the parallel wait / stop patch; re-read this ledger on its arms.

**Read more.** [plans/2026-10-16-cap-ledger-prereg.md](plans/2026-10-16-cap-ledger-prereg.md) (Chinese, pushed before any read).
Rows: all 2 Hz frames of the 147 NAVSIM test logs (navtest filter not applied), sampled per cell, 5 731 tokens, split `navsim/capledger_test`;
box `$DATA_DIR/runs/cap_ledger/` (select, plans/H|F, report). Shared code: wait_cap's `wc.py` and long_contact's `lc_ctx.label_log`,
imported unchanged (not moved to lib/ while a parallel lane edits wait_cap).

<!-- files:begin -->
## Files

- `2026-10-16-cap-ledger-prereg.md` (plans): CAP-LEDGER …
- `led.py` (scripts): which driving capabilities of the …
- `led_chain.sh` (scripts): Box, repo root, after `led.py select`. …
- `wc.py` (experiments/wait_cap/scripts): does each stage of our lineage wait …
- `326.md` (research/decisions): WAIT-CAP：出厂 Cinque 会等（人等的帧上 …

[results/](results/) 3 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
