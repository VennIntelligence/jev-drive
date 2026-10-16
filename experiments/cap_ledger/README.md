# cap_ledger: which capabilities each training stage lost

status: live
decisions: pending
index: capability-loss ledger, 13 behaviour cells x 10 stages (pending)

**Question.** Besides waiting (decision 326), which driving capabilities of the shipped Cinque (P0) did the first adapter stage P2-F, or a
later stage of our lineage, train away: stopping for a red light / stop line / stopped lead, passing on green, following, free-road cruise
speed, lane change, junction turns, curve keeping, yielding to pedestrians / crossing vehicles?

**Conclusion.** Pending (prereg pushed before any read).

**Next.** Run the chain, read the ledger, write the decision entry.

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

[results/](results/) 0 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
