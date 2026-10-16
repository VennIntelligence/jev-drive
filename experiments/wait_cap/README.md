# wait_cap: does each training stage wait where the human waits

status: live
decisions: pending
index: open-loop wait / launch / bypass per lineage stage on test-log frames (read pending)

**Question.** On frames where the human waits (stopped car ahead, light / stop line) or stops, and contrasting frames where the human goes,
does each stage of our lineage (shipped Cinque P0 -> P2-F -> P2H10-F -> YR10m10-F / APY10m10-AB -> YS-L18 -> RB0-F -> RSG-L18 -> RBSG2-L18)
plan to wait, launch straight, or bypass laterally; which stage introduces launching, and does bypass rise with any stage?

**Conclusion.** Pending (pre-registered, no read yet).

**Next.** Rows -> index -> pp_prep (W protocol) -> route memory -> plans of every checkpoint -> report; closed-loop counterpart from stored rollouts.

**Read more.** [plans/2026-10-16-wait-cap-prereg.md](plans/2026-10-16-wait-cap-prereg.md) (Chinese, pushed before any read).
Rows: 4 674 frames of the 147 NAVSIM test logs (navtest filter not applied), split `navsim/waitcap_test@v1:a9a1dbdd6617`; box
`$DATA_DIR/runs/wait_cap/` (select, plans), token cache `runs/op_parity/cache/idx_waitcap_test[@warp]`.

<!-- files:begin -->
## Files

- `2026-10-16-wait-cap-prereg.md` (plans): WAIT-CAP 预登记：底模会不会等，哪一段训练让它起步 / …
- `wc.py` (scripts): does each stage of our lineage wait …

[plans/](plans/) 1 live plans · [scripts/](scripts/) 1 entry points
<!-- files:end -->
