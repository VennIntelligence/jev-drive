# wait_cap: does each training stage wait where the human waits

status: live
decisions: 326
index: shipped Cinque waits 0.978, P2-F onward 0.012; no stage bypasses (<= 0.004)

**Question.** On frames where the human waits (stopped car ahead, light / stop line) or stops, and contrasting frames where the human goes,
does each stage of our lineage (shipped Cinque P0 -> P2-F -> P2H10-F -> YR10m10-F / APY10m10-AB -> YS-L18 -> RB0-F -> RSG-L18 -> RBSG2-L18)
plan to wait, launch straight, or bypass laterally; which stage introduces launching, and does bypass rise with any stage?

**Conclusion.** On 4 674 unfiltered frames of the 147 NAVSIM test logs, shipped Cinque (P0) waits where the human waits (0.978) and mostly
goes where the human goes (waits 0.334, discrimination 0.64); the first adapter stage P2-F turns waiting into launching (0.012, -0.966
[-0.978, -0.950]) and no later stage changes it (all 0.005-0.012). No stage bypasses laterally (stopped-car / queue frames <= 0.004, never
above P0): the launch goes straight along the lane, 30-50 % of plans reach the stopped car. Closed loop, the one human-static public scene:
all 14 of our labels drive into the car, their lateral offset appears only after contact (decision 326).

**Next.** Training design for the user to critique, not launched: [plans/2026-10-16-wait-train-design.md](plans/2026-10-16-wait-train-design.md)
(same-domain wait / launch / stop rows from a trainval camera download, arc-length-only loss, one joint arm + weight-0 control, 1 000-step dose pilot, ~22 card-h).

**Read more.** [plans/2026-10-16-wait-cap-prereg.md](plans/2026-10-16-wait-cap-prereg.md) (Chinese, pushed before any read); results/lines.md (registered lines), results/wait_cap.md (every arm and cell), results/closed.md (stored rollouts by stratum).
Rows: 4 674 frames of the 147 NAVSIM test logs (navtest filter not applied), split `navsim/waitcap_test@v1:a9a1dbdd6617`; box
`$DATA_DIR/runs/wait_cap/` (select, plans), token cache `runs/op_parity/cache/idx_waitcap_test[@warp]`.

<!-- files:begin -->
## Files

- `2026-10-16-wait-cap-prereg.md` (plans): WAIT-CAP 预登记：底模会不会等，哪一段训练让它起步 / …
- `wc.py` (scripts): does each stage of our lineage wait …

[results/](results/) 6 result files · [plans/](plans/) 2 live plans · [scripts/](scripts/) 3 entry points
<!-- files:end -->
