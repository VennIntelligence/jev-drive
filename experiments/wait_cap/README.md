# wait_cap: does each training stage wait where the human waits

status: live
decisions: 326, 335, 336
index: shipped Cinque waits 0.978, P2-F onward 0.012 [d326]; WAIT-TRAIN pilot STOP on dev ADE, the rise is a real closed-loop cost (held -0.112, offroad x2), no closed-loop waiting [d335]; P0 waits in closed loop, J holds a stop but cannot make one on arrival [d336]

**Question.** On frames where the human waits (stopped car ahead, light / stop line) or stops, and contrasting frames where the human goes,
does each stage of our lineage (shipped Cinque P0 -> P2-F -> P2H10-F -> YR10m10-F / APY10m10-AB -> YS-L18 -> RB0-F -> RSG-L18 -> RBSG2-L18)
plan to wait, launch straight, or bypass laterally; which stage introduces launching, and does bypass rise with any stage?

**Conclusion.** On 4 674 unfiltered frames of the 147 NAVSIM test logs, shipped Cinque (P0) waits where the human waits (0.978) and mostly
goes where the human goes (waits 0.334, discrimination 0.64); the first adapter stage P2-F turns waiting into launching (0.012, -0.966
[-0.978, -0.950]) and no later stage changes it (all 0.005-0.012). No stage bypasses laterally (stopped-car / queue frames <= 0.004, never
above P0): the launch goes straight along the lane, 30-50 % of plans reach the stopped car. Closed loop, the one human-static public scene:
all 14 of our labels drive into the car, their lateral offset appears only after contact (decision 326).

**WAIT-TRAIN (decision 335).** Same-domain wait / launch / stop / yield rows with an arc-length-only loss on the RSG-L18 recipe. The
1 000-step dose pilot stopped at the registered gate on dev ADE alone (J2 0.608, J5 0.762 vs C 0.529 + 0.01), while waiting rose from
0.007 to 0.86 / 0.91 ([results/wait_train/pilot_gate.md](results/wait_train/pilot_gate.md)). The exploratory reads after the stop
([results/wait_train/explore_d.md](results/wait_train/explore_d.md), prereg 补记 D) show the rise is real:
- the plans are shorter along the path on ordinary low-speed free-road rows: 95 % of the rise is along-track, 4 s bias -0.49 vs -0.03 m;
- on held 460 @18 J2 scores -0.112 [-0.146, -0.072], with 53 vs 41 zeros, 28 vs 14 offroad and 11 % less distance;
- the static public scene and PAI stop17 do not change.

No formal was run.

**补记 E (decision 336, diagnostics).** The closed loop can show waiting: shipped Cinque stops in the static scene (0.48 m, the human
0.50 m), and every adapter stage, the WAIT pilots included, drives into the lead. Replayed on P0's own standing inputs, J2P / J5P wait
(4 s plans of 0.1-0.4 m). On arrival, though, they plan 2.0-2.3 m where P0 plans 0.43 m, so the car never reaches standstill.
Zeroing the adapter's speed history does not help. J2P's offroad doubling is mostly a +0.08 m closed-loop left drift on edge-hugging
paths. Details: [results/wait_train/explore_e.md](results/wait_train/explore_e.md).

**Prepared, not launched (2026-10-17 night).** Two follow-up pilots for the morning choice between decision 335's options (iii) and (iv),
draft lines in [plans/2026-10-17-retain-anticopy-draft.md](plans/2026-10-17-retain-anticopy-draft.md) (Chinese); chain `scripts/nx.sh`, reads and lines `scripts/nx.py`:
- (iii) RETAIN: the RB0-F base stage + LwF-style distillation toward shipped Cinque's plan on the on1_WAIT rows + D2 desire distillation, lambda 1 / 3 (`nx.sh rt-pilot`);
- (iv) ANTI-COPY: the RSG-L18 recipe + speed-history scale noise on the imitation rows, 1 000 steps (`nx.sh ac-pilot`).

**Next (before WAIT-TRAIN).** Training design for the user to critique, not launched: [plans/2026-10-16-wait-train-design.md](plans/2026-10-16-wait-train-design.md)
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
