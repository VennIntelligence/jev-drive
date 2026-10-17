# wait_cap: does each training stage wait where the human waits

status: live
decisions: 326, 335, 336, 337, 339
index: shipped Cinque waits 0.978, P2-F onward 0.012 [d326]; WAIT-TRAIN pilot STOP on dev ADE, the rise is a real closed-loop cost (held -0.112, offroad x2), no closed-loop waiting [d335]; P0 waits in closed loop, J holds a stop but cannot make one on arrival [d336]; RTB1 retain base keeps the arrival stop at a cost [d337], V2 (rolling-arrival rows only) fixes the public cost, the left cost is lateral [d339]

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

**RTB1 retain line (decisions 337, 339, exploratory).** The RB0-F base plus a P0 longitudinal-only KD on arrival rows, continued with the RSG-L18 recipe,
stops in the static scene without contact (0.53-0.63 m, RSG hits the lead) but pays held zeros +4 to +7, at-fault +5 and public rear contacts 24 vs 18
([results/next/rtb_cont.md](results/next/rtb_cont.md), d337). Variant round ([results/next/rtb_v.md](results/next/rtb_v.md), d339): dropping the standing-wait rows (V2)
keeps the stop (static 0.86 m, stop17 15 / 17), brings public 1491 to -0.002 [-0.011, 0.007] vs RSG with rear contacts 17, and held to +0.007 [-0.005, 0.025];
halving lambda (V1) does not help. What is left is lateral (public offroad 36 vs 31). Start-delay diagnosis: [results/next/explore_f.md](results/next/explore_f.md).
V2 PAI 281 in the submission config (one seed; the second base seed was cancelled by the user): 79 zeros / 12 at-fault, -0.007 [-0.038, 0.023] vs RSG, stop17 15 / 17,
none of the flags fixed before the read (prereg amendment 3) crossed ([results/next/rtb_v2_pai.md](results/next/rtb_v2_pai.md)).

**Explore G (turn tracking and lane lines, exploratory, no training).** The retain line (RTB1 / V2) does not track turns measurably worse than RSG-L18. On identical
inputs it plans the same turns: logged dev rows +0.01 m at 10 m; replayed closed-loop states a 1.4–1.7 % under-turn, 2–3 % in large narrow bends, none in the seed-1 pair.
Its entry speed equals RSG's, and paired hard-failure flips in bends are balanced (held 11 / 6, 8 / 7, 9 / 6; public 11 / 10, 9 / 9). The remaining corner-cutting /
opposite-lane failures are shared with RSG, so no lateral work this round. The shipped lane lines read 0.2–0.3 m on continuous straight roads; at junctions and on irregular
roads they are 1–6 m off. [results/next/explore_g.md](results/next/explore_g.md).

**Red-light and rear-end census (handoff 2026-10-17 step 0, CPU).** [results/next/census_red_rear.md](results/next/census_red_rear.md): the scorer has no
red-light term, a rear contact is never at fault and only freezes progress, and a car that moves ~3 m where the human stands still leaves the corridor.
The MTGS renders keep the lamps red; RSG / V2 launch with the human at red lights as elsewhere. 24 % of d325's uniform windows are red-light waits
(public 0.4 %); a log replay of an "ignore red" car scores 0.03 there, so the switch would cost about 0.11 per scene on a private-like mix. Rear
contacts (public 18 / 17 / 24 for RSG / V2 / a) are mostly the ego slower than the log while moving.

**Prepared, not launched (2026-10-17 night).** Follow-up pilots for the morning choice between decision 335's options (iii) and (iv), re-aimed
by decision 336; draft lines in [plans/2026-10-17-retain-anticopy-draft.md](plans/2026-10-17-retain-anticopy-draft.md) (Chinese); chain `scripts/nx.sh`, reads and lines `scripts/nx.py`:
- (iii) RETAIN, favoured: distil shipped Cinque's speed profile only (arc length, the lateral path untouched) on rolling-arrival / standing-wait
  rows (on1_WAIT) and navtrain approach rows fed a flat history, + D2 desire distillation, lambda 1 / 3; (iii-a) a 1 000-step patch on the
  RSG-L18 recipe (`nx.sh rta-pilot`, ~0.6 card-h), (iii-b) the RB0-F base rebuilt (`nx.sh rtb-pilot`, ~1.6 card-h); arrival-replay line;
- (iv) ANTI-COPY, lower priority: the RSG-L18 recipe + speed-history scale noise on the imitation rows, 1 000 steps (`nx.sh ac-pilot`).

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
