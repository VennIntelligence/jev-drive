# desire_cap: did adapter training kill openpilot's desire pathway

status: closed
decisions: 332, 334
index: P0 laneChange 1.04 m at 4 s (partial); P2-F cuts 69 %, left 97 %

**Question.** Does the shipped Cinque (P0) move its plan a lane / into a turn when fed openpilot's desire pulse (laneChangeLeft / Right,
turnLeft / Right) on our training input protocol, and which stage of our lineage (P2-F, P2H10-F, YR10m10-F, YS-L18; RB0-F, RSG-L18,
RBSG2-L18) keeps, weakens or kills that response?

**Conclusion.** Shipped Cinque has a partial, graded laneChange pathway (open loop 1.04 m [0.92, 1.19] at 4 s, 1.54 m at 8 s, direction
100 %); the first adapter stage P2-F cuts it 69 % and the left lane change 97 % (P2H10-F 0.28 crosses the kill line), YR10m10-F onward recover
55-66 % on the right only; turn desire is no turn pathway at junctions (decision 332).

**D0 (decision 334).** The route-derived laneChange desire fed only on the driver side (lib/route_desire.py, `JEV_DESIRE=1`; RSG-L18-s0,
PAI submission config) does not carry lane changes (map lane-change sites 7 vs 6 of 31, right 1 / 12 on both arms) and costs safety (PAI 281
at-fault collision zeros 12 vs 9, side contacts 7 vs 4; train160 rule scenes -0.111 [-0.212, -0.021]); it does push the car back to the
route's lane (+0.22 rule-level done share, both sides). Open loop below 8.9 m/s RSG-L18 has no pathway in the registered sense (right 0.72 m,
direction 0.8; left 0.15 m); P0 keeps its left one. Line stopped; training (D1 / D2) is the remaining route, gate 8.9 m/s, D2 on both sides.
Prereg [plans/2026-10-17-desire-d0-prereg.md](plans/2026-10-17-desire-d0-prereg.md), results [results/d0/](results/d0/d0.md), scripts `dd.py`, `dd_chain.sh`.

**Read more.** [plans/2026-10-16-desire-cap-prereg.md](plans/2026-10-16-desire-cap-prereg.md) (Chinese, pushed before any read); results/desire_cap.md (every table).
Rows: CAP-LEDGER's NAVSIM test-log cells (`navsim/capledger_test`: C cruise, T turns, LC lane changes) and PAI on-policy rows (`on1_PAIonp`);
box `$DATA_DIR/runs/desire_cap/`. The desire pulse is placed in the policy's desire queue (`_to_copy`, 33 x 0.2 s blocks) by wrapping
`jevdrive.op_adapt.policy_feeds`; nothing shared is edited.

<!-- files:begin -->
## Files

- `2026-10-16-desire-cap-prereg.md` (plans): DESIRE-CAP 预登记：底模有没有 desire 通路，我们的 …
- `dc.py` (scripts): does the shipped Cinque have a desire …
- `dc_chain.sh` (scripts): Box, repo root
- `led.py` (experiments/cap_ledger/scripts): which driving capabilities of the …
- `wc.py` (experiments/wait_cap/scripts): does each stage of our lineage wait …
- `op_adapt.py` (jevdrive): adapting openpilot's vision layer
- `pp_train.py` (experiments/op_parity/scripts): Cinque fine-tuned on navtrain with …

[results/](results/) 2 result files · [plans/](plans/) 2 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
