# desire_cap: did adapter training kill openpilot's desire pathway

status: live
decisions: 332
index: P0 laneChange 1.04 m at 4 s (partial); P2-F cuts 69 %, left 97 %

**Question.** Does the shipped Cinque (P0) move its plan a lane / into a turn when fed openpilot's desire pulse (laneChangeLeft / Right,
turnLeft / Right) on our training input protocol, and which stage of our lineage (P2-F, P2H10-F, YR10m10-F, YS-L18; RB0-F, RSG-L18,
RBSG2-L18) keeps, weakens or kills that response?

**Conclusion.** Shipped Cinque has a partial, graded laneChange pathway (open loop 1.04 m [0.92, 1.19] at 4 s, 1.54 m at 8 s, direction
100 %); the first adapter stage P2-F cuts it 69 % and the left lane change 97 % (P2H10-F 0.28 crosses the kill line), YR10m10-F onward recover
55-66 % on the right only; turn desire is no turn pathway at junctions (decision 332).

**Next.** Design for the user, not launched: [plans/2026-10-16-desire-route-design.md](plans/2026-10-16-desire-route-design.md) (route-derived
lane-change desire in the driver, desire-delta distillation in training; first step = driver-only desire on PAI, no training).

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
