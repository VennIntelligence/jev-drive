# desire_cap: did adapter training kill openpilot's desire pathway

status: live
decisions: pending
index: desire pathway per stage (laneChange / turn desire), pending

**Question.** Does the shipped Cinque (P0) move its plan a lane / into a turn when fed openpilot's desire pulse (laneChangeLeft / Right,
turnLeft / Right) on our training input protocol, and which stage of our lineage (P2-F, P2H10-F, YR10m10-F, YS-L18; RB0-F, RSG-L18,
RBSG2-L18) keeps, weakens or kills that response?

**Conclusion.** Pending (prereg pushed before any read).

**Next.** Run the chain after CAP-LEDGER's row cache, read, write the decision entry and the route -> desire design.

**Read more.** [plans/2026-10-16-desire-cap-prereg.md](plans/2026-10-16-desire-cap-prereg.md) (Chinese, pushed before any read).
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

[results/](results/) 0 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
