# statepol: Public state-space RL policies on WOMD

status: concluded
decisions: 51
index: Only 2 BehaviorBench PPO ran: yield 43.0% vs null 2.0%
key: scripts/statepol_build.py, scripts/statepol_convert.py, scripts/statepol_eval.py, scripts/statepol_readout.py

**Question.** Can public self-play RL policies serve as a bypass/negotiation/recovery executor after openpilot desire?

**Conclusion.** Only 2 BehaviorBench weights ran: yield 43.0% (null 2.0%), aggressive PPO bypasses 79.5% with 15.8% crashes, none recover; no direct executor (decisions 51). PufferDrive v1 numbers voided; v2 only.

**Read more.** research/state-space-policies.md, `git show bcbdde4:todos/2026-09-26-state-space-policies.md`

<!-- files:begin -->
<!-- files:end -->
