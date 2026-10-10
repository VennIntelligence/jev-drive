# flowhead: flow-matching plan head against regression

status: live
decisions: (pending) (inputs: 144, 148, 172, 204, 207, 240, 243)
index: Flow-matching trajectory head on SH30's hidden state vs the same head as a regression; running
key: experiments/flowhead/plans/2026-10-10-flow-head-prereg.md, lib/traj_head.py, experiments/op_parity/scripts/pp_train.py, experiments/flowhead/scripts/flow_train_chain.sh, experiments/flowhead/scripts/flow1.py, jevdrive/bench/navsim.py

**Question.** With everything else as the SH30 recipe (frozen Cinque encoder), does a flow-matching trajectory head in place of the regressed plan lower the off-road failures of navtest turns above 45 deg?

**Conclusion.** None yet (lane FLOW1 running).

**Next.** Train `FMH-F-s{0,1}` (flow) and `RGH-F-s{0,1}` (the same head as a regression), read navtest / navhard, the turn failure classes, the equal-arc curve offset and the sample diversity against the registered lines.

**Read more.** Pre-registration (Chinese): [plans/2026-10-10-flow-head-prereg.md](plans/2026-10-10-flow-head-prereg.md).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points, `results/` small result files, `plans/` plan notes. Refresh the file list and INDEX.md with `python tools/topic_index.py`.
