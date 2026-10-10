# flowhead: flow-matching plan head against regression

status: closed
decisions: 246 (inputs: 144, 148, 172, 204, 207, 240, 243)
index: Flow-matching trajectory head vs the same head as a regression on SH30 features: off-road on turns -5 pp but EPDMS -1.1, not a candidate; closed
key: experiments/flowhead/plans/2026-10-10-flow-head-prereg.md, lib/traj_head.py, experiments/op_parity/scripts/pp_train.py, experiments/flowhead/scripts/flow_train_chain.sh, experiments/flowhead/scripts/flow1.py, jevdrive/bench/navsim.py

**Question.** With everything else as the SH30 recipe (frozen Cinque encoder), does a flow-matching trajectory head in place of the regressed plan lower the off-road failures of navtest turns above 45 deg?

**Conclusion.** The flow head lowers navtest > 45 deg off-road by 4.98 pp [-6.52, -3.32] and lifts navhard (+7.6), but lowers whole-board EPDMS (-1.10), EP, LK, EC and the < 5 deg bucket (-2.33) through a shorter, wider and laterally noisier plan; by the registered no-regression rule it is not a candidate and the branch is closed (decision 246). The mean-of-modes mechanism is not supported (|C| on > 45 deg is +0.20 m higher). HUGSIM condition met but no serving path exists for a head, not built.

**Next.** None on this branch. Only if a closed-loop reading of the navhard gain is wanted: build a head serving option behind `select_4` in `jevdrive.bench` (equivalence gate first). Full read: [results/flow1.md](results/flow1.md).

**Read more.** Pre-registration (Chinese): [plans/2026-10-10-flow-head-prereg.md](plans/2026-10-10-flow-head-prereg.md).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points, `results/` small result files, `plans/` plan notes. Refresh the file list and INDEX.md with `python tools/topic_index.py`.
