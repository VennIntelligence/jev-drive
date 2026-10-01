# hugsim: HUGSIM install, controller, zero-shot exam, I3

status: concluded
decisions: 19, 44
index: fixed2 controller passes acceptance; 4 Hz clock +25-38% lateral
key: scripts/hugsim/install.sh, scripts/hugsim/zs_exam.sh, scripts/hugsim/zs_run.py, scripts/hugsim/zs_agent.py, scripts/hugsim/preset_eval.py, scripts/hugsim/pairs_run.sh, jevdrive/hugsim_zs.py, jevdrive/hugsim_pairs.py, jevdrive/hugsim_preset.py

**Question.** Can HUGSIM be the real-appearance closed-loop column, and how do Alpamayo and openpilot score on it?

**Conclusion.** Upstream trackers fail acceptance, `fixed2` passes; openpilot 4 Hz clock +25-38% lateral error; 64 scenarios scored, no HD-Score entry (decisions 19). I3 pairs: no head beats the prior (decisions 44).

**Read more.** docs/hugsim.md, `git show bcbdde4:todos/2026-09-25-hugsim-exam/README.md`

<!-- files:begin -->
<!-- files:end -->
