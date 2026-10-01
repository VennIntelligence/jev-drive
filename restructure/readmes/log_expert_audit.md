# log_expert_audit: independent human start/stop/turn events in real logs

status: concluded
decisions: 77
headline: openpilot native plan captures start 0.58/0.73/0.64, stop 0.29/0.87/0.38 on WOD/navtrain/nuScenes; motivated op_adapt_l

**Question.** Do real driving logs hold enough independent expert start / stop / turn / nudge events to supervise them, and where does the native plan miss them?

**Conclusion.** Start, stop and turn onset each have about 0.7-3.7 thousand independent events in WOD train / navtrain; native plan capture is start 0.58 / 0.73 / 0.64 and stop 0.29 / 0.87 / 0.38 (WOD train / navtrain / nuScenes), turn onset 0.74 / 0.72 / 0.61 without navigation input; nudge and lane change only a few hundred and mixed with curves (decisions 77). The pre-registered ratio-to-control readout is invalid; capture rates are post-hoc. This motivated op_adapt_l (decisions 78).

**Read more.** research/openpilot-openloop-standing.md; pre-registration: `git show bcbdde4:todos/2026-10-01-log-expert-audit.md`

<!-- files:begin -->
<!-- files:end -->
