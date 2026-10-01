# zeroshot_openloop: Zero-shot open-loop exams, Alpamayo and openpilot

status: concluded
decisions: 34, 37, 39
index: WOD RFS Cinque 8.005, Alpamayo 8.034, above cv 7.103
key: scripts/wod_zeroshot.py, scripts/wod_zeroshot_alpamayo.py, scripts/navsim_zs_chain.sh, scripts/navsim_zs_alpamayo.py, scripts/navsim_zs_report.py, scripts/nusc_zs.py, scripts/nusc_zs_alpamayo.py, scripts/pai_openpilot.py, jevdrive/navsim_rig.py, jevdrive/navsim_agent.py

**Question.** How do untrained Alpamayo 1.5 and openpilot score on WOD-E2E, NAVSIM, nuScenes, PhysicalAI-AV?

**Conclusion.** WOD RFS: Cinque 8.005, Alpamayo 8.034, Lebowski 7.886, cv 7.103 (decisions 34); NAVSIM PDMS 44-52 vs cv 20.7, navhard EPDMS 9-11 (decisions 37); nuScenes L2 worse than cv (decisions 39).

**Read more.** research/openpilot-openloop-standing.md, research/openpilot-openloop-integration.md, docs/navsim.md

<!-- files:begin -->
<!-- files:end -->
