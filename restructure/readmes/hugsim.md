# hugsim: HUGSIM install, controller acceptance, zero-shot exam, I3 pairs

status: concluded
decisions: 19, 44
headline: HUGSIM fixed2 controller passes acceptance, upstream trackers fail; openpilot 4 Hz clock costs +25-38% lateral error

**Question.** Can HUGSIM (3DGS-rendered closed loop) serve as the real-appearance control column, and how do Alpamayo 1.5 / openpilot score zero-shot on it, with the controller fixed first?

**Conclusion.** Entry 19 made HUGSIM the missing third column (NAVSIM + HUGSIM + Bench2Drive; confirmed as plan in decisions 21), since a HUGSIM-vs-Bench2Drive comparison is the cleanest real-vs-sim pair. Controller acceptance (docs/hugsim.md): upstream and PR #57 trackers fail pre-registered thresholds, `fixed2` (PR #57 + `lqr-tracker-v2.patch`) passes (lateral @0.5 s 0.016 / 0.26 m median / p95). Zero-shot exam: openpilot 4 Hz dilated clock costs +25-38% 2 s lateral error; scored runs for 64 scenarios exist ([results/hugsim-exam/](results/hugsim-exam/) `scored_op.csv`, `scored_base.csv`) but no decision entry summarises HD-Scores, and the exam README stops at the adapter checklist. I3 (65 HUGSIM 3DGS vehicle-pair scenes) is a pair set for the reaction-head work: retraining on it removes most of the harm there but no head beats the prior, and it does not transfer to WOD / NAVSIM (decisions 44).

**Read more.** docs/hugsim.md, docs/zeroshot-adapters.md, research/ablation-matrix-inventory.md (section 5, I3), `git show bcbdde4:todos/2026-09-25-hugsim-exam/README.md`, `git show bcbdde4:todos/2026-09-24-zeroshot-exam/hugsim.md`, `git show bcbdde4:todos/2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md`

<!-- files:begin -->
<!-- files:end -->
