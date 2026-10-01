# op_adapt_l: Cinque adapted by imitating real-log human futures

status: live
decisions: 77, 78, 79, 80, 81
headline: Adapter lifts open-loop stop capture 0.252 to 0.559, start 0.531 to 0.632, but B2D closed loop shows no gain

**Question.** Can a lightly adapted Cinque (stage 4 unfrozen, WOD intent adapter, distillation to the original on other frames) learn start / stop / turn-onset from real-log human futures without becoming conservative, in open loop and in the B2D closed loop?

**Conclusion.** Open loop on WOD val, `main` 3 seeds: capture start 0.531 -> 0.632, stop 0.252 -> 0.559, turn onset 0.726 -> 0.843; the only failed line is stay false-start +2.7 pp (line +2 pp), and distillation weight is the single knob (dw 10 passes but start gain shrinks to +0.039) (decisions 78). The adapter reads route intent, not scene cues (decisions 78 follow-up). RFS barely moves (+0.009; perfect log imitation caps near +0.1, CI includes 0) (decisions 79). About 110 segments per class already give 82-95% of the full gain; a start gate (AUC 0.77) stays on the same frontier (decisions 80). In B2D dev closed loop there is no score gain: paired difference to the pacing control is -4.1 / -7.8 / +9.8 over three training seeds, phantom stops halve but vehicle collisions rise (decisions 81).

**Next.** Separate domain from closed loop: run the same checkpoints in HUGSIM (real-appearance closed loop) and measure open-loop capture on CARLA frames (decisions 81).

**Read more.** research/openpilot-closedloop-integration.md, plans/2026-10-01-op-adapt-L-prereg.md, plans/2026-10-01-op-adapt-L-followup.md, plans/2026-10-01-op-adapt-L-rfs-diagnosis.md, plans/2026-10-02-op-adapt-L-gate-and-curve.md, plans/2026-10-01-op-adapt-L-b2d-prereg.md; review note: `git show bcbdde4:tmp/2026-10-01-op-adapt-L-review-and-next-steps.md`

<!-- files:begin -->
<!-- files:end -->
