# night_queue_2: P6 behaviour exam, openpilot bypass/desire, heads, backbones

status: concluded
decisions: 47, 48, 49, 50, 52, 53
headline: P6 v0 holds (PDM-Lite bypasses 9 classes at 100%) but placement null misses gate; V-JEPA 2 flips pedestrians 47%

**Question.** Night queue 2 (N1-N6): build the post-judgement behaviour exam (bypass, negotiation), test whether openpilot carries bypass information and can execute it, and check the leaderboard head, detector input and backbones on the paired exam.

**Conclusion.** N1: P6 v0 holds (PDM-Lite bypasses 9 obstacle classes at 100%, 0% after deleting the registration; negotiation 65% wait first), but the placement null misses its gate (0.889 vs 0.90) (decisions 52). N2: openpilot features read static obstacle AUC 0.972-0.992, but desire pulses give only 0.61-0.68 m lateral shift at 5-10 m/s (decisions 49). N3: the Hydra head is incomparable in CARLA and adding the gated CARLA delta costs 5-8 NAVSIM PDMS (decisions 53). N4: image-plane detection tokens match lifted ones (decisions 50). N6: V-JEPA 2 47.0% and Qwen 41.5% pedestrian flips, SigLIP2 33.8%, DINOv2 7.1%, openpilot small 3.0% (decisions 48). Motivation and instrument survey: decisions 47.

**Read more.** research/behavior-layer-instruments.md, research/ablation-matrix-inventory.md, research/nohack-mechanisms.md, `git show bcbdde4:todos/2026-09-26-night-queue-2.md`

<!-- files:begin -->
<!-- files:end -->
