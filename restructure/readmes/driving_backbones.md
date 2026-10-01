# driving_backbones: openpilot / Alpamayo 1.5 as frozen backbones

status: concluded
decisions: 40
headline: openpilot temporal beats V-JEPA 2 and Qwen at WOD pre-onset (-0.294 vs -0.030); Alpamayo mid-layer different, not better

**Question.** In the P3 ladder protocol (frozen features, thin head, same subset and judge), do driving-trained representations (openpilot `temporal`, Alpamayo 1.5 VLM taps) beat generic backbones, and does it replicate on nuScenes?

**Conclusion.** openpilot `temporal` (512-d) is judged "better" for Cinque, Lebowski and small: WOD train-fit pre-onset -0.294 [-0.424, -0.168] (Cinque) / -0.318 (Lebowski) versus V-JEPA 2 -0.030 and Qwen -0.005; nuScenes replicates (-0.091 / -0.087 m, collision about halved). Alpamayo 1.5 mid-layer tap is "different, not better"; `L27_last` passes both directions but is one of 10 secondary arrays and still unreplicated (open, decisions 40). Corrected in place: the `cls_late` gap to the native plan is seed dependent (Cinque never closes it in 3 seeds, Lebowski only in seed 0), and the 6-point NAVSIM gap to TransFuser is recipe (Hydra-style sub-score heads give 84.2 PDMS), not representation.

**Read more.** research/openpilot-openloop-standing.md, research/openpilot-and-open-driving-models.md, research/feature-adapter-domain-shift.md, `git show bcbdde4:todos/2026-09-24-driving-backbones/README.md`

<!-- files:begin -->
<!-- files:end -->
