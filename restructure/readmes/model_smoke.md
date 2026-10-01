# model_smoke: Alpamayo 1.5 and openpilot smoke runs, rig robustness

status: concluded
decisions: 33, 36
headline: Smoke only: openpilot 1-3 ms/step on TensorRT; 2 deg yaw error gives 4.6x lateral error; Alpamayo B2D DS 60.8, SR 2/5

**Question.** Do Alpamayo 1.5 (10B VLA) and the three openpilot models run on our box with sane outputs and latency, and how much does openpilot degrade when the camera rig changes?

**Conclusion.** Smoke runs only, not benchmarks. openpilot small / Cinque v3 / Lebowski run on onnxruntime TensorRT EP at p50 0.99 / 2.26 / 3.33 ms per step (batch 1); Alpamayo 1.5 minADE_6@6.4 s is 0.74 m on 31 PhysicalAI-AV clips (todo smoke READMEs). Rig study on comma1M (44 synthetic input variants, Cinque native 2 s lateral / longitudinal error 0.16 / 0.86 m): other pinhole/fisheye/multi-camera rigs, resolution, blur, JPEG stay within +8%; what hurts is yaw calibration error of 2 deg (lateral x4.6), frame-time jitter (+83%) and a NAVSIM-style 2 Hz x 1.5 s time axis (lateral x9, longitudinal x25); real-data camera height does not matter (decisions 36). B2D closed-loop smoke (n = 5): Alpamayo DS 60.8, SR 2/5; the openpilot DS 2.7 was an adapter bug and is void (corrected 2026-09-25, decisions 33).

**Read more.** research/openpilot-and-open-driving-models.md, docs/zeroshot-adapters.md, `git show bcbdde4:todos/2026-09-24-alpamayo-smoke/README.md`, `git show bcbdde4:todos/2026-09-24-zeroshot-exam/openpilot-migration.md`, `git show bcbdde4:todos/2026-09-24-zeroshot-exam/bench2drive.md` (entry 33 closed-loop smoke); openpilot smoke README is kept as [results/openpilot-smoke/README.md](results/openpilot-smoke/README.md)

<!-- files:begin -->
<!-- files:end -->
