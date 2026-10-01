# world_model: Latent world model on openpilot + V-JEPA 2

status: concluded
decisions: 54, 60, 61, 65, 76
index: W failed from action-scene confounding; WL-2 held-out no-go (H 0.38)
key: jevdrive/wl.py, jevdrive/wl_traj.py, jevdrive/wl2_model.py, jevdrive/wl2_feat.py, jevdrive/wl2_report.py, jevdrive/nq4_w.py, jevdrive/wl_model.py, scripts/render_diag.py, scripts/wl_pipeline.sh, scripts/wl2_train.sh, tests/test_wl2.py

**Question.** Can an action-conditioned world model on frozen latents support a critic and action selection?

**Conclusion.** W failed from action-scene confounding, not the latent (decisions 54). WL: critic AUC 0.909 but lateral 61.7%, ped flips 0% (61). WL-2 held-out C3a H 0.38 [0.12, 0.60]: no-go (76).

**Read more.** research/wl2-results.md, research/results/wl2/, `git show bcbdde4:todos/2026-09-28-wm-loop.md`

<!-- files:begin -->
<!-- files:end -->
