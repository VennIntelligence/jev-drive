# world_model: Latent world model on openpilot + V-JEPA 2

status: concluded
decisions: 54, 60, 61, 65, 76, 123
index: W failed from action-scene confounding; WL-2 held-out no-go (H 0.38)

**Question.** Can an action-conditioned world model on frozen latents support a critic and action selection?

**Conclusion.** W failed from action-scene confounding, not the latent (decisions 54). WL: critic AUC 0.909 but lateral 61.7%, ped flips 0% (61). WL-2 held-out C3a H 0.38 [0.12, 0.60]: no-go (76).

**Read more.** research/wl2-results.md, experiments/world_model/results/wl2/, `git show bcbdde4:todos/2026-09-28-wm-loop.md`

<!-- files:begin -->
## Files

- `wl.py` (lib): openpilot proposes, a latent world …
- `wl_traj.py` (lib): NumPy only
- `wl2_model.py` (archive): the arms, the k-fold seeds, the …
- `wl2_feat.py` (archive): WL-2 feature extraction, pipelined with …
- `wl2_report.py` (archive): WL-2 readouts, exactly as registered
- `nq4_w.py` (archive): an action-conditioned latent world …
- `wl_model.py` (archive): WL world model, critics and the …
- `render_diag.py` (lib): Read-out
- `wl_pipeline.sh` (archive): WL features, training and readouts …
- `wl2_train.sh` (archive): one self-advancing chain of …
- `test_wl2.py` (archive): Unit checks of the WL-2 training and …

[archive/](archive/) 26 one-off code · [results/](results/) 48 result files · [figs/](figs/) 10 figures · [lib/](lib/) 3 library
<!-- files:end -->
