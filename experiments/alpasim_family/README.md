# alpasim_family: why only openpilot drivers fall on private

status: live
decisions: 263
index: reported pitch +2-3 deg reproduces YR's private drop; APY ordering fails; no code path

**Question.** Is the openpilot family's private-only loss (YR10m10-F-s0 0.925 -> 0.80, GT 0.94 -> 1.73 m; APY 0.935 -> 0.640) explained by
the one thing only our driver does: warping CAM_F0 into openpilot's camera with the calibration the session reports?

**Conclusion.** On the nuPlan track only our driver reads the camera calibration (LTF / DD / garage crop fixed pixels), and it is
sensitive to it: a reported pitch 2-3 deg further down than the rendered camera reproduces YR's private shape on 96 public scenes
(-0.148 / -0.175 score, GT x1.58 / x2.16, drift on 15 / 16 vehicles; yaw 4 deg also), while the LTF sample is untouched. But it is not
the explanation as it stands: under the same error APY loses less than YR (-0.067 against -0.148), the reverse of private, and no
shipped code path makes the report differ from the render (same pickle fields; the omitted distortion is harmless, +0.003; the
flat-pickle fallback camera collapses the driver to 0.46, not 0.80) (decision 263).

**Next.** A serving-side guard is free on public: ignoring the reported mount for a fixed nominal one scores -0.003 [-0.034, +0.028]
(96 scenes); confirming on all 1491 scenes costs about 2 card-hours, and only a warm-up pass count (32 private scenes, 2 left until
2026-11-10, the user's action) could show whether it matters on private. Open: why APY falls further than YR on private.

**Read more.** Dependence table [results/dependence.md](results/dependence.md) (task 1; also the renderer's undisclosed distortion and
the level, fixed-altitude closed-loop ego), sensitivity [results/sensitivity.md](results/sensitivity.md) (task 2), warm-up rules
[results/warmup.md](results/warmup.md) (task 3), pre-registration and amendment 1 [plans/2026-10-12-family-prereg.md](plans/2026-10-12-family-prereg.md),
render against the real frame [figs/render_vs_real.jpg](figs/render_vs_real.jpg).

<!-- files:begin -->
## Files

- `2026-10-12-family-prereg.md` (plans): FAMILY 预注册：openpilot 家族在 private …
- `dependence.md` (results): what each nuPlan-track driver reads …
- `sensitivity.md` (results): calibration and image-geometry …
- `warmup.md` (results): what the nuPlan warm-up track is
- `fam_perturb.py` (scripts): what a driver is told about CAM_F0, or …
- `fam_driver.py` (scripts): the served sh30 driver with FAM_ARM …
- `fam_ltf.py` (scripts): the shipped SimScale LTF sample with …
- `fam_report.py` (scripts): the sensitivity table from the …
- `sensitivity.csv` (results): 
- `asset_calib_public.csv` (results): 
- `fam.sh` (experiments/alpasim/scripts/drivers): the sh30 driver through, which applies …

[results/](results/) 7 result files · [figs/](figs/) 1 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 6 entry points
<!-- files:end -->

How to run an arm (box; `run.sh` driver `fam` = our sh30 driver, `FAM_FAMILY=ap2` + `AP2_TAG` for AP2-family checkpoints; `famltf` = the
LTF sample; the scene list must sit in `runs/alpasim/c0b/lists/`):

```bash
C=$DATA_DIR/runs/alpasim_family/asset_calib_public.csv; E="SH30_TAG=YR10m10-F-s0,SH30_COMPILE=0,FAM_CALIB=$C"
scripts/tmux_run.sh fam-x env OT_LANE=family OT_OUT=$DATA_DIR/runs/alpasim_family OT_OWNER=alpasim-family \
  python3 experiments/alpasim/scripts/ot2_loop.py <stage> "Pp2:fam:$E,FAM_ARM=pitch=2::fam_s96" "Lcal:famltf:FAM_ARM=pitch=2+yaw=2::fam_s96"
$DATA_DIR/third_party/alpasim/.venv/bin/python experiments/alpasim_proxy/scripts/proxy_collect.py --out <dir> "Pp2=<stage>/runs/Pp2-*/2*" ...
.venv/bin/python experiments/alpasim_family/scripts/fam_report.py <collected dirs> --out experiments/alpasim_family/results/sens   # Mac
```
