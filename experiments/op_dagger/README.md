# op_dagger: DAgger fine-tune on reprojection rollouts

status: paused (2026-10-11 index review: stopped in research/next-round/plan.md section 5)
decisions: 132 (inputs: 92, 98, 100, 111, 118, 123, 124)
index: Reprojection rollout engine on real WOD logs + DAgger pilot (paper §3 fig 8)

**Question.** If geometric reprojection is the rollout engine, and openpilot (shipped Cinque) drives 1-2 s closed loop on real WOD logs
(action curvature -> lib/op_ctrl -> kinematic step at the logged speed -> re-projected logged frames), does fine-tuning on its own visited
states (labelled with the recovery path back to the log) shrink the closed-loop-only gaps (launch yaw loop gain, offset recovery) more than
the same states and labels with a scripted history (layer-3 static O pairs)?

**Conclusion.** See [results/pilot.md](results/pilot.md).

**Next.** See results/pilot.md.

**Read more.** [plans/2026-10-06-dagger-prereg.md](plans/2026-10-06-dagger-prereg.md) (design, readouts, lines, written before training).

Pipeline (box, op-train env): `dg_clips.py select / render train heldout sf` (CPU) -> `dg_roll.py check` (engine identity, port vs ONNX,
decision 123 gain) -> `dg_roll.py collect --model shipped --set train` -> `dg_train.py --tag dg1 --rolls shipped` / `--tag st1 --static`
-> `dg_roll.py collect --model dg1` -> `dg_train.py --tag dg2 --rolls shipped,dg1` -> `dg_roll.py eval --model <m> --set heldout`
-> `dg_report.py --models shipped dg1 st1 dg2 --control st1 --out experiments/op_dagger/results/pilot`.
Data: `$DATA_DIR/runs/op_dagger/{clips,roll,runs,check}`.

<!-- files:begin -->
## Files

- `2026-10-06-dagger-prereg.md` (plans): op_dagger 预登记：重投影 rollout 上的 DAgger …
- `pilot.md` (results): results so far
- `dg_common.py` (scripts): clip sets, logged poses, the …
- `dg_roll.py` (scripts): op_dagger rollout engine runs
- `dg_train.py` (scripts): the op_route_ft recipe without a route …
- `dg_report.py` (scripts): op_dagger readouts, exactly as defines …
- `dg_clips.py` (scripts): 20 frames on the 5 Hz lattice per clip

[results/](results/) 9 result files · [figs/](figs/) 3 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 7 entry points
<!-- files:end -->
