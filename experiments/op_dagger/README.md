# op_dagger: DAgger fine-tune on reprojection rollouts

status: live
decisions: (inputs: 92, 98, 100, 111, 118, 123, 124)
index: Reprojection rollout engine on real WOD logs + DAgger pilot (paper §3 fig 8)
key: experiments/op_dagger/plans/2026-10-06-dagger-prereg.md, experiments/op_dagger/results/pilot.md, experiments/op_dagger/scripts/dg_common.py, experiments/op_dagger/scripts/dg_roll.py, experiments/op_dagger/scripts/dg_train.py, experiments/op_dagger/scripts/dg_report.py, experiments/op_dagger/scripts/dg_clips.py

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
<!-- files:end -->
