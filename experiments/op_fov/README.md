# op_fov: wider field of view for openpilot, zero-shot

status: concluded
decisions: (pending)
index: Wide FOV 90/116 deg: turn gain -0.005..-0.033, early stop; wide frame barely used
key: experiments/op_fov/README.md experiments/op_fov/results/pilot.md experiments/op_fov/figs/fov_inputs.png experiments/op_fov/plans/2026-10-05-fov-prereg.md experiments/op_fov/scripts/fov_replay.py experiments/op_fov/scripts/fov_report.py jevdrive/openpilot/frames.py

**Question.** Does feeding the shipped Cinque a wider field of view (smaller model-frame focal, same 512x256 frames) help it on
sharp / 90 deg turns on real road data, without breaking its scale on straight roads?

**Conclusion.** No. On 8 real comma1M turns (pilot, pre-registered early stop) wide 90 / 116 deg / horizontal-only 90 deg change the
action-curvature gain by -0.005 to -0.033 and its timing by -0.15 to -0.23 s; scale guardrails hold. Freezing the wide frame changes
the turn as little (-0.035): shipped Cinque barely uses wide content in turns. A wider road frame (39 deg) breaks scale (plan speed
x1.22, ADE x8.7). With its own camera the model already turns about right on real turns (A_act 0.89, A_plan 1.04). (decision pending)

**Read more.** Results [results/pilot.md](results/pilot.md). Pre-registration: [plans/2026-10-05-fov-prereg.md](plans/2026-10-05-fov-prereg.md).

<!-- files:begin -->
<!-- files:end -->
