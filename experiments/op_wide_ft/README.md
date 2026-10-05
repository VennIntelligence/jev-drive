# op_wide_ft: fine-tune openpilot with a 116 deg wide slot

status: live
decisions: (pending) (inputs: 127, 128, 130, 133, 134, 135, 136)
index: Wide slot 58.7 -> 116 deg in fine-tuning, paired arms; B2D 25 turns primary
key: experiments/op_wide_ft/plans/2026-10-05-wide-ft-prereg.md, experiments/op_wide_ft/scripts/wide_run.py, experiments/op_wide_ft/scripts/wide_chain.sh, experiments/op_wide_ft/scripts/wide_lane.py, experiments/op_wide_ft/scripts/wide_report.py, experiments/op_wide_ft/scripts/wide_ol.py, experiments/op_wide_ft/scripts/wide_real.py, experiments/op_route_ft/scripts/rft.py, jevdrive/openpilot/frames.py

**Question.** Once fine-tuned with its wide slot widened to 116 deg (same 512x256 tensor, wide model focal 455 -> 160, horizon row fixed, road
camera unchanged), does openpilot use the wide view to turn on sharp / 90 deg B2D junction turns where the shipped model under-turns (decision 127)?

**Design.** Two arms identical except the wide FOV of the CARLA rows: the op_route_ft rc-bear-fix recipe (bear route adapter, decision-130 action
target, 4000 steps, seed 0) on the 8 607 CARLA exit-pair poses re-rendered once at the `spec` rig, with the wide frame cut from the same wide-sensor
capture at 58.7 deg (W58) and 116 deg (W116); same rows, same targets (teacher = the original on the 58.7 deg input), same seed. Real (navtrain /
WOD) rows keep 58.7 deg in both arms (their cameras have no vertical coverage for 116 deg), so W116 is a mixed-FOV model. Pre-registration with
the gate and decision lines: [plans/2026-10-05-wide-ft-prereg.md](plans/2026-10-05-wide-ft-prereg.md).

**Conclusion.** Pending.

**Read more.** [plans/2026-10-05-wide-ft-prereg.md](plans/2026-10-05-wide-ft-prereg.md), [../op_fov/README.md](../op_fov/README.md) (zero-shot
wide FOV, decision 135), [../alpamayo_turns/README.md](../alpamayo_turns/README.md) (decision 136), [../op_route_ft/README.md](../op_route_ft/README.md)
(the fine-tune recipe and the B2D turn harness).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
