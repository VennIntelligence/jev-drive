# op_resume: emulated driver resume from a held standstill

status: live
decisions: (pending) (inputs: 90, 96, 107, 113, 117, 118, 119, 124, 126)
index: Shared driver-resume rule (standstill 10 s + lead head clear) for HUGSIM stuck runs; pilot pending
key: experiments/op_resume/plans/2026-10-05-op-resume-prereg.md, jevdrive/openpilot/resume.py, jevdrive/openpilot/interface.py, tests/test_openpilot_resume.py, experiments/hugsim/lib/zs_agent.py, lib/op_arb_agent.py, experiments/op_resume/scripts/or_submit.py, experiments/op_resume/scripts/or_hugsim.sh, experiments/op_resume/scripts/or_offline.py, experiments/op_resume/scripts/or_report.py

**Question.** Does a product-faithful emulation of the driver's resume press (the car held >= 10 s, the model's own lead head
clear within 15 m; launch to 2.5 m/s, then hand back) fix HUGSIM `spec`'s runs stuck on stop plans (decisions 118 / 119) without
costing non-stuck HUGSIM scenes, B2D red-light / stop / lead-stop cases or stopped-lead collisions?

**Conclusion.** Pending (pre-registered, `plans/2026-10-05-op-resume-prereg.md`).

**Next.** Pilot (6 stuck + 2 non-stuck HUGSIM scenes, 3 B2D routes), gate, then HUGSIM 64 + B2D guard subset.

**Read more.** [plans/2026-10-05-op-resume-prereg.md](plans/2026-10-05-op-resume-prereg.md), [docs/openpilot-interface.md](../../docs/openpilot-interface.md)
(key `lon.resume`), [../hugsim/results/op_control_stack.md](../hugsim/results/op_control_stack.md) (decision 118),
[../hugsim/results/op_control_stack_long.md](../hugsim/results/op_control_stack_long.md) (decision 119).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
