# long_contact: is the plan's longitudinal consequence readable

status: live
decisions: pending (inputs: 196, 203, 218, 219, 220, 223, 237, 238, 241, 252)
index: LONG-PROBE: A / B / C consequence of the own plan's speed profile, S0 head + input ladder (read pending)
key: experiments/long_contact/plans/2026-10-11-long-probe-prereg.md, experiments/long_contact/results/ladder.md, experiments/long_contact/lib/lc.py, experiments/long_contact/lib/lc_head.py, experiments/long_contact/scripts/lc_ctx.py, experiments/long_contact/scripts/lc_rows.py, experiments/long_contact/scripts/lc_native.py, experiments/long_contact/scripts/lc_train.py, experiments/long_contact/scripts/lc_read.py, experiments/long_contact/scripts/lc_ood.py, experiments/long_contact/scripts/lc_pai.py, experiments/long_contact/scripts/lc_chain.sh

**Question.** Is the longitudinal consequence of the student's own plan, (A) reaching a stopped or slow vehicle ahead in its lane, (B) passing
a mandatory stop point while the stop is mandatory, (C) entering a > 45 deg turn faster than the logged driver, readable from what the
driver has at inference time, with a read directly comparable to BODY1 arm S0's lateral one (decision 223)?

**Conclusion.** Pending (the registered read runs in `scripts/lc_chain.sh`).

**Next.** The registered read (10 full trainings, 5 rungs x 2 seeds), then the out-of-distribution rows.

**Read more.** Pre-registration [plans/2026-10-11-long-probe-prereg.md](plans/2026-10-11-long-probe-prereg.md) (labels, ladder, populations,
registered lines, pilot checklist).

<!-- files:begin -->
<!-- files:end -->

Layout: `lib/` labels and route geometry (`lc.py`) and the ladder head (`lc_head.py`, BODY1's `contact_head` modules imported unchanged);
`scripts/` the context labeler (`lc_ctx.py`: hindsight route, stop lines with light states, turns from the nuPlan logs and maps), the row
labels and inputs (`lc_rows.py`, `lc_native.py`), the trainer (`lc_train.py`), the reads (`lc_read.py`, `lc_ood.py`, `lc_pai.py`), the
label figure (`lc_fig.py`) and the chain (`lc_chain.sh`); `results/` tables; `figs/` figures. Box: `$DATA_DIR/runs/long_contact/`.
