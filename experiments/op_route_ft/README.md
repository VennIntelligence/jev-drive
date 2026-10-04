# op_route_ft: route-choice adapter fine-tune of openpilot

status: live
decisions: (inputs: 118, 121, 122, 127)
index: Route-choice adapter (bear / poly) + action pathway fine-tune; B2D 25 junction turns primary
key: experiments/op_route_ft/plans/2026-10-05-route-ft-prereg.md, experiments/op_route_ft/scripts/rft.py, experiments/op_route_ft/scripts/rft_eval.py, lib/route_adapter.py, tests/test_route_adapter.py

**Question.** Does a lightly fine-tuned openpilot (shipped Cinque, stage 4 + plan / action pathways + a small route adapter) turn at B2D junctions by itself when it is told the route?

**Conclusion.** Pending. Open loop (results/openloop.md): the adapter is read (CARLA dev exit correct 0.58-0.59 vs shipped 0.16 and rc-ctl 0.29; drift <= 0.06 m) but misses the 0.8 line and executes CARLA N1 negatives (0.44-0.51 m vs 0.3); B2D 25 turns running.

**Next.** Pilot rc-bear (400 steps), then rc-bear / rc-poly full, rc-ctl, rc-all (exploratory); open-loop readouts, B2D 25 turns, guard subset. Pre-registered lines: [plans/2026-10-05-route-ft-prereg.md](plans/2026-10-05-route-ft-prereg.md).

**Read more.** [plans/2026-10-05-route-ft-prereg.md](plans/2026-10-05-route-ft-prereg.md) (arms, data, readouts, lines), [../op_route_cmd/README.md](../op_route_cmd/README.md) (route material),
[../op_adapt_h/README.md](../op_adapt_h/README.md) (layer-3 recipe reused), [../op_adapt_l/README.md](../op_adapt_l/README.md) (intent adapter, T4 slices), [../op_guard/](../op_guard/) (guard set).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
