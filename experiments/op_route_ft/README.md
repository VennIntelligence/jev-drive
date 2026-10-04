# op_route_ft: route-choice adapter fine-tune of openpilot

status: live
decisions: (inputs: 118, 121, 122, 127)
index: Route-choice adapter (bear / poly) + action pathway fine-tune; B2D 25 junction turns primary
key: experiments/op_route_ft/plans/2026-10-05-route-ft-prereg.md, experiments/op_route_ft/scripts/rft.py, experiments/op_route_ft/scripts/rft_eval.py, lib/route_adapter.py, tests/test_route_adapter.py

**Question.** Does a lightly fine-tuned openpilot (shipped Cinque, stage 4 + plan / action pathways + a small route adapter) turn at B2D junctions by itself when it is told the route?

**Conclusion.** The route-choice fine-tune does not reach the primary line: on the 25 B2D junction turns (zones off) rc-bear takes 4 / 25 (shipped 1, rc-ctl 6, rc-poly 7, rc-all 6; line >= 13 and above rc-ctl), and the adapter adds nothing over the fine-tune itself in closed loop (rc-bear minus rc-ctl -0.08 [-0.28, +0.15]). The adapter is read open loop (CARLA exits correct 0.58 vs 0.16 shipped, 0.29 rc-ctl; line 0.8 fails) and was fed in the closed loop (checked); the fine-tuned heads steer 1.2-3.4 x the needed curvature in the right direction but mostly after the turn start (late), with every arm crawling at about 1 m/s and stopped 40% of the window. Guard subset: rc-bear fails navhard (-0.43) and early-turn; rc-all fails early-turn, one HUGSIM spin and window collisions (8 vs 5), navhard DAC flat, EP +3.9 pp. Single seed. Full tables, failure split, doubts: [results/report.md](results/report.md).

**Next.** Not started (no new arms tonight): an action-supervision ablation at approach poses (the likely gap between plan and steering), a no-desire closed-loop arm to isolate the adapter, and the cause of the 40% stopped share in the harness.

**Read more.** [plans/2026-10-05-route-ft-prereg.md](plans/2026-10-05-route-ft-prereg.md) (arms, data, readouts, lines), [../op_route_cmd/README.md](../op_route_cmd/README.md) (route material),
[../op_adapt_h/README.md](../op_adapt_h/README.md) (layer-3 recipe reused), [../op_adapt_l/README.md](../op_adapt_l/README.md) (intent adapter, T4 slices), [../op_guard/](../op_guard/) (guard set).

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
