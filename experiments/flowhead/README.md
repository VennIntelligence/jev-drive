# flowhead: flow-matching plan head against regression

status: closed
decisions: 246 (inputs: 144, 148, 172, 204, 207, 240, 243)
index: Flow-matching trajectory head vs the same head as a regression on SH30 features: off-road on turns -5 pp but EPDMS -1.1, not a candidate; closed

**Question.** With everything else as the SH30 recipe (frozen Cinque encoder), does a flow-matching trajectory head in place of the regressed plan lower the off-road failures of navtest turns above 45 deg?

**Conclusion.** The flow head lowers navtest > 45 deg off-road by 4.98 pp [-6.52, -3.32] and lifts navhard (+7.6), but lowers whole-board EPDMS (-1.10), EP, LK, EC and the < 5 deg bucket (-2.33) through a shorter, wider and laterally noisier plan; by the registered no-regression rule it is not a candidate and the branch is closed (decision 246). The mean-of-modes mechanism is not supported (|C| on > 45 deg is +0.20 m higher). HUGSIM condition met but no serving path exists for a head, not built.

**Closed loop (2026-10-15, mode-averaging Reading 2).** The AlpaSim sh30 driver now serves `thead.pt` the way bench serves it on
navtest (tap `select_4`; the flow head uses fixed noise row 0 and logs its samples from rows 0-7 as `th_k`). The served poses match bench's
navtest head poses: mean max difference 0.003-0.004 m on 300 tokens per tag. On held 460 at 18 decisions **the flow head fails**:
- **F1 refuted.** It completes 0.5 of the 24 strict held turns against its regression twin's 11 (-10.5 [-16.5, -5.5]; FMH-F-s0 / s1 1 and 0,
  RGH-F 10 and 12, RSG-L18 16 and 14).
- **F2 refuted.** Its approach deficit is larger: +0.27 vs +0.09, FM - RG +0.18 [+0.07, +0.26].
- **F3.** Mean score is 0.462 vs 0.694 (-0.233 [-0.276, -0.185]), zeros 229.5 vs 119.5 (+110 [+90, +128], mostly corridor exits:
  148.5 vs 63.5), at-fault +30.5 [+14.5, +45.5].

The regression twin itself sits far below RSG-L18 (0.694 vs 0.882). At the failing decisions (served deficit >= 0.5), the full turn
(>= 0.6 of the human's heading) is among the 8 samples on 25 % of decisions ("partial" by the registered rule; 49 / 65 have none). Over
all failed-turn window decisions it is on 72 %. So the mode is often present but not selected, and absent where the plan fails most. On
straight decisions every sample sits right of the lane (4 s lateral -0.14 to -0.91 m). The served row 0 is the most biased: -0.91 m, 0.43 m
from the samples' median. That fits d246's laterally noisy plan and the corridor zeros. Full read:
[results/closed_loop/cl.md](results/closed_loop/cl.md); held table [results/closed_loop/held18.md](results/closed_loop/held18.md).
Look at [results/closed_loop/cl_samples.png](results/closed_loop/cl_samples.png). Left: how many of the 8 samples ask for the full turn;
the failing decisions pile at 0, the other failed-turn decisions at 8. Right: the served row's lateral offset against the other rows.
Chain: `scripts/flow_cl.sh`; analysis `experiments/route_follow/scripts/rf_modeavg.py flow`; reads under `$DATA_DIR/runs/route_mem/read/{FMH,RGH}-F-s{0,1}`,
report and extracts under `$DATA_DIR/runs/flowhead/closed_loop/`.

**Next.** None on this branch. Only if a closed-loop reading of the navhard gain is wanted: build a head serving option behind `select_4` in `jevdrive.bench` (equivalence gate first). Full read: [results/flow1.md](results/flow1.md).

**Read more.** Pre-registration (Chinese): [plans/2026-10-10-flow-head-prereg.md](plans/2026-10-10-flow-head-prereg.md).

<!-- files:begin -->
## Files


[results/](results/) 10 result files · [plans/](plans/) 2 live plans · [scripts/](scripts/) 4 entry points
<!-- files:end -->

Layout: `scripts/` entry points, `results/` small result files, `plans/` plan notes. Refresh the file list and INDEX.md with `python tools/topic_index.py`.
