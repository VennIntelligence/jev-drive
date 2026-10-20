# round2: NTV0 continuation, joint arm J vs control

status: live
decisions: none yet
index: J + ACC + GAP PAI 281 0.689 vs NTV0 0.744 (-0.056); launches 21 / 37; gates FAIL

**Question.** Does one joint continuation of NTV0-L18-s0 (SIMP-G onset rows flattened with probability 0.5 + PAI human-state imitation rows with
their time-dilated stretch variants, 32 of the 83 navtrain rows of a batch) beat the same-step continuation without changes on the PAI-FIX
native exam and the two offline gates (inertia, lead_break), and does it scale from half A to all 853 scenes?

**Conclusion.** Round 1 read (9 500 / 4 700 steps, results/): on the PAI-FIX native exam J does not beat CTRL on TARGET (0.511 vs 0.523,
-0.013 [-0.118, +0.094], zeros 22 vs 25: overshoot 9 vs 13, route-turn 0 vs 5, offroad 6 vs 3), gains on T281 (+0.152 [+0.048, +0.265],
zeros 28 vs 40) and loses G-LAUNCH (-0.330 [-0.638, -0.074], launched 3 / 6 vs 5 / 6); J-half is lower on TURN (-0.174). Offline: inertia
STOP ratio 0.753 vs 0.892 (TURN 0.942 vs 0.906; free-road slope 0.947), lead_break BRK +0.074 [+0.034, +0.123]: both registered gates FAIL.
The main session picked J for PAI 281. Stage 2 (results/pai281_R2J_ag.md): R2J + ACC + GAP on 280 clean scenes 0.689 vs stored NTV0 + ACC + GAP
0.744, -0.056 [-0.107, -0.005], zeros 64 vs 59 (route-turn 2 vs 15, but offroad 20 vs 12, AF-front 8 vs 3, zero-other 13 vs 6), launched 21 / 37
vs 33 / 37: worse by the registered stop rule, so the other stage-2 reads were not run.

**Next.** Main session: J not adopted on PAI 281; launch loss (G-LAUNCH, launched 21 / 37) is the regression to explain.

**Read more.** [plans/2026-10-20-round2-prereg.md](plans/2026-10-20-round2-prereg.md) (arms, draw shares, step rule and passes per row, gates,
read-outs, cost).

Box: run dir `$DATA_DIR/runs/round2/` (jobs.txt, stages.txt, steps.json, DONE); merged pool `$DATA_DIR/runs/op_parity/cache/on1_JTPH`
(on1_JT's five parts + on1_PAIH / on1_PAIH05 / on1_PAIH15 as parts ph / ph05 / ph15); draws `runs/defect_fix/w/R2JF.npz` (J), `R2JA.npz`
(J-half). Tags R2C-L18-s0 (CTRL), R2J-L18-s0 (J), R2JH-L18-s0 (J-half).

<!-- files:begin -->
## Files

- `2026-10-20-round2-prereg.md` (plans): ROUND2 预登记：NTV0 续训的联合臂 J（历史拉平 p = 0.5 + …
- `r2.sh` (scripts): three continuations of NTV0-L18-s0 with …
- `r2.py` (scripts): the merged pool's fix-up and checks …
- `rm_train.py` (experiments/route_mem/scripts): onpol_nuplan's on_train.py `train` (its …
- `README.md` (experiments/pai_rows): PAI human-state and frame-stretch …
- `README.md` (experiments/pai_fix): targeted PAI exam for stop and turn …
- `inertia.py` (experiments/defect_fix/scripts): does the plan's stop / turn / open-road …
- `lead_break.py` (experiments/defect_fix/scripts): why does RB0-F-s0's own plan not brake …

[results/](results/) 24 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
