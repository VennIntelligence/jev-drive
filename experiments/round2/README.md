# round2: NTV0 continuation, joint arm J vs control

status: live
decisions: none yet
index: registered: CTRL / J / J-half from NTV0-L18-s0, PAI human rows + onset flattening

**Question.** Does one joint continuation of NTV0-L18-s0 (SIMP-G onset rows flattened with probability 0.5 + PAI human-state imitation rows with
their time-dilated stretch variants, 32 of the 83 navtrain rows of a batch) beat the same-step continuation without changes on the PAI-FIX
native exam and the two offline gates (inertia, lead_break), and does it scale from half A to all 853 scenes?

**Conclusion.** Not yet: registered, identity / smoke first, training gated on the PAI-ROWS H pools.

**Next.** `r2.sh pre` (identity, smoke) -> `r2.sh launch` (queued on PAI-ROWS H/DONE) -> results/.

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

[plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
