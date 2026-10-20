# round3: remove round 2's launch loss, keep its turn fix

status: live
decisions: 414 (round 3); round 3b not yet recorded
index: offline: J's launch loss tracks PAI standing rows (2.5 / batch vs 0.16), not the flattening

**Question.** Is round 2's launch loss (R2J: PAI 281 launches 33 / 37 -> 21 / 37, PAI-FIX G-LAUNCH 0.865 -> 0.535) caused by the PAI human rows whose
4 s label stands throughout (H1), by the onset history flattening (H2) or by the stretch rows at standstill (H3), and does removing the cause
keep the route-turn fix (PAI 281 route-turn zeros 15 -> 2)?

**Conclusion.** Offline, before any training (results/rows_state.md, results/launch_plans.md): J's batch carries 2.54 PAI rows standing
through their label (navtrain has 22 of 103 288, the S pool 0.14 a batch), so "standstill and stay" goes from 1.1 % to 18.5 % of the standstill
rows (stay + launch); at the human's launch J is still in 15 / 37 PAI 281 launch scenes (NTV0 7) and its plan from standstill covers 0.39 x the
human's 4 s arc (NTV0 0.62 x). The flattening alone moves none of the training probes in two stored pairs (SIMPF / SIMPB, B2F / B2C). Points to
H1; H3 is not separable from H1. Arms J-W (standing-label rows at weight 0) and J-NF (no flattening) are prepared and their 60-step smokes pass; not yet trained.

**Round 3b** ([plans/2026-10-21-round3b-prereg.md](plans/2026-10-21-round3b-prereg.md); results gate3b.md, paifix3b_vl_vs_ctrl.md, inertia3b.md, launch_rows.md). On the J-NF base, NS (no stretch rows, draw R3NS) and MV (PAI human rows fed >= 2 m/s only, R3MV), 9 500 steps from NTV0-L18-s0, vs stored R2C. Offline: plain PAI launch rows cover 4.16 m in 4 s vs navtrain's 8.28 m; f0.5 / f1.5 launch rows are 0.27x / 3.03x of their base row. **Both fail the binding G-LAUNCH line**: NS 0.631 (-0.234 [-0.532, -0.038], launched 4 / 6, rear-hit 3), MV 0.569 (-0.295 [-0.585, -0.035], 3 / 6, rear-hit 3) vs R2C 0.865 (5 / 6, rear-hit 1); T281 route-turn zeros 2 / 4 vs 12 (pass); TARGET 0.526 / 0.574 vs 0.523 (pass). Inertia stop / turn ratios NS 0.801 / 0.808, MV 0.744 / 0.948 (R2C 0.892 / 0.906), both fail the 0.6 line. MV sees no PAI launch or standstill rows and still loses the launch, so the loss is not carried by static / low-speed PAI labels; by the stop rule no PAI 281 read was run. Snapshots R3{NS,MV}x{2500,5000,7500}-L18-s0 are on the box (unread).

**Next.** Main session: the launch loss follows PAI human rows even when they are moving-only; what they change on moving states (speed level, following gap) is the open question.

**Read more.** [plans/2026-10-21-round3-prereg.md](plans/2026-10-21-round3-prereg.md) (diagnosis, arms, rule, lines, cost).

Box: run dir `$DATA_DIR/runs/round3/` (jobs.txt, stages.txt, DONE); draw file `runs/defect_fix/w/R3JW.npz` (from R2JF); tags R3JW-L18-s0, R3JNF-L18-s0;
control is round 2's stored R2C-L18-s0.

<!-- files:begin -->
## Files

- `2026-10-21-round3-prereg.md` (plans): ROUND3 预登记：找出并去掉第 2 轮 J …
- `r3.py` (scripts): where does round 2's launch loss come …
- `r3.sh` (scripts): two continuations of NTV0-L18-s0 with …
- `rows_state.md` (results): PAI human rows by state
- `launch_plans.md` (results): model plans at the human's launch
- `w_R3JW.json` (results): 
- `README.md` (experiments/round2): NTV0 continuation, joint arm J vs …
- `README.md` (experiments/pai_rows): PAI human-state and frame-stretch …
- `README.md` (experiments/pai_fix): targeted PAI exam for stop and turn …

[results/](results/) 8 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
