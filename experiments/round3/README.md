# round3: remove round 2's launch loss, keep its turn fix

status: live
decisions: none yet
index: offline: J's launch loss tracks PAI standing rows (2.5 / batch vs 0.16), not the flattening

**Question.** Is round 2's launch loss (R2J: PAI 281 launches 33 / 37 -> 21 / 37, PAI-FIX G-LAUNCH 0.865 -> 0.535) caused by the PAI human rows whose
4 s label stands throughout (H1), by the onset history flattening (H2) or by the stretch rows at standstill (H3), and does removing the cause
keep the route-turn fix (PAI 281 route-turn zeros 15 -> 2)?

**Conclusion.** Offline, before any training (results/rows_state.md, results/launch_plans.md): J's batch carries 2.54 PAI rows standing
through their label (navtrain has 22 of 103 288, the S pool 0.14 a batch), so "standstill and stay" goes from 1.1 % to 18.5 % of the standstill
rows (stay + launch); at the human's launch J is still in 15 / 37 PAI 281 launch scenes (NTV0 7) and its plan from standstill covers 0.39 x the
human's 4 s arc (NTV0 0.62 x). The flattening alone moves none of the training probes in two stored pairs (SIMPF / SIMPB, B2F / B2C). Points to
H1; H3 is not separable from H1. Arms J-W (standing-label rows at weight 0) and J-NF (no flattening) are prepared, not yet trained.

**Next.** Main session: launch both arms (`r3.sh launch`) or J-W only; the prereg's lines decide the PAI 281 read.

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

[results/](results/) 6 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
