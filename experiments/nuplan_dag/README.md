# nuplan_dag: PAI DAgger map teacher moved to nuPlan

status: concluded
decisions: 444, 445
index: DAgger map teacher does not transfer to nuPlan; arena lateral zeros 95/107 vs 89/99

**Question.** Two parts. (1) Does PAI training hurt nuPlan? R8K-L18-s2 and R6DAG-L18-s1 are read on held 460 and the arena with the native serve, paired against B2 (`STANDFMXH-L18-s1`) and NTV0. (2) Does round 6's DAgger with a map lateral teacher (decision 434) transfer to nuPlan? The student is B2, rolled on the 1031 onpol-train scenes; the teacher is the route-lane centre plus the human's offset; the extra rows are added to B2's own recipe. Can a lateral teacher alone move the lateral zeros, which decision 395 found are mostly over-speed?

**Conclusion.** Part 1: the PAI checkpoints read at NTV0's level on the nuPlan arena (0.763 / 0.767 vs 0.770, CIs span 0). On held 460 they show a small loss against NTV0 that the CI does not separate from 0, with 8-9 more zeros, mostly offroad and front collisions (decision 444). Part 2: the teacher is sound on nuPlan (T1 0.072 / 0.092 m, T2 0.970). NDAG still does not lower arena lateral zeros (95 / 107 vs B2's 89 / 99, mostly over-speed) and does not beat B2 on the arena (-0.002 / -0.005). It is better only on held 460 (35 / 38 vs 41 zeros), which was not the registered read. The method does not transfer (decision 445).

**Next.** None in this lane. Pools `on1_JTXD`, `on1_ND*`, the rows and the rollouts are kept until a deletion is approved.

**Read more.** Pre-registration (Chinese, amendments 1-3): [plans/2026-10-22-nuplan-dag.md](plans/2026-10-22-nuplan-dag.md). Part 1: [results/p1.md](results/p1.md). Part 2: [results/p2.md](results/p2.md) and [results/p2_quick.md](results/p2_quick.md). Gates: [results/t1/t1.md](results/t1/t1.md), [results/label_NDP.md](results/label_NDP.md), [results/gates.md](results/gates.md), [results/identity.json](results/identity.json), [results/smoke.json](results/smoke.json). Probe BEV plots: [results/figs/probe/](results/figs/probe/) (row frame; red = B2's served plan, blue = teacher, grey = the human's log; look for B2 swinging wide in right turns and the teacher pulling it back). Box: `$DATA_DIR/runs/nuplan_dag/`.

<!-- files:begin -->
## Files

- `2026-10-22-nuplan-dag.md` (plans): NUPLAN-DAG：PAI 的 DAgger + 地图横向教师搬到 …
- `nd.sh` (scripts): the PAI DAgger map-teacher method moved …
- `nd_teach.py` (scripts): the PAI DAgger map teacher on nuPlan …
- `nd.py` (scripts): tables of the nuPlan DAgger map-teacher …
- `dag.py` (experiments/round6/scripts): DAgger rows for the LATERAL part of the …
- `pn.sh` (experiments/practice_nuplan/scripts): Box, repo root
- `stand_b.sh` (experiments/defect_fix/scripts): arm B of the JTX STAND pilot with

[results/](results/) 38 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 3 entry points
<!-- files:end -->
