# nuplan_dag: PAI DAgger map teacher moved to nuPlan

status: live
decisions: 391, 395, 434, 443
index: Part 1 PAI checkpoints on nuPlan reading; Part 2 NDAG chain (probe first)

**Question.** Two parts. (1) Does PAI training hurt nuPlan? R8K-L18-s2 and R6DAG-L18-s1 are read on held 460 and the arena with the native serve, paired against B2 (`STANDFMXH-L18-s1`) and NTV0. (2) Does round 6's DAgger with a map lateral teacher (decision 434) transfer to nuPlan? The student is B2, rolled on the 1031 onpol-train scenes; the teacher is the route-lane centre plus the human's offset; the extra rows are added to B2's own recipe. Can a lateral teacher alone move the lateral zeros, which decision 395 found are mostly over-speed?

**Conclusion.** Pending.

**Next.** Part 1 summary (p1sum); Part 2 runs probe c000 (6 scenes) -> gates T1-T4 -> full roll -> NDAG-L18-s0 / s1 -> reads.

**Read more.** Pre-registration (Chinese): [plans/2026-10-22-nuplan-dag.md](plans/2026-10-22-nuplan-dag.md). Box: `$DATA_DIR/runs/nuplan_dag/`.

<!-- files:begin -->
## Files

- `2026-10-22-nuplan-dag.md` (plans): NUPLAN-DAG：PAI 的 DAgger + 地图横向教师搬到 …
- `nd.sh` (scripts): the PAI DAgger map-teacher method moved …
- `nd_teach.py` (scripts): the PAI DAgger map teacher on nuPlan …
- `nd.py` (scripts): tables of the nuPlan DAgger map-teacher …
- `dag.py` (experiments/round6/scripts): DAgger rows for the LATERAL part of the …
- `pn.sh` (experiments/practice_nuplan/scripts): Box, repo root
- `stand_b.sh` (experiments/defect_fix/scripts): arm B of the JTX STAND pilot with

[results/](results/) 0 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 3 entry points
<!-- files:end -->
