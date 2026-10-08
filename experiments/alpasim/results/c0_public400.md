# C0: SH30 and AP2 on the 400 landed public AlpaSim scenes (2026-10-09, decision 199)

Local closed-loop runs, execution evidence for the nuPlan track; nothing was submitted to AlpaSim. Scenes: every scene of the four asset shards with a `.done` marker when the runs
started (part001, 002, 003, 005; 400 scenes, list `$DATA_DIR/runs/alpasim/scenes_public_landed4.txt`). Part007 got its `.done` marker afterwards and is **not covered**; no run was added for it.
Drivers: `SH30-F-s0` (`SH30_COLD=backwarp`) and `AP2-AB-s0`, no tap, wizard overrides as in docs/alpasim.md (8 concurrent rollouts, one stack per card). Read-out: `scripts/c0_report.py`
(numbers in `c0_public400.json`). Run dirs on the box under `$DATA_DIR/runs/alpasim/c0/`: `sh30_400_rerun/20261009-020755`, `ap2_400_r1/20261008-233048`, `ap2_400_r2/20261008-233048`.
Wall per run 28.7 min (SH30), 28.7 and 26.3 min (AP2), one card each, about 1.4 card-hours in all (the 8-scene pilot adds 0.03).

The first SH30 attempt (`sh30_400/20261008-233048`, 151 scene dirs) hung at 23:38: a runtime worker was killed with exit code -9 (SIGKILL; the box was heavily loaded, cause not identified), the
runtime's result pump then failed and the process sat idle for 2.6 h holding the job. It was cancelled through the pool and rerun from scratch (the runtime has no resume); only the rerun is reported.

## Results

Scenes common to all three runs: 400.

| run | scenes | mean scene score | score 1 | score 0 | at-fault collision | offroad | left corridor | zero, other | slow (0 < score < 1) | at-fault events | mean progress | no-score rollouts |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30-F-s0 | 400 | 0.9306 | 327 | 23 | 8 | 7 | 8 | 0 | 50 | 15 | 0.953 | 23 |
| AP2-AB-s0 run 1 | 400 | 0.9335 | 349 | 24 | 9 | 6 | 9 | 0 | 27 | 15 | 0.998 | 24 |
| AP2-AB-s0 run 2 | 400 | 0.9335 | 349 | 24 | 9 | 6 | 9 | 0 | 27 | 15 | 0.998 | 24 |

AP2 repeat: mean 0.9335 vs 0.9335, difference +0.0000 (paired s.e. 0.0000); 0 of 400 scenes differ; mean |diff| 0.0000; zero in run 1 only 0, in run 2 only 0, both 24.

| scene | AP2 run 1 | AP2 run 2 | zero reasons run 1 / run 2 |
|:--|--:|--:|:--|

Best-of-two (per scene maximum, an oracle selector):

| pair | oracle mean | oracle minus best single driver (by mean) |
|:--|--:|--:|
| SH30 vs AP2 run 1 | 0.9627 | +0.0292 |
| SH30 vs AP2 run 2 | 0.9627 | +0.0292 |
| SH30 vs AP2 (mean of runs) | 0.9627 | +0.0292 |
| AP2 run 1 vs AP2 run 2 (same driver) | 0.9335 | +0.0000 |

Same-driver oracle (AP2 run 1 vs run 2) gains +0.0000: that is what a selector gets from run-to-run noise alone. Cross-driver oracle (SH30 vs AP2 run 1) minus best single, net of that noise: +0.0292. C2 line: >= 0.015 -> MET.

Zero in all three runs: 13; zero in SH30 only (AP2 both runs > 0): 10; zero in AP2 both runs only (SH30 > 0): 11.

Zero-score scenes:

| scene | SH30 | AP2 run 1 | AP2 run 2 |
|:--|:--|:--|:--|
| 2021.05.25.14.16.10_veh-35_01100_01664-368cb65e8fef57b7 | 0.99 | collision_at_fault | collision_at_fault |
| 2021.05.25.14.16.10_veh-35_01100_01664-6fed9368351f54d5 | collision_at_fault | 1.00 | 1.00 |
| 2021.05.25.14.16.10_veh-35_01100_01664-9215555823945665 | collision_at_fault | collision_at_fault | collision_at_fault |
| 2021.05.25.14.16.10_veh-35_02482_02649-8e9f743d92c05d10 | 1.00 | collision_at_fault | collision_at_fault |
| 2021.05.25.14.16.10_veh-35_04097_04328-d90cd23a434f5c55 | collision_at_fault | 0.99 | 0.99 |
| 2021.05.25.14.24.08_veh-25_03764_04034-23ee145aa4de582b | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.05.25.14.24.08_veh-25_03764_04034-5e9e8c31277d5edc | offroad | offroad | offroad |
| 2021.05.25.14.24.08_veh-25_04059_04203-6225b347244658c1 | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.05.25.14.24.08_veh-25_04059_04203-bca002ce93bd5997 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.05.25.14.26.37_veh-27_04122_04279-690d5fcd5dd056dc | collision_at_fault | collision_at_fault | collision_at_fault |
| 2021.05.25.15.59.03_veh-30_03499_03671-00ff8eeb53cc598b | offroad | offroad | offroad |
| 2021.05.25.15.59.03_veh-30_04027_04200-476c37463b4f580e | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.05.25.15.59.03_veh-30_04027_04200-5abe25231fd05639 | offroad | 0.91 | 0.91 |
| 2021.05.25.15.59.03_veh-30_04463_04606-6e3c7a34388e5ae3 | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.05.25.16.37.23_veh-25_00005_00217-60681597a59d5cf9 | left_corridor_laterally | 0.87 | 0.87 |
| 2021.05.25.16.37.23_veh-25_03311_03550-40d43586b1195366 | offroad | offroad | offroad |
| 2021.05.25.17.54.41_veh-35_01654_01850-77155a60b2ae5e75 | left_corridor_laterally | 1.00 | 1.00 |
| 2021.05.25.17.54.41_veh-35_01905_02121-4749b2486da65268 | collision_at_fault | collision_at_fault | collision_at_fault |
| 2021.06.03.12.02.06_veh-35_00233_00609-be7331d3f05e5d16 | 1.00 | collision_at_fault | collision_at_fault |
| 2021.06.03.13.55.17_veh-35_02419_02561-4f5d9ee9c2915058 | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.06.03.13.55.17_veh-35_02866_03582-0854af027e06530a | collision_at_fault | 1.00 | 1.00 |
| 2021.06.03.17.06.58_veh-35_02571_02742-9a89dccc70835d69 | collision_at_fault | 1.00 | 1.00 |
| 2021.06.03.17.06.58_veh-35_02943_03220-4495218e41b35f25 | 1.00 | collision_at_fault | collision_at_fault |
| 2021.06.03.17.06.58_veh-35_02943_03220-56da58294b3d53c5 | 0.99 | collision_at_fault | collision_at_fault |
| 2021.06.03.17.06.58_veh-35_02943_03220-f1b802f6e9a559af | offroad | offroad | offroad |
| 2021.06.28.13.53.26_veh-26_00492_00696-b411a654aa215f1b | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.06.28.20.24.43_veh-38_03385_04952-1347c91c511a5918 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.06.28.20.24.43_veh-38_03385_04952-927b73fea33f5218 | left_corridor_laterally | 1.00 | 1.00 |
| 2021.06.28.20.24.43_veh-38_03385_04952-d6353a288d0b545d | 1.00 | collision_at_fault | collision_at_fault |
| 2021.06.28.20.24.43_veh-38_03385_04952-e933d70dd378598f | left_corridor_laterally | 1.00 | 1.00 |
| 2021.06.28.21.16.05_veh-14_00957_01198-c768a604b14e5956 | offroad | offroad | offroad |
| 2021.06.28.21.16.05_veh-14_00957_01198-ef300f8a9cf254bc | offroad | offroad | offroad |
| 2021.06.28.21.47.53_veh-35_00280_00424-f4da0138413c595a | collision_at_fault | 1.00 | 1.00 |
| 2021.08.16.14.23.37_veh-45_00015_00132-9fdd329b72e85179 | 1.00 | left_corridor_laterally | left_corridor_laterally |

## Reading

- The two drivers score alike (SH30 0.9306, AP2 0.9335) with the same number of at-fault events (15 each). AP2 makes progress (0.998 vs 0.953; 27 vs 50 slow scenes, 349 vs 327 scenes at 1.0) and loses nothing net on zeros.
- Zeros are mostly not shared: only 13 of the 23 / 24 are zero in all three runs; 10 scenes are zero for SH30 only and 11 for AP2 only. That is the complementarity behind best-of-two.
- **The AP2 repeat is bit-identical**: all 400 scene scores equal in both runs. The simulator and driver are deterministic given the same inputs, so "repeat noise" here is exactly 0 and says nothing about seed or training variance (one training seed each, d189). The oracle gain of 0.0292 therefore needs no noise correction, but it is an upper bound from a per-scene selector with hindsight, on scenes that are 4 of ~15 drives and strongly clustered (many scenes per drive segment, e.g. veh-35_01100_01664 with three scenes).
- C2 line (oracle minus best single driver >= 0.015): **met** (+0.0292). The line is a go condition for building a disagreement-gated arbiter; whether a gate can pick the right driver without hindsight is not measured here.
- Zero causes (AP2 / SH30): at-fault collision 9 / 8, offroad 6 / 7, left corridor 9 / 8. No zero without a named flag. Collision details and the slow scenes were not examined case by case.
- Reference only: the 48-scene numbers of d185 / d188 / d189 (WA-JEPA 0.978, SH30 0.947, AP2 0.932) are the part001 subset of this list; SH30 and AP2 on those 48 are not recomputed here.
