# CB2 verdict (2026-10-25)

Registered lines: [../../plans/2026-10-25-cb2.md](../../plans/2026-10-25-cb2.md) section 3 (E1) and 5. Primary scorer = fixed (1b024f0).

| gate / line | value | line | result |
|:--|--:|--:|:--|
| E1 arena held native, P1N mean | 0.7844 | >= B2-s1 re-read 0.9412 - 0.060 = 0.8812 | **FAIL** |
| E1 held 460 zeros, P1N | 44 | <= 60 | pass |

E1 fails, so stage 2 (P1N's own DAgger rows, D1N) was not submitted, and neither was the D1N seed 1 or the PAI 281 side read.
CB2 does not tie B2. No adoption. Mechanism: STAND-eligible S rows are 0.2 % of the draw vs 14.5 % in B2's (the static practice-start
rows that the static-row rule removes). Numbers: [README.md](README.md), [sum1.json](sum1.json).

## Amendment 2: P1N' (B2's S-row rule)

| gate / line | value | line | result |
|:--|--:|--:|:--|
| E1 arena, P1N' | 0.9424 | >= 0.8812 | pass |
| E1 held 460 zeros | 42 | <= 60 | pass |
| stage 2 pool gate R1 on P1N''s practice shards | (c) control nan (no CB2P1NB-F-s1); (a) 0.986-0.988 on 3 shards | (a) >= 0.99, (c) control < 0.20 | **FAIL** |

D1N' not trained. P1N' itself is level with B2 on the arena (+0.001 [-0.006, +0.009] fixed scorer) and 0.027 below on held 460
(zeros 42 vs 40, inside the registered +4 band); it was not registered as an adoption arm.

## Amendment 4: stage 2 R1 re-run

| gate | value | line | result |
|:--|--:|--:|:--|
| R1 (a) onpol-train L18CBPB | one-ulp share 0.9976, max 0.0625 m | one-ulp share >= 0.99 and max <= 0.047 m (nuplan_dag amendment 2) | **FAIL** |
| R1 (a) practice CBPB4 | 0.9994, max 0.0625 m | same | **FAIL** |
| R1 (a) practice CBPB0-3, 5, 6 | | same | pass |
| R1 (c) all pools, control CB2P1N-F-s0 | pose share 0.001-0.005 | < 0.20 | pass |

D1N' not trained. P1N' against the adoption rule: arena tie (+0.001 [-0.006, +0.009] vs B2-s1), held 460 zeros 42 <= 44, so both
registered lines are met; held 460 mean -0.027 [-0.049, -0.006] vs B2-s1, -0.018 [-0.041, +0.004] vs B2-s0. One seed, not a registered
adoption arm.

## Amendment 6: R1 with the per-point fp16-step cap

| gate | value | line | result |
|:--|--:|--:|:--|
| R1 (a), 8 pools | max 3 300-4 000 fp16 steps; nearly every row over one step | every point <= one step at its own magnitude, share >= 0.99 | **FAIL** (all 8) |
| diagnostic: max(0.03126 m, own step) | max ratio 1.5-2.0; 0-7 rows per practice shard, 42 rows onpol | (not a gate) | - |

D1N' not trained. Final standing: P1N' ties B2 on the arena and is 0.027 below B2-s1 on held 460; D1N' could not be built under any of
the three registered versions of R1 (a).
