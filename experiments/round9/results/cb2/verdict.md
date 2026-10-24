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
