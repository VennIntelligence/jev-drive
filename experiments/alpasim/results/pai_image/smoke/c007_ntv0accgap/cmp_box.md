# image vs box_native: 29 scenes

- mean image 0.5403 (13 zeros), box_native 0.6090 (11 zeros); paired diff -0.0687 [-0.1724, +0.0005] (bootstrap 95 %)
- scores equal 25, within 0.01 27, within 0.05 27; zero / non-zero agree 27 of 29
- served plans identical (all decisions, tol 0.0001 m) in 0 of 29 scenes; plan difference at decision 0: median 0.0006 m, max 1.2834 m

| scene | image | box_native | diff | decisions | first k differing | plan diff k=0 (m) |
|---|--:|--:|--:|--:|--:|--:|
| 1a5da90a | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 1.2834 |
| 1a69f757 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0006 |
| 1bfad87c | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0008 |
| 2466266e | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0007 |
| 26004ca8 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0014 |
| 268949d2 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0001 |
| 2bbf1efc | 0.0000 left_corridor_laterally | 1.0000  | -1.0000 | 199 | 0 | 0.0003 |
| 2c263e19 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0012 |
| 2d023524 | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0001 |
| 2d10a08d | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0002 |
| 2fb5b2b5 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0001 |
| 2fe500f7 | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0003 |
| 300465f7 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0004 |
| 302c5c99 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0060 |
| 3232fdb7 | 0.0000 left_corridor_laterally | 1.0000  | -1.0000 | 198 | 0 | 0.0031 |
| 32be03e4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0006 |
| 35e082a4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 187 | 0 | 0.0002 |
| 382a7247 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0017 |
| 3ce13eed | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0137 |
| 3fa5e06b | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0007 |
| 3ffad395 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0002 |
| 404541da | 0.7650  | 0.7570  | +0.0080 | 199 | 0 | 0.0021 |
| 4176f31d | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0003 |
| 478d6ec4 | 0.9028  | 0.9028  | -0.0000 | 199 | 0 | 0.0005 |
| 4a863a41 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0058 |
| 4cb7f7ac | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0004 |
| 4e4a2a51 | 0.0000 collision_at_fault | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0006 |
| 51400458 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0006 |
| 52bd89ce | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0001 |
