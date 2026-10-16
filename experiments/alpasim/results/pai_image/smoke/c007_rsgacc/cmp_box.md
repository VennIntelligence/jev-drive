# image vs box_native: 29 scenes

- mean image 0.5109 (13 zeros), box_native 0.4774 (14 zeros); paired diff +0.0334 [-0.0031, +0.1034] (bootstrap 95 %)
- scores equal 25, within 0.01 27, within 0.05 28; zero / non-zero agree 28 of 29
- served plans identical (all decisions, tol 0.0001 m) in 0 of 29 scenes; plan difference at decision 0: median 0.0012 m, max 18.8360 m

| scene | image | box_native | diff | decisions | first k differing | plan diff k=0 (m) |
|---|--:|--:|--:|--:|--:|--:|
| 1a5da90a | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0010 |
| 1a69f757 | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0002 |
| 1bfad87c | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0009 |
| 2466266e | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0019 |
| 26004ca8 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0035 |
| 268949d2 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0002 |
| 2bbf1efc | 1.0000  | 1.0000  | +0.0000 | 199 | 1 | 0.0001 |
| 2c263e19 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0054 |
| 2d023524 | 1.0000  | 1.0000  | +0.0000 | 198 | 1 | 0.0000 |
| 2d10a08d | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0002 |
| 2fb5b2b5 | 1.0000  | 1.0000  | +0.0000 | 199 | 1 | 0.0000 |
| 2fe500f7 | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0006 |
| 300465f7 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0027 |
| 302c5c99 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0041 |
| 3232fdb7 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0031 |
| 32be03e4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0049 |
| 35e082a4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 187 | 0 | 18.8360 |
| 382a7247 | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0098 |
| 3ce13eed | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0025 |
| 3fa5e06b | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0007 |
| 3ffad395 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0006 |
| 404541da | 0.6879  | 0.7176  | -0.0297 | 199 | 0 | 0.0043 |
| 4176f31d | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0001 |
| 478d6ec4 | 0.9030  | 0.9030  | +0.0000 | 199 | 0 | 0.0012 |
| 4a863a41 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0018 |
| 4cb7f7ac | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.7471 |
| 4e4a2a51 | 1.0000  | 0.0000 offroad | +1.0000 | 199 | 0 | 0.0045 |
| 51400458 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0003 |
| 52bd89ce | 0.2239  | 0.2242  | -0.0003 | 199 | 0 | 0.0002 |
