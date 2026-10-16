# image vs box_native: 29 scenes

- mean image 0.4491 (12 zeros), box_native 0.4502 (12 zeros); paired diff -0.0012 [-0.1042, +0.1019] (bootstrap 95 %)
- scores equal 18, within 0.01 25, within 0.05 27; zero / non-zero agree 27 of 29
- served plans identical (all decisions, tol 0.0001 m) in 0 of 29 scenes; plan difference at decision 0: median 0.0054 m, max 13.6218 m

| scene | image | box_native | diff | decisions | first k differing | plan diff k=0 (m) |
|---|--:|--:|--:|--:|--:|--:|
| 1a5da90a | 0.4582  | 0.4582  | -0.0000 | 198 | 0 | 0.0010 |
| 1a69f757 | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0019 |
| 1bfad87c | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0049 |
| 2466266e | 0.0000 offroad | 0.0000 offroad | +0.0000 | 198 | 0 | 0.0273 |
| 26004ca8 | 0.3035  | 0.3166  | -0.0131 | 199 | 0 | 0.0028 |
| 268949d2 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0002 |
| 2bbf1efc | 1.0000  | 1.0000  | +0.0000 | 199 | 1 | 0.0001 |
| 2c263e19 | 0.0000 left_corridor_laterally | 1.0000  | -1.0000 | 199 | 0 | 0.0054 |
| 2d023524 | 1.0000  | 1.0000  | +0.0000 | 198 | 2 | 0.0000 |
| 2d10a08d | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0150 |
| 2fb5b2b5 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0128 |
| 2fe500f7 | 0.5031  | 0.5027  | +0.0003 | 198 | 0 | 0.0005 |
| 300465f7 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0376 |
| 302c5c99 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0055 |
| 3232fdb7 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0035 |
| 32be03e4 | 0.5663  | 0.5657  | +0.0006 | 199 | 0 | 0.0017 |
| 35e082a4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 187 | 0 | 13.6218 |
| 382a7247 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0118 |
| 3ce13eed | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0034 |
| 3fa5e06b | 0.6779  | 0.6765  | +0.0014 | 199 | 0 | 0.0193 |
| 3ffad395 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0005 |
| 404541da | 0.8870  | 0.9105  | -0.0235 | 199 | 0 | 0.0044 |
| 4176f31d | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0021 |
| 478d6ec4 | 0.8941  | 0.8942  | -0.0000 | 199 | 0 | 0.0325 |
| 4a863a41 | 0.6195  | 0.6192  | +0.0004 | 199 | 0 | 0.0211 |
| 4cb7f7ac | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.7471 |
| 4e4a2a51 | 1.0000  | 0.0000 offroad | +1.0000 | 199 | 0 | 0.0103 |
| 51400458 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0150 |
| 52bd89ce | 0.1136  | 0.1135  | +0.0001 | 199 | 0 | 0.0080 |
