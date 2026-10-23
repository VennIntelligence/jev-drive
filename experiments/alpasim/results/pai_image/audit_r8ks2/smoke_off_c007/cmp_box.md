# image vs native: 29 scenes

- mean image 0.6865 (9 zeros), native 0.7103 (8 zeros); paired diff -0.0238 [-0.0714, +0.0001] (bootstrap 95 %)
- scores equal 27, within 0.01 28, within 0.05 28; zero / non-zero agree 28 of 29
- served plans identical (all decisions, tol 0.0001 m) in 0 of 29 scenes; plan difference at decision 0: median 0.0011 m, max 0.0310 m

| scene | image | native | diff | decisions | first k differing | plan diff k=0 (m) |
|---|--:|--:|--:|--:|--:|--:|
| 1a5da90a | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0058 |
| 1a69f757 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0004 |
| 1bfad87c | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0007 |
| 2466266e | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0013 |
| 26004ca8 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0138 |
| 268949d2 | 0.0000 collision_at_fault | 0.6898  | -0.6898 | 199 | 0 | 0.0001 |
| 2bbf1efc | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0057 |
| 2c263e19 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0090 |
| 2d023524 | 1.0000  | 1.0000  | +0.0000 | 198 | 1 | 0.0001 |
| 2d10a08d | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0003 |
| 2fb5b2b5 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0001 |
| 2fe500f7 | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0016 |
| 300465f7 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0003 |
| 302c5c99 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0031 |
| 3232fdb7 | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0038 |
| 32be03e4 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0009 |
| 35e082a4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 187 | 0 | 0.0001 |
| 382a7247 | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0078 |
| 3ce13eed | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0039 |
| 3fa5e06b | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0011 |
| 3ffad395 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0031 |
| 404541da | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0067 |
| 4176f31d | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0001 |
| 478d6ec4 | 0.9097  | 0.9092  | +0.0005 | 199 | 0 | 0.0003 |
| 4a863a41 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0029 |
| 4cb7f7ac | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0310 |
| 4e4a2a51 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0010 |
| 51400458 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0005 |
| 52bd89ce | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0001 |
