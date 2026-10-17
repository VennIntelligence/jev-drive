# image vs box_native: 29 scenes

- mean image 0.5519 (12 zeros), box_native 0.5502 (12 zeros); paired diff +0.0017 [-0.0014, +0.0066] (bootstrap 95 %)
- scores equal 25, within 0.01 27, within 0.05 28; zero / non-zero agree 29 of 29
- served plans identical (all decisions, tol 0.0001 m) in 0 of 29 scenes; plan difference at decision 0: median 0.0011 m, max 1.9418 m

| scene | image | box_native | diff | decisions | first k differing | plan diff k=0 (m) |
|---|--:|--:|--:|--:|--:|--:|
| 1a5da90a | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0010 |
| 1a69f757 | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0002 |
| 1bfad87c | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0009 |
| 2466266e | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0019 |
| 26004ca8 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0035 |
| 268949d2 | 0.8044  | 0.8187  | -0.0143 | 199 | 0 | 0.0002 |
| 2bbf1efc | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0001 |
| 2c263e19 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0118 |
| 2d023524 | 1.0000  | 1.0000  | +0.0000 | 198 | 2 | 0.0000 |
| 2d10a08d | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0002 |
| 2fb5b2b5 | 1.0000  | 1.0000  | +0.0000 | 199 | 1 | 0.0000 |
| 2fe500f7 | 1.0000  | 1.0000  | +0.0000 | 198 | 0 | 0.0006 |
| 300465f7 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0027 |
| 302c5c99 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0037 |
| 3232fdb7 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0031 |
| 32be03e4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0049 |
| 35e082a4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 187 | 0 | 1.9418 |
| 382a7247 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0081 |
| 3ce13eed | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0011 |
| 3fa5e06b | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0007 |
| 3ffad395 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0006 |
| 404541da | 0.9478  | 0.9473  | +0.0005 | 199 | 0 | 0.0046 |
| 4176f31d | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0001 |
| 478d6ec4 | 0.9031  | 0.9030  | +0.0000 | 199 | 0 | 0.0012 |
| 4a863a41 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0018 |
| 4cb7f7ac | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0028 |
| 4e4a2a51 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0046 |
| 51400458 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0003 |
| 52bd89ce | 0.3505  | 0.2872  | +0.0633 | 199 | 0 | 0.0002 |
