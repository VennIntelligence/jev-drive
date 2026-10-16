# image vs box_native: 29 scenes

- mean image 0.3630 (13 zeros), box_native 0.3368 (14 zeros); paired diff +0.0262 [-0.0013, +0.0801] (bootstrap 95 %)
- scores equal 18, within 0.01 28, within 0.05 28; zero / non-zero agree 28 of 29
- served plans identical (all decisions, tol 0.0001 m) in 0 of 29 scenes; plan difference at decision 0: median 0.0066 m, max 13.6624 m

| scene | image | box_native | diff | decisions | first k differing | plan diff k=0 (m) |
|---|--:|--:|--:|--:|--:|--:|
| 1a5da90a | 0.4102  | 0.4101  | +0.0001 | 198 | 0 | 0.0303 |
| 1a69f757 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0238 |
| 1bfad87c | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0497 |
| 2466266e | 0.0000 offroad | 0.0000 offroad | +0.0000 | 198 | 0 | 0.0095 |
| 26004ca8 | 0.3089  | 0.3030  | +0.0058 | 199 | 0 | 0.0591 |
| 268949d2 | 0.2523  | 0.2519  | +0.0004 | 199 | 3 | 0.0001 |
| 2bbf1efc | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0008 |
| 2c263e19 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0008 |
| 2d023524 | 1.0000  | 1.0000  | +0.0000 | 198 | 1 | 0.0000 |
| 2d10a08d | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0108 |
| 2fb5b2b5 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0072 |
| 2fe500f7 | 0.5334  | 0.5334  | +0.0000 | 198 | 0 | 0.0014 |
| 300465f7 | 0.0000 offroad | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0028 |
| 302c5c99 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0007 |
| 3232fdb7 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0036 |
| 32be03e4 | 0.4439  | 0.4539  | -0.0099 | 199 | 0 | 0.0039 |
| 35e082a4 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 187 | 0 | 13.6624 |
| 382a7247 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | 0 | 0.0016 |
| 3ce13eed | 0.0000 left_corridor_laterally | 0.0000 offroad | +0.0000 | 199 | 0 | 0.0018 |
| 3fa5e06b | 0.6218  | 0.6218  | -0.0000 | 199 | 0 | 0.0325 |
| 3ffad395 | 0.0000 collision_at_fault | 0.0000 collision_at_fault | +0.0000 | 199 | 0 | 0.0241 |
| 404541da | 0.7170  | 0.7243  | -0.0073 | 199 | 0 | 0.0078 |
| 4176f31d | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0108 |
| 478d6ec4 | 0.8699  | 0.8699  | +0.0000 | 199 | 0 | 0.0005 |
| 4a863a41 | 0.4763  | 0.4745  | +0.0018 | 199 | 0 | 0.0010 |
| 4cb7f7ac | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | 0 | 0.0066 |
| 4e4a2a51 | 0.7740  | 0.0000 offroad | +0.7740 | 199 | 0 | 0.0003 |
| 51400458 | 1.0000  | 1.0000  | +0.0000 | 199 | 0 | 0.0296 |
| 52bd89ce | 0.1189  | 0.1251  | -0.0062 | 199 | 0 | 0.0080 |
