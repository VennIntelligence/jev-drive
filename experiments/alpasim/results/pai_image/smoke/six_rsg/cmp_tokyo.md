# image vs tokyo_mounted: 6 scenes

- mean image 0.0876 (5 zeros), tokyo_mounted 0.2543 (4 zeros); paired diff -0.1667 [-0.5000, +0.0000] (bootstrap 95 %)
- scores equal 5, within 0.01 5, within 0.05 5; zero / non-zero agree 5 of 6
- served plans identical (all decisions, tol 0.0001 m) in 5 of 6 scenes; plan difference at decision 0: median 0.0000 m, max 0.0000 m

| scene | image | tokyo_mounted | diff | decisions | first k differing | plan diff k=0 (m) |
|---|--:|--:|--:|--:|--:|--:|
| 054b5901 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | - | 0.0000 |
| 2f227f2c | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 199 | - | 0.0000 |
| 49bcab6f | 0.0000 <AioRpcError of RPC that terminated with | 1.0000  | -1.0000 | 0 | - | nan |
| 90ab8c24 | 0.5259  | 0.5259  | +0.0000 | 199 | - | 0.0000 |
| a174faed | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | - | 0.0000 |
| fc9fde87 | 0.0000 left_corridor_laterally | 0.0000 left_corridor_laterally | +0.0000 | 198 | - | 0.0000 |
