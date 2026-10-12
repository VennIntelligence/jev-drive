# junction_eyes row tables (step 0, executor 1)

Box: `$DATA_DIR/runs/junction_eyes/step0/rows/` (built by `scripts/je0.py rows`; definitions in
[plans/2026-10-12-prereg-A.md](../plans/2026-10-12-prereg-A.md), amendment 1 items 1-2). `all.parquet` has every row. The other files
are views of it with the same schema: `J_nav`, `J_on`, `val`, `P`, `navtest16`, `held452`, `zero98`. `rows.json` holds the counts and G-split.

| set | rows | scenes / segments | units |
|:--|--:|--:|--:|
| J_nav (navtrain) | 99 028 | 1 192 | 162 |
| J_on (AlpaSim train on-policy) | 33 899 | 868 | 28 |
| val (J_nav 10 % of drive logs + 3 J_on units) | 12 812 | 223 | 17 |
| P = J minus val | 120 115 | 1 837 | 173 |
| navtest16 (read) | 3 712 | 39 | 16 |
| held452 (read) | 8 129 | 452 | 16 |
| zero98 (read) | 457 | 72 | 12 |

Also present in `all.parquet`: `navtest_rest` (navtest tokens of the other 28 drive logs; L-F0-nav reads them, the default eye does not)
and `zero_rest` (the other decisions of the zero rollouts; not read).

Columns:

| column | meaning |
|:--|:--|
| rid | global row id; **labels are aligned to it**: `labels/ra.npy` (N, 30) float32, `labels/rb.npy` (N, 48, 48) uint8 |
| set | J_nav, J_on, navtest16, navtest_rest, held452, zero98, zero_rest |
| source | navtrain, onpol, navtest, held, zero |
| split | fit (in P), val, read, none |
| scene | navsim log segment (navtrain / navtest) or AlpaSim scene id |
| log | navsim log segment (for AlpaSim rows: the scene's segment) |
| unit | drive log, date_vehicle |
| token | navsim token; on-policy row name `on:<scene>:<seed>:<k>`; `held:` / `zero:<tag>:<scene>:<k>` |
| tag, k | rollout label and decision index (AlpaSim rows), else "log", -1 |
| t_log_us | absolute log time of the decision (J_on), else -1 |
| jdist, turn40 | route distance to the junction (m, NaN = none within 60 m) and route turn over 40 m (deg), from decision 275's label tables |
| cand | J_nav: the row meets the junction / turn rule (False = a straight filler row) |
| w_scene | J_on: 1 / rows of its scene (the trainer's scene-uniform sampling); 1 elsewhere |
| tok_file, tok_row | frozen-token cache: `runs/op_parity/cache/<tok_file>/front.npy` row (`road_probe/tok/*` live under `runs/`) |
| tab_file | the cache's `tab.npz` (ego, cam, ...) at row `tok_row` |
| old_lab, old_row | decision 275 / rp_real label table (`runs/road_probe/<old_lab>.parquet`) and row |
| dump | J_on: the driver dump npz of the decision |

Label layout. R-A: `ra[rid].reshape(5, 3, 2)` = s in (5, 10, 15, 20, 30) m x (centre, left, right) x (x, y). Frame: rear axle, y left,
NaN = undefined. R-B: `rb[rid]` indexed [x, y], with cell centres x = 0.25 + 0.5 i and y = -11.75 + 0.5 j. Classes: 0 off-road, 1 drivable,
2 oncoming. Pixel rows for J_nav: navtrain cache order = `tok_file` shard + `tok_row`.
