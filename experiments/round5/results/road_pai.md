# PAI road raster for the human-state rows (ROUND5)

Plan: [plans/2026-10-21-pai-road-raster.md](../plans/2026-10-21-pai-road-raster.md). Code: `scripts/road_pai.py` / `road_pai.sh`. Trainer: `rm_train.py` / `on_train.py --on-road-pai <index> [--on-road-pai-w W]` (default off).

## Format

nuPlan road.npz layout: SDF (m, + inside) 128 x 96 cells of 0.5 m, x -8 .. 56 m, y -24 .. 24 m, rear axle (= PAI rig) frame at t0, float16, clipped +-20 m, opb_labels canvas (0.25 m fill, EDT, 2 x 2 mean); surface (plan amendment 1) = the evaluator's PAI offroad rule: the ego box centre's 4-connected component of the canvas cut by road edges within 3 m of the ego's elevation, plus the lane union (rf_tax.Map); built 851 scenes in 159 s on 36 cores (median 30 ms a raster). 12.6 % of rasters have > 90 % of the canvas inside (road edges with gaps; the edge lines stay thin negative walls, as the scorer only flags touching an edge). Stored once per (scene, k): `$DATA_DIR/runs/round5/road_pai/sdf.npy` (135793 rasters, 3.34 GB); one small index per pool (`road_<pool>.npz`: tokens, ok, ix, sdf_file; drivable_hinge.Hinge reads it).

## Fallback and G1 (human's own label path, margin 0.3 m, Pacifica footprint)

| pool | PAI rows | no surface | (a) t0 off | (b) label > 0.5 m out | hinged (a-c) | G1: label triggers, of hinged | passed | shipped ok |
|---|--:|--:|--:|--:|--:|--:|:-:|--:|
| on1_PAIH | 135793 | 0.0 % | 0.1 % | 2.4 % | 97.4 % | 6.9 % | no (rule d added) | 90.7 % |
| on1_PAIH05 | 123879 | 0.0 % | 0.1 % | 0.0 % | 99.8 % | 4.1 % | no (rule d added) | 95.7 % |
| on1_PAIH15 | 99269 | 0.0 % | 0.2 % | 8.5 % | 91.3 % | 10.4 % | no (rule d added) | 81.8 % |
| on1_JTPH | 358941 | 0.0 % | 0.1 % | 3.3 % | 96.6 % | 6.8 % | no (rule d added) | 90.0 % |

**G1 fails under the primary rule (a)-(c)**: the human's own label path comes within 0.3 m of the surface edge on 4.1-10.4 % of the hinged rows (on1_PAIH 6.9 %; it grows with the stretch factor and with speed). The registered rule (d) is therefore in the shipped `ok`: rows whose own label triggers the hinge carry no road hinge (G1 = 0 by construction; on1_JTPH keeps 90.0 % of its PAI rows hinged). `ok_primary` (a)-(c) is in every index file for a switch. The 1 % line is stricter than nuPlan itself: the same read on navtrain's human rows gives 12 % (below).

nuPlan reference (navtrain rows' own fut, same geometry):

| raster | labelled | label triggers (all labelled) | t0 off | label triggers, of rows the PAI rule (a, b) keeps |
|---|--:|--:|--:|--:|
| navtrain_drivable | 103288 | 12.1 % | 0.5 % | 11.6 % |
| navtrain_road | 103288 | 12.6 % | 0.5 % | 12.1 % |

## G2: R2J 281 ACC + GAP offroad zeros, the hinge on R2J's own plan 2.5 s before the event

- zeros 20: hinge triggers on **12 (60.0 %)** (line > 50 %: pass); t0 footprint already off 0 (rule a would leave those unhinged), triggers on a hinged row 12; plan corner below 0 m 4
- control: R2J's solved scenes (172), one decision every 2 s (1720): triggers 10.1 % (hinged rows 9.7 %), t0 off 0.5 %

Per zero: [road_pai_g2.csv](road_pai_g2.csv). The hinge on the triggering zeros is shallow (median hinge value 0.005; 4 of 20 plans have a corner past the edge 2.5 s before); 8 zeros do not trigger (plan still >= 0.3 m inside 2.5 s before the event). The control false-trigger rate is 10 %.

## G3: identity (R2J's command, 60 steps eager)

- off: identical **True** (scalars 278, max diff 0.0, weights 0.0, on_hash_60 ['43bfc742366790f8', '43bfc742366790f8'])
- w0: identical **True** (scalars 278, max diff 0.0, weights 0.0, on_hash_60 ['43bfc742366790f8', '43bfc742366790f8'])

Disk under `$DATA_DIR/runs/round5/road_pai/`: 4.01 GB (sdf.npy 3.34 GB, parts 0.40 GB, indices 0.20 GB) + the three 60-step G3 run dirs (87 MB each). Cost: about 2.8 booked core-hours (build 36 cores x 3 min, the rest < 1) and 3 x 2 min of GPU for G3; no training.

Use: `rm_train.py ... --on on1_JTPH --on-road-pai $DATA_DIR/runs/round5/road_pai/road_on1_JTPH.npz` (W default 1). A new merged pool needs `road_pai.py index --pool on1_<NEW>` (seconds; rasters are shared). PAI S rows (on1_CPAIX) need a `build` on their own anchors first.
