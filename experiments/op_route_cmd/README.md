# op_route_cmd: route polyline input material

status: live
decisions: (inputs: 93, 102, 118, 121)
index: Route polyline material: navtrain 103 k / WOD 522 k labelled, 32 k / 62 k turns >= 25 deg; noise + negatives + CARLA plan, no training

**Question.** Can openpilot be told which junction exit to take by a navigation-level route polyline (ego frame, 10 m vertices, 150 m ahead; the geometry
alone encodes "third road from the left", no ordinal, no HD lane geometry at test time)? This topic only prepares the fine-tune material: clean hindsight labels
for the real-data frame pools, the train-time noise function, negatives, and the CARLA counterfactual-pair plan. Nothing is trained here.

**Conclusion.** Material is ready (results/data_prep.md). Hindsight polylines exist for all 103 288 navtrain tokens and all 522 023 WOD-E2E train + val frames;
a complete turn >= 25 deg within 150 m of the driven path: navtrain 32 269 samples (817 of 1 192 logs, 6 205 turn events; 4 713 of the 11 310 decision-93
junction frames), WOD train 52 135 / val 9 799 (949 / 232 events; WOD has no map and only 5 s future per frame, chained to <= 22 s). The route fields attach to the
op_adapt_H sample tables by `id` (100% coverage of nav 2 400 and wod 2 780 rows). The noise function (`NavNoise`) is bit-identical to the closed-loop option C in
its C preset and has unit tests; a negative-route generator (4 kinds, 158 samples, 135 flagged for a visual check) and a CARLA pair plan (10 628 held-out-safe
(road, exit) polylines, ~2 card-h for 10 k poses) are written. The CARLA pair material itself is rendered (results/carla_pairs.md): 8 607 poses / 21 151 (pose, exit) rows (3 917 poses with 3 exits), 1 310 junctions, none of the 172 Bench2Drive junctions, openpilot rig at 1.22 m, 0.63 s per pose per server; a 2 000-pose subset set is separate.

**Negatives (2026-10-05).** `route_neg.npz` sidecar (63 693 map-screened rows on 32 754 navtrain frames, N1-A / N2 / N3 / N4; tier B kept apart as `lane_change_needed`), results/negatives.md.

**Next.** Merge lane (the CARLA rows join as `samples/route_carla` + `route.npz`, see results/carla_pairs.md): read `route.npz` through `lib/route_poly.attach`, call `noise_polyline` per sample at train time, decide the model-side encoding. Spot check of `figs/negatives_screened_sheet.png` / `negatives_borderline_sheet.png`.

**Read more.** [results/data_prep.md](results/data_prep.md) (counts, format, merge notes, doubts), [results/negatives.md](results/negatives.md),
[results/carla_pairs_plan.md](results/carla_pairs_plan.md), [results/carla_pairs.md](results/carla_pairs.md) (rendered CARLA pairs: format, rates, doubts), [../op_img_cmd/README.md](../op_img_cmd/README.md) (the image-side command channel this competes with),
[../op_common_cause/results/pair_inventory.md](../op_common_cause/results/pair_inventory.md) (decision 93), [../op_closed_loop/scripts/turn_calibration_options.py](../op_closed_loop/scripts/turn_calibration_options.py) (option C).

<!-- files:begin -->
## Files

- `route_poly.py` (lib): the navigation-level input "which way …
- `route_neg.py` (lib): a route that contradicts the scene
- `data_prep.md` (results): counts, format, merge notes
- `negatives.md` (results): spec, generator, sample
- `carla_pairs_plan.md` (results): CARLA counterfactual route-polyline …
- `route_nav.py` (scripts): Route polylines for NAVSIM navtrain
- `route_wod.py` (scripts): Route polylines for WOD-E2E
- `route_report.py` (scripts): Counts of the route-polyline material
- `route_samples.png` (figs): 

[results/](results/) 9 result files · [figs/](figs/) 4 figures · [plans/](plans/) 0 live plans · [scripts/](scripts/) 9 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Shared code lives in the repo-level `lib/route_poly.py` and `lib/route_neg.py`
(imported by this topic and by op_closed_loop); tests `tests/test_route_poly.py`, `tests/test_route_wod.py`.
