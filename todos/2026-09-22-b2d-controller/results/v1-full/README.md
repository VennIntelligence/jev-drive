# v1 frozen evidence

New final evidence directory; previous intermediate and seed-specific reports remain unchanged.
No qualified new controller default is selected. Independent v2 development continues.

- [routes.csv](routes.csv): 78 selected official route×preset×seed results: 60 Dev10 plus18 predeclared held-out results.
- [attempts.csv](attempts.csv): 81 attempts, retaining all3 infrastructure failures and their known costs/missing data.
- [comparison.json](comparison.json), [comparison.md](comparison.md): nine campaign groups and nine within-dataset/seed preset pairs; G2 sources remain separate.
- [slope-summary.json](slope-summary.json): exact copy of the completed Town04 stationary full-brake slope hold diagnostic.
- [s-curve-summary.json](s-curve-summary.json): exact copy of all3 frozen-v1 real-S-curve development results; all3 cruise-speed gates failed.
- [s-curve-speed-analysis.json](s-curve-speed-analysis.json): strict recomputation of the existing speed gate from raw independent truth, plus measured command/actuation context and source hashes.
- [verification.json](verification.json): counts, finalized-result completeness and exact copy/source checks.

The78 official rows include69 completion100 results. These are route-oracle diagnostics with policy=none, not learned-planner leaderboard scores.
Do not collapse datasets into a tuning objective. Internal vendor TickRuntime remains an official failure; artificial harness caps are distinct.
Source report paths, hashes, campaign commits and missingness are retained in comparison.json. Original raw logs were not rewritten.

Reproduction (choose a new output directory; do not overwrite this one):

```bash
DATA_DIR=/data /data/envs/carla/bin/python scripts/b2d_controller_compare.py \
  --campaign /data/runs/b2d/controller/dev10 \
  --campaign /data/runs/b2d/controller/confirmation \
  --campaign /data/runs/b2d/controller/confirmation/holdout \
  --g2 /data/runs/b2d/controller/development3/summary.json \
  --g2 /data/runs/b2d/controller/development-s-v1/summary.json \
  --out NEW_OUTPUT_DIRECTORY
```

The[held-out figures](../../figures/v1-holdout/README.md) have their own source/code provenance and exact CSV.
The[confirmation](../raw-file-index/confirmation-v1.json) and[S-curve](../raw-file-index/development-s-v1.json) raw-file indexes
were captured after completed end events and after the recorded server PIDs had exited.
Slope hold covers5s after settling on a measured6.357° incline; it does not test approach braking.
