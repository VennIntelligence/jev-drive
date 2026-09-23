# Controller diagnostic figures

Diagnostic route oracle; policy = none.

PNG: 300 dpi. PDF: vector output with embedded TrueType fonts.
Exact plotted observations are in adjacent CSV files; blank values mean unavailable.
The manifest records source paths, SHA256 hashes, snapshots of JSON summaries, versions, and warnings.
This output is immutable to the plot CLI: use a new versioned --out directory for subsequent campaigns.
Raw trajectories/telemetry remain at the paths in the manifest. All raw samples used by the trajectory and stop plots are also exported to CSV.

Run:

```bash
/data/envs/carla/bin/python scripts/b2d_controller_plot.py --development /data/runs/b2d/controller/development-v4 --campaign /data/runs/b2d/controller/formal-v4 --out /data/worktrees/jev-drive-controller-v2/todos/2026-09-22-b2d-controller/figures/formal-v4 --route-id 26966 --label 'Frozen v4 route oracle'
```
