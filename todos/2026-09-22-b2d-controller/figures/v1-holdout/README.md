# Controller diagnostic figures

Diagnostic route oracle; policy = none.

PNG: 300 dpi. PDF: vector output with embedded TrueType fonts.
Exact plotted observations are in adjacent CSV files; blank values mean unavailable.
The manifest records source paths, SHA256 hashes, snapshots of JSON summaries, versions, and warnings.
This output is immutable to the plot CLI: use a new versioned --out directory for subsequent campaigns.
Raw trajectories/telemetry remain at the paths in the manifest. All raw samples used by the trajectory and stop plots are also exported to CSV.

Run:

```bash
/data/envs/carla/bin/python scripts/b2d_controller_plot.py --development /data/runs/b2d/controller/development3 --campaign /data/runs/b2d/controller/confirmation/holdout --out /home/ujs/mycode/jev-drive/todos/2026-09-22-b2d-controller/figures/v1-holdout --route-id 26966 --label Holdout --campaign-only
```

The missing control.jsonl for pursuit/2084/attempt1 is an infrastructure failure before telemetry. It remains in the19-attempt CSV and source warnings; no zero-error sample is substituted.
