# jev-drive

Autonomous driving research (VennIntelligence). Rules for working in this repo: [CLAUDE.md](CLAUDE.md).

This file is the top-level index: every doc in the repo is reachable from here.

| Where | What |
|---|---|
| [docs/README.md](docs/README.md) | how-to docs: the GPU box, storage, Python env, data, long runs, baselines |
| [research/README.md](research/README.md) | research topics and the writing conventions for them |
| [research/decisions.md](research/decisions.md) | the shared decision log: what we decided, why, and whether it is still provisional |
| [research/frozen-vlm-planner.md](research/frozen-vlm-planner.md) | current main topic: frozen VLM + thin head trajectory planning |
| [research/qwen-latent-driving.md](research/qwen-latent-driving.md) | layer probe study, now the analysis part of the main topic |
| [research/survey-thin-head.md](research/survey-thin-head.md) | literature survey for group meetings: world models, VLAs, thin heads, the Waymo E2E leaderboard and how crowded the line is |
| [research/lit/](research/lit) | literature and deep-research reports |
| [research/figs/](research/figs) | figures referenced by the research docs and todos (PNG, committed) |
| [todos/README.md](todos/README.md) | experiment plans, one file per plan |
| [scripts/](scripts) | shell entry points run on the box |
| [jevdrive/](jevdrive) | the Python package: data, features, probes, planner |

Adding a doc, a topic or a figure: link it from its section's README, and from this table if it is a
new top-level place.
