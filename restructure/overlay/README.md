# jev-drive

Autonomous driving research (VennIntelligence). Working rules: [CLAUDE.md](CLAUDE.md).
Every doc in the repo is reachable from this page through one of the places below.

| Place | What |
|---|---|
| [experiments/INDEX.md](experiments/INDEX.md) | every experiment, one line each; its README links code, results, figures and narrative docs |
| [research/decisions.md](research/decisions.md) | the decision log: one line per standing decision, full entries in `research/decisions/` |
| [research/README.md](research/README.md) | narrative layer (Chinese): writing rules, surveys and cross-cutting essays |
| [docs/README.md](docs/README.md) | how-to docs: GPU box, Tokyo box, storage, env, data, long runs, closed loop, shared libraries |
| [jevdrive/](jevdrive) | shared Python package ([docs/lib.md](docs/lib.md)) |
| [scripts/](scripts) | shared box entry points (tmux, downloads, CARLA server, Bench2Drive harness) |
| [tests/](tests), [patches/](patches), [tools/](tools) | shared tests, third-party patches, repo tooling |

Adding a doc: link it from the README of the place it belongs to (topic README, research/README.md or
docs/README.md), not from here.
