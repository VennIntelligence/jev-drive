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
| [research/trajectory-to-control.md](research/trajectory-to-control.md) | how a planner's trajectory becomes CARLA's 20 Hz `VehicleControl`: what CARLA ships, what the Bench2Drive baselines and openpilot do, who learns what, and the recommendation |
| [research/survey-counterfactual-video-gen.md](research/survey-counterfactual-video-gen.md) | literature status as of 2026-09-22: counterfactual pair evaluation and training, generative editing of real frames, video-generation models as planner representations, MiniMax H3 facts; what is done and what is still open |
| [research/openpilot-and-open-driving-models.md](research/openpilot-and-open-driving-models.md) | landscape as of 2026-09-24: openpilot model generations and its absence from every public leaderboard, specialist vs VLM-based planners on Bench2Drive / NAVSIM / WOD-E2E, and which open driving stacks have real commercial deployment |
| [research/openpilot-openloop-standing.md](research/openpilot-openloop-standing.md) | openpilot's open-loop standing (2026-09-26): WOD-E2E and NAVSIM side by side with our heads, Alpamayo, cv and the leaderboards; what NAVSIM's 2 Hz input does to the native plan; frozen `temporal` + thin head on NAVSIM; the continuation-share table |
| [research/prediag-2026-09/](research/prediag-2026-09/README.md) | the September 2026 pre-diagnostic round before method selection: train-split recheck, judge proposal, readout ladder, backbone ladder, CARLA gap and paired exam; results and decile figures collected in one place |
| [research/capability-vs-leaderboard.md](research/capability-vs-leaderboard.md) | direction skeleton (2026-09-24): driving ability split into a routine (R) and an emergent-reaction (E) layer, leaderboard score vs ability, the four measurements (hack audit, the zero-shot exam, R layer, P5 v1 E layer) and candidate extraction methods |
| [research/leaderboard-vs-ability.md](research/leaderboard-vs-ability.md) | synthesis of the two read-only leaderboard audits (2026-09-24): where the high scores come from per board, which boards' scores can stand as R-layer or E-layer evidence, cross-board consistency, reproduction gaps and noise, what to port vs extract, and the attack surfaces our CARLA paired exam must guard; the 45 open questions are answered in [synthesis_answers.md](research/results/leaderboard-text-analysis/synthesis_answers.md) |
| [research/behavior-layer-instruments.md](research/behavior-layer-instruments.md) | survey (2026-09-26) of instruments for the third layer, behaviour after the judgement (bypass, yielding / negotiation, recovery): what public boards can and cannot separate, how PDM-Lite and BehaviorAgent actually bypass, a zero-cost WOD-E2E nudge subset, and the P6 paired behaviour-mode exam draft |
| [research/state-space-policies.md](research/state-space-policies.md) | publicly released state-space (structured-state, no image) multi-agent driving RL policies as of 2026-09-26: who ships weights, licences, input formats, what runs on our data; BehaviorBench PPO / conditioned PPO installed and given a pre-registered bypass / negotiation / recovery exam on WOMD interactive val |
| [research/nohack-mechanisms.md](research/nohack-mechanisms.md) | mechanisms behind the methods that score high without leaderboard tricks (RAM, DAgger, tokenization, gate, V-JEPA 2 pretraining, ...): which gains reach the E layer, which are R-layer recipes, and how each maps onto our frozen-feature + thin-head paradigm |
| [research/ablation-matrix-inventory.md](research/ablation-matrix-inventory.md) | inventory (2026-09-26) of the backbone x head x exam ablation matrix: filled cells with sources, empty cells graded by cost, and 15 protocol inconsistencies to settle before a paper table |
| [research/articles/](research/articles/) | curated deep-dive articles and primers for humans (e.g. [trajectory-to-control](research/articles/trajectory-to-control/)) |
| [research/lit/](research/lit) | literature and deep-research reports |
| [research/figs/](research/figs) | figures referenced by the research docs and todos (PNG, committed) |
| [research/results/](research/results) | the small result files (CSV, metrics, timings) the decision log's tables were built from |
| [todos/README.md](todos/README.md) | experiment plans, one file per plan |
| [scripts/](scripts) | shell entry points run on the box |
| [jevdrive/](jevdrive) | the Python package: data, features, probes, planner |

Adding a doc, a topic or a figure: link it from its section's README, and from this table if it is a
new top-level place.
