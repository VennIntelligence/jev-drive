# Docs index

How-to docs, one line each. Add a line when you add a doc. Each doc starts with "Read this when ..." and ends
with "Last verified: <date>"; one topic per file, keep it short.

| Doc | Read this when |
|---|---|
| [remote-box.md](remote-box.md) | you log in to or use the GPU box |
| [tokyo-box.md](tokyo-box.md) | you need to see CARLA on a monitor |
| [network-proxy.md](network-proxy.md) | a download fails or is slow on the box |
| [storage.md](storage.md) | you decide where models, data or checkpoints go |
| [python-env.md](python-env.md) | you run project code on the box or add a dependency |
| [lib.md](lib.md) | you start a new experiment (Run, splits, cache, par, stats) |
| [long-runs.md](long-runs.md) | you start a job over a minute: tmux, logs, curves, the pre-run checklist |
| [web-reader.md](web-reader.md) | you show docs to someone without pushing |
| [closed-loop-runbook.md](closed-loop-runbook.md) | any CARLA / Bench2Drive run: entry points, `jevdrive.cl`, profile default, capacity |
| [closed-loop-acceptance.md](closed-loop-acceptance.md) | before scoring a model in closed loop: what is accepted |
| [carla.md](carla.md) | you need a CARLA server on the box |
| [bench2drive-cost.md](bench2drive-cost.md) | you need what a Bench2Drive evaluation costs and why |
| [b2d-controller.md](b2d-controller.md), [b2d-controller-lateral.md](b2d-controller-lateral.md), [b2d-tcp-controller.md](b2d-tcp-controller.md) | you use the fixed trajectory controller or its TCP / lateral variants |
| [zeroshot-adapters.md](zeroshot-adapters.md) | you score an external driving model in closed loop |
| [driving-runtime.md](driving-runtime.md), [tokyo-python-optimization.md](tokyo-python-optimization.md) | you need frame / preview helpers or the TCP Python speed-ups |
| [waymo-e2e.md](waymo-e2e.md), [navsim.md](navsim.md), [hugsim.md](hugsim.md) | you need WOD-E2E, NAVSIM / OpenScene or HUGSIM data and setup |
| [baselines.md](baselines.md) | you need a baseline model's latency on our card |

Old paths from before the 2026-10 restructure: [path-map.tsv](path-map.tsv).
