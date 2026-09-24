# Docs index

One line per doc. Add a line here whenever you add a doc.

| Doc | Read this when |
|---|---|
| [remote-box.md](remote-box.md) | you need to log in to or use the GPU box |
| [tokyo-box.md](tokyo-box.md) | you need to *see* CARLA on a real monitor: a window, a manual drive, a screenshot |
| [network-proxy.md](network-proxy.md) | a download fails or is slow on the box |
| [storage.md](storage.md) | you need to decide where models, data or checkpoints go |
| [python-env.md](python-env.md) | you need to run project code on the box or add a dependency |
| [baselines.md](baselines.md) | you need a competitor model's latency on our card, or want to re-run one |
| [long-runs.md](long-runs.md) | you start a job that takes more than a minute, or need its logs or curves |
| [waymo-e2e.md](waymo-e2e.md) | you need the Waymo E2E driving data, or need to download or re-fetch it |
| [carla.md](carla.md) | you need a CARLA simulator on the box, or want the cost of a Bench2Drive evaluation |
| [bench2drive-cost.md](bench2drive-cost.md) | you need to run a Bench2Drive closed-loop evaluation, or what one costs and why |
| [b2d-controller.md](b2d-controller.md) | you need the fixed controller API, diagnostic campaigns, acceptance boundaries, or preserved plots |
| [b2d-tcp-controller.md](b2d-tcp-controller.md) | you need the actual TCP longitudinal comparison, its shared execution limits, or telemetry contract |
| [b2d-controller-lateral.md](b2d-controller-lateral.md) | you need fixed-window turn tests, optional rear pose propagation, or the isolated GPU-library recovery |
| [driving-runtime.md](driving-runtime.md) | you need reusable frame/preview helpers or want to connect another driving model |
| [tokyo-python-optimization.md](tokyo-python-optimization.md) | you need the measured TCP Python optimizations and their switches |
| [navsim.md](navsim.md) | you need the NAVSIM / OpenScene data on the box, or need to re-download it |
| [hugsim.md](hugsim.md) | you need the HUGSIM closed-loop benchmark data on the box, or need to re-download it |

Rules:
- One topic per file.
- Start each doc with "Read this when ...".
- End each doc with "Last verified: <date>".
- Keep it short.
