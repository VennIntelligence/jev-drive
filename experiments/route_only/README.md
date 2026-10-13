# route_only (ROUTE-ONLY, JEV_ROUTE, route as path, longitudinal-only): route path, model speed

status: concluded
decisions: pending
index: route-as-path not supported: held ties RSG, public at-fault +9; route covers 48 % of decisions

**Question.** If the served path is the AlpaSim route carried as memory and only the speed profile comes from the model's plan
(openpilot-style longitudinal-only), what does RSG-L18-s0 score on held 460 / 18 decisions and public 1491 / 10 decisions?

**Conclusion.** Not supported. R-own ties RSG-L18-s0 on held 452 / 18 (score -0.003 [-0.013, +0.006], turn completion identical
66.7 %) and fails the public guard (score -0.0068, at-fault zeros +9.0 [+3.1, +15.0]). The route reaches only 48 % / 30 % of
decisions (the messages start 40 m ahead; 23 / 24 turns lie in the gap), and 9-13 % of served decisions are > 6 m off the route.
Bridging the gap (R-bridge) gives 90 zeros vs 26. The privileged log-speed arm (+0.052, 92 % turns) points at speed (draft decision 287).



**Read more.** [plans/2026-10-13-route-only-prereg.md](plans/2026-10-13-route-only-prereg.md). Switch `JEV_ROUTE=off|own|log`
(experiments/alpasim/lib/route_path.py, default off). Results: [results/route_only.md](results/route_only.md). PAI 281 not run (Tokyo full; pai_driver.py not hooked). Box: `$DATA_DIR/runs/route_only/`.

<!-- files:begin -->
## Files


[results/](results/) 14 result files · [figs/](figs/) 7 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 5 entry points
<!-- files:end -->
