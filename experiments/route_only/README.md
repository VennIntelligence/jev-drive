# route_only (ROUTE-ONLY, JEV_ROUTE, route as path, longitudinal-only): route path, model speed

status: live
decisions: pending
index: served path = carried AlpaSim route, speed = RSG-L18-s0 plan; held + public read

**Question.** If the served path is the AlpaSim route carried as memory and only the speed profile comes from the model's plan
(openpilot-style longitudinal-only), what does RSG-L18-s0 score on held 460 / 18 decisions and public 1491 / 10 decisions?

**Conclusion.** Pending.

**Next.** Held and public reads; PAI 281 is a follow-up (not run: Tokyo full; pai_driver.py not hooked).

**Read more.** [plans/2026-10-13-route-only-prereg.md](plans/2026-10-13-route-only-prereg.md). Switch `JEV_ROUTE=off|own|log`
(experiments/alpasim/lib/route_path.py, default off). Box: `$DATA_DIR/runs/route_only/`.

<!-- files:begin -->
## Files


[results/](results/) 0 result files · [figs/](figs/) 0 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 1 entry points
<!-- files:end -->
