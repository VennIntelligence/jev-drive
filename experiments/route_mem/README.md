# route_mem: route-memory input branch for the nuPlan driver

status: live
decisions: (inputs 266, 272, 275, 278, 279; contrast 127-134, 189, 204)
index: route memory as an adapter branch on YS2G's recipe; pre-registered, training queued

**Question.** Decision 279 showed that the AlpaSim route waypoints the driver received earlier, carried into the current frame by its own ego
poses, place the lane centre to 0.2-0.35 m inside the junction stretch where the frozen eyes miss by 1.1-1.7 m (decision 275). Fed to the
adapter as a route-memory branch (Cinque frozen, command kept) on `YS2G`'s recipe, do the junction road-exit zeros fall where the memory reaches
and stay where it does not?

**Next.** Training `YRM-L18-s{0,1}` (YS2G's recipe + the branch), gates G-init / G-serve / G-dev / G-rely, then the held 452 read at 18
decisions against `YS2G` and `YS-L18` (plans/2026-10-12-route-mem-prereg.md).

**Read more.** [plans/2026-10-12-route-mem-prereg.md](plans/2026-10-12-route-mem-prereg.md) (Chinese; design, gates, registered lines, how it
differs from decisions 127-134 / 189 / 204). Memory = every route message of the episode, carried by the ego poses, merged, 40 points at -6 .. +72 m
with a validity mask (`lib/rmem.py`, the same code offline and in the served driver); the served driver uses it only for a checkpoint with a
`route_mem.json` sidecar, `SH30_RMEM=0` feeds an empty memory. Box: `$DATA_DIR/runs/route_mem/` (feat/, rely/, served/, read/, cover/).

<!-- files:begin -->
## Files

- `2026-10-12-route-mem-prereg.md` (plans): ROUTE-MEM 预登记：driver 记住自己收到过的 …
- `rmem.py` (lib): the route memory a driver keeps of the …
- `rm_build.py` (scripts): the route memory of every training / …
- `rm_train.py` (scripts): onpol_nuplan's on_train.py `train` (its …
- `rm_chain.sh` (scripts): Every job goes through the GPU pool
- `sh30_driver.py` (experiments/alpasim/lib): SH30 as an AlpaSim nuPlan-track driver
- `sh30_core.py` (experiments/alpasim/lib): SH30 as a causal 2 Hz planner
- `README.md` (experiments/local_map): carried probe and route at the zeros

[results/](results/) 0 result files · [figs/](figs/) 0 figures · [plans/](plans/) 1 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 3 entry points
<!-- files:end -->
