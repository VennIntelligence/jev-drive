# route_follow: does the served policy use its route

status: live
decisions: (none yet; inputs 237, 245, 268, 282, 283, body1 Amendment 7)
index: route counterfactual on served tokens; route enters only as a 3-way command
key: research/next-round/route-follow.md, experiments/route_follow/scripts/rf_cf.py, experiments/alpasim/lib/sh30_driver.py, experiments/alpasim/lib/pai_driver.py, lib/parity_adapter.py, experiments/road_probe/scripts/rp.py

**Question.** At junction turns and lane changes, does the plan of the served checkpoints move at the controller's horizon (1-2 s) when the
route they are given changes, i.e. is the route unused, used too weakly, or used too late?

**Conclusion.** (pending; the registered read is in research/next-round/route-follow.md section 2.1)

**Next.** Read, then the ranked proposals of research/next-round/route-follow.md section 3.

**Read more.** [research/next-round/route-follow.md](../../research/next-round/route-follow.md) (Chinese; train / serve chain, registration,
reads, proposals). Box: `$DATA_DIR/runs/route_follow/` (cf/, nav.json, read/<ts>/). Inputs reused unedited: `$DATA_DIR/runs/road_probe/tok/{held,zero}`
(served tokens) and `labels/{held,zero}.parquet` (decision 275).

<!-- files:begin -->
<!-- files:end -->
