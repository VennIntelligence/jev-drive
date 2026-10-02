# op_img_cmd: route command drawn into openpilot's image

status: live
decisions: (pending)
index: Route drawn on openpilot input frames: zero-shot test (running)
key: experiments/op_img_cmd/plans/2026-10-04-img-cmd-prereg.md, experiments/op_img_cmd/scripts/img_overlay.py, experiments/op_img_cmd/scripts/img_run.py, experiments/op_img_cmd/scripts/img_geom_nav.py, experiments/op_img_cmd/scripts/img_sheet.py

**Question.** Does openpilot (Cinque, frozen) follow a route command given through the image (painted route, blocked
other branches, a sign) instead of the desire input, zero-shot; and if only partly, can light fine-tuning teach it?

**Conclusion.** Pending.

**Next.** Zero-shot on navtrain junction approaches (decision 93 frames in the op_lb pool), then CARLA.

**Read more.** [plans/2026-10-04-img-cmd-prereg.md](plans/2026-10-04-img-cmd-prereg.md) (samples, overlay families,
metrics and verdict rules, fixed before the full run).

<!-- files:begin -->
<!-- files:end -->
