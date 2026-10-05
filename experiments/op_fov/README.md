# op_fov: wider field of view for openpilot, zero-shot

status: live
decisions: (pending)
index: Wider wide-camera FOV on real comma1M turns, shipped Cinque, zero-shot
key: experiments/op_fov/README.md experiments/op_fov/plans/2026-10-05-fov-prereg.md experiments/op_fov/scripts/fov_replay.py experiments/op_fov/scripts/fov_report.py jevdrive/openpilot/frames.py

**Question.** Does feeding the shipped Cinque a wider field of view (smaller model-frame focal, same 512x256 frames) help it on
sharp / 90 deg turns on real road data, without breaking its scale on straight roads?

**Conclusion.** Pending.

**Next.** Pilot (8 turns + 4 straights x 4 arms), then the full event set if not a clear negative.

**Read more.** Pre-registration: [plans/2026-10-05-fov-prereg.md](plans/2026-10-05-fov-prereg.md).

<!-- files:begin -->
<!-- files:end -->
