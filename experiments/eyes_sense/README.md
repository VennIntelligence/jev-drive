# eyes_sense: same eyes, better car sense (2 x 2)

status: live
decisions: (pending) (inputs: 136, 232, 233, 236, 243, 244, 245, 248)
index: EYES (frozen 3-view bank) x SENSE (body lesson), one joint trainer, 2 seeds
key: experiments/eyes_sense/plans/2026-10-11-eyes-sense-prereg.md, experiments/alpasim/scripts/joint_train.py, experiments/alpasim/plans/2026-10-11-joint1-prereg.md, experiments/vis_train/scripts/vt_read.py, experiments/vis_train/scripts/vt_r.py, experiments/body1/lib/loss43.py, experiments/body1/scripts/bd4_train.py, experiments/op_parity/scripts/pp_train.py, jevdrive/bench/navsim.py

**Question.** In one joint policy training from the shipped weights (lambda-10 recipe), do two more frozen views (EYES: the untrained 3-view token
bank of decision 248's RW0) lower the > 45 deg cut-inside and off-road failure rates without a < 5 deg, NC or EC cost, and how do they compare
with the body lesson (SENSE: `P2H10S`) read in the same tables?

**Conclusion.** None yet (pre-registered 2026-10-11, [plans/2026-10-11-eyes-sense-prereg.md](plans/2026-10-11-eyes-sense-prereg.md)).

**Next.** Arms (a) P2H10, (b) + EYES, (c) + SENSE, (d) both, two seeds each, from `experiments/alpasim/scripts/joint_train.py`; (a) / (c) are the stored
`P2H10-F` / `P2H10S-F` if the trainer reproduces them bit for bit. Reads: navtest by flips and decision 248's no-regression table, navhard, test-time
`:noside` / `:mshuf` / `:sideoff`, nuPlan 700-scene closed loop for (a) / (c).

**Read more.** [plans/2026-10-11-eyes-sense-prereg.md](plans/2026-10-11-eyes-sense-prereg.md); decision 248 and experiments/vis_train/results/reads.md.

<!-- files:begin -->
<!-- files:end -->
