# factor_wm: Factorized world model, exact ego, pretrained exogenous

status: live
decisions: (inputs: 54, 76, 103, 123, 132, 138, 139)
index: Stage 1 pre-registered: on-policy openpilot in exact-ego reprojection sim; no runs yet
key: experiments/factor_wm/plans/2026-10-05-stage1-prereg.md, research/factorized-world-model.md, experiments/op_dagger/scripts/dg_common.py, experiments/op_dagger/scripts/dg_roll.py, experiments/op_dagger/results/pilot.md, experiments/world_model/README.md

**Question.** Does a driving world model whose ego part is exact (kinematics + geometric reprojection of real frames) and whose exogenous part
comes from fine-tuned industrial pretrained models train openpilot better than learned world models? Stage 1 (ego part only): does on-policy
training of openpilot in the exact-ego simulator, with closed longitudinal control and pose-indexed source frames, fix launch / stall / spin /
turn failures on HUGSIM and NAVSIM beyond decision 132's lateral-only DAgger?

**Conclusion.** None yet; nothing has been run. Literature check: the full combination is unpublished, every component exists
([research/factorized-world-model.md](../../research/factorized-world-model.md)).

**Next.** Gate G0 of the stage 1 pre-registration (engine fidelity at launch / sharp turns, and whether shipped Cinque fails in the engine at all).

**Read more.** [plans/2026-10-05-stage1-prereg.md](plans/2026-10-05-stage1-prereg.md) (stage 1 design, lines, compute; stage 2 sketch on WL-2
CARLA fork pairs), [research/factorized-world-model.md](../../research/factorized-world-model.md) (novelty check, ~40 works).

<!-- files:begin -->
<!-- files:end -->
