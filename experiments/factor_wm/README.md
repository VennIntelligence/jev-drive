# factor_wm: Factorized world model, exact ego, pretrained exogenous

status: live
decisions: (inputs: 54, 76, 103, 123, 132, 138, 139)
index: G1 fails: exact-ego DAgger fixes launch in-engine (0.18->0.02) but HUGSIM stalls 18->48
key: experiments/factor_wm/plans/2026-10-05-stage1-prereg.md, experiments/factor_wm/results/g0-verdict.md, experiments/factor_wm/results/g1.md, experiments/factor_wm/scripts/fw_common.py, experiments/factor_wm/scripts/fw_g0.py, research/factorized-world-model.md, experiments/op_dagger/scripts/dg_common.py, experiments/op_dagger/scripts/dg_roll.py, experiments/op_dagger/results/pilot.md, experiments/world_model/README.md

**Question.** Does a driving world model whose ego part is exact (kinematics + geometric reprojection of real frames) and whose exogenous part
comes from fine-tuned industrial pretrained models train openpilot better than learned world models? Stage 1 (ego part only): does on-policy
training of openpilot in the exact-ego simulator, with closed longitudinal control and pose-indexed source frames, fix launch / stall / spin /
turn failures on HUGSIM and NAVSIM beyond decision 132's lateral-only DAgger?

**Conclusion.** G0 (no training, WOD val, [results/g0-verdict.md](results/g0-verdict.md)): shipped Cinque fails in the exact-ego engine
(time-synchronous logged frames, SPEC lateral + longitudinal, 8 s) on 0.52 of perturbed rollouts, with stall 0.17, heading 0.17 and lane 0.18
(G0b passes). On real frames it already under-accelerates at launch (t0 median -0.09 m/s2) and understeers in sharp turns (gain 0.63).
Depth reprojection with wide fill is not better than the plane warp at launch (0.99x) or in sharp turns (0.85x), so G0a fails and the engine
stays plane. Literature: the full combination is unpublished, every component exists
([research/factorized-world-model.md](../../research/factorized-world-model.md)).

**G1** ([results/g1.md](results/g1.md)): fails as registered. S3 (exact-ego engine, SPEC lateral + longitudinal, 8 s, DAgger) fixes launch inside its
own engine (stall 0.18 to 0.02), but on HUGSIM 64 launch stalls rise 18 to 48 and HD falls 0.286 to 0.182. 0 of the 12 ego-failure scenes are fixed, and
S0 ego failures turn into collisions (14) rather than fixes (2). Open loop it gains (navhard +2.44, navtest +0.26). S2 (decision 132's lateral 2 s engine) is
the best closed-loop arm (HD 0.328, 15 completes) but fixes only 2 of 12.

**Next.** None scheduled. Two cheap checks would explain the transfer failure: leaked log motion in the time-synchronous frames, and action[1] vs plan speed
(results/g1.md).

**Read more.** [plans/2026-10-05-stage1-prereg.md](plans/2026-10-05-stage1-prereg.md) (stage 1 design, lines, compute; stage 2 sketch on WL-2
CARLA fork pairs), [research/factorized-world-model.md](../../research/factorized-world-model.md) (novelty check, ~40 works).

<!-- files:begin -->
<!-- files:end -->
