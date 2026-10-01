# b2d_controller: Fixed-trajectory controller for Bench2Drive

status: concluded
decisions: 26, 27, 29, 30
headline: No first-round controller qualified as default; PI had higher DS (59.1 vs 53.8) but worse completion, +10.8% lateral RMS

**Question.** Can a controller that tracks a fixed time-parameterised trajectory beat the CARLA/TCP default in Bench2Drive closed loop, judged per driving behaviour and not by DS alone.

**Conclusion.** No first-round candidate qualified as default (decisions 26): in the Dev10 pursuit/CARLA/TCP comparison all three completed 9/10, and lateral RMS did not improve across the board. The v3/v4 PI candidate had higher DS (59.147 vs 53.811) but 16/20 vs 17/20 completion and +10.84% lateral RMS, so CLI carla/vendor stays (decisions 27). Hermite interpolation and the rear-axle propagation term were rejected or failed one heading guard (decisions 29). Turn error traced to two control-law mismatches, slip-frame pursuit plus Ackermann inversion: window CTE RMS on the sharp right turn 26966 fell .392 to .089 m, but the steer-rate and lateral-acceleration action guards failed, so the default is unchanged (decisions 30). Later TFv6 runs found no controller main effect (decisions 31, see b2d_tfv6).

**Read more.** docs/b2d-controller.md, docs/b2d-controller-lateral.md, research/trajectory-to-control.md, `git show bcbdde4:todos/2026-09-22-b2d-controller.md`, todos/2026-09-22-b2d-controller/final-report.md, todos/2026-09-23-lateral-followup/final-report.md, todos/2026-09-23-controller-next/lateral-v2-report.md.

<!-- files:begin -->
<!-- files:end -->
