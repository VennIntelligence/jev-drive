# b2d_privileged: privileged-rule ceilings on low-scoring B2D routes

status: live
decisions: 82
headline: Diagnostic only: red light + green release lifts DS 75.0 to 95.0 (+20); all rules together hurt (collisions 5 to 17)

**Question.** How much DS and how many infractions come back if op-drive's arbitration gets CARLA ground truth (light state, obstacle, other vehicles) on 7 historically low-scoring dev routes?

**Conclusion.** Diagnostic only (44 runs, 7 routes x 2 traffic seeds; the planned confirmation batch was stopped after a serialization bug). Red light + green release: DS 75.0 -> 95.0 (+20.0 [+15, +30]) and official red-light runs 5 -> 1; bypass raises RC on the construction route 31.9 -> 100 but collisions follow; junction rule +2.0 [-0.2, +6.2] with no ceiling measured (three different scenarios); all three together collisions 5 -> 17 and DS -1.6, rules hurt each other (decisions 82).

**Next.** Validate the red-light rule on all dev and more red-light routes (paired CI lower bound > 0, no new deadlocks), then inject skills one scene type at a time with a no-harm test on non-target routes.

**Read more.** research/openpilot-closedloop-integration.md, docs/closed-loop-runbook.md, plans/2026-10-01-b2d-privileged-ceiling.md; task prompts: `git show bcbdde4:tmp/2026-10-01-b2d-privileged-ceiling-prompt.md`, `git show bcbdde4:tmp/2026-10-01-b2d-red-light-prompt.md`

<!-- files:begin -->
<!-- files:end -->
