# statepol: public state-space RL policies on WOMD behaviour exams

status: concluded
decisions: 51
headline: Only 2 BehaviorBench PPO weights ran: yield 43.0% vs null 2.0%, aggressive PPO bypasses 79.5% with 15.8% crashes

**Question.** Are there public state-space (self-play RL) policies that could serve as a bypass / negotiation / recovery executor after openpilot's desire, measured with counterfactual pairs on WOMD validation_interactive?

**Conclusion.** Only the two BehaviorBench weights (`simple_ppo.pt`, `conditioned_ppo.pt`) could be installed and run (Gigaflow has no weights, GPUDrive needs a Madrona build). Negotiation: PPO yields 43.0% (null 2.0%) with lateral avoidance 37.0%, about 3x IDM's 13.8%; bypass depends on the reward coefficients (aggressive conditioned PPO bypasses 79.5%, null 19.9%, 15.8% crashes); no policy recovers. Neither can serve directly as an executor after openpilot desire (decisions 51). Note: a PufferDrive eval bug makes goal-reaching vehicles invisible but collidable, so v1 numbers were voided and only the fixed v2 is reported.

**Read more.** research/state-space-policies.md, research/behavior-layer-instruments.md, `git show bcbdde4:todos/2026-09-26-state-space-policies.md`

<!-- files:begin -->
<!-- files:end -->
