# critic_pilot: a learned consequence critic for on-policy decisions

status: concluded
decisions: pending
index: learned critic fails L1-L3: any-zero 2 s AUC 0.85 vs hand rule 0.83

**Question.** From a decision's frozen tokens, ego vector, route memory and a candidate plan, can a small learned critic predict the
closed-loop consequence (zero events within 2 / 4 s by type, lateral offset from the route at +2 s, progress at +4 s, yaw-rate peak) well
enough on a policy it never saw to replace the hand-written hinges (S, route hinge, stop-line B, progress P)?

**Conclusion.** Not feasible as specified: on RSG held (unseen policy, trained on RB0 + RBSG) any-zero-within-2 s AUC 0.846 [0.738, 0.935]
(L1 >= 0.90 fail; the hand rule |served plan offset from route at 10 m| reaches 0.825 and 0.996 on corridor), lat2 R^2 0.445 (L2 fail),
counterfactual flip 0.41 (L3 fail); vision tokens add nothing beyond offroad, and with one executed plan per state the critic ranks plans
weakly (gradient right-leaning on 55-81 % of rows, 8-30x smaller than S). Decision entry pending (coordinator).

**Read more.** Prereg [2026-10-13-critic-pilot-prereg.md](../route_follow/plans/2026-10-13-critic-pilot-prereg.md); pool contract
[on_pool_format.md](../onpol_nuplan/plans/on_pool_format.md); the hinge framework the critic would replace
[research/self-consequence/design.md](../../research/self-consequence/design.md).

<!-- files:begin -->
## Files

- `2026-10-13-critic-pilot-prereg.md` (experiments/route_follow/plans): can a learned consequence critic …
- `critic_pilot.md` (results): CRITIC-PILOT read
- `cp_labels.py` (scripts): the closed-loop consequence of every
- `cp_train.py` (scripts): the five arms, the held reads
- `critic.py` (lib): a small transformer that reads one
- `critic_pilot.json` (results): 
- `design.md` (research/self-consequence): SELF-CON 设计稿：冻结编码器上的 policy 学自己 plan …
- `286.md` (research/decisions): ROUTE-NIGHT：从出厂权重重训的路线记忆底座 RB0-F …
- `266.md` (research/decisions): ANCHOR-E1：served plan …

[results/](results/) 2 result files · [lib/](lib/) 1 library · [scripts/](scripts/) 2 entry points
<!-- files:end -->
