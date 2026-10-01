# b2d_controller_eval: Frozen L1/L2/L3 controller evaluation

status: concluded
decisions: 41
index: Ours win L1 (ramp 0.92 vs 2.38), not closed loop (DS 86 vs 95)

**Question.** How to judge a controller when DS cannot: L1, L2, L3 layers?

**Conclusion.** L1 ramp C/D 0.92/0.90 vs TFv6 A/B 2.38/2.10; closed-loop DS A 94.1, B 95.0, C 86.5, D 86.0: none better (decisions 41, pending).

**Read more.** research/trajectory-to-control.md, docs/b2d-controller.md

<!-- files:begin -->
## Files

- `b2d_controller_eval_campaign.py` (archive): Resumable Task 10 CARLA campaign with …
- `b2d_controller_eval_prepare_v2.py` (archive): Task 10 amendment v2 inputs
- `b2d_controller_eval_l1_v2.py` (archive): one validator per route, N routes in …
- `b2d_controller_eval_l1_v3_score.py` (archive): merges the v2, P and v3 batches per …
- `b2d_controller_eval_l23_score.py` (archive): Frozen plan-conditioned Task 10 L2 and …
- `b2d_controller_eval_closed_loop.sh` (archive): Closed loop for one candidate config as …
- `b2d_controller_eval_v3_l1.sh` (archive): crawl reference + plan interfaces
- `b2d_controller_eval_refs.py` (archive): Controller-neutral, time-parameterized …
- `b2d_controller_eval_tune_next.py` (archive): Pre-registered round 2 of the dev L1 …
- `controller-scorecard.md` (experiments/b2d_tfv6/results/tfv6-controller): Controller scorecard（Task 10 v2）

[archive/](archive/) 23 one-off code
<!-- files:end -->
