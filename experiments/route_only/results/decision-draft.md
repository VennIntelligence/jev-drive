# Draft decision entry (PENDING, for the coordinator; proposed number 287)

Index line (research/decisions.md):

| 287 | ROUTE-ONLY (not trained; RSG-L18-s0 with the served path replaced by the AlpaSim route carried as memory, speed = the plan's own arc lengths; pre-registered, box only, PAI not run): **not supported**. R-own (carried route where the memory covers the ego, model plan in the gap) ties RSG on held 452 / 18 (score -0.003 [-0.013, +0.006], zeros +1, turn completion 66.7 % identical, H1-H4 fail by the letter) and fails the public guard (score -0.0068 [-0.0105, -0.0031], at-fault zeros +9.0 [+3.1, +15.0], 10 broken / 0 fixed). The route path reaches only 48 % / 30 % of decisions because the messages start 40 m ahead and 23 / 24 held turns lie in that gap; where it is served, 9-13 % of decisions put the ego more than 6 m off its own carried route (snaps, loops), and 7 / 10 public breaks come from there. Bridging the gap (R-bridge, Amendment 1) is far worse (zeros 90 vs 26, offroad 74, turn completion 46 %). The privileged log-speed arm gains +0.052 and 92 % turn completion, mostly from speed on the model's own path (decision 285); see [287](decisions/287.md) | medium (one checkpoint, one read per arm; switch-off re-run bitwise, so the paired reference is exact) | holds; route-as-path closed in this form; speed is the lever |

Entry body (research/decisions/287.md), English:

**287. ROUTE-ONLY: the AlpaSim route carried as memory cannot be served as the path. Where it is served it ties RSG-L18-s0 at best
and costs public at-fault zeros; bridging the gap is far worse; speed, not path, is the lever the privileged arm shows.**

- Date 2026-10-13. Prereg `experiments/route_only/plans/2026-10-13-route-only-prereg.md` (`0689dd80`, Amendment 1 `dddbf628`).
  Results `experiments/route_only/results/route_only.md`. Switch `JEV_ROUTE` (`experiments/alpasim/lib/route_path.py`).
- Setting, reads, lines, mechanisms and caveats: as in the results file. Registered verdicts: H1, H2 (by the letter), H3, H4 and G1
  fail for R-own; all fail for R-bridge.
- Evidence that would overturn it: a route source that covers the near 40 m (e.g. the route regenerated from the map at each
  decision, which a submission does not receive), or a plausibility-gated R-own that passes G1 (only removes the off-route breaks;
  cannot raise turn completion, since the turns sit in the gap).
- Follow-ups (not run): PAI 281 at gains 20 when Tokyo frees (pai_driver.py needs the same hook); a gated R-own only if a reason
  beyond "remove the damage" appears.
