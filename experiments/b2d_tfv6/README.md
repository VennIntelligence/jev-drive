# b2d_tfv6: TFv6 controller campaign and diagnoses

status: concluded
decisions: 31
index: representation +14.3 DS [+5.1, +25.9]; controller +1.0, not detected

**Question.** Does TFv6's B2D score come from waypoint tracking or the route + speed representation, and does our controller change DS?

**Conclusion.** Representation A - B = +14.3 DS [+5.1, +25.9]; controller C - B -6.1 was voided (tangent phantom targets), W2b rerun +1.0 [-12.2, +15.4], not detected; D - C +0.4 (decisions 31).

**Read more.** research/trajectory-to-control.md, experiments/b2d_tfv6/results/tfv6-controller/report-w2b.md

<!-- files:begin -->
## Files

- `b2d_tfv6_campaign.py` (lib): Resumable, route-paired TFv6 campaign …
- `b2d_tfv6_w2b.py` (lib): W2b staged campaign with immediate …
- `b2d_tfv6_w2b_analyze.py` (archive): Pair W2b with canonical W2
- `b2d_tfv6_controller_agent.py` (archive): Four-arm TFv6 controller experiment
- `b2d_tfv6_d3.py` (archive): D3b diagnostic reruns, with per-case …
- `b2d_tfv6_d3_analyze.py` (archive): Summarize the D3b factorial and Kalman …
- `b2d_tfv6_plant_diagnosis.py` (archive): Read logged truth and controls
- `test_b2d_tfv6_w2b.py` (archive): W2b stage partition and immediate …

[archive/](archive/) 40 one-off code · [results/](results/) 200 result files · [lib/](lib/) 2 library
<!-- files:end -->
