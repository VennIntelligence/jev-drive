# b2d_tfv6: TFv6 controller campaign and diagnoses

status: concluded
decisions: 31
index: representation +14.3 DS [+5.1, +25.9]; controller +1.0, not detected
key: scripts/b2d_tfv6_campaign.py, scripts/b2d_tfv6_w2b.py, scripts/b2d_tfv6_w2b_analyze.py, scripts/b2d_tfv6_controller_agent.py, scripts/b2d_tfv6_d3.py, scripts/b2d_tfv6_d3_analyze.py, scripts/b2d_tfv6_plant_diagnosis.py, scripts/test_b2d_tfv6_w2b.py

**Question.** Does TFv6's B2D score come from waypoint tracking or the route + speed representation, and does our controller change DS?

**Conclusion.** Representation A - B = +14.3 DS [+5.1, +25.9]; controller C - B -6.1 was voided (tangent phantom targets), W2b rerun +1.0 [-12.2, +15.4], not detected; D - C +0.4 (decisions 31).

**Read more.** research/trajectory-to-control.md, todos/2026-09-23-tfv6-controller/report-w2b.md

<!-- files:begin -->
<!-- files:end -->
