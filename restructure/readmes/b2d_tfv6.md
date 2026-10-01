# b2d_tfv6: TFv6 controller campaign and diagnoses

status: concluded
decisions: 31
headline: TFv6 representation effect +14.3 DS [+5.1, +25.9]; controller main effect +1.0 [-12.2, +15.4], not detected

**Question.** Does the high TFv6 Bench2Drive score come from waypoint tracking or from the route + target-speed representation, and does our controller (C) or its lateral correction (D) change DS.

**Conclusion.** Representation effect A - B = +14.3 DS [+5.1, +25.9] over 48 route-seed pairs (decisions 31). Controller main effect C - B was first reported as -6.1 [-20.8, +9.6], then voided: the tangent rule created phantom targets at standstill. After the W2b rerun (96 cases) it is +1.0 [-12.2, +15.4], still not detected (not equivalent), and D - C is +0.4 [-2.7, +3.5]; the 220-route tier was not run. D3 diagnosis of the remaining failure modes (held-out route deviation, 2091 start stall) is in the same entry.

**Read more.** research/trajectory-to-control.md, `git show bcbdde4:todos/2026-09-23-tfv6-controller/protocol.md`, todos/2026-09-23-tfv6-controller/report.md, todos/2026-09-23-tfv6-controller/report-w2b.md, todos/2026-09-23-tfv6-controller/diagnosis-tangent.md.

<!-- files:begin -->
<!-- files:end -->
