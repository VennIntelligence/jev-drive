# Frozen five-route controller development comparison

All 20 v2 cases and 15 fixed v1 cases are included, including failures. See cases.csv for all original gates and matched-comparisons.csv for paired differences.

The v1/v2 comparison changes the adapter and generated path. The v2 pursuit additive/max comparison holds that adapter fixed. These are privileged route diagnostics, not leaderboard scores.

Moving CTE uses |true signed speed| >= 0.5 m/s with no startup exclusion, and supplements the unchanged full-route gates. Timing covers controller.step only. report.json records exact raw paths, SHA-256 hashes, metric definitions and limitations.
