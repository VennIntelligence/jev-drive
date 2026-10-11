# Official submission #2 (YR10m10-F-s0, nuPlan track): status read 2026-10-12 01:30 JST

Read by the ONPOL-PAI session on the Tokyo box with the competitor CLI's `status` command (read-only; no other call was made).
Submission id `cdf145f9-c8fb-474d-ba8a-22499f1b4a59`, image `teams/lao-siji:yr10m10-f-s0-v1`, queued 2026-10-12 00:56 JST, evaluated
00:56 to 01:22 JST (26 min; observed wall time 1 166 s of the 3 000 s limit), 1 000 of 1 000 rollouts scored.

| | GT distance (CLI `score`), m | Average scene score | At-fault distance, km |
|---|--:|--:|--:|
| Submission #2, private (`YR10m10-F-s0`) | **1.733** | not in the status JSON | not in the status JSON |
| Submission #1, private (`APY10m10-AB-s0`) | 2.29 | 0.640 | 0.132 |
| `YR10m10-F` on the public scenes (as given by the coordinating session) | 0.94 | 0.925 | 0.79 |

- The status JSON carries only `score`, which is the GT distance (lower is better), and the rollout coverage. **The average scene score and
  the at-fault distance of submission #2 are not in it and are not recorded here**; they have to be read from the leaderboard page by the
  user or by the session that holds that access.
- On the one number available, submission #2 is 0.56 m closer to the log than submission #1 on the private suite (1.733 against 2.29),
  and 0.79 m further than the same checkpoint on the public scenes (1.733 against 0.94): the public-to-private gap is still there, smaller
  than submission #1's (2.29 against 1.13 public).
- One official slot remains.

Raw status:

```json
{
  "competition_id": "alpasim-end2end-2026",
  "controller_gains": null,
  "created_at": "2026-10-11T15:56:08+00:00",
  "evaluation_completed_at": "2026-10-11T16:22:32+00:00",
  "evaluation_started_at": "2026-10-11T15:56:19+00:00",
  "failure_code": null,
  "failure_diagnostics": null,
  "failure_reason": null,
  "grace_submission": false,
  "image_digest": "sha256:3f55a7dcdec32d8a624b01d12390256278b8616869654df1cbb78de7c5bfc722",
  "image_size_bytes": 4782501934,
  "rollout_coverage": {"expected_rollout_count": 1000, "valid_scored_rollout_count": 1000, "valid_scored_rollout_rate": 1.0},
  "score": 1.73267758643548,
  "status": "SUCCEEDED",
  "submission_id": "cdf145f9-c8fb-474d-ba8a-22499f1b4a59",
  "team_id": "lao-siji",
  "throughput_limit": {"limit_wall_time_s": 3000.0, "margin_s": 1834.435, "observed_wall_time_s": 1165.565, "passed": true, "track": "nuplan"},
  "track": "nuplan",
  "updated_at": "2026-10-11T16:22:32+00:00"
}
```

The image URIs are left out of this copy (registry address).
