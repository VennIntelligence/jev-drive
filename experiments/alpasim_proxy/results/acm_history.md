# ACM teams on the AlpaSim e2e board: climb history and probe count

Read 2026-10-23 23:40 UTC from the public API (`GET /leaderboard?track=pai|nuplan`, the same call as `ap <profile> leaderboard`; no login).
Snapshots: `leaderboard_pai_20261023T2340Z.json`, `leaderboard_nuplan_20261023T2340Z.json`. The board returns one row per
SUCCEEDED submission (`record_kind=submission`, with `submission_id`, `image_tag`, `submitted_at`), not best-per-team. Failed or
rejected submissions do not appear, so every count below is a lower bound.

## Which teams

Only one team id/name starts with "ACM": `acmlab-dow` (display name `ACM_Lab_D_Team`), PAI track only. No ACM team on nuPlan
(48 team ids on the nuPlan board, none matches). "ACM NYCU" is not a separate team: it is the image tag of their third PAI
submission (`NYCU-collab-with-HonHai-and-HHRI-AI`). Team `hri-cmr` (Team HRI-CMR, PAI, two 2026-09-18/19 `distiltcm4b-lateral8*`
rows, PCS ~1985-2000) is possibly the same HHRI group named in that tag; this is a guess from the tag, not shown by the API,
and it does not start with ACM.

## Submissions (PAI, board of 23:40 UTC; PCS refit on every call)

| # | image tag | submission id | submitted (UTC) | mean scene | PCS [95%] | at-fault km | rank now (interval) |
|---|---|---|---|---|---|---|---|
| 1 | `c8209ad8f2be` | c7a1bd82... | 2026-09-28 12:08 | 0.688 | 2315 [2257, 2373] | 0.530 | 2 (1-5) |
| 2 | `913d1f4fa2da` | 9be754dc... | 2026-09-30 02:18 | 0.679 | 2260 [2206, 2313] | 0.511 | 5 (2-9) |
| 3 | `NYCU-collab-with-HonHai-and-HHRI-AI` | 5684d797... | 2026-10-16 12:13 | 0.747 | 2242 [2196, 2289] | 0.885 | 6 (3-10) |

## Climb over time

PCS of the same three rows in our stored snapshots (not comparable in absolute level; the zoib fit moves with the pool, e.g. the
10-23 refit after the corridor fix lifted everyone by ~60-100):

| board read | #1 (09-28) | #2 (09-30) | #3 (10-16) | best-so-far at that read |
|---|---|---|---|---|
| 2026-10-19 | 2250 (rank 1) | 2203 (rank 4) | 2211 (rank 3) | #1 2250, team at row 1 |
| 2026-10-23 early | 2316 (rank 1) | 2261 (rank 3) | 2240 (rank 4) | #1 2316, row 1 |
| 2026-10-23 23:40 | 2315 (rank 2) | 2260 (rank 5) | 2242 (rank 6) | #1 2315, row 2 behind our p1023a |

Best-PCS-over-time (from submit times; ordering within the team on the same ruler): the first probe (09-28) is the team's best on
PCS in every read; probes 2 and 3 are lower on PCS (by ~55 and ~73 on the latest read) but mean scene score and at-fault distance
move differently: the 09-28 row has mean 0.688 / 0.53 km, the 10-16 row has mean 0.747 / 0.885 km (+0.059 mean, +67 % at-fault
distance) yet 73 PCS lower. The board's PCS rewards something other than mean score and at-fault distance alone, and the NYCU
row traded PCS for a higher mean. No snapshot before 10-19 exists in our store for PAI (the 10-12 snapshot is nuPlan only), so the
09-28 to 10-16 curve between those dates is known only from the submit times.
What changed between submissions: the API shows only tags; the first two are opaque hashes (apparently image digests, 2 days
apart), the third is a named collaboration image submitted 16 days later. Nothing more can be said from public data.

## Probe counts (both tracks, full board)

| team | PAI | nuPlan | total | Sep | Oct |
|---|---|---|---|---|---|
| acmlab-dow | 3 | 0 | 3 | 2 | 1 |
| vennai-org (us) | 2 | 0 | 2 | 0 | 2 |
| ujs-iv (us) | 1 | 2 | 3 | 0 | 3 |
| lao-siji (us) | 0 | 2 | 2 | 0 | 2 |

Quota is 3 per team per UTC month shared across tracks: acmlab-dow is at most 2 in September and 1 in October, no sign of
several accounts or of exceeding it. Teams over 3 in one month on the full board (both tracks): September: py123d-garage 29,
opendrivelab-org 10, alpain-autonomy 9, host 9 (all hosts or organisers' baselines), metamon 5, fox-srdc 4, simforge-synthetic 4,
axe 4, lastga 4, team-lucifer 4, symphi 4; October: py123d-garage 16 (host), team-zyt 4, team-andrew-fox 4. The September
counts above 3 suggest the monthly limit was not enforced (or differed) before October; the non-host October cases (team-zyt,
team-andrew-fox, 4 each) may be several accounts or an exception. Only SUCCEEDED rows are visible.

## Versus our entries (23:40 UTC read)

| entry | track | PCS | mean | rank |
|---|---|---|---|---|
| vennai-org p1023a (10-23) | PAI | 2353 | 0.757 | 1 |
| acmlab-dow best (`c8209ad8f2be`) | PAI | 2315 | 0.688 | 2 |
| ujs-iv p1021a (10-21) | PAI | 2264 | 0.716 | 4 |
| acmlab-dow `913d1f4fa2da` | PAI | 2260 | 0.679 | 5 |
| acmlab-dow NYCU row | PAI | 2242 | 0.747 | 6 |
| vennai-org rsg-l18-s0-acc-gap-v1 (10-17) | PAI | 2198 | 0.680 | 9 |
| ujs-iv jtn-l18-s1-holdstop3 / native (10-19) | nuPlan | 1460 / 1317 | 0.955 / 0.915 | 34 / 42 |
| lao-siji yr10m10-f-s0-v1 / apy10m10-ab-s0-v1 (10-11) | nuPlan | 1029 / 780 | 0.797 / 0.637 | 78 / 123 |

Our p1023a leads their best by 38 PCS with overlapping intervals (rank interval 1-3 vs 1-5). Their NYCU row has mean 0.747 near our
p1023a 0.757 but at-fault 0.885 km against our 0.663 km. ACM has had no new probe since 10-16 (one in October used).
