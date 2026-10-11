# FAMILY task 3: what the nuPlan warm-up track is (read 2026-10-12; nothing submitted, no CLI call made)

Sources: alpasim branch `e2e_challenge` head 2023534 (2026-09-18, still the head on 2026-10-12; the CLI clone on the Tokyo box
`/data/third_party/alpasim-cli`), the HF Space `app.py` (last commit 504d5f65, 2026-10-02), NVlabs/alpasim issues #185, #196, #200, #203, #204.

| question | answer | source |
|---|---|---|
| Uses an official slot? | No. "Each warmup track has a separate allowance of three submissions per team in a rolling 30-day window. Warmup submissions do not consume the official allowance." | Space app.py:230-232; e2e_challenge/README.md:50-52 |
| How often | 3 per warm-up track (`nuplan-warmup`) per rolling 30 days. Ours: 1 used on 2026-10-11, so 2 left until 2026-11-10. Whether a FAILED warm-up counts is not stated (issue #203's author treats it as spent). An early bug that charged warm-ups to the official quota was fixed 2026-09-18 (#185) | README.md, #185, #203 |
| What runs | "one fixed wave only: 32 private scenes (16 driver replicas with two simultaneous scenes per replica)", 45 min limit. Fixed wave: the same 32 each time. Whether they are a subset of the 1000-rollout official suite is not stated (#204, unanswered) | README.md:38-41 |
| What comes back | the normal submission status only: state, expected / valid scored rollouts, a passed count, the throughput check (wall time vs limit). Not scored, never on a board, score `null`, no per-scene data, no at-fault or GT distance. A run with 0 passed is marked FAILED / NO_SUCCESSFUL_ROLLOUTS | README.md:43-50; #203 |
| "passed" | not defined in any public source; the evaluator's per-rollout `passed` is `failure_reason is None`, i.e. no at-fault collision, no offroad, no lateral corridor exit (progress does not matter) — inference that the server count is this | src/eval aggregation/scene_score.py:66-96, processing.py:305 (inference) |
| Gains | the CLI sends `--controller-gains` for warm-ups too; whether they are applied is not stated | competitor_cli/alpasim_challenge.py:136-145 |
| Our first warm-up | `2065ba22-8525-41a6-b445-a6c09611d9a4` (APY10m10-AB-s0 image): SUCCEEDED, 32 / 32 scored, throughput 270.6 s / 2700 s. The passed count was not recorded; `status 2065ba22...` (the user's CLI) would show it | docs/alpasim.md "First warm-up" |

What it can tell us: whether an image boots and meets throughput, and one pass count out of 32 fixed private scenes. One scene is about
3 points of pass rate, so two images separate only by a large gap (YR vs APY on private differ by ~0.16 in score, about 4-6 passes out of
32 if the zero rate scales). It says nothing about calibration, progress or which scenes fail.
