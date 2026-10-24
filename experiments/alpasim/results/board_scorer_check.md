# Is the private board scored with the corridor fix (PR #215)? (2026-10-25)

CPU only, no new rollouts. Decision 480. Question: are the AlpaSim nuPlan and PAI private boards computed with the fixed corridor scorer
(NVlabs/alpasim `e2e_challenge` 1b024f0, PR #215, our [scripts/scorer_v2.py](../scripts/scorer_v2.py), decision 453) or with the old
one (0bb4c4b)?

**Verdict: recomputed, on both tracks. Evidence strong.** Three independent lines agree: the organisers say so, every stored board
row's mean scene score moved in the window of that statement while nothing else in the row moved, and on nuPlan the size and the
mixed sign of the moves are what the fix does to our stored arena reads (it is not a pure gain there). The board has been stable since.

## 1. Upstream (public)

- PR #215 "Bugfix: Lateral Corridor Exit", merged 2026-10-20 10:16 UTC into `e2e_challenge` (merge 1b024f0), changes only
  `src/eval/src/eval/scorers/ground_truth.py` (+ its test). The only later commit on the branch is e121e59 "Enable report uploading"
  (#217, 10-20 13:35 UTC), not a scorer change.
- Issue #198 (stationary GT tail jitter causes false lateral corridor failures; nuPlan example `clipgt-9b6c68b4...` from public_2604):
  organiser (mwatson-nvidia) 10-19 16:49 UTC "we have a fix and will be working on recomputing the metrics and updating the
  leaderboard this week"; **10-22 15:33 UTC "This has now been fixed. The metrics have been recalculated (users can still see the frozen
  v1 version of the metrics) and all future submissions include the fix."** Issue closed at the same time. The statement names no track;
  the report in the issue is a nuPlan scene.

## 2. Board snapshots (`experiments/alpasim_proxy/results/leaderboard_*.json`, plus a fresh read 2026-10-25 via `ap ujs-iv leaderboard`)

Same `submission_id` compared between consecutive snapshots. Between the 10-19 and 10-23 snapshots `average_scene_score` changed for
almost every row while `avg_dist_between_incidents_at_fault`, `dist_to_gt_trajectory`, `legacy_score`, `evaluation_backend` and
`metric_algorithm` stayed bit-identical: the stored rollouts were rescored, not re-run, and only the scene score moved (corridor exits
zero the scene but are not at-fault incidents, so at-fault distance cannot move under this fix).

| track | snapshots | rows in both | mean changed | change in mean scene score |
|---|---|--:|--:|---|
| nuPlan | 10-12 -> 10-19 -> 10-19 13:49Z | 108 / 130 | 0 / 0 | none |
| nuPlan | **10-19 13:49Z -> 10-23** | 132 | **126** | median +0.006, range -0.012 .. +0.019 (114 up, 12 down, 6 same) |
| nuPlan | 10-23 -> 10-23 23:40Z -> 10-25 | 146 / 148 | 0 / 0 | none |
| PAI | **10-19 -> 10-23** | 104 | **102** | median +0.028, range 0 .. +0.054 (none down) |
| PAI | 10-23 -> 10-23 23:03Z -> 23:40Z -> 10-25 | 120 / 122 / 122 | 0 | none |

nuPlan changes are whole multiples of 0.001 (1 000 private windows, one window = 0.001). PCS moved only by the usual refit noise on
nuPlan (median +1), +27.5 median on PAI.

Our and the anchor submissions (old -> new board mean):

| track | entry | before | after | change |
|---|---|--:|--:|--:|
| nuPlan | ujs-iv `jtn-l18-s1-holdstop3` | 0.9503 | 0.9553 | +0.005 |
| nuPlan | ujs-iv `jtn-l18-s1-native` | 0.9155 | 0.9145 | -0.001 |
| nuPlan | lao-siji `yr10m10-f-s0-v1` | 0.7999 | 0.7969 | -0.003 |
| nuPlan | lao-siji `apy10m10-ab-s0-v1` | 0.6401 | 0.6371 | -0.003 |
| nuPlan | garage `014-nuplan-0014-v1` | 0.8479 | 0.8529 | +0.005 |
| nuPlan | `simscale-diffusiondrive-release-v0` / `simscale-lft-release-v0` | 0.7717 / 0.7377 | 0.7777 / 0.7407 | +0.006 / +0.003 |
| nuPlan | host `go_straight_with_delay_v2` | 0.7995 | 0.8165 | +0.017 |
| PAI | vennai-org `rsg-l18-s0-acc-gap-v1` | 0.6316 | 0.6802 | +0.049 |
| PAI | ujs-iv `p1021a` (read 10-21, decision 426 / 10-25) | 0.6649 | 0.7155 | +0.051 |

PAI fits the fix directly: our local rescore (decision 453) of the same models on PAI 281 gives RSG-s0 +0.064 and NTV0-s0 +0.071, all
from overshoot zeros flipping to 1.0, never down; the board gains are about 0.7-0.75 of the local ones. A PAI gain with no row going
down and at-fault untouched is the signature of this fix.

## 3. nuPlan: does the fix explain the small, mixed-sign move?

On nuPlan the fix is not monotone. Arena static windows have a human that stands still for the whole 5.5 s window (scorer GT length
median 0.003 m, [corridor_rule.md](../../alpasim_proxy/results/corridor_rule.md)). Old scorer: past the end it used the perpendicular to
the last, sub-mm jitter segment, so a car that rolled >= 4 m sometimes passed by luck of the jitter direction. New scorer: a GT path that
never gets 1 m from its end has no direction, the lateral distance is the plain distance, so **every** roll to 4 m on such a window is
now a corridor zero. Windows where the human moved >= 1 m and then stopped can flip the other way (the PAI case).

From the stored per-step metrics of the arena reads (`metrics_unprocessed.parquet`; ASLs are kept for only ~55 rollouts per read, so no
full rescore) this splits exactly into:
- **exact new zeros**: GT length < 1 m (direction None under the fix), the plain distance reached 4 m, old scorer did not fire, score > 0;
  under 1b024f0 these are zeros with certainty.
- **possible flips** (upper bound): old corridor zeros with GT length >= 1 m; each can go 0 -> 1 or stay 0 (needs the ego position).

| arena read (all 7 698 windows unless noted) | old mean | exact new zeros | possible flips | new - old, bound | board change of the same submission |
|---|--:|--:|--:|---|--:|
| APY10m10-AB-s0 | 0.621 | 127 (-0.017) | 179 | [-0.017, +0.007] | -0.003 |
| YR10m10-F-s0 | 0.745 | 105 (-0.014) | 162 | [-0.014, +0.007] | -0.003 |
| G0014 (7 146) | 0.819 | 26 (-0.004) | 195 | [-0.004, +0.024] | +0.005 |
| DD-sample | 0.710 | 94 (-0.012) | 380 | [-0.012, +0.037] | +0.006 |
| LTF-sample | 0.693 | 101 (-0.013) | 369 | [-0.013, +0.035] | +0.003 |
| RSG-L18-s0 | 0.773 | 89 (-0.012) | 97 | [-0.012, +0.001] | not submitted |
| NR1F / NR1HV / NR1HV2-L18-s0 | 0.80 | 84-87 (-0.011) | 98-119 | [-0.011, +0.001 .. +0.005] | not submitted |

Held split (2 228 windows): same picture, our openpilot family [-0.010, +0.005..+0.009], APY [-0.014, +0.012], YR [-0.010, +0.015],
G0014 [-0.002, +0.028], DD / LTF [-0.012, +0.055].

Every board change falls inside its arena bound, and the old scorer predicts no change at all. The implied share of possible flips that
did flip is 0.3-0.6 for all five anchors (APY 0.58, YR 0.50, G0014 0.31, DD 0.37, LTF 0.34), the same order for all, as one fixed rule
should give. The ordering is as the mechanism predicts: policies that creep on static windows (our APY / YR / native JTN) lose a little,
policies that hold still or overshoot moving-then-stopped humans gain (garage, DD, go-straight, which stays at standstill: +0.017, the
largest gain; JTN + stop3 holds still: +0.005 vs native -0.001). This is not a proof that the board used exactly 1b024f0 (we cannot
rescore the private windows), but nothing else in the public record changed scoring, and the organisers' statement covers it.

## Implications

- **Treat scorer_v2 (1b024f0) as primary on both tracks.** Old-scorer numbers are the board's frozen v1, not the live board.
- **PAI:** decision 453 / 455 already use the new scorer; nothing changes. Local-to-board ratio of the corridor gain is about 0.7.
- **nuPlan:** the fix moves a read by about +-0.01, smaller than most of our model gaps, so old arena differences are not void, but it
  is not a constant shift: it adds about 1.1 % exact zeros for any model that rolls 4 m on a fully static window (about 85-130 per 7 698
  for our family, 26 for garage) and returns some overshoot zeros, which favours holding still. Model pairs that differ in static-window
  creep (native vs hold, D3-type "cannot wait" models, decision 476) can move relative to each other by up to ~0.01.
- **Our stored nuPlan reads cannot be fully rescored**: the arena keeps ASLs for ~55 rollouts per read. The new-zero term is exact from the
  stored parquets (rule above), the flip term needs the ego position. For new nuPlan reads, score with the 1b024f0 eval (the clone
  `$DATA_DIR/third_party/alpasim-e2e-1b024f0`, as scorer_v2 does) or keep ASLs for every corridor zero, so selection is on the board's
  scorer.
- **Decision 355's six-anchor order was fitted on the old board.** On the new board go-straight (0.8165) is above YR (0.797); the old
  board had YR 0.800 > go-straight 0.799 (a 0.001 tie). The arena read of go-straight is not in the per-step store (`starter` has 142
  windows), so whether the arena still orders all six is untested; garage > {go-straight, YR} > DD > LTF > APY still holds on the board.

## Files and rerun

Box `$DATA_DIR/runs/alpasim_proxy/arena/corridor/`: `ar_<tag>.csv.gz` (stored, corridor rule 10-18) and `v2check/ar_<tag>.csv.gz` (new
extracts for G0014, DD-sample, LTF-sample, NR1HV, NR1HV2, NR1FB10), `v2check/v2bound.py`, `v2check/bounds.txt`.
```bash
P=/root/autodl-tmp/ujs/third_party/alpasim/.venv/bin/python; R=$DATA_DIR/runs/alpasim_proxy/arena
$P experiments/alpasim_proxy/scripts/corridor_rule.py extract --runs $R/reads/<tag>/c*/run --tag <tag> --out $R/corridor/v2check/ar_<tag>.csv.gz
.venv/bin/python $R/corridor/v2check/v2bound.py   # exact new zeros: gt_len < 1 & tripped & corridor == 0 & score > 0; flips: gt_len >= 1 & corridor == 1
```
