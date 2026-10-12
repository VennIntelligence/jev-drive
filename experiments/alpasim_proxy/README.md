# alpasim_proxy: a local selection set that orders like private

status: live
decisions: 262
index: no public reweighting orders 5 private anchors (frozen 0/84); public GT distance does (tau 0.6)

**Question.** Is there a reweighting or stratum of the 1491 public AlpaSim nuPlan scenes that reproduces the private leaderboard ordering and
gaps of policies whose private numbers are known, so the last official slot can be chosen on it?

**Conclusion.** No stratification or reweighting of the public scenes is a valid stand-in for the private suite: proxies fitted on the two submissions alone order none of the three new anchors right (0 of 84 frozen proxies pass), and refits on all five anchors pass only with 20-70 effective scenes and fail leave-one-anchor-out (at most 3 of 5). The gap is driver-specific, not scene-mix: the shipped LTF and DiffusionDrive samples keep their public numbers on private (score -0.09 / -0.03, at-fault km x0.92 / x1.10), the garage 0014 checkpoint gains (+0.11), our openpilot students lose (-0.13 / -0.30); of all public statistics only GT distance (and the share of clips leaving the path by 4 m) ranks the anchors like the private score (Kendall tau 0.6, one discordant family pair) and the private GT distance (tau 0.6, the one real discordance is APY against DD), while public scene score and at-fault distance rank them backwards (tau -0.6) (decision 262).

**Next.** The last slot is chosen by the rule in research/next-round/plan.md section 2.3 (public GT distance within the openpilot family, a warm-up read by the user, the long-horizon read when it lands). Untested hypothesis for the family-specific loss: the openpilot input path (calibration-driven warp) on private rigs; the samples and the garage read raw images.

**Read more.** Pre-registration and Amendment 1: [plans/2026-10-12-proxy-prereg.md](plans/2026-10-12-proxy-prereg.md). Context: the AUD1
audits in [../alpasim/results/](../alpasim/results/) (`aud1_private_gap.md`, `aud1_drift_by_decision.md`). Scene length is owned by the
long-horizon lane ([../alpasim_long/](../alpasim_long/)); this topic takes its length stratum from there and does not run long scenes.

<!-- files:begin -->
## Files

- `2026-10-12-proxy-prereg.md` (plans): PROXY 预注册：一个行为像 private …
- `proxy_fit.py` (scripts): can a reweighting of the 1491 public …
- `proxy_transfer.py` (scripts): which PUBLIC statistic orders the …
- `proxy_collect.py` (scripts): per-scene and per-decision tables of …

[results/](results/) 22 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 5 entry points
<!-- files:end -->

## Anchors: five policies whose private numbers are known

Private numbers from the leaderboard JSON pulled 2026-10-12 ([results/leaderboard_nuplan_20261012.json](results/leaderboard_nuplan_20261012.json)).
Public numbers: all 1491 public scenes, native on the GPU box (pool), metrics as in AUD1 (src/eval @ 0bb4c4b; 12 route-fold scenes count 0 in
the score, GT distance = mean truncated per-clip maximum, at-fault distance = driven km / at-fault events).

| anchor | public score / at-fault km / GT m | private score / at-fault km / GT m | change |
|---|--:|--:|---|
| `APY10m10-AB-s0` (submission 1) | 0.935 / 0.863 / 1.130 | 0.640 / 0.132 / 2.288 | -0.295, km x0.15, GT x2.0 |
| `YR10m10-F-s0` (submission 2) | 0.925 / 0.789 / 0.941 | 0.800 / 0.529 / 1.733 | -0.125, km x0.67, GT x1.84 |
| SimScale LTF sample (`simscale-lft-release-v0`) | 0.824 / 0.294 / 1.876 | 0.738 / 0.270 / 2.266 | -0.086, km x0.92, GT x1.21 |
| SimScale DiffusionDrive sample (`simscale-diffusiondrive-release-v0`) | 0.797 / 0.243 / 1.843 | 0.772 / 0.267 / 2.158 | -0.025, km x1.10, GT x1.17 |
| garage `model_0014` (five identical team submissions) | 0.740 / 0.183 / 0.869 | 0.848 / 0.228 / 0.718 | +0.108, km x1.25, GT x0.83 |

Public score ranks the anchors APY > YR > LTF > DD > garage; private ranks them garage > YR > DD > LTF > APY. The third-party drivers keep
roughly their public level, ours fall: the private suite is not uniformly harder, it is harder for the openpilot-student family.

How they were run: LTF and DD are the shipped samples (`e2e_challenge/sample_submission_simscale_navsim_{transfuser,diffusiondrive}`) run natively
in `envs/alpasim-ltf` (torch 2.8 for sm_120 instead of the image's 2.6; driver file `experiments/alpasim/scripts/drivers/dd.sh`, LTF is
`run.sh ltf`). Fidelity check on the Tokyo box: both shipped images built unmodified and run through `experiments/alpasim/docker/smoke.sh` on
the 48 local scenes ([results/tokyo_samples/](results/tokyo_samples/)): LTF container 0.8736 vs native 0.8735 (no scene differs by > 0.011);
DD container 0.8308 vs native 0.8515 (one scene flips to 0, the rest agree). Garage: `py123d_garage` v0.1.1 (src identical to the 0.1.0 release)
with the public `model_0014.pth` (sha256 d536614c...), native env `envs/garage`, entrypoint and env of its `submission.Dockerfile`, wrapped by
`scripts/garage_driver.py` (an observer that counts plan failures per session; no model or driver logic changed; `scripts/setup_garage.sh`,
`experiments/alpasim/scripts/drivers/garage.sh`). Six public scenes have no route left after the anchor frame; the garage driver raises
`InsufficientRouteError` there and serves its fallback, as its image would (60 of 14 790 decisions; kept, `OT_HEALTH_WARN=1`). The sample drivers
had 0 inference errors and 0 fallbacks. Pool jobs: 15 runs, about 4 card-hours. Box: `$DATA_DIR/runs/alpasim_proxy/anchors/`.

## Reweighting the public scenes (P0-P3)

Strata: city (LV 481 / BOS 467 / PIT 323 / SG 220 = left-hand), start speed, near-stationary start (log path < 5 m and v0 < 1 m/s, 25 scenes),
4-way command, first route waypoint lateral / longitudinal, log path length, start acceleration. Proxies: single factor (P1), factor pairs
with a KL penalty (P2), max-entropy tilt on all of them (P3); loss = squared error of anchor-centred log failure rate / log km / log GT.

- **Two anchors only** ([results/fit_a2/](results/fit_a2/), frozen before any new anchor was read): city, left-hand traffic, command, route
  geometry and acceleration cannot even flip the sign of the score gap (best +0.008 with Pittsburgh only; Singapore alone -0.026). Only the
  near-stationary stratum carries the private gap: on its 25 scenes YR - APY is +0.160 score, -1.0 m GT, at-fault km x2.1 (the private gap is
  +0.16, -0.56 m, x4.0); proxies that put 66-77 % of the weight there pass V1 + V2 with 47-58 effective scenes.
- **Out of sample (V5, the main read):** none of the 84 frozen proxies orders the five anchors right ([results/fit_a5/frozen_eval.csv](results/fit_a5/frozen_eval.csv)).
  The near-stationary proxy gets the private GT order exactly (garage 0.84 < YR 1.94 < DD 2.36 < LTF 2.48 < APY 2.66 m) but puts the garage last
  on score (0.807 against YR 0.924; private: garage first).
- **Refit on all five** ([results/fit_a5/fits.csv](results/fit_a5/fits.csv), [loo.csv](results/fit_a5/loo.csv)): the best fits reach score tau 0.8
  only with 20-70 effective scenes (V4 line 300) and predict a held-out anchor's rank in at most 3 of 5 cases. Verdict: **no valid scene proxy.**

## Which public statistic transfers (P4)

[results/transfer_a5/](results/transfer_a5/) (`proxy_transfer.py`; CIs log-clustered over 44 logs):

| public statistic | Kendall tau vs private score | vs private GT |
|---|--:|--:|
| GT distance (lower better) | **+0.6** | **+0.6** (two discordant pairs, both APY against a sample: LTF / APY 2.27 / 2.29 m is a private near-tie, DD / APY 2.16 / 2.29 m is not) |
| share of clips >= 4 m off the path | +0.6 | +0.6 |
| per-decision drift slope k = 4..10 | +0.2 | +0.2 |
| scene score | -0.6 | -0.6 |
| at-fault distance | -0.6 | -0.6 |
| zeros | -0.2 | -0.2 |

None predicts the private at-fault distance (|tau| <= 0.2). Public GT distance passes the registered score line (tau >= 0.6) with one
discordant family pair: APY's public GT (1.13 m) beats LTF / DD (1.84-1.88 m) but its private score is lower. Within our family it has
the right sign every time it was tested (APY vs YR: public -0.19 m, private -0.56 m and +0.16 score).

Ranking of every stored candidate by public GT distance (1491 scenes, 95 % CI): garage 0.87 [0.77, 0.95]; YR10m10-F s1 / s0 0.93 / 0.94
[0.83, 1.04]; P2H10-F-s1 0.96; P2H10S-F s0-s3 0.99; JAPSnoA 1.00 / 1.02; P2H10-F-s0 1.02; APY10m10-AB s0-s3 1.11-1.16; JAPSnoY 1.14-1.18;
WA-JEPA 1.19; JAPS 1.24 / 1.26; SH30 1.34; AP2 1.51; DD 1.84; LTF 1.88. No stored candidate is separable from `YR10m10-F-s0` on it.

## What the proxy cannot see, and what was not done

- Scene length: every public scene is 10 decisions. The long-horizon lane ([../alpasim_long/](../alpasim_long/)) rebuilds the trajdata
  cache at longer windows; once it has a held read, the anchors (LTF, DD, garage: about 15 min per 400 scenes each) should be rerun on
  it, and that read is tested against the same five private triples.
- The family-specific loss is unexplained. Candidates not testable on public scenes: the openpilot warp's reliance on the reported
  calibration on private rigs (AUD1 found the MTGS renderer ignores calibration overrides, so a rig mismatch cannot be simulated here),
  route content. The warm-up track (32 private scenes, status only, the user's action) is the only private probe left.
- Tokyo was used only for the container fidelity check above. The two submitted images were not rerun: the stored native rollouts cover
  all 1491 public scenes, the image and native runs agree (AUD1: 0.9708 vs 0.9701 on 48 scenes; interpreted vs compiled passes +0.0002 on
  220), and no public stratum is missing except length.

## Files

`scripts/proxy_fit.py` (reweighting fits, V1-V5), `scripts/proxy_transfer.py` (P4), `scripts/proxy_collect.py` (box: anchor tables through
`aud1_drift_collect.py`), `scripts/garage_driver.py`, `scripts/setup_garage.sh`; results: `anchors/*.csv` (per-scene anchor tables),
`fit_a2/`, `fit_a5/`, `transfer_a5/`, `tokyo_samples/`, `scene_city.csv`, the leaderboard snapshot. Mac inputs for stored drivers:
`tmp/aud1_drift/`, `tmp/aud1_drift_x/` (box `$DATA_DIR/runs/alpasim/aud1_drift*`). Commands:

```bash
# box: anchors (pool), then tables
OT_LANE=proxy OT_OUT=$DATA_DIR/runs/alpasim_proxy/anchors OT_OWNER=alpasim-proxy python3 experiments/alpasim/scripts/ot2_loop.py dd \
  "DD-o:dd:::ot3new" "DD-f:dd:::cf1fresh" "DD-c:dd:::chunks"          # ltf / garage likewise (garage with OT_HEALTH_WARN=1)
$DATA_DIR/third_party/alpasim/.venv/bin/python experiments/alpasim_proxy/scripts/proxy_collect.py --out <dir> "DD-sample=<anchors>/dd/runs/DD-*/2*"
# Mac
.venv/bin/python experiments/alpasim_proxy/scripts/proxy_fit.py tmp/aud1_drift/per_scene.csv experiments/alpasim_proxy/results/anchors/{ltf,dd,g0014}.csv \
  --anchors APY10m10-AB-s0,YR10m10-F-s0,LTF-sample,DD-sample,G0014 --out experiments/alpasim_proxy/results/fit_a5 --frozen experiments/alpasim_proxy/results/fit_a2/weights.npz
```
