# alpasim_proxy: a local selection set that orders like private

status: live
decisions: 262, 325
index: no public reweighting orders 5 private anchors (frozen 0/84); public GT distance does (tau 0.6); SCENE-MIX: go-straight public 0.52 vs private 0.80, private is waiting-rich; WAIT-SET: 316 windows where the human waits, any tag in one pool job

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
- `scene_mix.py` (scripts): human-state statistics of 5.5 s windows …
- `wait_set.md` (results): a local closed-loop set where the …
- `wait_set.py` (scripts): a local closed-loop set where the …
- `wait_set.sh` (scripts): the local closed-loop set where the …

[results/](results/) 40 result files · [plans/](plans/) 2 live plans · [scripts/](scripts/) 8 entry points
<!-- files:end -->

## SCENE-MIX: the private suite is rich in scenes where the human waits (decision 325)

Pre-registration and amendments 1-3: [plans/2026-10-16-scene-mix-prereg.md](plans/2026-10-16-scene-mix-prereg.md); results
[results/scene_mix/](results/scene_mix/) (`n1.md`, `n2.md`); script `scripts/scene_mix.py` (`extract` on the box, `report` / `sample` / `n1` on the Mac).

- **N1.** The host's go-straight is the AlpaSim starter kit (straight along the heading at the current speed, stays at standstill), run
  unmodified on a 300-scene log-stratified public sample (seed 20261016) plus the 25 near-stationary scenes, one pool job (1824 s): public
  **0.522 [0.465, 0.583]** against private 0.799 (line <= 0.70: supported). Its zeros are geometry (offroad 49, corridor 30, collision 18 of
  99). On the 25 near-stationary scenes it scores 1.000 (no progress penalty within 5.5 s), YR 0.920, APY 0.760 (collisions).
- **N2.** 70 052 uniform 5.5 s windows on the 136 raw test logs against the 1491 public windows (a public scene starts 1.5 s before its
  central token): near-stationary start 0.328 vs 0.016 (x20), human static the whole window 0.301 vs one scene, ends stopped 0.354 vs
  0.017, cruise 0.153 vs 0.111. On the one public static scene every openpilot-family driver (and LTF, DD) scores 0 with a rear-end
  collision; garage, WA-JEPA and go-straight pass.
- **Reading.** Drivers that hold at standstill gain on private (go-straight +0.28, garage +0.11), drivers that launch lose (DD -0.03, LTF
  -0.09, YR -0.125, APY -0.295). Size is not measurable from public (the static stratum has one scene); LTF / DD losing only a little bounds
  how static-rich private can be. This narrows decision 262's "driver-specific, not scene mix" to an interaction of mix and launch behaviour.

## WAIT-SET: a closed-loop set where the human waits at standstill

[results/wait_set.md](results/wait_set.md); code `scripts/wait_set.py` (definitions in the docstring), `scripts/wait_set.sh`; the set
`results/wait_set/set.json` (split `alpasim_nuplan/waitset@v1`), cache `$DATA_DIR/datasets/alpasim_nuplan_wait/v1` on the box.

- **Set.** The 199 public scenes whose human starts from a standstill, with the trajdata window moved into the human's wait (same asset,
  same scorer and serving): variant **a** = 147 windows where the human never moves (window ends 0.5 s before the movement onset), variant
  **b** = 169 windows where the human starts moving at 3.0 s. Classes: red light first in line / queued, other with a lead / free (75-84
  each); 95 windows come from held-split scenes. Every window lies inside its reconstruction's camera video; 34 windows were dropped for an
  undrawn vehicle in the ego's lane, 48 for a standstill too short.
- **Read a tag:** `bash experiments/alpasim_proxy/scripts/wait_set.sh read <tag>` (one pool job, public 10-decision driver path + the cache),
  then `wait_set.sh report`.
- **Validated before any model read:** the rebuilt cache equals the shipped one on all 171 source scenes' public windows; all 316 windows
  pass the cached-track check; smoke (4 windows): go-straight and human replay 1.0, the replay launches 0.02-0.07 s after the human.

**Handoff (2026-10-17 15:48 CST; the agent ended before the reads, the main session collects).** One chain in the pool
(`bash experiments/alpasim_proxy/scripts/wait_set.sh chain anchors`; ids in `$DATA_DIR/runs/alpasim_proxy/wait_set/{jobs.txt, stages.txt}`,
logs `pool/<name>/`):
- validation: `ws-read-starter` 1017-154757-5c1c, `ws-read-replay` 1017-154757-bf74 -> `ws-gate` 1017-154758-2d2f (go-straight >= 0.99 on
  variant a, replay >= 0.98 on a and b; verdict in `wait_set/gate.txt`; a failed gate fails every read below);
- reads after the gate: `ws-read-P0` 1017-154759-a61f, `ws-read-RSG-L18-s0` 1017-154759-83e1, `ws-read-RTB1ARO-L18-s0` 1017-154800-a9a6;
  private anchors `ws-read-APY10m10-AB-s0` 1017-154800-3689 (ap2 driver), `ws-read-YR10m10-F-s0` 1017-154801-3e4e, `ws-read-LTF-sample`
  1017-154801-8c82, `ws-read-DD-sample` 1017-154802-c808, `ws-read-G0014` 1017-154802-c336 (third-party drivers native, as their public
  reads) -> `ws-report` 1017-154803-8789 over every read.
- The first chain (15:23) died at 15:41: the platform SIGKILLed the runtime workers of both validation reads (and of the defect-fix read)
  at the box's memory line; a 316-window runtime outgrew its 30 GB booking. A read now runs variant a and variant b as two runtimes of
  about a public chunk each inside its one job, and a killed worker exits 137 so the pool retries it (logs of the dead jobs:
  `pool/failed-1541/`).

Collect (box, repo root): `bash experiments/alpasim_proxy/scripts/wait_set.sh report` -> `wait_set/report/{tables.md, all.csv}` (per tag and
variant: score, zeros by type, roll while the human waits, wait success, b: early / never / launch delay; by class; held-only; paired
differences). Anchor check: `$DATA_DIR/third_party/alpasim/.venv/bin/python experiments/alpasim_proxy/scripts/wait_set.py anchors --csv
$DATA_DIR/runs/alpasim_proxy/wait_set/report/all.csv --out $DATA_DIR/runs/alpasim_proxy/wait_set/report/anchors.md`: per anchor the variant-a
zero rate z_a and score s_a = 1 - z_a next to public and private, the fits private = public - k z_a and private = (1 - p) public + p s_a
(p = share of waiting windows a private-like mix would need), in-sample and leave-one-anchor-out predictions, and Kendall tau-b of each
against private (public alone ranks the five non-trivial anchors at tau -0.6, decision 262). Six anchors: the five plus go-straight
(`starter`, private 0.799; its public is N1's 300-scene sample, 0.522). One read per anchor; results go to `results/wait_set.md`.

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
