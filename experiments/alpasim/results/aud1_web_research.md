# aud1: web research on the nuPlan public-to-private gap (0.935 local -> 0.640 private)

Read-only web research, 2026-10-11. Sources: NVlabs/alpasim issues, discussions, PRs and branches (via `gh`), the HF
Space source (`app.py` and its commit history), the HF dataset `OpenDriveLab/AlpasimChallenge2026_nuplan_track`,
kesai-labs `py123d_garage` and `drive-irt`, the navhard-leaderboard docs, the NVIDIA developer forum, arXiv, and the
public nuPlan leaderboard JSON pulled from the Space's own Gradio endpoint (`lambda_5`, "all successful submissions",
105 entries, fetched 2026-10-11). Nothing was posted or submitted. Quotes are verbatim. Anything marked **[inference]** is
my reading, not a source statement.

## 1. What the organisers have said about the PRIVATE nuPlan suite

**Composition and provenance**
- The private set was swapped out wholesale at the September reopening. mwatson-nvidia, 2026-09-15: "The leaderboard
  was replaced, yes, because the test set changed drastically. A totally different set is now being used for nuPlan
  and we increased the PAI track test set by almost an order of magnitude."
  https://github.com/NVlabs/alpasim/issues/182
- Distribution, per two organisers in reply to a team (team-lucifer) whose model "performs better than the
  organizer-released checkpoint on nearly every metric across a select local nuPlan scenes, yet performs substantially
  worse on the official evaluation":
  - mwatson-nvidia, 2026-09-25: "The private dataset is unpublished anywhere. And, while I'm not super acquainted with
    the distributions (I know high level statistics), I wouldn't expect the distributions to be too far off for the
    nuPlan private test set. This is not true with the PAI test set, where the private test set is much more difficult
    than public set."
  - ln2697 (Long Nguyen, py123d_garage lead), same thread: "The nuPlan test set is not a previously unseen
    distribution. But as the main person behind py123d_garage, I don't have much more information, either."
  - Source: https://github.com/NVlabs/alpasim/issues/191
- The organisers have not published which logs, cities or splits the private set comes from, or how many scenes it has.
  None of the HF Space versions, READMEs, issues or the KE:SAI blog give these. What is known:
  - The HF Space says only "The nuPlan track uses a private evaluation dataset" (current `app.py`, last commit
    2026-10-02). https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026
  - The track-wide statement "Competition scenes come from Denmark, France, Germany, Japan, South Korea, Spain, Sweden,
    the United Kingdom, and the United States" sits under the PAI track. The KE:SAI blog (2026-09-08) adds Singapore
    for the challenge as a whole and calls the nuPlan track "120 hours of publicly released sensor data".
    https://kesai.eu/blog/2026-09-08-common-task-framework/
  - The public HF dataset carries only `MTGS_asset/navtest` and `trajdata_cache/nuplan_test.tar.gz`. Its README
    describes the latter as "Pre-built trajdata cache for the nuPlan test split". The cache was replaced on 2026-08-26
    to remove ghost pedestrian and cyclist actors, and mwatson-nvidia said: "I've also taken the updated cache for the
    private/final set we plan to use for the competition after the re-open."
    https://huggingface.co/datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track,
    https://github.com/NVlabs/alpasim/issues/156
  - **[inference]** "Not a previously unseen distribution" plus the shared `nuplan_test` trajdata path suggest nuPlan
    logs rendered with MTGS, probably from the nuPlan test or val pool outside navtest. Nothing confirms this.
- Route generation for private scenes: the shared nuPlan base config, which the ec2 preset inherits, says: "Request up
  to 80 m of route lookahead by extending the recorded trajectory along connected map lanes. **All public and private
  NuPlan challenge presets inherit this shared setting.**" (`route_generator_type: MAP`, `route_start_offset_m: 40.0`).
  https://github.com/NVlabs/alpasim/blob/0bb4c4bfe10951ea5589aa8ea514e422cf3c3506/src/wizard/configs/e2e_challenge_nuplan_common/base.yaml
- The MAP route foldback known issue does not touch the private set. HF Space, known issues for 0bb4c4b (added
  2026-09-23): "In 10 of 1,485 public scenarios, near-stationary position noise can make MAP route generation select
  the wrong lane and fail its foldback check before the driver runs. **This does not affect the private competition
  set.** ... e2e_challenge will remain unchanged during the competition." This is the official confirmation that the
  public suite has 1,485 scenarios. (HF Space commit 7f673c4a, 2026-09-23)
- Traffic is replayed, not reactive, and vehicles only. "Traffic on this track consists of vehicles only; pedestrians
  and cyclists are not simulated." (e2e_challenge README). "For this iteration of the challenge, the behaviors of the
  contenders are fixed based on the recording" (mwatson-nvidia, 2026-07-27, https://github.com/NVlabs/alpasim/issues/123).
- Timestep: the nuPlan track keeps NAVSIM parameters on purpose: "No--the purpose of the nuplan track is to give teams
  who might have participated in the NAVSIM challenge a quicker entrypoint and so we have retained those parameters."
  (2026-08-18, https://github.com/NVlabs/alpasim/issues/155). Config: `control_timestep_us: 500_000`,
  `force_gt_duration_us: 500_000`, `n_sim_steps: 200`, 8 cameras at 1920x1080 at 2 Hz, `RENDER_AGGREGATED`, trajdata
  `num_timesteps_before: 30`, `num_timesteps_after: 80`.

**Official harness versus the local `full` preset (the only config deltas)**
- `e2e_challenge_nuplan/ec2.yaml` and `full.yaml` both use `/nuplan_scenes: navtest_full` and the same base. They differ
  in topology and concurrency: ec2 has `topology: 8gpu_32rollouts_mtgs`, `runtime.nr_workers: 4`,
  `driver.n_concurrent_rollouts: 2`, `renderer.n_concurrent_rollouts: 8`, `controller.n_concurrent_rollouts: 4`.
  `full` keeps `nr_workers: 1` and one rollout per endpoint.
  https://github.com/NVlabs/alpasim/tree/0bb4c4bfe10951ea5589aa8ea514e422cf3c3506/src/wizard/configs/e2e_challenge_nuplan
- The README says: "The EC2 configs start 16 replicas of the submitted image across GPUs 4-7 with 2 concurrent rollouts
  per replica". It also says each image must "support multiple concurrent calls on multiple instances (replicas) of the
  same image". GPU: "It is an H100 GPU" (2026-09-24, https://github.com/NVlabs/alpasim/issues/189). There is a 16 GiB
  VRAM cap per driver instance, and a per-track throughput budget enforces the 0.1 s per `Drive` target.
  https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md
- The renderer image in production is not the public one: "The direct runner supplies the complete external driver
  endpoint list and any production-only renderer image/entrypoint overrides for official evaluation."
  `deploy/competition_ec2.yaml` sets `renderer_entrypoint: "/app/internal/scripts/pycena/runtime/pycena_nrm_full"`
  (the nuPlan base config overrides the renderer to `alpasim-mtgs-server` from the trusted base image).
  https://github.com/NVlabs/alpasim/blob/e2e_challenge/src/wizard/configs/e2e_challenge/README.md

**Deployed commits, bugs, reruns, metric changes**
- Soft open ran on `6b24e01` with a "provisional dataset". Maintenance downtime ran from 2026-08-31 and the reopening
  was 2026-09-09 on `cd713e0`. Per the HF Space, `0bb4c4b` (2026-09-14) is the current deployment. The branch head
  `20235345` (2026-09-18) changes only the README and CLI (warmup tracks, allocation message); I checked this with
  `gh api compare 0bb4c4bf...20235345`.
- nuPlan off-road was disabled until 2026-09-14. mwatson-nvidia, 2026-09-11: "The config for the nuplan track had an
  issue where the off-road penalty was basically disabled. This will require a retroactive modification of the existing
  submission in the nuplan track." The fix is PR #181 (2026-09-14). The HF Space adds: "Organizers will rerun only the
  evaluation stage for existing nuPlan submissions." https://github.com/NVlabs/alpasim/issues/180,
  https://github.com/NVlabs/alpasim/pull/181
- Open scoring bug, 2026-10-08/09 (issue #198, reproduced at `0bb4c4b`): stationary GT tail jitter can turn forward
  overshoot into `left_corridor_laterally`, which scores the scene 0. maximilianigl, 2026-10-09: "This is indeed a bug
  and we do have a fix (extrapolating the final orientation from the last 1m of actual movement). We have our orga
  meeting on Monday and will discuss how to handle this during the ongoing challenge." The two cases reported are PAI
  scenes, but `ground_truth.py` is shared by both tracks. https://github.com/NVlabs/alpasim/issues/198
- Final-run protocol (HF Space commit 9bd0e8ec, 2026-10-02): "At least the top three teams in each track will be
  selected for final runs ... Each selected contestant will receive ... two additional evaluation runs in nuPlan to
  reduce potential variance in the final results." The Harmonizer "will ... remain disabled for nuPlan final runs."
  Registration closes 2026-10-18. Final results are a 2026-11-15 target "subject to available compute".
- Quota is 3 official submissions per UTC calendar month, shared across tracks. Failed evaluations count. Warmups get
  3 per track per rolling 30 days and run "one fixed wave only: 32 private scenes". Their status shows
  expected/valid/passed rollout counts. Issue #203 (2026-10-11) quotes "32 expected / 32 valid scored rollouts, 0
  passed". https://github.com/NVlabs/alpasim/issues/196, https://github.com/NVlabs/alpasim/issues/203
- Organisers do not hand out per-scene results. Discussion #199 (2026-10-09) asks for nuPlan per-scene gate flags or a
  nuPlan reference bundle and has no reply yet. The local-evaluation README says "nuPlan does not yet have a reference
  bundle". https://github.com/NVlabs/alpasim/discussions/199
- Reruns are granted only when the platform is at fault: "the precedent we have set is to limit the issuance of reruns
  to cases where the platform was at fault" (2026-10-01, https://github.com/NVlabs/alpasim/issues/194).

## 2. Reports of public-vs-private gaps by other participants

- **team-lucifer (skr3178), 2026-09-25, #191:** their model beats the organiser checkpoint locally "on nearly every
  metric" but "performs substantially worse on the official evaluation". Organisers answered that the private set is
  unpublished and not a new distribution. No technical cause was offered. Their WA-JEPA entry `wajepa-s2-20260904` is
  at 0.762 on the board. Their later `lf-007` (2026-09-30) is at 0.879.
  https://github.com/NVlabs/alpasim/issues/191
- **VF-Team (Eashwar93), 2026-10-01, #194:** "My latest submission from October indeed performs quite well locally, but
  underperformed heavily on the AlpaSim benchmark. This tells me that ... local evaluation can't be used as a proxy for
  performance on the official benchmark." maximilianigl, 2026-10-02: "Note that the numerical values will not be
  comparable between local and benchmark, but hopefully the relative ranking should be somewhat indicative (although of
  course it cannot be perfect)." On the board, VF-Team is at `v2.1` 0.891 (2026-09-21) and `v4.0` 0.769 (2026-10-02),
  so their newer model dropped 0.12 on private. https://github.com/NVlabs/alpasim/issues/194
- **PAI track, team-alphamale, 2026-08-28, #166:** two policies came out in the opposite order locally (pinned
  evaluator) and on the leaderboard. mwatson-nvidia gave three reasons: "1. the dataset used is a private dataset not
  published anywhere ... 2. a policy can get a very high 'distance between at-fault incidents' by simply braking hard
  ... 3. the competition was still using an older version of the repo ... if you ran your tests locally on a more recent
  version, that could explain some of the differences". https://github.com/NVlabs/alpasim/issues/166
- **PAI, alexcodigo84, #166:** a model "roughly 2x better" than the VaVAM sample locally gets a "substantially lower
  policy capability score". mwatson-nvidia: "The scene score really is the most important aspect, and it's not
  necessarily the average of the scene score that matters due to weighting in the fit."
- **PAI, trexinator37, 2026-08-12, #134:** an assertion on a fixed 1920x1080 image failed on the server. The server
  delivers 1916 or 1920 widths depending on the scene. The driver returned empty trajectories, which gave "a very bad
  policy capability score (but strangely, a much higher/better avg_dist_at_fault metric)". The organiser confirmed:
  "the current official evaluation set produces both 1916x1080 and 1920x1080 JPEGs". This is PAI only; nuPlan requests
  a fixed 1920x1080. https://github.com/NVlabs/alpasim/issues/134
- **Organiser-authored PAI numbers (OPTED, arXiv 2609.20756, 2026-09-17; authors include Igl and Chitta):**
  LTFv6+OPTED scores 41.8 locally on the 441-scene curated val split versus 32.2 on the private set. VaVAM+OPTED scores
  37.1 versus 35.8. The paper does not comment on the gap. https://arxiv.org/html/2609.20756
- **Infrastructure failures that only show up on the server:** a GPU-name whitelist crash on "nvidia h100 80gb hbm3"
  (#151). A CUDA-graph replay error under InferenceMode (#201, 2026-10-10). `terminate_session=True` invalidating
  rollouts (#159). None of these is a silent score drop like ours; they fail outright or invalidate rollouts.

## 3. Documented pitfalls in driver implementation

- **Route contract (likely what "routepad" and "routebridge" refer to).** At `0bb4c4b`,
  `RouteGenerator.generate_route` resamples 20 waypoints over 80 m (about 4.21 m spacing) and transforms them to the
  **rig frame**. With `route_start_offset_m: 40` it then drops every waypoint closer than 40 m, so roughly the 40-80 m
  segment is left. `prepare_for_policy` then truncates at the first segment outside [3.5, 4.5] m and pads to 20 points
  with **NaN**. After the recording ends, the route is extended along map lanes, choosing "arbitrarily ... the next lane
  with the closest heading". The route can be empty when the remaining polyline is shorter than 40 m.
  https://github.com/NVlabs/alpasim/blob/0bb4c4bfe10951ea5589aa8ea514e422cf3c3506/src/runtime/alpasim_runtime/route_generator.py,
  https://github.com/NVlabs/alpasim/blob/0bb4c4bfe10951ea5589aa8ea514e422cf3c3506/src/runtime/alpasim_runtime/events/policy.py
- The host garage driver handles exactly these cases. It drops NaN rows, prepends the ego origin `(0,0)` when the first
  waypoint is farther than the resolution (a "bridge" over the 0-40 m gap), and extends the route along its final
  1 m heading until it reaches the trained target-point distance + 2 m. The code comment reads: "Extends the route along
  its final heading so target points keep their trained distance near the clip end" (a "pad"). Its LTF takes "one target
  point 45 m along the route".
  https://github.com/kesai-labs/py123d_garage/blob/main/src/py123d_garage/evaluation/alpasim/help/driver_service.py,
  https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints/blob/main/resnet34_v0.1.0/README.md.
  **[inference]** The board tags `v4-e20-s12-routepad` and `v5j-e17-s9-routepad` (tongji-zeroone-av, 2026-10-03/05,
  0.705 and 0.725) almost certainly name this padding. No public source defines "routepad" or "routebridge"; neither
  term appears on GitHub, the forum or the Space.
- **Route semantics.** The route comes from the recorded ego path over the whole recording and is fixed at
  initialisation. "Extrapolation is only done after the end of the recording (for which no data exists)." (2026-08-07,
  https://github.com/NVlabs/alpasim/issues/138). Accumulating route observations over a rollout is allowed on PAI ("you
  can use all the information provided over the interface", 2026-08-24). There is no nuPlan ruling.
  https://forums.developer.nvidia.com/t/alpasim-e2e-challenge-2026-is-temporal-accumulation-of-route-observations-allowed/380855
- **Egomotion and dynamics.** NAVSIM adaptation guide: "For model dynamics, use only its same-timestamp RPC
  `DynamicState`, which provides rig-frame velocity and acceleration. Use pose history only when that state is missing
  ... This contract requires official runtime `4d352ba` or newer." Trajectories are returned in the rig frame and
  anchored in the local frame. The guide also says to "confirm that the entire run did not use fallback trajectories"
  and to check `inference_error = 0`.
  https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/NAVSIM_MODEL_ADAPTATION.md
- **Cold start / history.** nuPlan force-GT is only 0.5 s, so the policy drives from the second 2 Hz tick. For PAI the
  local-evaluation README stresses that `+e2e_challenge=dev` "is what pins the contract ... Selecting a driver alone is
  not enough, because a driver config can carry its own camera and timing overrides". The garage's host entries
  `044-hist01/02/05/10(-single)-0062` (2026-09-18) span 0.874-0.896 on private. Same-checkpoint garage configs differ by
  up to 0.12: `0062` runs `039-042` sit at 0.77, `hist` runs at 0.88-0.90. Driver-side settings move the private score
  as much as the gap between models.
- **Controller.** Official is the nonlinear MPC. Linear MPC is "not available in the actual competition". Gains are
  tunable only through the CLI `--controller-gains` (weights 0-10, `idx_start_penalty` 0-19). CONTROLLER_TUNING.md:
  "when a reference trajectory requests extremely harsh, physically unrealistic braking ... the nonlinear optimization
  can produce a zig-zagging solution ... and the vehicle may not recover". It is "not a general trajectory validator,
  trajectory-repair system, or safety filter". Issue #152 (2026-08-17) showed the straight-line driver steering
  "~90°" after the handover under the nonlinear MPC. PR #153 fixed that.
  https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/CONTROLLER_TUNING.md,
  https://github.com/NVlabs/alpasim/issues/152
- **Camera calibration.** Scale intrinsics to the decoded JPEG and never assert a width (PAI, #134). nuPlan delivers
  1920x1080 for `CAM_L0/F0/R0` and the guide uses those three.
- **Scoring.** A scene is 0 on `collision_at_fault`, `offroad` or `left_corridor_laterally`. Otherwise it equals
  progress, clipped and divided by a saturation threshold, and is 1.0 when the GT distance is below a minimum.
  Timesteps after the ego leaves the 4 m GT tunnel are cut ("no incidents that occur count against the score, but no
  progress is counted", #138; the 4 m tunnel is described in #192). Hard braking inflates km per incident without
  helping scene score (#166).
  https://github.com/NVlabs/alpasim/blob/0bb4c4bfe10951ea5589aa8ea514e422cf3c3506/src/eval/src/eval/aggregation/scene_score.py
- **Timing and concurrency.** Two concurrent sessions per replica in official runs. "Missing/malformed rollout results
  now cause `RESULTS_INCOMPLETE`" (#170). Wall time on the server was about 3x a contestant's offline estimate in #170.
  Warmups "do not establish that an image meets the timing, throughput, or full-workload requirements".

## 4. Teams on the board: write-ups and what made scores jump

No team listed has published a write-up, repo or paper on its AlpaSim nuPlan entry (searched GitHub, arXiv, the HF and
NVIDIA forums). Technical reports are due 2026-10-31. The team-lucifer notes repo
(github.com/skr3178/AlphaSim_challenge) is indexed by search but no longer resolves. The board itself is evidence
(snapshot 2026-10-11, average scene score / PCS):

| team | progression (tag: scene score) | reading |
|---|---|---|
| SymPhi | vavam-gain1075 0.569 (09-16), vavam-nav-medoid 0.629, vavam-nav-brake4 0.652, sp-0926b 0.702 (09-26), **sp-1002a 0.940** (10-02) | +0.24 in a week. Probably a different model or driver path, not a tweak. No public detail. |
| tongji_zero (my-drive-cosmos) | visual12 0.774 (10-08), **simwam-rl10 0.920** (10-11 tag, 10-10 submit) | Simulator-trained / RL world-action model. Highest km/incident outside the garage (0.929). |
| tongji-zeroone-av | v4 routepad 0.705, v5j routepad 0.725 | Route padding alone did not lift them past 0.73. |
| team-ready (RDY Mobility) | fold4-lf4 0.875 (09-23), rdyc2-fix-v1 0.902 (09-30) | Tag says "fix". |
| team-lucifer (Lucifer AI) | vavam-stock 0.549, wajepa-s2 0.762, garage 0014 0.848, **lf-007 0.879** | Best result builds on the garage baseline. |
| wam-cbf (one-wam/one-wam2) | drivewam 0.616, step32250 0.650, v3nl80600 0.805, refod80600 0.796 | CBF safety filter. Large gain in km/incident (0.08 to 0.86). |
| VF-Team | v2.1 0.891, v4.0 0.769 | Newer model worse on private (their #194 report). |
| Team VT | smokepass 0.782, V2 0.849, V3 0.750, V4 0.801 | |
| amanupg | v0.3-a15int8-1hz 0.827 | Alpamayo-1.5 int8 at 1 Hz. |
| ACM_Lab_D_Team, Fox SRDC (team-foxhihi 0.560), Apex Autonomous | not on the nuPlan board or no detail | |

- **Host baselines.** "py123d-garage (host)" is KE:SAI, which is ineligible for prizes. `resnet34_v0.1.0` /
  `model_0014.pth` is the only public checkpoint: "Camera-only latent TransFuser with a ResNet-34 image backbone",
  trained on nuPlan + PAI train/val (nuPlan source weight 4, about 46% per batch). It is two-stage (perception
  pretraining, then planning posttraining for 15 epochs; 0014 is epoch 14). Inputs are three cameras stitched to
  1024x256, ego velocity, and one target point 45 m along the route. The output is a 4 s trajectory at 0.5 s.
  `0062` is not in the public HF repo and is an unreleased host checkpoint **[inference: likely a longer or
  nuPlan-specific posttraining run]**.
  https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints
- **The private evaluation looks deterministic.** Four independent submissions of the public `0014` checkpoint
  (wj-team `resnet34_v0.1.0-v1`, yax-nam `base0014-v1`, team-lucifer untagged, ulsan-alps `garage-ltf-r34-v011`) score
  identically: scene score 0.848, km 0.228, GT distance 0.72, PCS 1521-1523. The host's own `014-nuplan-0014-v1` also
  scores 0.848.
- **The host frontier on private** is `093-nuplan-v1` (2026-10-07) at 0.938 and `092` at 0.925. These are newer
  unreleased garage policies, so a nuPlan camera policy can score 0.94 on the private suite.
- **"go_straight_with_delay_v2"** (Host, 2026-09-01) scores **0.799** with GT distance 1.15 m. A policy that drives
  straight beats every entry below 0.80, including ours at 0.640.
- **Ulsan-alps GTRS and opendrivelab simscale-gtrs-dense** (0.761 and 0.761/0.675/0.563) and the
  **simscale-diffusiondrive/LTF releases** (0.772/0.738) are the SimScale sample submissions run by hosts.

## 5. Predecessor competitions

- **NAVSIM v1 challenge (CVPR 2024).** It used a private test split (`private_test_e2e`) with "only sensor inputs, ...
  withholding all annotations", with 463 submissions from 143 teams. Hydra-MDP (NVIDIA) won. After the challenge the
  public server re-opened on navtest. I found no published public-vs-private gap or organiser retrospective on
  overfitting. https://arxiv.org/abs/2406.15349, https://blogs.nvidia.com/blog/auto-research-cvpr-2024/
- **NAVSIM v2 challenge (CVPR 2025, then ICCV 2025).** Public `navhard_two_stage` (450 real Stage-1 and 5,462
  synthetic Stage-2 samples) was used against a private `private_test_hard_two_stage` ("provided by nuPlan from
  Motional"), with a `warmup_two_stage` HF split and EPDMS two-stage scoring. Winners: NVIDIA GTRS (CVPR 2025) and
  SimpleVSF at 53.06 (ICCV 2025). SimpleVSF's best navhard ablation is 47.18-47.68, while its private_test_hard score
  is 53.06, so the private score was **higher** than public. That is a different scene set, not an overfit drop. I
  found no organiser retrospective on overfitting.
  https://github.com/autonomousvision/navsim/blob/main/docs/splits.md, https://opendrivelab.com/challenge2025/,
  https://arxiv.org/html/2510.17191v1, https://arxiv.org/pdf/2506.04218
- **Reproducibility on navhard.** One team could not reproduce a published 48.3 navhard EPDMS (got 42.17) until it
  forced fp32 inference. https://github.com/WilliamXuanYu/CLOVER/issues/2. Numeric precision on a different GPU
  changes closed-loop scores by points.
- **CARLA / Bench2Drive / HUGSIM 2025.** I found no challenge with a public/private split and a published gap or
  retrospective (searched; results were benchmark papers only). This item is thin.
- **KE:SAI's stated lessons (2026-09-08 blog).** Displacement error "can reward the wrong behavior". NAVSIM ignored
  deployment constraints. "Incorporating significant methodological complexity is now common on the benchmark, even
  for marginal gains." https://kesai.eu/blog/2026-09-08-common-task-framework/

## 6. Drive-IRT / Policy Capability Score

- **Input** is the per-rollout scene score (`scene_score.py`). The algorithm is "zoib (a continuous Beta-IRT variant),
  applied uniformly to all eligible submissions" (2026-08-03, https://github.com/NVlabs/alpasim/issues/133).
  Repeated rollouts are averaged per scene before the fit (local-evaluation README).
- **Model.** Molenaar (2022) zero-and-one-inflated Beta-IRT with **per-item discrimination** (2PL by default:
  "fix_discrimination=False learns a per-item discrimination (default, 2PL)"). Exact zeros follow
  P(X=0) = sigma(g0_n - a_n theta_s) and exact ones follow P(X=1) = sigma(a_n theta_s - g1_n). The interior is Beta
  with mean sigma(a_n theta_s - b_n) and a per-item precision. The fit uses SVI with 1000 epochs and seed 42, and an
  average over the last 2.5% of iterates.
  https://github.com/kesai-labs/drive-irt/blob/main/src/drive_irt/irt/beta_irt.py,
  https://ln2697.github.io/navhard-leaderboard-2/docs/preliminary/10_zero_inflated_model.html,
  https://github.com/NVlabs/alpasim/blob/0bb4c4bfe10951ea5589aa8ea514e422cf3c3506/e2e_challenge/local_evaluation/evaluate.py
- **Weighting.** "A policy's estimated ability rises quickly when it passes routes that the rest of the leaderboard
  fails, and falls harder when it stumbles on routes that the rest passes." "Pushing from 90 to 100 on an easy route
  barely moves the rank, while pushing from 40 to 50 on a hard route moves it substantially."
  https://ln2697.github.io/navhard-leaderboard-2/docs/preliminary/01_motivation.html
- **Anchors.** PCS is an affine map of ability fixed by two anchor subjects and their target scores. `EloScale`
  defaults to 1000/1600, and the local PAI bundle uses 600/2000. The official nuPlan anchors are not published.
  **[inference]** Host `policy_4` sits at 600/607, which fits a low anchor at 600.
- **best_score** equals the score for every entry, in both #166 and our snapshot. The Space shows each team's best with
  a tiebreaker (Space commit c65d7edc, 2026-09-28).
- **Rank intervals.** These are a Monte-Carlo posterior rank interval at 95% with 100k samples. **Display order is by
  the upper rank bound `rank_hi` ascending**, then by km between at-fault incidents, then by point rank (#166). That is
  why rank 9 (1672) sits above rank 10 (1684) in the snapshot.
- **Pool dependence.** "The leaderboard is recomputed each time a new submission comes in and, due to the nature of
  the algorithm, ordering can change as more and more diverse submissions are received" (#166). The docs list "Moving
  Baseline" and decoy attacks as known drawbacks. Fit-to-fit noise looks like ±1-2 PCS (four identical 0014 runs at
  1521-1523, two identical go-straight runs at 1314/1316). The 95% PCS interval for an entry near ours is about ±35.
  No source quantifies drift over time.
- **Our PCS 788 at 0.640.** The other entries between 0.629 and 0.669 have PCS 795-1009, not only 838-1009: wj-team
  `drivor-nav1` 0.641 gives 797, team-adr `wta-ep18` 0.669 gives 795, symphi 0.629 gives 834. We are at the low edge
  but inside the band. The documented explanation is the weighting above. Under the zoib 2PL fit, a zero on a scene
  most of the pool passes, or that has high discrimination, costs more ability than a zero on a scene most policies
  fail. **[inference]** Our zeros fall more on easy scenes, which fits being 0.16 below go-straight. Non-monotonic
  cases are common on this board: go-straight scores 0.799 for PCS 1316, VF-Team 0.891 for PCS 1232, and tongji
  simwam 0.920 for PCS 1321.

## Implications for our 0.935 -> 0.640 gap (ranked by how well the evidence supports each)

1. **An image- or harness-specific behaviour failure on the official evaluator, not distribution shift (strongest).**
   - The organisers say the private set is not a new distribution (#191).
   - Host policies reach 0.938 there.
   - A go-straight policy gets 0.799. We are 0.16 below driving straight, and our GT distance (2.29 m) is about twice
     go-straight's (1.15 m) and three to five times the garage's (0.44-0.75 m). Our car leaves the recorded path far
     more than a policy that ignores the scene.
   - Evaluation is deterministic (four identical 0014 runs), so this is not noise.
   - About 0.12 of our 0.295 drop matches the baselines' shared drop. The remaining ~0.17 is specific to us.
2. **Concurrency is the one documented config delta between local `full` and official `ec2`.** The official run has
   16 replicas x 2 concurrent sessions per replica, `nr_workers 4`, and a shared H100 throughput budget. Any per-process
   state that is not keyed by `session_uuid` would be invisible locally and corrupt rollouts there: history buffers,
   cached plans, route, dynamic state, CUDA-graph or batch buffers, RNG. So would timeouts that push us to fallback
   trajectories. Our local runs are Docker-free native AlpaSim, so this path was never exercised. Testing two
   interleaved sessions per driver process locally is the cheapest check.
3. **Route handling on private scenes.** The suites share the MAP generator with a 40 m start offset, rig-frame
   waypoints and NaN padding. The private set may hit the end-of-recording extension, short or empty routes, and NaN
   tails more often than navtest. Garage-style bridge/pad handling is the documented robust form, and other teams named
   versions after it ("routepad"). Check how our driver behaves on NaN-only or empty routes and routes that start at
   40 m.
4. **Renderer and GPU numerics.** Production uses an internal renderer entrypoint and H100 instead of our GPU. The
   navhard fp32 episode shows precision can move closed-loop scores by points. This hurts a policy that is fragile to
   image statistics, but the shared baseline drop already bounds the effect for robust policies.
5. **Scoring bugs (#198 tail jitter) and the off-road switch-on (#180).** These hit all entries. Our local runs use
   `0bb4c4b`, so off-road is on in both. They are not a differential explanation.
6. **PCS 788.** This is consistent with the zoib weighting (zeros on easy scenes are expensive) and sits inside the
   band of peers at 0.63-0.67. It is not an anomaly needing a separate explanation.

**Concrete, dated facts we may not have known**
- 2026-09-15 (#182): the nuPlan private set is "a totally different set" from the soft-open one. 2026-09-25 (#191):
  organisers say it is "not a previously unseen distribution" and expect the distribution to be close to public.
- 2026-09-23 (HF Space): public suite = 1,485 scenarios. The MAP-foldback issue (10 public scenes) "does not affect the
  private competition set". `e2e_challenge` stays frozen for the competition.
- The base config comment says "All public and private NuPlan challenge presets inherit" the MAP route with a 40 m
  start offset.
- Official nuPlan runs: 16 replicas x 2 concurrent rollouts, H100 (2026-09-24, #189), and a production-only renderer
  entrypoint.
- The leaderboard JSON is public via the Space's Gradio endpoint `lambda_5`. Fields: `rank_lo/hi`, PCS interval,
  `dist_to_gt_trajectory`, `best_score`, `metric_algorithm: zoib`. Ours: PCS 788 [750, 825], rank [76, 90], km 0.132,
  GT distance 2.29 m, submitted 2026-10-11.
- Host `go_straight_with_delay_v2` = 0.799 on private. The public `0014` checkpoint = 0.848, reproduced exactly by
  five submissions. The best host policy = 0.938 (2026-10-07).
- Displayed rank is ordered by the PCS 95% rank upper bound, then km/incident, then point rank (#166, 2026-08-28).
- 2026-10-09 (#198): the organisers acknowledge a corridor-scoring bug, have a fix, and were to decide on 2026-10-12
  how to handle it mid-challenge.
- 2026-10-02 (HF Space): nuPlan finalists get two extra evaluation runs, the Harmonizer stays off for nuPlan, at least
  the top 3 teams per track go to final runs, and registration closes on 2026-10-18.
- Warmup status reports expected/valid/passed rollout counts on 32 private scenes (#203). Our warmup quota is 3 per
  rolling 30 days and does not use official quota. It is a cheap probe of whether a fix changes private pass counts.
- No nuPlan per-scene results or reference bundle are available (Discussion #199, 2026-10-09, unanswered).
