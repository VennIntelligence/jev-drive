# P0 subjects: third-party navtest drivers for the usage probe (shortlist, 2026-10-22)

Measurement subjects / reference rows for the paper's C2 ("grafted foundation models are surplus"), never part of our
method. Criterion: public checkpoint, runs on NAVSIM navtest with the authors' own code, published navtest score to
reproduce. Repos and checkpoints below were checked live on GitHub / Hugging Face on 2026-10-22 (API listing, sizes,
gating). All published numbers are **NAVSIM v1.1 PDMS on navtest**: the reproduction gate is v1 PDMS (devkit
`third_party/navsim-v1.1`, env `envs/navsim1`, cache `runs/navsim/metric_cache/v1_navtest`, both exist); our board
number (v2 EPDMS, `jevdrive.bench` navtest) is read from the same poses.

Box root below: `$DATA_DIR = /root/autodl-tmp/ujs`.

## Shortlist

| # | Model | Role | Published navtest (metric) | Checkpoint (size) | Licence / gating | Verdict |
|---|---|---|---|---|---|---|
| 1 | DiffusionDrive | strong non-foundation baseline (ResNet-34 TransFuser + truncated diffusion, 60 M) | **88.1 PDMS** v1, per-token CSV released | HF `hustvl/DiffusionDrive` file `diffusiondrive_navsim_88p1_PDMS` (0.73 GB) | repo MIT, weights card Apache-2.0, not gated | required, set up first |
| 2 | ReCogDrive-2B-RL | LLM / VLM driver (InternVL3-2B, i.e. InternViT + Qwen2.5-1.5B LLM, hidden states -> DiT diffusion planner, DiffGRPO) | **90.8 PDMS** v1 (README table, Base-RL) | HF `owl10/ReCogDrive-VLM-2B` (4.18 GB) + `owl10/ReCogDrive-2B-RL` `ReCogDrive_Diffusion_Planner_2B_RL.ckpt` (0.55 GB) | Apache-2.0, not gated | LLM subject, probe target |
| 3 | WoTE | world-model driver (BEV world model rolls out candidate trajectories, reward scorer; TransFuser base) | **88.3 PDMS** v1 (README) | official Google Drive folder `1dIHK8nXkzhIhGCRQOpKibaizwH-7fHqs` (ckpt `epoch=29-step=19950.ckpt` + extra data); unofficial HF mirror `dfrmx/WoTE` with the same file names (0.77 GB ckpt + 1.53 GB `formatted_pdm_score_256.npy`) | Apache-2.0, not gated | world-model subject (cheap) |
| 4 | DriveVLA-W0 | foundation world model + VLA (Emu3 8B, world-model pretraining on video, flow-matching action expert) | **87.2 PDMS** v1 for the released checkpoint (folder name; paper headline 88.4 / 93.0 belong to other variants) | HF `liyingyan/DriveVLA-W0` folder `Emu3_Flow_Matching_Action_Expert_PDMS_87.2` (4 safetensors, 18.15 GB) + `json_output/` (their own outputs, 5 021 tokens, 0.23 GB) | repo licence not declared, weights card none; not gated | world-model subject closest to the paper's claim; **conditional** (artifacts below) |
| B | AutoVLA | backup LLM driver (Qwen2.5-VL-3B, action tokens, RFT) | 89.1 PDMS (checkpoint name `AutoVLA_PDMS_89`; check the paper table before gating) | HF `Zewei-Zhou/AutoVLA` `AutoVLA_PDMS_89.ckpt` (16.29 GB) | UCLA Academic Software License (academic only), not gated | backup only: needs its own nuPlan preprocessing of navtest; set up only if #2 fails |

Rejected: **World4Drive** (`ucaszyp/World4Drive`, no public NAVSIM checkpoint linked; nuScenes `dist_test` only), **LAW**
(`BraveGroup/LAW`, Google Drive checkpoints are nuScenes perception-free; no NAVSIM checkpoint), **Epona** (`Kevin-thu/Epona`,
checkpoints public but its NAVSIM number comes from its own nuPlan test meta, no NAVSIM agent / navtest eval path),
Drive-JEPA (no public code / weights found), Qwen-Drive-1.0-4B (WOD-E2E model, no navtest path; latency already in
docs/baselines.md), DiffusionDriveV2 (91.2, same family as #1, adds nothing as a reference row).

Why these: #1 is the required non-foundation reference (same ResNet-34 scale as most navtest entries, released
per-token CSV makes the gate exact). #2 is the only LLM-based navtest driver with public weights whose planner reads
VLM hidden states directly (the P1 hook point is a single tensor: the VLM hidden state handed to the DiT), small (2B),
Apache. #3 is a cheap world-model driver run through the stock NAVSIM agent path; #4 is the one subject that is a
*foundation* world model grafted onto driving, which is exactly C2's target, but it is heavy and has open artifact
questions, so it is the stretch row.

## Data (shared by all)

| Need | Box path | Status |
|---|---|---|
| navtest logs (147) | `$DATA_DIR/datasets/navsim/navsim_logs/test` | exists |
| navtest cameras (8 cams, current + history) | `$DATA_DIR/datasets/navsim/sensor_blobs/test` (121 GB) | exists |
| maps | `$DATA_DIR/datasets/navsim/maps` (`nuplan-maps-v1.0`) | exists |
| v1 metric cache (PDMS gate) | `$DATA_DIR/runs/navsim/metric_cache/v1_navtest` (3.0 GB) | exists |
| v2 metric cache (EPDMS board) | `$DATA_DIR/runs/navsim/metric_cache/v2_navtest` (3.6 GB) | exists |
| **navtest LiDAR (`MergedPointCloud`)** | `$DATA_DIR/datasets/navsim/sensor_blobs/test/<log>/MergedPointCloud` | **missing** (docs/navsim.md: camera-only by design). Needed by #1 and #3 (TransFuser backbones, `latent=False`, `lidar_seq_len 1` = current frame only). Source: HF dataset `OpenDriveLab/OpenScene` `openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_{0..31}.tgz`, 32 archives, **92.7 GB to download**. Extract only `MergedPointCloud` of the frames the agents read (navtest current frames, ~12 k files; est. ~10-15 GB on disk), one archive at a time, delete each archive after extraction (peak ~3 GB). ~2.5-3 h at the box's ~10 MB/s via `proxy_on`. |
| NAVSIM devkit env for scoring | `envs/navsim1` (v1.1, PDMS), `envs/navsim2` (v2, EPDMS) | exist; scoring path `experiments/zeroshot_openloop/archive/navsim_zs_score.sh` + PrecomputedAgent |

## Per model: setup recipe for the setup agent

Common rules: Blackwell RTX 6000D (sm_120) needs torch >= 2.7 with cu128; every model uses **torch 2.8.0 + cu128**
instead of its pin (the only allowed deviation; record it). numpy 1.x envs need `OPENBLAS_CORETYPE=Haswell`
(sitecustomize, as `scripts/setup_navsim_devkit.sh` does). pip mirror `UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple`;
GitHub via `source /etc/network_turbo`, Hugging Face via `proxy_on` (docs/network-proxy.md). Code checkouts in
`$DATA_DIR/third_party/<name>` at the commit below; weights in `$DATA_DIR/models/<name>`. Path constants that the READMEs
tell users to edit (checkpoint, backbone, anchor paths) may be set; nothing else in the model code changes.

### 1. DiffusionDrive

- Repo: https://github.com/hustvl/DiffusionDrive @ `9b52ed0ec06b073d82d6f392ab084c7b301c8681` (2025-12-08). A navsim v1.1 fork
  (agent `diffusiondrive_agent`, `navsim/agents/diffusiondrive/`).
- Checkpoint: https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_navsim_88p1_PDMS (729 518 199 B,
  repo sha `8e3cc29`). Skip `diffusiondrive_nusc_stage2.pth` (nuScenes).
- Extra files: anchors https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy ;
  published per-token eval CSV https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/diffusiondrive_88p1_PDMS.csv
  (the reproduction reference, store next to the weights); ResNet-34 init `timm/resnet34.a1_in1k` exists in
  `$DATA_DIR/cache/huggingface/hub/models--timm--resnet34.a1_in1k` (only `bkb_path` init; the checkpoint overrides it).
  Set `bkb_path` / `plan_anchor_path` in `navsim/agents/diffusiondrive/transfuser_config.py` (README step).
- Env: their `environment.yml` = python 3.9 + navsim v1.1 `requirements.txt` (nuplan-devkit v1.2, diffusers, timm,
  pytorch-lightning, hydra-core 1.2.0, numpy 1.23.4). Proposed target `envs/p0-navsim1-gpu` (shared with WoTE, see #3):
  `uv venv -p 3.10`, torch 2.8.0+cu128 / torchvision 0.23.0, then the fork's requirements minus the nuplan git pin,
  `uv pip install --no-deps -e $DATA_DIR/third_party/nuplan-devkit` (exists), diffusers, timm. Fork imported via PYTHONPATH.
- Eval as shipped: `run_pdm_score.py train_test_split=navtest agent=diffusiondrive_agent agent.checkpoint_path=$CKPT`.
- Needs LiDAR (see Data).
- Already on the box, **not the same model**: `$DATA_DIR/models/alpasim_dd/diffusiondrive_sim_navhard.ckpt` and
  `$DATA_DIR/third_party/alpasim/e2e_challenge/sample_submission_simscale_navsim_diffusiondrive` are the SimScale /
  AlpaSim camera-only DiffusionDrive sample (trained for navhard-sim, docs/alpasim.md), env `envs/alpasim-ltf`
  (torch 2.8.0, timm 1.0.27, no nuplan / navsim). Not a substitute for the published 88.1 checkpoint; `envs/alpasim-ltf`
  is only a reference for a working torch 2.8 + timm install. `envs/garage` (torch 2.8, timm, lightning) likewise.
- Published: 88.1 PDMS (NC 98.2, DAC 96.2, TTC 94.7, C 100, EP 82.2, paper Table 1); v2 EPDMS 88.2 as quoted in
  `jevdrive/openloop_standing.py` (SimWAM Table 2, secondary).

### 2. ReCogDrive-2B-RL

- Repo: https://github.com/xiaomi-research/recogdrive @ `5831ee83c7a6a40bf6919715b96b0e9ba2b48335` (2026-10-14). A navsim fork
  (README points `NAVSIM_DEVKIT_ROOT` at `navsim-main`: confirm whether its scorer is v1 PDMS or v2 before quoting).
- Checkpoints: https://huggingface.co/owl10/ReCogDrive-VLM-2B (sha `16873ac`, 4.18 GB, `agent.vlm_path`) and
  https://huggingface.co/owl10/ReCogDrive-2B-RL `ReCogDrive_Diffusion_Planner_2B_RL.ckpt` (sha `7346ede`, 0.55 GB,
  `agent.checkpoint_path`). The 8B variant (90.4) is not needed.
- Eval as shipped (`scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh`): torchrun
  `navsim/planning/script/run_pdm_score_recogdrive.py train_test_split=navtest agent=recogdrive_agent
  agent.checkpoint_path=... agent.vlm_path=... agent.cam_type=single agent.grpo=False agent.cache_hidden_state=False
  agent.vlm_type=internvl agent.dit_type=small agent.vlm_size=small agent.sampling_method=ddim`. Front camera only (no LiDAR).
- Env: `environment.yml` python 3.9 + `requirements.txt` (navsim deps + diffusers, timm, torch unpinned). transformers /
  flash-attn versions for InternVL3 are not pinned in requirements.txt: take them from `internvl_chat/` (check
  `internvl_chat/pyproject.toml` / requirements) and record. Target `envs/p0-recogdrive`, weights `models/recogdrive/`.
- Published: 90.8 PDMS (README); `openloop_standing.py` quotes "ReCogDrive (VLM)" 90.8 PDMS / 83.6 EPDMS / navhard 25.7.
- Nothing on the box yet.

### 3. WoTE

- Repo: https://github.com/liyingyanUCAS/WoTE @ `298957c128a91d41a1c6075bd0bb6e7e845e093f` (2025-06-29). navsim v1.1 fork, agent
  `WoTE_agent`, config `navsim.agents.WoTE.configs.default.WoTEConfig`.
- Checkpoint + extra data: official Google Drive folder https://drive.google.com/drive/folders/1dIHK8nXkzhIhGCRQOpKibaizwH-7fHqs
  (ckpt `epoch=29-step=19950.ckpt`, `trajectory_anchors_256.npy`, `formatted_pdm_score_256.npy`, `resnet34.pth`); use `gdown --folder`
  via `proxy_on`. Fallback: HF `dfrmx/WoTE` (unofficial mirror, sha `818b2c9`, same ckpt name + `formatted_pdm_score_256.npy`;
  anchors would then be regenerated with `scripts/miscs/k_means_trajs.py` -> record which source was used). Set
  `resnet34_path` etc. in `navsim/agents/WoTE/configs/default.py`.
- Eval as shipped (`scripts/evaluation/eval_wote.sh`): `run_pdm_score.py agent=WoTE_agent agent.checkpoint_path=...
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig split=test scene_filter=navtest`.
- Env: python 3.8 pin in `environment.yml`, otherwise navsim v1.1 requirements: share `envs/p0-navsim1-gpu` with
  DiffusionDrive if both import cleanly (3.10, torch 2.8 cu128); else a separate `envs/p0-wote`. Weights `models/wote/`.
- Needs LiDAR (TransFuser backbone, `lidar_seq_len 1`).
- Published: 88.3 PDMS (NC 98.5, DAC 96.8, EP 81.9, TTC 94.9, C 99.9).
- Nothing on the box yet.

### 4. DriveVLA-W0 (conditional, stretch)

- Repo: https://github.com/BraveGroup/DriveVLA-W0 @ `a55766ce46ee55d9035b93816e57b8d933166482` (2026-02-11). Not a NAVSIM agent:
  its own inference writes one JSON per token (`inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh`), then
  `inference/vla/eval_navsim_metric_from_json.sh` scores them with navsim v1.1. Bench integration = convert their JSONs to
  the poses file (tokens, (N, 8, 3)).
- Checkpoint: https://huggingface.co/liyingyan/DriveVLA-W0 (sha `0bd601d`) folder `Emu3_Flow_Matching_Action_Expert_PDMS_87.2`,
  4 safetensors = 18.15 GB. `json_output/` in the same repo = their outputs on 5 021 tokens (0.23 GB): a per-token reference.
- **Open artifacts (check before downloading the 18 GB):** README says to also download `navsim_emu_vla_256_144_test_pre_1s.pkl`
  from that HF repo, but it is **not in the listing** (generate with `tools/pickle_gen/pickle_generation_navsim_pre_1s.py`?);
  the inference script reads `VLA_ACTION_TOKENIZER` (FAST tokenizer, `pretrained_models/fast`), `VLA_NORM_STATS`
  (`configs/normalizer_navsim_*`, in repo), and anchor files `cluster_centers_8192.npy` / `formatted_pdm_score_8192.npy`
  at private paths (unclear whether the flow-matching expert uses them); inputs are Emu3 VQ tokens of the navtest images
  (`scripts/tokenizer/extract_vq_emu3_navsim.sh`, needs `BAAI/Emu3-VisionTokenizer`, a GPU pass over navtest frames).
  If any of these is unobtainable, report DriveVLA-W0 as "cannot run as shipped" and stop on it.
- Env: python 3.10, torch 2.4.0 cu124 pin -> 2.8.0 cu128; `transformers==4.44.0`, `tiktoken==0.6.0`, `flash-attn==2.5.7`
  (needs a sm_120 build: use a flash-attn wheel matching torch 2.8 / cu128, or sdpa if the code allows it; record which).
  Target `envs/p0-drivevla-w0`, weights `models/drivevla-w0/`.
- Published: 87.2 PDMS (released checkpoint name). Nothing on the box yet.

### B. AutoVLA (backup)

- Code exists: `$DATA_DIR/third_party/autovla` (ucla-mobility/AutoVLA @ `ba34eed`, 40 MB). Env `envs/autovla` and the
  checkpoint were **deleted** (not present in `envs/` or the HF cache; recipe in
  `experiments/baselines_latency/archive/setup_autovla.sh`, torch 2.8.0+cu128, transformers 4.49.0). Checkpoint
  https://huggingface.co/Zewei-Zhou/AutoVLA `AutoVLA_PDMS_89.ckpt` 16.29 GB + base `Qwen/Qwen2.5-VL-3B-Instruct`. NAVSIM eval needs
  their nuPlan preprocessing of navtest and `navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh`. Only if #2 fails.

## Inventory of the box (2026-10-22, nothing set up twice)

- `envs/`: alpasim-ltf, carla, garage, hugsim, jevdrive, navsim1, navsim2, nuplan_devkit_overlay, op-train, openpilot,
  p5v1-pdm, svlm-extra (fla only), vfi, wajepa, waymo. No env for any shortlisted model; `envs/autovla` and
  `envs/qwen-drive` were removed (code remains in `third_party/autovla`, `third_party/qwen-drive`).
- `models/`: alpasim_dd (SimScale DiffusionDrive sample, not the 88.1 checkpoint), alpasim_ltf, bench2drive, blue,
  bridgedrive, garage, gemma-4-E2B-it, imagenet, lama, openpilot, vjepa2_1, wajepa. HF cache: Qwen3-VL-2B / 4B,
  timm resnet34, dinov2, vjepa2, siglip2 (no shortlisted weights).
- `runs/`: `bench_baselines/{autovla,qwen-drive-1.0-4b,openjev}` are latency runs only (docs/baselines.md), no navtest
  scores. No previous navtest run of any shortlisted model.
- Bench registry (`jevdrive/bench/models.py`): onnx / parity / vt / adapt_h / wajepa families; a `pred` family
  (`PROBE-*`, supplied pose files) is being added by the P1 sibling. No third-party NAVSIM agent family yet.
- In flight: no download or env build has been started by this lane.
- Added by this lane (not yet wired into the bench): `jevdrive/bench/agent_poses.py`, a standalone script that runs a
  fork's NAVSIM agent exactly as its `run_pdm_score.py` does (hydra config of the fork + its eval overrides, SceneLoader,
  `compute_trajectory`) and writes the poses file for the v1 PDMS gate and the v2 EPDMS board.

## Reproduction gate (to be applied once envs exist)

Smoke slice: 8 whole navtest logs (a frozen `jevdrive.data.splits` log split), v1 PDMS. DiffusionDrive: per-token
comparison with its released CSV on those tokens, mean within +-1.0 PDMS. Others: no per-token reference (DriveVLA-W0:
its `json_output` tokens), smoke must not be below published - 3. Full navtest: |reproduced - published| <= 0.5 PDMS
(diffusion samplers are unseeded in the forks; we fix seed 0). A model outside tolerance is reported, not tuned.

## Disk estimate (cap 200 GB)

LiDAR extract ~10-15 GB (92.7 GB streamed); DiffusionDrive 0.8 GB; ReCogDrive 4.7 GB; WoTE ~2.5 GB; DriveVLA-W0 18.2 GB +
VQ tokens of navtest (unknown, est. < 10 GB) + Emu3 tokenizer 0.3 GB; envs 3-4 x ~8 GB. Total ~75-90 GB without AutoVLA
(+~25 GB with it). Outputs: poses files and per-token CSVs, < 1 GB.
