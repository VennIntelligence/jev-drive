# P0 subjects: setup log (2026-10-23)

Box root `$DATA_DIR=/root/autodl-tmp/ujs`. State as of the last contact with the box (the SSH endpoint went down during
builds; items marked PENDING were running in tmux `jev` and must be re-checked after the box is back).
Only deviation from as-shipped: torch 2.8.0+cu128 / torchvision 0.23.0 (Blackwell sm_120), plus the notes per model.

## Scope change (user, 2026-10-23): camera-only subjects only
LiDAR dropped. The partial navtest LiDAR work was removed: 22 GB of half-downloaded archives and 2.4 GB of extracted
`MergedPointCloud` dirs deleted (only ones created here; none existed before). Freed ~24.5 GB.
WoTE and the hustvl DiffusionDrive (camera + LiDAR) are abandoned, left in place, no further work:
- DiffusionDrive: repo `third_party/diffusiondrive` @ 9b52ed0; ckpt `models/diffusiondrive/diffusiondrive_navsim_88p1_PDMS` (729,518,199 B), `kmeans_navsim_traj_20.npy`, `diffusiondrive_88p1_PDMS.csv`; env `envs/diffusiondrive` (8.1 GB, py3.9, torch installed, requirements step interrupted = incomplete).
- WoTE: repo `third_party/wote` @ 298957c; `models/wote` 2.3 GB (ckpt + extra_data; resnet34.pth download was cut); env `envs/p0-wote` (8.0 GB, incomplete).

## ReCogDrive-2B-RL (camera-only in the forward pass)
- Repo `third_party/recogdrive` @ 5831ee83c7a6a40bf6919715b96b0e9ba2b48335.
- Weights `models/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt` (550 MB), `models/recogdrive/ReCogDrive-VLM-2B/` (4.18 GB). Total 4.5 GB.
- LiDAR finding: `navsim/agents/recogdrive/recogdrive_agent.py:111-112` `get_sensor_config()` returns `SensorConfig.build_all_sensors(include=[0,1,2,3])`,
  which sets `lidar_pc=[0,1,2,3]` (`navsim/common/dataclasses.py:558`); the loader then opens the .pcd of every history frame
  (`dataclasses.py:134-136`, called from `AgentInput` building at ~:218-224) and errors if the file is missing. No file under
  `navsim/agents/recogdrive/` references lidar: the feature builder / backbone use camera images only. So LiDAR is requested, never consumed.
  A camera-only run needs the sensor config with `lidar_pc=False` (not changed in the repo; `agent_poses.py` / a check script can override it).
- Env: `envs/p0-recogdrive`. The first build (py3.9) failed at import: `recogdrive_dit.py:218` uses `torch.Tensor | tuple[...]` annotations, so the code needs python >= 3.10
  (environment.yml says 3.9). Rebuilt on python 3.10 (tmux `rc_build3`, PENDING), repo requirements.txt as shipped + transformers 4.37.2, tokenizers 0.15.1, timm 0.9.12,
  sentencepiece 0.1.99, einops 0.6.1, accelerate<1 (from internvl_chat.txt), plus `pytest` (nuplan import) and `decord` (imported by internvl_preprocess.py), and `peft>=0.17,<0.18`
  (newest diffusers refuses the internvl_chat pin peft 0.10.0). flash_attn not installed (backbone asks `use_flash_attn=True`; falls back if absent: to verify).
- Check script `$DATA_DIR/tmp/rc_check.py` (one navtest token, camera-only config copy, `compute_trajectory` on GPU): PENDING until the 3.10 env exists.

## DriveVLA-W0 (camera-only)
- Repo `third_party/drivevla-w0` @ a55766ce46ee55d9035b93816e57b8d933166482.
- Artifacts exist publicly on HF `liyingyan/DriveVLA-W0` root: `navsim_emu_vla_256_144_test_pre_1s.pkl` (35 MB), `cluster_centers_8192.npy`, `test_vq_codes.zip` (152 MB, VQ codes of the test split:
  no GPU tokenization pass needed), `formatted_pdm_score_8192.zip` (6.4 GB, not downloaded; the flow-matching script only sets it via env var), `flow_matching_json_output_PDMS_87.2.zip` (544 MB, their outputs).
  The FAST action tokenizer config is in the repo (`pretrained_models/fast` per README tree), to verify.
- Download to `models/drivevla-w0` (checkpoint folder 18.15 GB + the files above): first attempt died on an HF connection error, retried with a loop (tmux `dl_w0c`, PENDING). Env `envs/p0-drivevla-w0`: NOT started.

## AutoVLA (camera-only)
- Code `third_party/autovla` @ ba34eed. `get_sensor_config()` has `lidar_pc=False` (navsim/navsim/agents/autovla_agent.py:405).
- Weights `models/autovla/AutoVLA_PDMS_89.ckpt` (16.29 GB) + `models/autovla/Qwen2.5-VL-3B-Instruct` (tmux `dl_av`, PENDING); env `envs/autovla` via `experiments/baselines_latency/archive/setup_autovla.sh` (tmux `av_build`, PENDING).
- Navtest needs their own preprocessing: `navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh` reads `JSON_DATA_PATH=./dataset/nuplan/navtest_nocot`
  (produced by `scripts/run_nuplan_preprocessing.sh` with the navtest split, `INCLUDE_COT=false`), the navtest metric cache, `SENSOR_DATA_PATH=.../sensor_blobs/test`,
  config `config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml`; entry `run_pdm_score_cot.py agent=autovla_agent`.

## Camera-only non-foundation references already on the box (nothing new downloaded)
- SimScale DiffusionDrive: `models/alpasim_dd/diffusiondrive_sim_navhard.ckpt` (243,596,717 B); code `third_party/alpasim/e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/navsim_diffusiondrive_challenge/` (`simscale_diffusiondrive/` model, `policy.py`).
  `latent=True` (no LiDAR, LTF-style input), cameras CAM_L0 / F0 / R0. Not a NAVSIM agent class: standalone `DiffusionDrivePolicy.predict_batch` (images + command one-hot + velocity + acceleration).
- SimScale LTF: `models/alpasim_ltf/ltf_sim_navtest.ckpt`; code `.../sample_submission_simscale_navsim_transfuser/navsim_transfuser_challenge/`.
- Both import and run in `envs/alpasim-ltf` (torch 2.8.0, no navsim installed). Check (GPU 0, random 1080x1920 images, command straight): DD loads strict, trajectory (8, 3), 0.95 s incl. first-call; LTF (8, 3), 0.63 s.
  Script `$DATA_DIR/tmp/dd_sim_check.py {dd|ltf}`. To run navtest they need a thin adapter (NAVSIM `AgentInput` -> `InferenceInput`; command from the navtest driving command, ego velocity / acceleration from `ego_statuses[-1]`), then the bench scores the poses; no existing NAVSIM-env agent wraps them.
