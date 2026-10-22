# P0 subjects: setup log (2026-10-23)

Box root `$DATA_DIR=/root/autodl-tmp/ujs`. State as of the last contact with the box (the SSH endpoint went down during
builds; the box has since been restarted and the four unfinished jobs were relaunched and finished).
Only deviation from as-shipped: torch 2.8.0+cu128 / torchvision 0.23.0 (Blackwell sm_120), plus the notes per model.

## Scope change (user, 2026-10-23): camera-only subjects only
LiDAR dropped. The partial navtest LiDAR work was removed: 22 GB of half-downloaded archives and 2.4 GB of extracted
`MergedPointCloud` dirs deleted (only ones created here; none existed before). Freed ~24.5 GB.
WoTE and the hustvl DiffusionDrive (camera + LiDAR) are abandoned, left in place, no further work:
- DiffusionDrive: repo `third_party/diffusiondrive` @ 9b52ed0; ckpt `models/diffusiondrive/diffusiondrive_navsim_88p1_PDMS` (729,518,199 B), `kmeans_navsim_traj_20.npy`, `diffusiondrive_88p1_PDMS.csv`; env `envs/diffusiondrive` (8.1 GB, py3.9, torch installed, requirements step interrupted = incomplete).
- WoTE: repo `third_party/wote` @ 298957c; `models/wote` 2.3 GB (ckpt + extra_data; resnet34.pth download was cut); env `envs/p0-wote` (8.0 GB, incomplete).

## ReCogDrive-2B-RL: READY (camera-only forward check passed)
- Repo `third_party/recogdrive` @ 5831ee83c7a6a40bf6919715b96b0e9ba2b48335. Env `envs/p0-recogdrive` (python 3.10.21, torch 2.8.0+cu128, transformers 4.37.2, 8.1 GB).
- Weights `models/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt` (550 MB; agent.checkpoint_path), `models/recogdrive/ReCogDrive-VLM-2B/` (4.18 GB; agent.vlm_path). 4.5 GB total.
- Deviations from as-shipped: torch 2.8.0+cu128; python 3.10 (the code uses `X | Y` annotations, recogdrive_dit.py:218; environment.yml says 3.9); extra packages `pytest` (nuplan import), `decord` (internvl_preprocess.py),
  `peft>=0.17,<0.18` (newest diffusers refuses internvl_chat's peft==0.10.0); transformers 4.37.2 / timm 0.9.12 / tokenizers 0.15.1 as in internvl_chat.txt; no flash-attn ("FlashAttention2 is not installed", the backbone falls back and loads).
  Run-time env: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch >= 2.6 rejects the lightning ckpt's omegaconf objects at recogdrive_agent.py:102; set from our side, repo untouched).
- LiDAR: requested, never consumed. `recogdrive_agent.py:111-112` `get_sensor_config()` = `SensorConfig.build_all_sensors(include=[0,1,2,3])` sets `lidar_pc=[0,1,2,3]` (`navsim/common/dataclasses.py:558`), the loader then opens each history .pcd (`dataclasses.py:134-136`) and errors if absent.
  No file in `navsim/agents/recogdrive/` mentions lidar. Check with the sensor config overridden at run time (`sc = agent.get_sensor_config(); sc.lidar_pc = False`, `agent_poses.py` needs the same override): `AgentInput.lidars` all empty, `compute_trajectory` returns poses (8, 3) fine.
- Forward check (pool job p0-recog-check, card 5, `$DATA_DIR/tmp/rc_check.py` + `rc_check.sh`, log `tmp/p0_logs4/log.txt`): token c600bf33d70453c4 (first log of navtest), poses (8, 3), 3.08 s incl. first call, peak 7.65 GB, lidar_loaded [False x4].
- The shipped scorer passes `sensor_config=agent.get_sensor_config()` (run_pdm_score_recogdrive.py:92, plus `load_image_path=True`): the registration side must apply the lidar_pc=False override there.

## DriveVLA-W0: weights + env READY, load check passed (no forward: needs the VQ-token dataset path)
- Repo `third_party/drivevla-w0` @ a55766ce46ee55d9035b93816e57b8d933166482. Env `envs/p0-drivevla-w0` (python 3.10, torch 2.8.0+cu128, transformers 4.44.0, numpy 2.1.2, 7.1 GB).
  Built from the repo's `environment.yaml` pip list as shipped minus torch / torchvision / torchaudio / triton / nvidia-* (replaced by torch 2.8 cu128), deepspeed (training only) and flash-attn
  (the inference config sets `attn_implementation: sdpa`, so it is not needed).
- Data in `models/drivevla-w0` (34 GB): `Emu3_Flow_Matching_Action_Expert_PDMS_87.2/` (4 safetensors 18.15 GB + config + tokenizer files + `json_output/` their per-token outputs),
  `Emu3_NuPlan_Pretrain_Cktps/` (16.9 GB, weights + config + tokenizer files; the shipped `Emu3Pi0.__init__` loads its VLM from this path = the script's `VLA_VLM_MODEL`/`pretrain_vlm_path`, then the hub weights overwrite it),
  `navsim_emu_vla_256_144_test_pre_1s.pkl` (35 MB, test meta), `test_vq_codes.zip` (152 MB, VQ codes of the test images, not unzipped yet), `cluster_centers_8192.npy`, `flow_matching_json_output_PDMS_87.2.zip` (544 MB).
  Not downloaded: `formatted_pdm_score_8192.zip` (6.4 GB; only the anchor-based variants use the anchors / scores, the flow-matching script lists them as env vars), `nuplan_pickle_meta_and_vq_codes.tar.gz`, trainval pickles.
  The FAST action tokenizer (`VLA_ACTION_TOKENIZER`) config dir is in the repo: `configs/fast/`; norm stats `VLA_NORM_STATS` in `configs/normalizer_navsim_*` (verify the path when wiring).
- Load check (pool job p0-w0-check, `$DATA_DIR/tmp/w0_check.py`, log `tmp/p0_logs6/log.txt`): `Emu3Pi0.from_pretrained(hub, pretrain_vlm_path=NuPlan_Pretrain, sdpa, bf16)` -> 9.07 B params, missing keys 0, unexpected 0, 13.4 s, 18.3 GB on the card.
- Pipeline shape: not a NAVSIM agent; `inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh` (torchrun, per-token JSON) then `eval_navsim_metric_from_json.sh`; paths are env-overridable (EMU_HUB, VLA_VLM_MODEL, TEST_DATA_PKL, VLA_ACTION_TOKENIZER, VLA_NORM_STATS, VLA_TOKEN_YAML, VLA_ANCHOR_*). Config code expects repo-relative `train/` + `reference/Emu3` (only the latter exists; `config.py` adds it to sys.path).

## AutoVLA: READY (load + one-sample inference passed)
- Code `third_party/autovla` @ ba34eed; `get_sensor_config()` has `lidar_pc=False` (navsim/navsim/agents/autovla_agent.py:405).
- Env `envs/autovla` (python 3.12, torch 2.8.0+cu128, transformers 4.49.0, 6.7 GB) from `experiments/baselines_latency/archive/setup_autovla.sh` unchanged (deviation: torch 2.8 / py3.12, listed in that script).
- Weights `models/autovla/AutoVLA_PDMS_89.ckpt` (16.29 GB) + `models/autovla/Qwen2.5-VL-3B-Instruct` (7.1 GB).
- Check: `experiments/baselines_latency/archive/autovla.py --checkpoint ... --base-model ... --iters 1 --warmup 0` (pool job p0-av-check): loads and returns 10 action tokens / traj points on the Waymo demo scene (nocot mode, ~2.0 s first call). Their NAVSIM eval path still needs the navtest preprocessing described above.

## Camera-only non-foundation references already on the box (nothing new downloaded)
- SimScale DiffusionDrive: `models/alpasim_dd/diffusiondrive_sim_navhard.ckpt` (243,596,717 B); code `third_party/alpasim/e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/navsim_diffusiondrive_challenge/` (`simscale_diffusiondrive/` model, `policy.py`).
  `latent=True` (no LiDAR, LTF-style input), cameras CAM_L0 / F0 / R0. Not a NAVSIM agent class: standalone `DiffusionDrivePolicy.predict_batch` (images + command one-hot + velocity + acceleration).
- SimScale LTF: `models/alpasim_ltf/ltf_sim_navtest.ckpt`; code `.../sample_submission_simscale_navsim_transfuser/navsim_transfuser_challenge/`.
- Both import and run in `envs/alpasim-ltf` (torch 2.8.0, no navsim installed). Check (GPU 0, random 1080x1920 images, command straight): DD loads strict, trajectory (8, 3), 0.95 s incl. first-call; LTF (8, 3), 0.63 s.
  Script `$DATA_DIR/tmp/dd_sim_check.py {dd|ltf}`. To run navtest they need a thin adapter (NAVSIM `AgentInput` -> `InferenceInput`; command from the navtest driving command, ego velocity / acceleration from `ego_statuses[-1]`), then the bench scores the poses; no existing NAVSIM-env agent wraps them.
