# Experiments index

One line per topic: name (aliases): status; key finding [d decision entries]. Open `<topic>/README.md`.

**live**

- [b2d_privileged](b2d_privileged/README.md): live; Red light + green release DS 75.0 to 95.0 [d82]
- [op_adapt_h](op_adapt_h/README.md): live; Fake-yaw following -71..-78%, navtest +0.82; HUGSIM spins 8 to 6 only [d98,(inputs:,92,94,96)]
- [op_adapt_l](op_adapt_l/README.md) (op-adapt L): live; Stop capture 0.252 to 0.559 open loop; B2D no gain [d77-81]
- [op_common_cause](op_common_cause/README.md): live; Plan follows fake history yaw, 3-7x more at low speed, real and CARLA alike [d92-93]
- [op_img_cmd](op_img_cmd/README.md): live; Image route: zero-shot uptake <= 0.14; fine-tuned 0.54-0.64, drift 0.29 m; sky arrow (no map) 0.29-0.39, junction drift 0.13-0.27 m; negatives: CARLA 0.30, wrong-exit control fails [d(pending)]
- [op_route_cmd](op_route_cmd/README.md) (route polyline, nav route): live; Route polyline material: navtrain 103 k / WOD 522 k labelled, 32 k / 62 k turns >= 25 deg; noise + negatives + CARLA plan, no training [d(inputs:,93,102,118)]
- [op_route_ft](op_route_ft/README.md) (route adapter fine-tune): live; Route-choice adapter (bear / poly) + action pathway fine-tune; B2D 25 junction turns primary [d(inputs:,118,121,122,127)]
- [op_dagger](op_dagger/README.md): live; Reprojection rollout engine on real WOD logs + DAgger pilot (paper §3 fig 8) [d(inputs:,92,98,100,111,118,123,124)]
- [factor_wm](factor_wm/README.md) (factorized world model, exact-ego sim): live; Stage 1 pre-registered: on-policy openpilot in exact-ego reprojection sim; no runs yet [d(inputs:,54,76,103,123,132,138,139)]
- [op_fov](op_fov/README.md) (wide FOV, fisheye): concluded; Wide FOV 90/116 deg: turn gain -0.005..-0.033, early stop; wide frame barely used [d(pending)]
- [op_parity](op_parity/README.md) (input parity, WA-JEPA inputs): live; Cinque + ego / pose / cmd / side cams vs WA-JEPA, equal inputs and data [d(inputs:,92,104,111,118,137,138,139)]
- [op_wide_ft](op_wide_ft/README.md) (wide FOV fine-tune, W116): concluded; Fine-tune with 116 deg wide: B2D small set 0/9 both arms, ol exit -0.002; gate stop [d(pending)]
- [alpamayo_turns](alpamayo_turns/README.md) (Alpamayo side cameras, multi-view turns): concluded; Cross cams off: dA_H +0.01 [-0.15,0.14] (nav); tele-slot drop -0.39 [d(pending)]
- [vlm_arb](vlm_arb/README.md) (vlm-arb, vlm_arb): live; Zero-shot Qwen3-VL-4B light reading: red-light infractions 13 to 6 (privileged 5), DS +5.0; fixed bypass pbyp2, stop-line R2 vred2, yellow rule vred3 [d84-87,89,91,95]

**openpilot adaptation**

- [op_adapt_r2](op_adapt_r2/README.md) (op-adapt r2, S_jev): superseded-by op_adapt_l; Stage 1 failed: drift 0.397 m (line 0.10) [d67]
- [op_adapt_r1](op_adapt_r1/README.md) (op-adapt r1, op_torch): Exact fp32 port; pedestrian AUC gain +0.076 nuScenes, +0.009 CARLA [d55]
- [op_closed_loop](op_closed_loop/README.md) (op-arb, op-drive): Native never starts (6/6); arbitration +9.7 DS is slowness [d57,74]
- [op_openloop](op_openloop/README.md) (op-interp, op-lb, navhard): NAVSIM score is input protocol: interpolation 52.1 to 84.2 [d34,36-37,39,66,73]
- [skill_pack](skill_pack/README.md) (N0-N4, navsim raise): Navtest PDMS 84.2 to 91.59 (N3), flat at N4 [d64,68-73,75,88,94,97]
- [log_expert_audit](log_expert_audit/README.md): Native stop capture 0.29 on WOD; motivated op_adapt_l [d77]
- [feature_adapter](feature_adapter/README.md) (E0, E1): CARLA P5 pedestrian AUC 0.51-0.53 vs 0.83 nuScenes [d62-63]

**closed-loop harness and controllers**

- [cl_infra](cl_infra/README.md) (cl-lib, infra): 220 routes take a measured 3.11 h; reduced profile [d16-17,83]
- [b2d_controller](b2d_controller/README.md): No controller qualified; PI DS 59.1 vs 53.8, lateral +10.8% [d26-27,29-30]
- [b2d_controller_eval](b2d_controller_eval/README.md) (Task 10): Ours win L1 (ramp 0.92 vs 2.38), not closed loop (DS 86 vs 95) [d41]
- [b2d_tcp](b2d_tcp/README.md): PI cuts jerk p95 84.5 to 45.1 m/s^3 on 3 routes [d28,30]
- [b2d_tfv6](b2d_tfv6/README.md) (TFv6 W2, D1-D3): representation +14.3 DS [+5.1, +25.9]; controller +1.0, not detected [d31]
- [tfv6_rules](tfv6_rules/README.md) (TFv6 rules): public B2D noise: single-eval DS SD 0.80; rules x interface not run [d38,31]
- [simlingo_catalogue](simlingo_catalogue/README.md): no result: 220-route x 2-arm batch never recorded [dnone]
- [carla_rewind](carla_rewind/README.md): rewind not equivalent (ego speed p95 0.305 vs 0.3); only 3.0x faster [d65]

**zero-shot exams and leaderboards**

- [zeroshot_openloop](zeroshot_openloop/README.md) (zero-shot): WOD RFS Cinque 8.005, Alpamayo 8.034, above cv 7.103 [d34,37,39]
- [zeroshot_b2d](zeroshot_b2d/README.md) (zero-shot B2D): n=5 smoke: Alpamayo DS 60.8, SR 2/5; openpilot DS 2.7 voided [d33]
- [model_smoke](model_smoke/README.md) (openpilot smoke, rigs): smoke: openpilot 1-3 ms/step; 2 deg yaw gives 4.6x lateral error [d33,36]
- [hugsim](hugsim/README.md) (HUGSIM, I3): fixed2 controller passes acceptance; 4 Hz clock +25-38% lateral [d19,44,90,96]
- [leaderboard_audit](leaderboard_audit/README.md) (hack audit): NAVSIM v2 +10.9 is the scorer; B2D DS SD 0.80 [d35,38]
- [top10](top10/README.md) (T1-T3): no top-10 family on all boards except SparseDrive [d46,58]
- [baselines_latency](baselines_latency/README.md): batch 1: Qwen-Drive-4B 702 ms, AutoVLA 1362 ms; our head <0.1 ms [d11,15,18]

**frozen features and the reaction line**

- [probe_planner_v0](probe_planner_v0/README.md) (probe v0, stage A): pre-onset vision delta null (CI [-0.062, +0.030]); K >= 1024 [d1-3,3b,3c,4-5,8-10,12-14]
- [prediag](prediag/README.md) (P0-P4, L0): pre-onset vision delta null (CI [-0.062, +0.030]) [d3d,20-24]
- [driving_backbones](driving_backbones/README.md): openpilot temporal -0.294 vs V-JEPA 2 -0.030, WOD pre-onset [d40]
- [reactivity](reactivity/README.md) (P5, M-C, I4): Dual-stream paired-diff head flips pedestrians 43.3%; ridge_late 0% [d32,42]
- [fusion_diag](fusion_diag/README.md) (fusion Q1-Q9): Qwen+openpilot complementary under paired-diff; SAM gate flips 29.2% [d43]
- [fastperc](fastperc/README.md): YOLO26x-seg 20 ms p95 for 3 cameras, 1/25 of SAM 3.1, equal recall [d45]
- [elicitation](elicitation/README.md) (E1-E6): Zero-shot elicitation harmful: WOD RFS -1.02, NAVSIM PDMS -8.2 [d42,44]
- [real_transfer](real_transfer/README.md) (G0-G3): No transfer: all 12 student zero-shot cells harmful (PDMS -1.1..-2.7) [d44]
- [statepol](statepol/README.md) (state-space): Only 2 BehaviorBench PPO ran: yield 43.0% vs null 2.0% [d51]

**night queues**

- [night_queue_2](night_queue_2/README.md) (nq2, N1-N6, P6): P6 v0 holds, placement null misses gate; V-JEPA 2 flips 47% [d47-50,52-53]
- [night_queue_3](night_queue_3/README.md) (nq3, Q1-Q6): No public lateral-readout examinee bypasses; zero speed -22.1 PDMS [d35,44,47-48,52-53]
- [night_queue_4](night_queue_4/README.md) (nq4, G K X OPL): No position memory in TFv6/BridgeDrive/BLUE/SimLingo (ghost 5-13%) [d35,44,58]

**real-appearance pairs and world models**

- [p3_ped_exam](p3_ped_exam/README.md) (P3, ped dose): 3DGS ped insertion/deletion stopped: donors slide, deletions smear [d44]
- [cosmos](cosmos/README.md): Cosmos v1 no-go (80% diff outside ped); G4 made 2004 pairs [d56,63]
- [controlnet_pair](controlnet_pair/README.md) (cn_pair): Paused, not no-go: deletion unclean, insertion fake-ish on 19 scenes [d59]
- [world_model](world_model/README.md) (W, WL, WL-2): W failed from action-scene confounding; WL-2 held-out no-go (H 0.38) [d54,60-61,65,76]
- [wm_policy](wm_policy/README.md) (worldmodel-4B as policy, WM-uncond): Stopped at step 0: anchors are fixed future video, no history-only mode [d(pending)]
