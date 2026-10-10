# Experiments index

One line per topic: name (aliases): status; key finding [d decision entries]. Open `<topic>/README.md`.

**live**

- [b2d_privileged](b2d_privileged/README.md): live; Red light + green release DS 75.0 to 95.0 [d82]
- [body1](body1/README.md) (self-model, body lesson, swept-footprint contact): live; Contact head passes offline (AUC 0.905 / 0.885); stop and lateral re-plan halted at one-chunk checks; loss arm 4.3 missed its first pilot gate, second attempt (Amendment 5, w = 3) met its pilot gate and G3 at full scale, closed loop L1 met but L2 and L3 not met (not promoted); third variant (Amendment 6, shape-only hinge gradient) met its pilot gate and arc lines, missed G3 (b) on one seed by one navtest token, no registered closed loop; descriptive 4-seed closed loop after the stop: mean +0.0040 [-0.0038, +0.0134] over the base, at-fault collision + offroad zeros 53 against 70 and corridor zeros 37 against 28 (each beyond the base's seed spread), slow 401 against 434, not promoted [d223, d224, d225, d227, d228, d229, d230, d231, d232, d233]
- [op_adapt_h](op_adapt_h/README.md): live; Fake-yaw following -71..-78%, navtest +0.82; HUGSIM spins 8 to 6 only [d98,(inputs:,92,94,96)]
- [op_adapt_l](op_adapt_l/README.md) (op-adapt L): live; Stop capture 0.252 to 0.559 open loop; B2D no gain [d77-81]
- [op_common_cause](op_common_cause/README.md): live; Plan follows fake history yaw, 3-7x more at low speed, real and CARLA alike [d92-93]
- [op_img_cmd](op_img_cmd/README.md): live; Image route: zero-shot uptake <= 0.14; fine-tuned 0.54-0.64, drift 0.29 m; sky arrow (no map) 0.29-0.39, junction drift 0.13-0.27 m; negatives: CARLA 0.30, wrong-exit control fails [d(pending)]
- [op_route_cmd](op_route_cmd/README.md) (route polyline, nav route): live; Route polyline material: navtrain 103 k / WOD 522 k labelled, 32 k / 62 k turns >= 25 deg; noise + negatives + CARLA plan, no training [d(inputs:,93,102,118)]
- [op_route_ft](op_route_ft/README.md) (route adapter fine-tune): live; Route-choice adapter (bear / poly) + action pathway fine-tune; B2D 25 junction turns primary [d(inputs:,118,121,122,127)]
- [op_dagger](op_dagger/README.md): live; Reprojection rollout engine on real WOD logs + DAgger pilot (paper §3 fig 8) [d(inputs:,92,98,100,111,118,123,124)]
- [factor_wm](factor_wm/README.md) (factorized world model, exact-ego sim): live; G1 and P2+DAgger fail: in-engine fixes, HUGSIM launch stalls (18->48; 1->9/12) [d(inputs:,54,76,103,123,132,138,139)]
- [op_fov](op_fov/README.md) (wide FOV, fisheye): concluded; Wide FOV 90/116 deg: turn gain -0.005..-0.033, early stop; wide frame barely used [d(pending)]
- [op_resume](op_resume/README.md) (driver resume, standstill launch): live; Driver-resume rule: 5/6 stuck HUGSIM runs launch (HD +0.12), missed-lead collision: gate stop [d(inputs:,90,96,107,113,117,118,119,124,126)]
- [op_parity](op_parity/README.md) (input parity, WA-JEPA inputs): live; Cinque + ego / pose / cmd / side cams vs WA-JEPA, equal inputs and data; geo-oracle (true SDF + agents as memory tokens): negative; geo-e2e (tokenizer trained with the adapter): read, worth +0.13 [-0.04, +0.29], ceiling < +0.3; NC failures (nc-taxonomy): 53% stopped / slow vehicle ahead, speed-scaling oracle recovers 74%; where to slow (nc-slow, scale-selection head on non-privileged inputs): fails the EP line, gain is a DAC horizon effect, AUC for recoverable collisions 0.66 (ego + plan) / 0.71 (openpilot lead outputs) / 0.76 (GT lead gap); off-track rows (ot-rows, 10 % re-projected rows): navhard +1.51 [+0.14, +3.10], below the +2.0 line, navtest +0.22; innovation gain (does SH30 lack scene-caused change): no, gain 0.94 / 0.87 vs WA-JEPA 0.97 / 0.94, gap stays a turn-precision gap; path / timing swap (pt-swap, is turn DAC a path-timing coupling): no, log timing on the plan's path removes 12.6% of > 20 deg DAC failures, the log path with the plan's timing 94.6%, the lever is the path shape; path-req (degraded logged-path fields as memory, oracle probes): oracle +1.38 read on 3 / 3 seeds, tolerant to 2 m cross-track / 1.5 m along-track error, a thin-head path on frozen features adds +0.09 (d204) [d(inputs:,92,104,111,118,137,138,139,196,197,198,206,207),217]; DIAG1 longitudinal audit: the adapter's speed profile is an ego-only prior (WOD loss all speed, navtest gain 1/4 speed), FIX1 fixes on WOD val +0.136 (decision 218) TR1 (decision 221): training the adapter's ego-only speed prior out, 5 arms, none keeps navtest and restores brake onset ([results/tr1_speed_prior.md](op_parity/results/tr1_speed_prior.md)).
- [op_probe](op_probe/README.md) (DAC localization, stage probes): live; navtest DAC gap starts in Cinque vision (thin heads +1.7 pp vs WA encoder); hinge head -0.9 pp; joint P2 / WA-JEPA page: gap is turns, each model's failures fixed at its own encoder [d(inputs:,104,112,144,145,146)]
- [b2d_collect](b2d_collect/README.md) (B2D collector, PDM-Lite data): live; 998 B2D clips, 902 k ticks at 20 Hz, 60% junction turns, all gates pass [d(inputs:,121,127,128,133,134,137,144,147,148)]
- [op_wide_ft](op_wide_ft/README.md) (wide FOV fine-tune, W116): concluded; Fine-tune with 116 deg wide: B2D small set 0/9 both arms, ol exit -0.002; gate stop [d(pending)]
- [alpamayo_turns](alpamayo_turns/README.md) (Alpamayo side cameras, multi-view turns): concluded; Cross cams off: dA_H +0.01 [-0.15,0.14] (nav); tele-slot drop -0.39 [d(pending)]
- [vlm_arb](vlm_arb/README.md) (vlm-arb, vlm_arb): live; Zero-shot Qwen3-VL-4B light reading: red-light infractions 13 to 6 (privileged 5), DS +5.0; fixed bypass pbyp2, stop-line R2 vred2, yellow rule vred3 [d84-87,89,91,95]
- [alpasim](alpasim/README.md) (AlpaSim challenge, nuPlan track): live; 48 public scenes: WA-JEPA 0.978, SH30 0.947, AlpaSim-aligned AP2 0.932, LTF 0.874; 700 landed scenes: SH30 0.914, AP2 0.922, OT30 0.926 / 0.934 (not a candidate by the line, gain is collisions), WA-JEPA reference 0.911, oracle SH30+AP2 0.950; zeros are a sideways heading drift on straight roads, not lead collisions: triggered by the strong hinge, lambda-10 P2H10-F-s0 0.948 on 700 scenes (held-out 300: +0.037 [+0.009, +0.067] over SH30, registered lines met); arbitration dropped; plan-averaging ensemble of two OT30 seeds 0.931, not adopted; AP2 + 10 % off-track rows 0.941 (+0.018 over AP2, candidate by the line, below lambda-10 P2H10), 25 % worse; shipped plan has no lateral recovery [d209-210, d184-185, 188-189, 199, 201-202, 205, 211]; OT3 (lambda-10 base, 700 scenes, 2 seeds): P2H10 0.9483, yaw-rate rows YR10m10 0.9545 (+0.0062 [-0.0022, +0.0157], line not met; later 400 scenes +0.0153 [+0.0043, +0.0267]), AP2 standard 0.9506, off-track +-1.5 m 0.8421 CF1 (decision 216): yaw-rate rows not confirmed on part012-015 (YR10m10 - P2H10 -0.0069), APY10m10 marginal candidate (+0.0146 over YR10m10 on 791 fresh scenes).; COL1 collisions (d219): nuPlan collisions are lateral drift (20 of 22 clear on the logged path), PAI 40 scenes 0.169 with 12 longitudinal collisions the lead head saw 6-8 s ahead, PAI trajectory speed step fixed in a diagnostic arm 0.169 -> 0.292 SWV1 (swept footprint, swerve clearance, decision 220): every at-fault collision is already in the plan's own swept ego box (66 / 66), lateral-acceleration cap avoids 0, privileged footprint stop 20 / 22 and 8 / 12: [results/swerve_clearance.md](alpasim/results/swerve_clearance.md). FIX1 serving switches (decision 226): PAI 0.215 -> 0.363 with speed-continuous serving + openpilot lead limit (60 scenes x 2 seeds), every arm negative on nuPlan 700 ([results/fix1_serving.md](alpasim/results/fix1_serving.md)).

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
