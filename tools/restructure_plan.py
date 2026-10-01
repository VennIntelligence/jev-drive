#!/usr/bin/env python3
"""Generate restructure/manifest.tsv from the topic rules below (docs/restructure-design.md).

The manifest is the reviewed artifact; this script only makes it reproducible. Edit the rules here,
regenerate, review the diff of the manifest. `tools/restructure.py` reads only the manifest.

    python tools/restructure_plan.py            # write restructure/manifest.tsv
    python tools/restructure_plan.py --graph    # also print cross-topic import edges
"""
from __future__ import annotations

import argparse
import ast
import collections
import fnmatch
import re
import subprocess
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
BASE_SHA = "bcbdde4"  # the tree this manifest was planned against

# ---------------------------------------------------------------------------------------------
# Topics. status: live (work continues; code in scripts/ + lib/) | concluded (code in archive/).
# code:   globs over scripts/** and jevdrive/** (first matching topic wins, in this order)
# results: research/results/<dir> names;  data: todos/<dir> data packages;  figs: research/figs globs
# plans:  todos/*.md plan files (deleted for concluded topics, moved to plan.md for live ones)
# tmp:    tracked tmp/ notes of this topic (deleted)
# ---------------------------------------------------------------------------------------------
T = collections.OrderedDict()


def topic(name, status, title, code=(), results=(), data=(), figs=(), plans=(), tmp=(), toplevel_results=(), decisions=""):
    T[name] = dict(status=status, title=title, code=list(code), results=list(results), data=list(data), figs=list(figs),
                   plans=list(plans), tmp=list(tmp), toplevel_results=list(toplevel_results), decisions=decisions)


topic("op_adapt_l", "live", "openpilot Cinque adapted by imitation of real-log human futures (op-adapt L), open loop and in B2D",
      code=["jevdrive/op_adapt_l.py", "jevdrive/op_adapt_l_arms.py", "jevdrive/op_l_b2d_report.py",
            "scripts/op_adapt_l_*", "scripts/op_l_b2d_*", "scripts/op_l_onnx.py", "scripts/make_op_adapt_l_figs.py"],
      results=["op-adapt-L"], figs=["op-adapt-L*"],
      plans=["2026-10-01-op-adapt-L-prereg.md", "2026-10-01-op-adapt-L-followup.md", "2026-10-01-op-adapt-L-rfs-diagnosis.md",
             "2026-10-01-op-adapt-L-b2d-prereg.md", "2026-10-02-op-adapt-L-gate-and-curve.md"],
      tmp=["2026-10-01-op-adapt-L-review-and-next-steps.md"], decisions="78-81")
topic("b2d_privileged", "live", "Privileged-rule ceilings on low-scoring Bench2Drive dev routes (red light, bypass, junction)",
      code=["scripts/b2d_privileged_*"], results=["b2d-privileged-ceiling"],
      plans=["2026-10-01-b2d-privileged-ceiling.md"],
      tmp=["2026-10-01-b2d-privileged-ceiling-prompt.md", "2026-10-01-b2d-red-light-prompt.md"], decisions="82")
topic("cl_infra", "concluded", "Closed-loop harness acceptance, CARLA scaling, thread census and the worker-profile experiment",
      code=["scripts/infra_*", "scripts/test_infra_*", "scripts/make_infra_scale_figs.py", "scripts/b2d_scale.py", "scripts/b2d_sweep.py",
            "scripts/b2d_optimization_report.py", "scripts/carla_knee.py", "scripts/carla_threads*", "scripts/carla_parallel.sh",
            "scripts/carla_bench.py", "scripts/lanes/*", "jevdrive/cl_profile_report.py", "scripts/b2d_wait_and_run.sh",
            "scripts/b2d_finish_watch.sh", "scripts/benchmark_b2d_preprocess.py", "scripts/pyspy_python.sh"],
      results=["infra", "infra-acceptance", "cl-lib", "b2d"], figs=["infra-scale*", "cl-workers*"],
      data=["2026-09-25-closed-loop-infra-acceptance"],
      plans=["2026-09-22-b2d-large-maps.md", "2026-09-25-closed-loop-infra-acceptance.md", "2026-10-01-cl-lib.md"],
      decisions="16-17, 83")
topic("b2d_controller", "concluded", "Fixed-trajectory controller for Bench2Drive: design, pose/lateral follow-ups, L1 diagnostics",
      code=["scripts/b2d_controller.py", "scripts/b2d_controller_[!e]*", "scripts/test_b2d_controller*", "scripts/testdata/*",
            "scripts/b2d_calibrate.py", "scripts/test_b2d_preview.py", "scripts/test_b2d_sensor_pipeline.py"],
      data=["2026-09-22-b2d-controller", "2026-09-23-lateral-followup", "2026-09-23-controller-next"],
      figs=["b2d-pose*"], plans=["2026-09-22-b2d-controller.md"], decisions="26-30, 41")
topic("b2d_controller_eval", "concluded", "Task 10: frozen L1/L2/L3 evaluation of controller candidates against expert references",
      code=["scripts/b2d_controller_eval_*"], decisions="41")
topic("b2d_tcp", "concluded", "Actual TCP inference with native lateral control and paired longitudinal controller tests",
      code=["scripts/b2d_tcp_campaign.py", "scripts/b2d_tcp_comparison_agent.py", "scripts/b2d_tcp_control.py", "scripts/b2d_tcp_eval_agent.py",
            "scripts/b2d_tcp_preprocess.py", "scripts/b2d_tcp_visual_agent.py", "scripts/test_b2d_tcp_*"],
      data=["2026-09-23-tcp-controller"], decisions="28")
topic("b2d_tfv6", "concluded", "TFv6 (LEAD) controller campaign W2/W2b/D1-D3, plant and steer diagnosis",
      code=["scripts/b2d_tfv6_*", "scripts/test_b2d_tfv6_*"], data=["2026-09-23-tfv6-controller"],
      toplevel_results=["results/diagnosis"], decisions="31")
topic("tfv6_rules", "concluded", "TFv6 rules x interface factorial and Bench2Drive by hazard family",
      code=["jevdrive/tfv6_rules.py", "scripts/tfv6_rules_*", "scripts/make_tfv6_rules_figs.py"],
      results=["tfv6-rules-interface"], data=["2026-09-25-tfv6-rules-interface"], decisions="31, 38")
topic("simlingo_catalogue", "concluded", "SimLingo catalogue experiment in Bench2Drive",
      code=["scripts/simlingo_*"], data=["2026-09-25-simlingo-catalogue"])
topic("zeroshot_b2d", "concluded", "Zero-shot Bench2Drive exam of Alpamayo 1.5 and openpilot (adapters, stall diagnosis)",
      code=["scripts/zeroshot_b2d_*", "scripts/b2d_zoo_*", "scripts/test_b2d_zoo_*", "scripts/test_zeroshot_openpilot_context.py",
            "scripts/zeroshot_b2d_alp_stall_probes/*", "scripts/wait_plan_line.sh"],
      results=["zeroshot-b2d"], figs=["alp-b2d*", "zeroshot-b2d*"], decisions="33")
topic("zeroshot_openloop", "concluded", "Zero-shot open-loop exams of Alpamayo 1.5 and openpilot on WOD-E2E, NAVSIM, nuScenes, PhysicalAI-AV",
      code=["jevdrive/wod_zeroshot.py", "jevdrive/navsim_zs.py", "jevdrive/navsim_rig.py", "jevdrive/nuscenes_zs.py", "jevdrive/navsim_agent.py",
            "scripts/wod_zeroshot*", "scripts/wod_test_*", "scripts/navsim_zs_*", "scripts/nusc_zs*", "scripts/pai_openpilot.py"],
      results=["wod-zeroshot", "navsim-zeroshot", "nuscenes-zeroshot", "pai-openpilot"],
      figs=["wod-zeroshot*", "navsim-zs*", "nusc-zeroshot*", "pai-openpilot*"],
      data=["2026-09-24-zeroshot-exam"], decisions="34, 37, 39")
topic("model_smoke", "concluded", "Alpamayo 1.5 and openpilot smoke runs, openpilot camera-rig robustness (migration study)",
      code=["scripts/alpamayo_*", "scripts/openpilot_*", "scripts/fetch_comma1m.py", "scripts/wod_openpilot_rigs.py"],
      results=["alpamayo-smoke", "openpilot-migration"],
      figs=["alpamayo-*", "openpilot-smoke*", "openpilot-migration*"],
      data=["2026-09-24-openpilot-smoke", "2026-09-24-alpamayo-smoke"], decisions="36")
topic("hugsim", "concluded", "HUGSIM: install, controller acceptance, zero-shot exam, I3 3DGS counterfactual pairs",
      code=["scripts/hugsim/*", "scripts/hugsim_*", "jevdrive/hugsim_*"], results=["hugsim", "hugsim-exam", "i3-hugsim-pairs"],
      figs=["hugsim-*", "i3-hugsim*"], data=["2026-09-25-hugsim-exam"], decisions="19")
topic("leaderboard_audit", "concluded", "Leaderboard hack audit, text analysis and B2D family breakdown (what the high scores are made of)",
      code=["scripts/make_leaderboard_ability_figs.py"], results=["hack-audit", "leaderboard-text-analysis", "b2d-family"],
      figs=["lb-ability*", "b2d-family*"],
      data=["2026-09-24-hack-audit", "2026-09-24-leaderboard-text-analysis"],
      plans=["2026-09-24-leaderboard-synthesis.md"], decisions="35, 38")
topic("probe_planner_v0", "concluded", "Probe v0, planner v0, Waymo stage A dry run and train features (the frozen-VLM start)",
      code=["jevdrive/probe.py", "jevdrive/probe_v0.py", "jevdrive/planner.py", "jevdrive/features.py", "jevdrive/labels.py", "jevdrive/planner_v0.py", "jevdrive/waymo_stage_a.py",
            "scripts/probe_v0.sh", "scripts/planner_v0.sh", "scripts/waymo_stage_a.sh", "scripts/waymo_features_watch.sh",
            "scripts/snapshot_processed.sh", "scripts/make_article_figs.py"],
      figs=["probe-layer*", "planner-*"],
      plans=["2026-09-19-probe-v0.md", "2026-09-20-planner-v0.md", "2026-09-20-waymo-stage-a-dryrun.md", "2026-09-21-waymo-train-features.md"],
      decisions="1-15")
topic("prediag", "concluded", "Pre-diagnostic round on Waymo E2E: L0, P0 train split, P1 judge, P2 readout ladder, P3 backbone ladder, P3e, P4 CARLA gap",
      code=["jevdrive/waymo_l0.py", "jevdrive/waymo_ladder.py", "jevdrive/waymo_p0.py", "jevdrive/waymo_p1.py", "jevdrive/waymo_heads.py",
            "jevdrive/waymo_qwenvid.py", "jevdrive/dit_features.py", "jevdrive/p4_carla.py",
            "scripts/waymo_p0.sh", "scripts/verify_top_decile.*", "scripts/p4_*", "scripts/make_top_decile_sheet.py",
            "scripts/make_article_critical_moment_figs.py"],
      results=["l0-surprise-weighting", "p0-train-split", "p1-judge", "p1-judge-ladder", "p2-readout-ladder", "p2p3-subset",
               "p3-backbone-ladder", "p3-qwenvid-trainfit", "p3-qwenvid2b", "p3e-heads", "p4-carla-gap", "qwenvid-train-profile",
               "top-decile-audit"],
      figs=["p0-*", "p2b-*", "p2c-*", "p3-vjepa*", "p3-all*", "p3-32b*", "l0-*", "p4-*", "mid-decile*", "top-decile*"],
      plans=["2026-09-22-p0-train-split-recheck.md", "2026-09-22-p1-judge.md", "2026-09-22-p2-readout-ladder.md",
             "2026-09-22-p3-backbone-ladder.md", "2026-09-23-p3e-multimodal-heads.md", "2026-09-23-p4-carla-feature-gap.md"],
      decisions="3d, 20-24")
topic("driving_backbones", "concluded", "openpilot / Alpamayo 1.5 as frozen backbones in the P3 ladder, replicated on nuScenes",
      code=["jevdrive/drive_backbones.py", "scripts/drive_backbones_*", "scripts/nusc_backbone_openpilot.py", "jevdrive/nusc_ladder.py", "scripts/make_driving_backbones_figs.py"],
      results=["driving-backbones"], figs=["driving-backbones*"], data=["2026-09-24-driving-backbones"], decisions="40")
topic("reactivity", "concluded", "P5 CARLA counterfactual pair exams (v0, v1), the reactivity program (M-C head), P5 VLM meta-action, I4",
      code=["jevdrive/p5_exam.py", "jevdrive/p5_pairs.py", "jevdrive/p5_qwen.py", "jevdrive/p5v1.py", "jevdrive/reactivity_*",
            "jevdrive/waymo_p5vlm.py", "jevdrive/op_route.py", "jevdrive/p5_openpilot.py", "scripts/p5_openpilot.py",
            "scripts/p5_pair_agent.py",
            "scripts/p5_gen*", "scripts/p5v1_*", "scripts/reactivity*", "scripts/i4_worldmodel_probe.py", "scripts/waymo_p5vlm.sh",
            "scripts/p5route.sh"],
      results=["p5-carla-pairs", "p5-v1", "p5-vlm-metaaction", "reactivity", "reactivity-i4"],
      figs=["p5-*", "reactivity-*", "i4-*", "i4_*"], data=["2026-09-25-reactivity-program"],
      plans=["2026-09-23-p5-vlm-metaaction-proto.md", "2026-09-24-p5-carla-pairs-v0.md", "2026-09-24-p5-v1-e-layer.md",
             "2026-09-24-r-layer-routine.md", "2026-09-24-zoo-in-carla.md", "2026-09-25-reactivity-program.md",
             "2026-09-25-openpilot-temporal-p5-and-route.md"],
      decisions="32, 42, 48")
topic("fusion_diag", "concluded", "Fusion diagnostics Q1-Q9: Qwen x openpilot complementarity, SAM 3.1 perception, reaction windows",
      code=["jevdrive/fusion_*", "jevdrive/sam_detect.py", "scripts/fd_*", "scripts/sam3_setup.sh"],
      results=["fusion-diagnostics"], figs=["fusion-*"], plans=["2026-09-25-fusion-diagnostics.md"], decisions="43")
topic("fastperc", "concluded", "Fast-channel perception: SAM 3.1 speed-ups vs YOLO-class detectors, latency x recall",
      code=["jevdrive/fastperc*", "scripts/fastperc*"], results=["fast-perception"], figs=["fastperc*"],
      plans=["2026-09-26-fast-perception.md"], decisions="45")
topic("elicitation", "concluded", "Elicitation program E1-E6 and I3: carrying the CARLA-elicited reaction to real data; seed checks",
      code=["jevdrive/elicit_*", "scripts/elicit_*", "scripts/seeds_overnight.sh"], results=["elicitation"], figs=["elicit-*"],
      plans=["2026-09-26-elicitation-program.md", "2026-09-26-overnight-queue.md"], decisions="44")
topic("real_transfer", "concluded", "Real-data transfer G0-G3: gates, HUGSIM vehicle pairs, edit-pair diagnostics",
      code=["jevdrive/real_g*", "scripts/real_g*"], results=["real-data-transfer"], figs=["real-g*"],
      plans=["2026-09-26-real-data-transfer.md"], decisions="44")
topic("night_queue_2", "concluded", "Night queue 2: N1 P6 v0 behaviour exam, N2 openpilot bypass/desire, N3 leaderboard head x reaction, N4-N6",
      code=["jevdrive/night2_*", "jevdrive/n5_depth.py", "jevdrive/n6_backbones.py", "jevdrive/p6.py", "scripts/night2_*", "scripts/n5_*",
            "scripts/n6_*", "scripts/p6_*"],
      results=["night2"], figs=["night2*"], plans=["2026-09-26-night-queue-2.md"], decisions="47-53")
topic("night_queue_3", "concluded", "Night queue 3: P6 generation (A), our heads in closed loop (B), examinees on P6 (C), Q4-Q6 (D), SCH scheduler",
      code=["jevdrive/nq3_*", "scripts/nq3_*", "scripts/nq3_d/*", "scripts/sch_cl_*", "scripts/test_sch_cl_*", "scripts/sch_gpu_helpers/*"],
      results=["nq3"], figs=["nq3-*"], plans=["2026-09-26-night-queue-3.md"], tmp=["2026-09-27-q4a-fp64.md"],
      decisions="35, 44, 46-48, 52-53 (Q readings)")
topic("p3_ped_exam", "concluded", "Real-appearance pedestrian exam (P3): OmniRe deletion/insertion pairs, filters, dose response, insertion tools",
      code=["scripts/p3/*", "jevdrive/nq4_p3*", "jevdrive/ped_dose.py", "scripts/nq4_p3.sh", "scripts/nq4_p1_count.py"],
      results=["p3"], figs=["p3/*"], plans=["2026-09-28-ped-dose-response.md"],
      tmp=["2026-09-27-p3-scenes.md", "2026-09-28-p3-filter.md", "2026-09-28-p3-handoff.md", "2026-09-28-p3-state.md"], decisions="44")
topic("world_model", "concluded", "Latent world model loop: W (nq4), WL fork data and critic, WL-2 representation and verdict, render diagnosis",
      code=["jevdrive/nq4_w*", "jevdrive/nq4_wdiag.py", "jevdrive/wl*", "scripts/nq4_w*", "scripts/wl*", "scripts/make_wl2_figs.py",
            "scripts/render_*", "tests/test_wl2.py"],
      results=["wl", "wl-dryrun", "wl2"], figs=["wl2-*"],
      plans=["2026-09-28-wm-loop.md", "2026-09-29-wl-dryrun.md", "2026-09-29-wl-spatial-tokens.md", "2026-09-29-wl2-prereg.md", "2026-09-30-wl2-feat.md"],
      tmp=["2026-09-27-wm-loop-diagnosis.md", "2026-09-28-wl-handoff.md", "2026-09-30-wl2-train-state.md", "2026-09-29-render-diag-state.md"],
      decisions="54, 60, 61, 76")
topic("night_queue_4", "concluded", "Night queue 4: G ghost/perturbation test, K recipe ladder, X mode head, OPL native plan; cx_* lane orchestration",
      code=["jevdrive/nq4_*", "scripts/nq4_*", "scripts/cx_*", "scripts/test_cx_*"],
      results=["nq4"], plans=["2026-09-26-night-queue-4.md"],
      tmp=["2026-09-26-astra-x-fix.md", "2026-09-28-g-lane-handoff.md", "2026-09-28-g-seed0-prelim.md", "2026-09-28-g-shift-swap-prelim.md",
           "2026-09-29-g-final.md"],
      decisions="58 (G), 35/44/52-53 (readings)")
topic("carla_rewind", "concluded", "In-place CARLA rewind vs from-scratch generation for fork data",
      code=["scripts/carla_rewind.py", "scripts/poc_carla_rewind.py", "scripts/rewind_*"], results=["carla-rewind"],
      plans=["2026-09-29-carla-rewind.md"], tmp=["2026-09-29-carla-rewind-state.md"], decisions="65")
topic("top10", "concluded", "Top-10 intersection across boards: T1-T3 executors run the leaderboard models on our exams",
      code=["jevdrive/top10_*", "scripts/top10_*", "scripts/top10_smoke/*", "scripts/top10_t2/*", "scripts/make_top10_*"],
      results=["top10-exams"], figs=["top10-*"], data=["2026-09-26-top10-intersection"],
      plans=["2026-09-26-top10-intersection.md"], decisions="46, 58")
topic("statepol", "concluded", "Public state-space RL policies (BehaviorBench) on a WOMD bypass / negotiation / recovery exam",
      code=["scripts/statepol_*"], plans=["2026-09-26-state-space-policies.md"], decisions="51")
topic("cosmos", "concluded", "Cosmos-Transfer2.5 re-rendering of CARLA pairs: pilot, v2 anchored repaint, G4 full run",
      code=["jevdrive/cosmos_*", "scripts/cosmos_*"], results=["cosmos"], figs=["cosmos*"],
      plans=["2026-09-28-cosmos-pilot.md"], tmp=["2026-09-29-cosmos-full-handoff.md"], decisions="56")
topic("feature_adapter", "concluded", "E0 layer probe and E1 CARLA-vs-Cosmos probe: where pedestrian information lives in openpilot",
      code=["scripts/op_layer_probe.py", "scripts/op_cosmos_probe.py"], results=["e0-layer", "e1-cosmos"],
      plans=["2026-09-29-e0-layer-probe.md", "2026-09-29-e1-cosmos-probe.md"], decisions="62, 63")
topic("controlnet_pair", "concluded", "Same-generator ControlNet pedestrian pairs on real WOD clips (paused)",
      code=["jevdrive/cn_pair.py", "scripts/cn_pair*"], results=["cn_pair"], figs=["cn_pair/*"],
      tmp=["2026-09-29-controlnet-pair-handoff.md"], decisions="59")
topic("op_adapt_r1", "concluded", "op-adapt round 1: PyTorch port of Cinque, stage-4 unfreeze + distillation, pedestrian aux heads",
      code=["jevdrive/op_adapt.py", "jevdrive/op_adapt_data.py", "jevdrive/op_torch.py", "scripts/op_adapt_bench.py", "scripts/op_adapt_cache.py", "scripts/op_adapt_equiv.py", "scripts/op_adapt_eval.py",
            "scripts/op_adapt_readout.py", "scripts/op_adapt_ref.py", "scripts/op_adapt_train.py", "scripts/op_adapt_next_cache.sh"],
      results=["op-adapt"], figs=["op-adapt-*"], plans=["2026-09-28-op-adapt.md"], decisions="55")
topic("op_adapt_r2", "superseded-by op_adapt_l", "op-adapt round 2: detection tokens, S_jev rule scorer, R0 tele view, staged lane",
      code=["jevdrive/op_adapt_r2*", "jevdrive/op_adapt_det.py", "jevdrive/op_adapt_score*", "scripts/op_adapt_r0.py", "scripts/op_adapt_r2_*",
            "scripts/op_adapt_det*", "scripts/op_adapt_score_*", "scripts/op_adapt_nav.py", "scripts/op_adapt_carla_*",
            "tests/test_op_adapt_score.py"],
      results=["op-adapt-r2"], plans=["2026-09-29-op-adapt-r2-prereg.md"], tmp=["2026-09-30-op-adapt-r2-state3.md"], decisions="67")
topic("op_closed_loop", "concluded", "openpilot in the B2D closed loop: arbitration study (op-arb) and op-drive with resume policy",
      code=["jevdrive/op_arb_*", "scripts/op_arb*", "scripts/op_drive_*", "scripts/check_op_calibration.py",
            "scripts/make_article_closed_loop_figs.py"],
      results=["op_arb", "op_drive"], figs=["op_arb*"], plans=["2026-09-28-op-closedloop.md", "2026-09-29-op-drive.md"],
      decisions="57, 74")
topic("op_openloop", "concluded", "openpilot on open-loop boards: 2 Hz contract and frame interpolation, NAVSIM navigation arms, standing, navhard deficit",
      code=["jevdrive/op_interp.py", "jevdrive/openloop_standing.py", "scripts/op_interp*", "scripts/op_lb*", "scripts/make_op_interp_figs.py",
            "scripts/make_openloop_standing_figs.py", "scripts/wod_openpilot_*", "scripts/navhard_deficit.py"],
      results=["op-interp", "op-lb", "openpilot-openloop", "navhard-deficit", "navsim-openblas-audit"],
      figs=["openloop-*", "op-interp*"],
      plans=["2026-09-25-openpilot-openloop-comparison.md", "2026-09-29-op-leaderboard.md"], decisions="37, 66, 73")
topic("skill_pack", "concluded", "NAVSIM skill pack N0-N4 and S: native plan + scorer head, navtrain metric cache",
      code=["jevdrive/skill_pack_*", "jevdrive/navsim_raise.py", "jevdrive/navsim_heads.py", "jevdrive/navsim_qwen.py",
            "scripts/skill_pack_*", "scripts/n1_*", "scripts/navsim_raise_*", "scripts/navtrain_mcache*"],
      results=["skill-pack"],
      plans=["2026-09-29-skill-pack-n0.md", "2026-09-29-n1-scorer.md", "2026-09-29-navtrain-mcache.md", "2026-09-30-navsim-raise.md"],
      decisions="64, 68-72, 75")
topic("log_expert_audit", "concluded", "How many independent expert start / stop / turn events real logs hold, and what the native plan misses",
      code=["scripts/log_expert_audit.py", "scripts/make_log_expert_audit_figs.py"], results=["log-expert-audit"], figs=["log-expert*"],
      plans=["2026-10-01-log-expert-audit.md"], decisions="77")
topic("baselines_latency", "concluded", "Latency of released baselines (AutoVLA, openjev, Qwen-Drive) and our backbones' load check",
      code=["scripts/bench_baselines/*"], decisions="11, 15, 18")

# Shared code: stays where it is. Decided by the import graph (imported by >= 2 topics, or infra by nature).
SHARED = [
    # jevdrive library
    "jevdrive/__init__.py", "jevdrive/common.py", "jevdrive/runlog.py", "jevdrive/plots.py", "jevdrive/camgeom.py",
    "jevdrive/hfdl.py", "jevdrive/traj.py", "jevdrive/nuscenes_index.py", "jevdrive/waymo.py",
    "jevdrive/cl/*", "jevdrive/openpilot/*", "jevdrive/alpamayo/*", "tests/test_cl.py",
    # lib-core round (landed on main after bcbdde4): shared by construction
    "jevdrive/run/*", "jevdrive/data/*", "jevdrive/cache.py", "jevdrive/par.py", "jevdrive/stats.py",
    "scripts/lib_demo.py", "tests/test_lib.py",
    # box ops and data
    "scripts/tmux_run.sh", "scripts/tensorboard.sh", "scripts/slot_run.sh", "scripts/sch_table.py", "scripts/boxwatch.sh",
    "scripts/download_*", "scripts/extract_nuscenes.sh", "scripts/setup_navsim_devkit.sh", "scripts/fetch_openpilot_models.py",
    "scripts/waymo_e2e.py", "scripts/waymo_prepare.sh", "scripts/carla_server.sh",
    # closed-loop harness (CARLA / Bench2Drive)
    "scripts/b2d_run.py", "scripts/b2d_route.py", "scripts/b2d_hooks.py", "scripts/b2d_report.py", "scripts/b2d_agent.py",
    "scripts/b2d_mapreuse.py", "scripts/b2d_expert_agent.py", "scripts/b2d_partner.py", "scripts/b2d_tcp_server.py",
    "scripts/b2d_policy_server.py", "scripts/b2d_viewer.py", "scripts/drive_runtime/*", "scripts/test_b2d_reaper.py",
    "scripts/test_b2d_runtime.py",
    "scripts/zeroshot_policy_server.py", "scripts/zeroshot_wire.py", "scripts/zeroshot_rigs.py", "scripts/b2d_zeroshot_agent.py",
    "research/plot_style.py", "research/figs/survey/*",
]

SHARED_MOVES: dict[str, str] = {}  # shared files that change place (none: shared code keeps its path)
EDGES: collections.Counter = collections.Counter()
UNSAFE_PROC = {"scripts/op_l_b2d_daemon.py": "pgrep -f use (unsafe-proc); not fixed by the restructure"}
VENDORED = {"scripts/b2d_zoo_pid.py", "scripts/b2d_zoo_planner.py"}


def git_files(*args):
    return subprocess.check_output(["git", "ls-files", *args], cwd=REPO, text=True).split("\n")[:-1]


def match(path, globs):
    return any(fnmatch.fnmatch(path, g) for g in globs)


def code_dest(topic_name, live, path):
    """New path of a code file: lib/ (python library, or a module other topics import), scripts/ (entry points of a
    live topic) or archive/ (one-off code of a concluded topic). A scripts/ subdir named after the topic is flattened."""
    p = Path(path)
    if p.parts[0] == "jevdrive":
        sub, rel = ("lib" if live else "archive"), p.name
    elif p.parts[0] == "tests":
        sub, rel = ("scripts" if live else "archive"), p.name
    else:
        sub = "scripts" if live else "archive"
        inner = p.parts[1:]
        if len(inner) > 1 and inner[0] in FLATTEN_SUBDIRS:
            inner = inner[1:]
        rel = "/".join(inner)
    return f"experiments/{topic_name}/{sub}/{rel}"


FLATTEN_SUBDIRS = {"p3", "hugsim", "bench_baselines", "lanes"}


def is_code(f):
    return f.startswith(("scripts/", "jevdrive/", "tests/")) or f in {"research/plot_style.py"}


def owners(files):
    """Topic owner of every code file: '-' for SHARED, else the first topic whose code globs match."""
    own = {}
    for f in files:
        if not is_code(f):
            continue
        if match(f, SHARED):
            own[f] = "-"
            continue
        for name, t in T.items():
            if match(f, t["code"]):
                own[f] = name
                break
    return own


def py_imports(own):
    """(importer, imported file) pairs between repo Python files: `jevdrive.x` modules and bare script-module names."""
    mods = {f[:-3].replace("/", ".").removesuffix(".__init__"): f for f in own if f.startswith("jevdrive/") and f.endswith(".py")}
    stems = collections.defaultdict(list)
    for f in own:
        if f.startswith("scripts/") and f.endswith(".py"):
            stems[Path(f).stem].append(f)
    out = collections.Counter()
    for f in own:
        if not f.endswith(".py"):
            continue
        try:
            tree = ast.parse((REPO / f).read_text())
        except Exception:
            continue
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level > 0 and f.startswith("jevdrive/"):
                pkg = f[:-3].replace("/", ".").split(".")[:-n.level]
                base = ".".join(pkg + ([n.module] if n.module else []))
                names = [base] + [f"{base}.{a.name}" for a in n.names]
            for m in names:
                tgt = mods.get(m)
                if tgt is None and m in stems:
                    c = stems[m]
                    same = [x for x in c if Path(x).parent == Path(f).parent]
                    tgt = (same or c)[0] if (same or len(c) == 1) else None
                if tgt and tgt != f:
                    out[(f, tgt)] += 1
    return out


def promote(own, edges):
    """Shared-vs-topic by the import graph, to a fixed point:
    - a module imported by shared code becomes shared (shared code never depends on a topic);
    - a module imported by three or more other topics becomes shared (two: exported, see below);
    - a module imported by one or two other topics is exported: it stays with its owner but in lib/, not archive/."""
    reason = {}
    while True:
        users = collections.defaultdict(set)
        for (f, tgt) in edges:
            if f in own and tgt in own and own[f] != own[tgt]:
                users[tgt].add(own[f])
        changed = False
        for tgt, us in users.items():
            if own[tgt] == "-":
                continue
            if "-" in us or len(us) >= 3:
                reason[tgt] = "shared by import graph: imported by " + ", ".join(sorted("shared code" if u == "-" else u for u in us))
                own[tgt] = "-"
                changed = True
        if not changed:
            exported = {tgt: us for tgt, us in users.items() if own[tgt] != "-"}
            return reason, exported


def build():
    files = git_files()
    tracked = set(files)
    rows, seen = [], set()

    def add(old, new, topic_name, kind, status, note=""):
        rows.append((old, new, topic_name, kind, status, note))
        if old.endswith("/"):
            seen.update(f for f in files if f.startswith(old))
        else:
            seen.add(old)

    own = owners(files)
    edges = py_imports(own)
    reason, exported = promote(own, edges)
    for f, t in own.items():
        note = UNSAFE_PROC.get(f, "")
        if f in VENDORED:
            note = "vendored third-party (moved as shipped, content not rewritten); " + note
        if t == "-":
            dest = SHARED_MOVES.get(f, f)
            add(f, dest, "-", "shared", "live", "; ".join(x for x in (reason.get(f, "shared infrastructure"), note) if x))
            continue
        live = T[t]["status"] == "live"
        sub = Path(f).parts[1] if f.startswith("scripts/") and len(Path(f).parts) > 2 and Path(f).parts[1] not in FLATTEN_SUBDIRS else None
        if sub and any(g in exported and Path(g).parts[1] == sub for g in own if g.startswith(f"scripts/{sub}/") and own[g] == t):
            note = "; ".join(x for x in (f"subdir scripts/{sub}/ kept together; a member is imported by another topic", note) if x)
            add(f, f"experiments/{t}/lib/{sub}/{Path(f).name}", t, "topic", "live", note)
            continue
        if f in exported:
            note = "; ".join(x for x in ("exported: imported by " + ", ".join(sorted(exported[f])), note) if x)
            dest = code_dest(t, True, f)
            if not f.startswith("jevdrive/") and not dest.split("/")[2] == "lib":
                dest = dest.replace(f"experiments/{t}/scripts/", f"experiments/{t}/lib/", 1)
            add(f, dest, t, "topic", "live", note)
        else:
            add(f, code_dest(t, live, f), t, "topic" if live else "archive", "live" if live else "one-off", note)
    EDGES.update(edges)
    # name collisions inside a topic (a jevdrive module and a script with the same name): the module goes to lib/
    dest = collections.Counter(r[1] for r in rows)
    for i, r in enumerate(rows):
        if dest[r[1]] > 1 and r[0].startswith("jevdrive/") and "/archive/" in r[1]:
            rows[i] = (r[0], r[1].replace("/archive/", "/lib/"), r[2], "topic", r[4], (r[5] + "; " if r[5] else "") + "lib/: a script of the same name sits in archive/")
    # results, figures, data packages, plans, tmp
    for name, t in T.items():
        for r in t["results"]:
            src = f"research/results/{r}/"
            dst = f"experiments/{name}/results/" if len(t["results"]) == 1 and not t["data"] else f"experiments/{name}/results/{r}/"
            add(src, dst, name, "data", "live" if t["status"] == "live" else "one-off", "small result files")
        for r in t["toplevel_results"]:
            add(r + "/", f"experiments/{name}/results/{Path(r).name}/", name, "data", "one-off", "top-level results/ dir")
        for d in t["data"]:
            sub = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", d)
            if sub in t["results"]:
                sub += "-plan"
            src = f"todos/{d}/"
            if all(f.endswith(".md") for f in files if f.startswith(src)):
                add(src, "", name, "delete", "superseded", "plan notes only (fold essentials into the README; git keeps them)")
            else:
                add(src, f"experiments/{name}/results/{sub}/", name, "data", "one-off", "data package from todos/ (reports kept with their data)")
        for g in t["figs"]:
            for f in files:
                if f.startswith("research/figs/") and fnmatch.fnmatch(f[len("research/figs/"):], g) and f not in seen:
                    rel = f[len("research/figs/"):]
                    if g.endswith("/*"):
                        rel = rel[len(g) - 1:]
                    elif g.endswith("*") and rel.startswith(g.rstrip("*") + "/"):
                        rel = rel[len(g.rstrip("*")) + 1:]
                    add(f, f"experiments/{name}/figs/{rel}", name, "data", "one-off", "figure")
        for p in t["plans"]:
            f = f"todos/{p}"
            if f not in tracked:
                continue
            if t["status"] == "live":
                add(f, f"experiments/{name}/plans/{p}", name, "doc", "live", "live plan (Chinese working note); fold into the README and delete when the lane closes")
            else:
                add(f, "", name, "delete", "superseded", "plan: essentials folded into the README; text stays in git history")
        for p in t["tmp"]:
            f = f"tmp/{p}"
            if f in tracked and f not in seen:
                add(f, "", name, "delete", "superseded", "tmp note: process log, git history keeps it")
    # every other tracked todos/ and tmp/ file is deleted
    for f in files:
        if f in seen:
            continue
        if f.startswith("todos/") or f.startswith("tmp/"):
            add(f, "", "-", "delete", "superseded", "todos/ and tmp/ are removed; git history keeps the text")
    return rows, files, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", action="store_true", help="print cross-topic import edges")
    ap.add_argument("--out", default="restructure/manifest.tsv")
    a = ap.parse_args()
    rows, files, seen = build()
    rest = [f for f in files if f not in seen and not match(f, SHARED)
            and (is_code(f) or f.startswith(("research/results/", "research/figs/", "results/")))]
    if a.graph:
        own = {r[0]: r[2] for r in rows}
        for (f, tgt), n in sorted(EDGES.items()):
            if own.get(f) != own.get(tgt):
                print(f"{own.get(f)}:{f} -> {own.get(tgt)}:{tgt} x{n}")
    print(f"{len(rows)} rows; unclassified code/results/figs: {len(rest)}")
    for f in rest:
        print("  UNCLASSIFIED", f)
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("old_path\tnew_path\ttopic\tkind\tstatus\tnote\n")
        for r in sorted(rows, key=lambda r: (r[2], r[3], r[0])):
            fh.write("\t".join(r) + "\n")


if __name__ == "__main__":
    main()
