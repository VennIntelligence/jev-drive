# WA-JEPA 公开 checkpoint 在我们 64 个 HUGSIM 场景上的参考分（预登记）

2026-10-05 写，在任何计分运行之前。目的：给 64 场景集做校准（对他们 436 场景 Table 2：WA-JEPA 0.4462、LTF 0.2310；我们集上 LTF 0.279、cv 0.292），
并回答「无状态 1.5 s 窗口 + GT ego 历史 + 路线命令的模型，在 PR #57 控制器下有没有起步打转 / 起步停滞」。对比对象：`cinque-fixed`（HD 0.278，已有 scored_op.csv，不重跑）。

## 协议（照他们发布的方式跑，偏离逐条记在 results/wajepa_ref.md）

- 模型：AFARI-Research/WA-JEPA HEAD bec2966 的 `close_loop/hugsim_planner.py` 原样；权重 `model_state_dict.pt`；配置 `configs/wa_jepa_hugsim.yaml` 不改（flow_inference_seed 1，num_inference_steps 4，bf16 autocast）。
- 相机：FRONT_LEFT / FRONT / FRONT_RIGHT / BACK，各 resize 到 512x256；KITTI-360 / Waymo 的 BACK 是黑的，如实记录，不补。
- 控制器：我们的 `HUGSIM-zs/fixed`（PR #57），不用他们的 monkey-patch；先离线验证两者逐元素等价。`run_upstream_compat` 的崩溃补丁不用，除非某场景真的崩，崩了就记哪一个。
- 同一 64 场景（`hugsim-exam-plan/scored.txt`）、同一打分器、同样 400 步上限，每场景一次，不重试挑分。
- 经 GPU pool 提交；分阶段 1 -> 约 10 -> 全部。

## 定义（先定死）

- spin：对最近记录路线点的航向误差在某步达到 60 度（`scripts/spin_analysis.py`，与 controller_spin.md 同一定义）。
- 起步停滞（launch stall）：前 40 步（10 s）内速度从未超过 1.6 m/s（op_control_stack_long.md 的 "never launch" 口径），或 end = max_steps。
- 失败类别（逐场景成对表）：spin / stuck（max_steps）/ bg 碰撞 / fg 碰撞 / off_route；spin 优先于其余类别（先看航向误差，再看 end）。「openpilot 失败而 WA-JEPA 成功」= Cinque HD < 0.5 且 WA-JEPA HD >= 0.5 且 WA-JEPA 的 end 是 complete；另给配对差 HD(WA) - HD(Cinque) 的 bootstrap CI（`jevdrive.stats.paired`，按场景为单位）。
- 难度（easy/medium/hard/extreme）与数据集分层只作描述，每格 16 个，不做显著性声明。

## 预测

1. 64 场景 HD-Score 落在 0.40–0.60（他们 436 场景 0.4462；我们的 LTF / cv 都比他们的 LTF 高，所以不排除偏高）。
2. spin（>= 60 度）<= 1 / 64。
3. 10 个 PR #57 Cinque 打转场景（`scripts/derot_spin10.txt`）里 WA-JEPA 打转 0 个。
4. 配对差 WA - Cinque 集中在中等难度（medium）、有交互（actor）的场景；easy 差距小，extreme 两者都低。

## 若预测失败怎么读

- 1 偏低：先查 fallback 步数（planner_stats.json n_failures）、命令映射、BACK 相机；不是模型弱之前先排除适配问题。
- 2/3 失败：说明无状态窗口也会起步打转，假设「循环 temporal 状态是起步打转的来源」不成立，回到场景内容（launch_lean.md）。

## 补充：早读门（2026-10-05 19:20 加，用户指示 GPU 0 专用 + 先小子集早读）

**时间线如实记录：这条在子集结果出来之后才写。** 之前的流水线是 smoke -> 10 个 spinner -> 全部 64 自动串联；门写入前，10 个 spinner 的 HD 与 end、以及全部 64 中先跑完的 25 个场景的一行日志（HD / end）我已经看过，spin 计数（航向误差）尚未算。
所以门的子集成员按只依赖 Cinque 数据的规则事后固定，门的判据保持用户给的原样；「看过 HD」这一点在报告里作为 caveat 写明。已 cancel 在跑的全量 job，剩余场景在 GPU 0 上续跑（pool 的 `--gpus 0`）。

- 子集（16）：10 个 spinner；cinque-fixed 失败（max_steps / 碰撞）的非 spinner 中按 scored_op.csv 顺序的 nuscenes 与 waymo 前 4 个（0041-medium-00、0411-medium-00、0254-hard-00、113792265837-easy-00）；cinque-fixed 完成的 easy 中按文件顺序的前 2 个（0051-easy-00、0166-easy-00）。
- 早停规则：WA-JEPA 在 10 个 spinner 中打转 >= 3 个，或子集平均 HD 不高于 cinque-fixed 同一子集 -> 子集后停下并报告；否则不等回复，直接续跑全 64。
- 早读消息给 main：逐场景 HD 与失败类（WA vs cinque-fixed）、spin 数、设置是否合理（车在动、相机对、BACK 的表现）。
