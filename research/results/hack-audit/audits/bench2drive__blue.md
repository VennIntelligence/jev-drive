# Bench2Drive · BLUE

- 固定源码：[George-Ling3/BLUE `6970cb69e05ef904b37f4264207ea0e1dab35ef2`](https://github.com/George-Ling3/BLUE/tree/6970cb69e05ef904b37f4264207ea0e1dab35ef2)；本地 `repos/bench2drive__blue/`。
- 论文：[BLUE](../../papers/blue.pdf)；README 报 3 seed 平均 90.58±0.12 DS。

## 已读与未读

已读 `README.md`、`docs/REPRODUCTION.md`、`gate/training/README.md`、`configs/blue_eval.yaml`、`gate/evaluation/eval_blue_full.sh`、`team_code/agent_simlingo.py` 的 setup、run_step 与控制段、`team_code/config_simlingo.py`、`simlingo_training/models/gate.py`，以及论文 §2–5、Appendix C/D 的数据划分、消融和评测段落。

未逐行读 SimLingo backbone、附带 Bench2Drive evaluator、全部评测 JSON 或未发布的 gate 训练实现；未运行任何脚本。公开训练入口明确留待 Stage 2，因此按“训练/评测/推理皆公开”给 A 档并不成立；当前能确认的是 B 档的推理/评测能力。

## 整体印象

BLUE 用独立训练路线上的闭环成功率差为 gate 构造标签，论文 Appendix C.1 明确说明与 Bench2Drive 220 条评测路线无重叠，所以不把它误记为测试路线泄漏。gate 在其他 CARLA benchmark 也评测，不能仅凭模拟标签认定其提升完全不迁移。明确可见的非模型控制增益是沿用 agent 中的卡住蠕行规则。

## 发现

- `B2D-BLUE-001` `manual_control_override`：代码在静止累计超过 800 帧后至少给 0.4 油门并取消刹车；[源码 L863–876](https://github.com/George-Ling3/BLUE/blob/6970cb69e05ef904b37f4264207ea0e1dab35ef2/team_code/agent_simlingo.py#L863-L876)。论文 §3–5 与 Appendix C/D、README 未披露该规则；影响数字 `none`。部署部分成立，固定 CARLA 帧数和无安全框的强制前进不可直接照搬；置信度中。

论文原文定位：Appendix C.1 “evaluation routes are entirely disjoint from the training routes”；Table 1 报 BLUE 比 SimLingo backbone +5.51 DS，但这是完整 gate 的变化，不能当作 creeping 效果。

## 局限与候选实测

静态检查不能知道蠕行触发多少次、是否影响 90.58 DS。候选实测：固定 gate、backbone 与 seed，仅关闭蠕行，比较 blocked、碰撞和 DS；此处未运行。公开 gate 训练入口缺失也限制对训练标签生成过程的代码复核。
