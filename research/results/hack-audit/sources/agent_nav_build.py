"""Build agent_nav stage-1 findings from checked, fixed-commit source ranges."""
import csv
import json
from pathlib import Path

ROOT = Path('/data/hack_audit')
OUT = ROOT / 'out'
REPOS = {
    (row['board'], row['slug']): row
    for row in csv.DictReader((OUT / 'repos.csv').open())
}


def finding(fid, board, slug, category, title, path, start, end,
            description, deploy_valid, disclosed, paper_quote,
            impact_evidence, confidence):
    repo = REPOS[(board, slug)]
    source = ROOT / repo['repo_path'] / path
    lines = source.read_text().splitlines()
    assert 1 <= start <= end <= len(lines) and end - start < 15, (fid, path, start, end)
    snippet = '\n'.join(lines[start - 1:end])
    assert snippet.strip()
    return dict(
        id=fid, board=board, repo=repo['repo_url'], commit=repo['commit'],
        category=category, title=title,
        location=f"{repo['repo_url']}/blob/{repo['commit']}/{path}#L{start}-L{end}",
        snippet=snippet, description=description, deploy_valid=deploy_valid,
        disclosed=disclosed, paper_quote=paper_quote,
        impact_evidence=impact_evidence, confidence=confidence,
    )


F = [
    finding(
        'NAV1-TOAD-001', 'navsim_v1', 'toad', 'metric_proxy_optimization',
        '测试时搜索优化 NAVSIM 学得的评分代理',
        'navsim/agents/drivoR/drivor_model.py', 357, 368,
        'CEM 对运动学展开后的轨迹调用学得的 PDM 子评分，并以该分数减舒适和锚定惩罚挑选采样控制。论文将同一 TOAD 机制用于 navtest PDMS，增益属于测试时搜索及评分代理，而非基础模型参数中的驾驶能力。固定 main 提交的 README 要求另切 nav1 分支，故此代码不能单独证明 v1 具体命令路径。',
        'partial：测试时优化可用于实车，但分数代理按 NAVSIM 的非反应式 PDMS 标签训练，换城和真实交互下的等额增益未知。',
        'yes',
        'TOAD 论文 §3、§4.2、Table 1："TOAD: CEM with trust region prior"；Table 1 的 DrivoR 行为 94.6 PDMS，加 TOAD 为 94.7。README §Evaluation 要求 NAVSIM-v1 另行 checkout nav1。',
        'Table 1，navtest PDMS 94.6→94.7，+0.1；仅对应论文版本，README 当前写 94.9。',
        'low',
    ),
    finding(
        'NAV1-DVM0-001', 'navsim_v1', 'drivevla_m0', 'metric_proxy_optimization',
        '64 候选由 PDMS 权重代理挑选',
        'navsim/agents/EpisodeDrive/action_decoder.py', 198, 211,
        '推理时把 NC、DAC、TTC、进度、舒适度等预测子分数按 NAVSIM PDMS 的形式聚合，取 argmax 候选。选轨得分中有针对榜单评分公式的部分，不能等同于真实驾驶质量；公开脚本仅为 Base 模型，论文 94.1 属 Scale，不能把此段直接量化为 94.1 的贡献。',
        'partial：候选轨迹打分仍可用于部署，但原权重与非反应式 PDMS 公式在真实场景未验证。',
        'yes',
        'DriveVLA-M0 论文 §3.1 和附录 A："each proposal is evaluated by the PDM scorer"；Table 1 区分 Base 92.3 与 Scale 94.1。',
        'none；论文未消融该评分头或权重对 navtest PDMS 的独立贡献。',
        'medium',
    ),
    finding(
        'NAV1-DVM0-002', 'navsim_v1', 'drivevla_m0', 'benchmark_prompt_specification',
        '系统提示明示 PDMS 公式和非反应式评测',
        'navsim/agents/EpisodeDrive/drivevla_backbone.py', 13, 23,
        '系统消息逐项列出榜单评分及权重，并告诉模型其轨迹将由 LQR 和沿日志轨迹运动的背景车进行非反应式四秒仿真。第 144–148 行把消息装入推理模板。若它影响输出，收益依赖评测约定；没有提示消融，也不能从静态代码证明实际增益。',
        'no：真实交通参与者会反应，固定日志回放与 PDMS 公式不能直接作为部署目标。',
        'partial',
        'DriveVLA-M0 论文 §3.1 只说明使用 "a system prompt T"，附录 C 给出 PDMS 定义；检查正文、附录 A/C，未见此完整提示或 LQR 非反应式文字。',
        'none；论文未比较有无该提示。',
        'low',
    ),
    finding(
        'NAV1-RAP-001', 'navsim_v1', 'rap', 'metric_proxy_optimization',
        '学习 PDMS 总分并据此选轨',
        'navsim/agents/rap_dino/rap_model.py', 155, 167,
        'RAP-DINO 在候选轨迹上预测最后一维 PDM 分数，推理取最大分候选；训练代码 rap_agent.py 第 419–421 行直接以仿真计算的 PDMS 标签监督。由此 navtest 分数的一部分来自拟合该榜单的评分器。',
        'partial：候选评估可以迁移，但原始 PDMS 标签、非反应式背景车及权重无法保证等效于部署质量。',
        'yes',
        'RAP 论文 §4 Models/Training："a trajectory scoring head trained with PDMS scores"；Table 1 报 RAP-DINO navtest PDMS 93.8。',
        'none；论文没有隔离 PDM 评分头相对同一视觉/轨迹模型的 navtest 消融。',
        'high',
    ),
    finding(
        'NAV1-DRIVOR-001', 'navsim_v1', 'drivor', 'benchmark_target_shaping',
        '用更远期轨迹造偏进度训练目标',
        'navsim/agents/drivoR/drivor_features.py', 224, 236,
        'v1 训练命令将 long_trajectory_additional_poses 设为 2；这里读取更远未来，再把采样位置逐步外推以构造较激进的第二目标。它提高偏进度的 navval PDMS，却在 v2 双阶段下降，显示增益与 v1 指标的进度偏好相连。',
        'partial：更积极前进在部分道路有用，但 v2 扰动场景的安全代价说明不能把 v1 增益原样迁移。',
        'yes',
        'DrivoR 论文 §4.2.4、Table 7："It improves NAVSIM-v1, which rewards progress over comfort, but hurts performance on NAVSIM-v2"。',
        'Table 7：navval PDMS 90.0→90.6（+0.6）；warmup 双阶段 EPDMS 39.4→37.8（−1.6）。',
        'high',
    ),
    finding(
        'NAV1-DRIVOR-002', 'navsim_v1', 'drivor', 'metric_proxy_optimization',
        'PDMS 子分数预测后以官方权重选轨',
        'navsim/agents/drivoR/drivor_model.py', 190, 203,
        '解耦评分器先预测 NC、DAC、TTC、EP、舒适度，再按 v1 PDMS 权重相加对 64 个候选取最大值。这个选择器可提升 navtest 分数，但其排序目标正是榜单的代理目标。',
        'partial：评分器可重标定用于部署；相同权重对新城市、真实反应交通的收益未被证明。',
        'yes',
        'DrivoR 论文 §3.4："the final trajectory is chosen from the proposal set via the max predicted score"；附录 Table 10 列出 v1 推理权重。',
        'Table 6：同为解耦评分器时，单总分 88.2→六个子分数 90.0 navval PDMS（+1.8）；这是子分数设计消融，非整个选择器贡献。',
        'high',
    ),
    finding(
        'NAV2-TOAD-001', 'navsim_v2', 'toad', 'metric_proxy_optimization',
        'CEM 在测试时反复优化学得的 PDM 评分器',
        'navsim/agents/drivoR/drivor_model.py', 357, 368,
        'navhard 双阶段配置在推理打开 5 轮、每轮 64 条控制序列的 CEM；循环用学得的子评分器评估运动学轨迹，并减去舒适和锚定惩罚。相同冻结模型的 EPDMS 增益由这套测试时优化产生，而非新学会驾驶。',
        'partial：搜索和舒适约束可以部署，但优化的是 NAVSIM 代理分数，且论文显示多迭代会偏离真实 EPDMS。',
        'yes',
        'TOAD 论文 §3、§4.2、Table 2："DrivoR + TOAD reaches 56.3 EPDMS"；§5 指出更多迭代会降低与实际 EPDMS 的相关性。',
        'Table 2，navhard 双阶段 EPDMS 54.6→56.3（+1.7）；Figure 4 的 scorer reward 消融在 iPad warmup 上 35.7→48.5。',
        'high',
    ),
    finding(
        'NAV2-TOAD-002', 'navsim_v2', 'toad', 'benchmark_split_adaptation',
        '在与 navhard 有交集的 warmup 上选 CEM 超参数',
        'navsim/planning/script/config/common/agent/drivoR.yaml', 43, 55,
        '最终配置固定 5 轮、64 样本、8 分之 1 精英、初始方差和锚定/舒适权重；论文 Figure 4 在 warmup-two-stage 上做这些组件和超参数消融，且论文脚注说明它与 navhard 有场景交集并获主办方认可。适配这一评测分布的收益不随模型迁往独立场景。',
        'no：对特定且有交集的验证划分调参本身不能迁移到新城市；CEM 算法可迁移。',
        'yes',
        'TOAD 论文 §4.1："For the ablation, we use warmup-two-stage as validation set ... containing 7 scenes"；脚注称 warmup 与 navhard 交叠、主办方验证使用；§5 Figure 4 展示参数消融。',
        'none；Figure 4 给 warmup 上的参数敏感性，但没有隔离「使用重叠划分」对 navhard 成绩的数字。',
        'medium',
    ),
    finding(
        'NAV2-DRIVOR-001', 'navsim_v2', 'drivor', 'benchmark_weight_tuning',
        'navhard 推理对子评分重新加权',
        'README.md', 203, 208,
        'v2 命令把 NC/DAC/DDC/TTC/EP 设为 10/13/6/14/15，明显偏离官方 EPDMS 的 1/1/1/5/5；论文说在 warmup-two-stage 上验证此改权方案，而 warmup 与 navhard 有交集。它把模型候选选择调向该榜的权重与分布。',
        'partial：安全和进度之间的权衡需要部署调校，但这些数字及重叠划分不会通用。',
        'yes',
        'DrivoR 论文 §4.2.4 "Safety-oriented agent"、附录 Table 8/10："For NAVSIM-v2, we adjust the weights"；§4.1 脚注说明 warmup 与 navhard 交叠且经主办方认可。',
        'none；Table 10 有权重、Figure 7 有定性变化，未给同一 v2 模型原权重与新权重的 navhard EPDMS 差。',
        'high',
    ),
    finding(
        'NAV2-DRIVOR-002', 'navsim_v2', 'drivor', 'metric_proxy_optimization',
        '用 v1 式子分数代理挑选 v2 候选',
        'navsim/agents/drivoR/drivor_model.py', 190, 203,
        'v2 README 要复制此 agent 到 NAVSIM-v2 环境并覆写权重；实际选择仍由预测的六项 PDM 子分数聚合后 argmax，而 v2 官方 EPDMS 还有 TLC、LK、HC、EC 等项。它是针对可学代理指标的择优，并非完整的真实驾驶价值函数。',
        'partial：候选打分可能迁移，但缺失的指标和人工权重需在目标车辆环境重新验证。',
        'yes',
        'DrivoR 论文 §3.4 说明六项 PDMS 子评分；§4.1、Table 3 报 NAVSIM-v2 双阶段结果；附录 Table 8/10 显示 EPDMS 指标与推理权重的区别。',
        'Table 3：相同 ViT-S 骨干、GTRS 头 45.8 与 DrivoR 头 48.3 EPDMS（+2.5）；论文称主要差异在评分管线，不等于完整 54.6 模型上的独立消融。',
        'medium',
    ),
    finding(
        'NAV2-GTRS-001', 'navsim_v2', 'gtrs', 'offline_candidate_library',
        '离线轨迹词表并入扩散候选再评分',
        'navsim/agents/gtrs_dense/hydra_model.py', 289, 298,
        '推理时把预先聚类的 8192/16384 条轨迹词表接在扩散模型生成的候选后，再由同一个 scorer 选一条（第 323–328 行）。分数的一部分来自离线候选覆盖范围和候选池规模，而不是即时感知和规划本身。',
        'partial：轨迹库和候选重排可以车载部署，但词表来自 nuPlan/OpenScene 轨迹统计，传感器、城市和运动分布改变会影响覆盖。',
        'yes',
        'GTRS 论文 §2.2、§4.3、Table 1："dynamic proposals ... are appended to the inference vocabulary"；Table 1 比较同骨干不同候选集。',
        'Table 1 旧版 navhard 协议：EVA-ViT-L 的 VXL 39.7→Vdp∪VXL 40.8 EPDMS（+1.1）。不能当作修复后 45.4 的增量。',
        'medium',
    ),
    finding(
        'NAV2-GTRS-002', 'navsim_v2', 'gtrs', 'benchmark_weight_tuning',
        'GTRS-Aug 重排使用手定子评分权重',
        'navsim/agents/gtrs_aug/hydra_model.py', 565, 574,
        '细排以 TLC、NC、DAC、DDC、TTC、进度、车道保持及历史舒适度的 sigmoid 分数手工拼接；系数 0.1/0.5/0.3/6 与官方 EPDMS 聚合不同。选轨因此受专为该榜单设置的排序公式影响。45.4 修复后对应具体模型权重未在本固定提交得到完整复现。',
        'partial：手调风险函数可上车，但这里的标签、系数与 navhard 分布不能原样迁移。',
        'partial',
        'GTRS 论文 §2.3 描述 refinement scorer，§4.1 描述 EPDMS；检查正文 §2–4 与 Table 1/2，未找到这一组 0.1/0.5/0.3/6 推理系数。README 仅列变体及旧分数。',
        'none；论文没有对这组细排权重做单独 ablation。',
        'medium',
    ),
]

assert len({item['id'] for item in F}) == len(F)
with (OUT / 'findings' / 'agent_nav.jsonl').open('w') as f:
    for item in F:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'Wrote {len(F)} findings')
