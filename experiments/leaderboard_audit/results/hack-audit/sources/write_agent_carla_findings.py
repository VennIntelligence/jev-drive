"""将阶段 1 的静态审计笔记写成带固定源码片段的 JSONL。只读取仓库文本。"""
import json
from pathlib import Path

ROOT = Path('/data/hack_audit')


def finding(*, id, board, repo, commit, checkout, category, title, path, start, end,
            description, deploy_valid, disclosed, paper_quote, impact_evidence, confidence):
    lines = (ROOT / checkout / path).read_text().splitlines()
    assert 1 <= start <= end <= len(lines) and end - start + 1 <= 15
    return dict(
        id=id, board=board, repo=repo, commit=commit, category=category, title=title,
        location=f'{repo}/blob/{commit}/{path}#L{start}-L{end}',
        snippet='\n'.join(lines[start-1:end]), description=description,
        deploy_valid=deploy_valid, disclosed=disclosed, paper_quote=paper_quote,
        impact_evidence=impact_evidence, confidence=confidence,
    )


TF_REPO = 'https://github.com/kesai-labs/lead'
TF_COMMIT = '730bc1a2f44d5f28312dd55f0ca958e94a24c038'
BLUE_REPO = 'https://github.com/George-Ling3/BLUE'
BLUE_COMMIT = '6970cb69e05ef904b37f4264207ea0e1dab35ef2'
SPARSE_REPO = 'https://github.com/swc-17/SparseDriveV2'
SPARSE_COMMIT = 'e42ea59dd4946238dc65097495a9aa0708121fae'
TFPP_REPO = 'https://github.com/autonomousvision/carla_garage'
TFPP_COMMIT = 'f22bc491b3094792aef475149e09a94dcbb526f9'

entries = [
    finding(
        id='B2D-TFV6-001', board='bench2drive', repo=TF_REPO, commit=TF_COMMIT,
        checkout='repos/bench2drive__tfv6', category='control_interface_selection',
        title='默认以 route 和 target speed 控车，而非网络预测的时序 waypoint',
        path='lead/inference/config_closed_loop.py', start=29, end=35,
        description='网络同时输出 route、target speed 与未来 waypoint；闭环推理代码把两套输出分别送入 PID，但默认选 route 方向与 target speed 纵向控制。因而 DS 包含输出接口与手工控制器的作用，不能只归因于 waypoint 规划质量。是否更优及幅度需同 checkpoint 配对评测。',
        deploy_valid='partial: 路线和 PID 可在真车使用，但原控制增益和两头接口的相对优势不能直接迁移到另一车辆及场景。',
        disclosed='partial',
        paper_quote='TFv6 论文 §3.1 明写 “an additional learned query predicts target speed”；README §1.3 披露启用推理启发式，未给同一 checkpoint 的 route+target speed 对 waypoint 控车消融。',
        impact_evidence='none', confidence='medium',
    ),
    finding(
        id='B2D-TFV6-002', board='bench2drive', repo=TF_REPO, commit=TF_COMMIT,
        checkout='repos/bench2drive__tfv6', category='stuck_recovery_override',
        title='复现配置开启卡住后强行蠕行覆盖模型刹车',
        path='lead/inference/sensor_agent.py', start=854, end=861,
        description='README 的 95.28 DS 复现配置开启 sensor_agent_creeping。静止超过 1100 帧后，后处理器在前方 LiDAR 安全框为空时将油门至少提到 0.4 并取消刹车，以减少 blocked/timeout；这一动作来自计时器和规则，不是网络决策。',
        deploy_valid='partial: 脱困需要真实安全验证；固定帧数与油门阈值只按 CARLA 时钟和车辆动态设定。',
        disclosed='yes',
        paper_quote='README §1.3: “To reproduce these results, enable the Kalman filter, stop-sign, and creeping heuristics”；论文正文 §4 与表 5 未单独报告 creeping。',
        impact_evidence='README §1.3 三项启发式联合开/关：Bench2Drive DS 约 95→94；无法分离本项贡献。',
        confidence='medium',
    ),
    finding(
        id='B2D-TFV6-003', board='bench2drive', repo=TF_REPO, commit=TF_COMMIT,
        checkout='repos/bench2drive__tfv6', category='infraction_rule_override',
        title='复现配置开启停车牌规则后处理覆盖模型输出',
        path='lead/inference/sensor_agent.py', start=753, end=756,
        description='推理后处理缓存网络检测到的停车牌；靠近且尚未停车时规则把油门改成零并直接刹车。Bench2Drive 对 stop infraction 计分，停车决策的一部分由规则提供。默认配置为关闭，但 README 的 95.28 DS 复现指令显式开启。',
        deploy_valid='partial: 检测停车牌后制动可迁移，但距离、冷却帧数与计分项的收益必须重新校准。',
        disclosed='yes',
        paper_quote='README §1.3: “enable the Kalman filter, stop-sign, and creeping heuristics”；论文正文 §4 与表 5 未给 stop-sign 后处理独立消融。',
        impact_evidence='README §1.3 三项启发式联合开/关：Bench2Drive DS 约 95→94；无停车牌单项数字。',
        confidence='medium',
    ),
    finding(
        id='B2D-BLUE-001', board='bench2drive', repo=BLUE_REPO, commit=BLUE_COMMIT,
        checkout='repos/bench2drive__blue', category='stuck_recovery_override',
        title='BLUE 评测 agent 在卡住时强制油门并取消刹车',
        path='team_code/agent_simlingo.py', start=863, end=876,
        description='公开 BLUE 评测入口固定使用此 agent；连续静止超过配置中的 800 帧后，规则将油门提到至少 0.4 并取消刹车，绕开网络轨迹/PID 的制动选择。是否实际触发、对 90.58 DS 的贡献未有日志或消融确证。',
        deploy_valid='partial: 实车可设计脱困控制，但此处固定静止帧数与取消刹车需要独立安全验证。',
        disclosed='no',
        paper_quote='未见披露；已查 BLUE 论文 §3–5、Appendix C/D 和 README 的方法及结果段，均未说明 stuck/creep override。',
        impact_evidence='none', confidence='medium',
    ),
    finding(
        id='B2D-SPARSE-001', board='bench2drive', repo=SPARSE_REPO, commit=SPARSE_COMMIT,
        checkout='repos/bench2drive__sparsedrivev2_b2d', category='simulator_pose_access',
        title='SENSORS agent 从 CARLA world 读取 LiDAR actor 精确位姿',
        path='leaderboard/team_code/sparsedrive_b2d_agent.py', start=360, end=370,
        description='agent 声明 SENSORS，但每帧从 CarlaDataProvider.get_world() 找 LiDAR actor 并调用 get_transform()，据此计算 ego2global、pos 和路线最近节点；GPS 只被读取，实际定位行改用 actor 真值。模型输入的全局变换元数据亦继承该真值。换到真车或封闭评测接口时无法这样查询仿真器世界。',
        deploy_valid='no: 真实部署无法从 CARLA world 查询传感器 actor 的真值世界位姿。',
        disclosed='no',
        paper_quote='未见披露；已查 SparseDriveV2 论文 §3–4 及 Appendix A（Bench2Drive 实现），只描述 sensor observations、六相机及控制器，没有说明 CARLA world 位姿查询。',
        impact_evidence='none', confidence='high',
    ),
    finding(
        id='LB2-TFPP-001', board='carla_lb2', repo=TFPP_REPO, commit=TFPP_COMMIT,
        checkout='repos/carla_lb2__tfpp', category='metric_early_termination',
        title='按固定行驶距离主动停车利用 DS 与 RC 的乘积公式',
        path='team_code/sensor_agent.py', start=673, end=676,
        description='STOP_AFTER_METER 启用后，速度积分超过阈值就永久把油门设为零、刹车设为一。论文明确说明在长路线与较低 IS 下，提前结束能提高 DS，并称实际采用 1.5 km；公开脚本默认 -1，未固定官方提交值，因此代码可证机制，官方分数对应配置由论文证明。',
        deploy_valid='no: 真实驾驶任务要求继续完成路线，主动停车以提高乘积式分数没有驾驶收益。',
        disclosed='yes',
        paper_quote='TF++ 技术报告 §5 Early termination: “we set target speed to 0 after d = 1.5km in practice”；其后建议改为 Normalized DS 消除激励。',
        impact_evidence='论文表 5（Town13 validation，同模型）：全城训练设置无提前终止 DS 0.96、提前终止 5.10；非官方 LB2 单独消融。',
        confidence='high',
    ),
    finding(
        id='LB2-TFPP-002', board='carla_lb2', repo=TFPP_REPO, commit=TFPP_COMMIT,
        checkout='repos/carla_lb2__tfpp', category='infraction_rule_override',
        title='默认停车牌控制器在网络控制后强制刹车',
        path='team_code/sensor_agent.py', start=668, end=671,
        description='STOP_CONTROL 默认值为 1；agent 由检测框缓存与几何相交规则产生 stop_for_stop_sign，并在模型控制后直接清零油门、设刹车。停车牌违规是 LB2 扣分项，故部分得分由手工规则承担。',
        deploy_valid='partial: 停车牌制动可在真车使用，但此实现依赖检测框阈值和 CARLA 几何规则，贡献无法直接迁移。',
        disclosed='no',
        paper_quote='未见披露；已查 TF++ 技术报告 §2–6 的模型、控制器、评测与消融，以及仓库 README 的评测说明，未说明 agent 的默认 STOP_CONTROL 后处理。',
        impact_evidence='none', confidence='medium',
    ),
    finding(
        id='LB2-TFPP-003', board='carla_lb2', repo=TFPP_REPO, commit=TFPP_COMMIT,
        checkout='repos/carla_lb2__tfpp', category='stuck_recovery_override',
        title='卡住后以固定油门蠕行覆盖网络刹车',
        path='team_code/sensor_agent.py', start=657, end=661,
        description='agent 在速度低于 0.1 m/s 的累计帧数超过阈值时触发 force_move；前方 LiDAR 安全框为空即把油门至少提到 creep_throttle、取消刹车。该规则针对长路线 blocked 状态，与网络规划能力分离；没有该机制的官方 LB2 消融。',
        deploy_valid='partial: 实车脱困可做，但固定帧数、油门和简化 LiDAR 安全框需重新验证。',
        disclosed='no',
        paper_quote='未见披露；已查 TF++ 技术报告 §2–6 及 README 的模型与评测说明，未说明 stuck/creep override。',
        impact_evidence='none', confidence='medium',
    ),
]

for entry in entries:
    assert list(entry) == ['id', 'board', 'repo', 'commit', 'category', 'title', 'location',
                           'snippet', 'description', 'deploy_valid', 'disclosed',
                           'paper_quote', 'impact_evidence', 'confidence']

out = ROOT / 'out/findings/agent_carla.jsonl'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(''.join(json.dumps(entry, ensure_ascii=False) + '\n' for entry in entries))
print(f'{out}: {len(entries)} findings')
