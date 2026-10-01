"""阶段 2 定稿：合并发现、统一类别、静态生成全量矩阵。绝不导入仓库代码。"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path('/data/hack_audit')
OUT = ROOT / 'out'

with (OUT / 'repos.csv').open(newline='') as f:
    repos = list(csv.DictReader(f))
by_board_slug = {(r['board'], r['slug']): r for r in repos}
by_board_url = {(r['board'], r['repo_url'].rstrip('/')): r for r in repos}

text = (OUT / 'taxonomy.md').read_text()
text = text.split('## 变更日志', 1)[0]
categories = re.findall(r'^### ([a-z][a-z0-9_]*)$', text, re.M)
assert len(categories) == 19 and len(categories) == len(set(categories)), categories

def recode(f):
    old = f['category']
    fid = f['id']
    if old == 'metric_proxy_optimization':
        return 'metric_proxy_test_search' if fid.startswith(('NAV1-TOAD-', 'NAV2-TOAD-')) else 'metric_proxy_candidate_selection'
    if old == 'metric_reward_training':
        return 'metric_reward_finetuning' if fid == 'REAL-WOD-001' else 'metric_proxy_candidate_selection'
    return {
        'stuck_recovery_override': 'manual_control_override',
        'infraction_rule_override': 'manual_control_override',
        'ego_kinematics_shortcut': 'ego_only_open_loop_planning' if fid == 'REAL-NUS-001' else 'ego_state_fusion',
        'benchmark_weight_tuning': 'manual_scorer_reweighting',
        'test_time_candidate_selection': 'multi_checkpoint_candidate_selection',
        'simulator_controller_patch': 'simulator_protocol_dependency',
    }.get(old, old)

merged = []
for file in sorted((OUT / 'findings').glob('*.jsonl')):
    for line in file.read_text().splitlines():
        if not line.strip():
            continue
        f = json.loads(line)
        if f['id'] == 'REAL-WOD-004':
            continue  # 阶段 2 裁决：普通训练标签提示，不进入确认矩阵。
        r = by_board_slug.get((f['board'], f['repo'])) or by_board_url.get((f['board'], f['repo'].rstrip('/')))
        assert r is not None, (file, f['id'], f['repo'])
        assert f['commit'] == r['commit'], (f['id'], f['commit'], r['commit'])
        f['repo'] = r['repo_url']
        f['category'] = recode(f)
        if f['id'] == 'REAL-NUS-001':
            f['title'] = '无环境感知的 ego 状态与导航命令预测短时轨迹'
            f['description'] = ('模型从预制 pkl 读取 21 维输入，不取相机或激光；论文说明其中既有历史 ego '
                                '运动状态，也有由未来真值生成的高层命令。完全不读环境感知的机制成立，'
                                '但完整 0.29 m 成绩不能全部归因于可部署的历史运动外推；未来命令另由 REAL-NUS-006 编码。')
            f['impact_evidence'] = ('论文 Table 1：仅历史轨迹 avg L2 0.97 m/碰撞 0.49%；增加速度和加速度后、'
                                    '仍无高层命令为 0.35 m/0.23%；加入未来命令后 0.29 m/0.19%，后一步由 REAL-NUS-006 说明。')
        assert f['category'] in categories, (f['id'], f['category'])
        merged.append(f)

ids = [f['id'] for f in merged]
assert len(ids) == len(set(ids))
merged.sort(key=lambda f: (f['board'], f['repo'], f['category'], f['id']))
with (OUT / 'findings.jsonl').open('w') as file:
    for f in merged:
        file.write(json.dumps(f, ensure_ascii=False) + '\n')

yes = {(f['board'], f['repo'], f['category']) for f in merged}
CLOSED = {'bench2drive', 'carla_lb2', 'hugsim'}
OPEN = {'navsim_v1', 'navsim_v2', 'nuscenes', 'wod_e2e'}
CLOSED_ONLY = {
    'control_interface_selection', 'manual_control_override', 'simulator_pose_access',
    'metric_early_termination', 'simulator_protocol_dependency', 'inference_failure_brake_fallback',
}
OPEN_ONLY = {'future_label_conditioning'}
NUSCENES_ONLY = {'ego_only_open_loop_planning', 'ego_state_fusion'}
TRAINING = {'metric_reward_finetuning', 'benchmark_target_shaping', 'metric_grid_alignment',
            'benchmark_split_adaptation'}
PROMPT_METHODS = {'drivevla_m0', 'sparseoccvla', 'drivema', 'autovla', 'blue'}
B_LIMITED = {('bench2drive', 'blue'), ('wod_e2e', 'autovla'), ('hugsim', 'uniad'), ('hugsim', 'ltf')}

def absent_value(r, category):
    board, slug = r['board'], r['slug']
    key = (board, slug)
    if category in CLOSED_ONLY and board not in CLOSED:
        return 'NA'
    if category in OPEN_ONLY and board not in OPEN:
        return 'NA'
    if category in NUSCENES_ONLY and board != 'nuscenes':
        return 'NA'
    if category == 'benchmark_prompt_specification' and slug not in PROMPT_METHODS:
        return 'NA'
    if category in TRAINING and key in B_LIMITED:
        return 'NA'
    if key == ('wod_e2e', 'autovla') and category in {
        'metric_proxy_candidate_selection', 'metric_proxy_test_search',
        'manual_scorer_reweighting', 'multi_checkpoint_candidate_selection',
        'future_label_conditioning', 'offline_candidate_library',
    }:
        return 'NA'  # Waymo 专用提交/评测入口缺失。
    if key == ('nuscenes', 'sparseoccvla') and category == 'future_label_conditioning':
        return 'NA'  # gt_planning_command 的生成链未公开，不能判 yes/no。
    return 'no'

matrix = []
for r in repos:
    for cat in categories:
        key = (r['board'], r['repo_url'], cat)
        matrix.append(dict(board=r['board'], repo=r['repo_url'], category=cat,
                           present='yes' if key in yes else absent_value(r, cat)))
assert len(matrix) == len(repos) * len(categories)
assert all(any(m['board'] == b and m['repo'] == u and m['category'] == c and m['present'] == 'yes'
               for m in matrix) for b, u, c in yes)
with (OUT / 'matrix.csv').open('w', newline='') as file:
    w = csv.DictWriter(file, ['board', 'repo', 'category', 'present'])
    w.writeheader()
    w.writerows(matrix)

counts = defaultdict(int)
for m in matrix:
    counts[m['present']] += 1
print(f'{len(merged)} findings, {len(categories)} categories, {len(repos)} units, {len(matrix)} matrix cells, values={dict(counts)}')
