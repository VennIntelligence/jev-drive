"""根据阶段 1 的代码可用性核验修正阶段 0 抽样，不运行模型。"""
import csv
from pathlib import Path

root = Path('/data/hack_audit')
out = root / 'out'

def read_csv(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

sf, sampling = read_csv(out / 'sampling.csv')
for row in sampling:
    key = (row['board'], row['method'])
    if key == ('bench2drive', 'BLUE'):
        row['code_grade'] = 'B'
    elif key == ('carla_lb2', 'TF++'):
        row['score'] = 'official DS 5.56; track disputed'
        row['split_protocol_note'] += ' 仓库运行脚本为 SENSORS，SOTA2 的 MAP 标签待核。'
    elif key == ('nuscenes', 'Senna'):
        row['code_grade'] = 'C'
        row['selected'] = 'no'
        row['not_selected_reason'] = '固定公开提交只有 Senna-VLM，缺少对应 0.22 L2 的 E2E 规划实现'
    elif key == ('hugsim', 'DrivoR'):
        row['code_grade'] = 'C'
        row['selected'] = 'no'
        row['not_selected_reason'] = '固定公开提交没有 HUGSIM 适配/评测代码；论文虽报告该分数'
    elif key == ('hugsim', 'LTF'):
        row['repo_url'] = 'https://github.com/hyzhou404/NAVSIM'
        row['code_grade'] = 'B'
        row['selected'] = 'yes'
        row['not_selected_reason'] = ''
    elif key == ('wod_e2e', 'AutoVLA'):
        row['code_grade'] = 'B'
        row['split_protocol_note'] += ' Waymo 专用提交/评测入口未公开；公开模型及通用训练路径可审计。'

sampling.append(dict(board='nuscenes', rank='6', method='BEV-Planner', venue_year='2023',
                     score='avg L2 0.35',
                     ranking_source_url='https://www.sota2.com/research/sota/open-loop-planning-on-nuscenes-v1-0-test',
                     snapshot_date='2026-09-24', repo_url='https://github.com/NVlabs/BEV-Planner',
                     code_grade='A', selected='yes', not_selected_reason='',
                     split_protocol_note='SOTA2 的 avg L2；与 Senna 等方法的协议未必相同；Senna 公开提交缺 E2E 规划后递补。'))
write_csv(out / 'sampling.csv', sf, sampling)

rf, repos = read_csv(out / 'repos.csv')
repos = [r for r in repos if (r['board'],r['method']) not in {('nuscenes','Senna'),('hugsim','DrivoR')}]
for row in repos:
    key = (row['board'],row['method'])
    if key == ('bench2drive','BLUE'):
        row['code_grade'] = 'B'
    elif key == ('bench2drive','SparseDriveV2'):
        row['repo_path'] = 'repos/bench2drive__sparsedrivev2_b2d'
        row['commit'] = 'e42ea59dd4946238dc65097495a9aa0708121fae'
        row['branch'] = 'bench2drive'
    elif key == ('wod_e2e','AutoVLA'):
        row['code_grade'] = 'B'

repos.append(dict(board='nuscenes', method='BEV-Planner', slug='bev_planner',
                  repo_url='https://github.com/NVlabs/BEV-Planner',
                  repo_path='repos/nuscenes__bev_planner',
                  commit='01c28d6db56a178ee3a65bf017fe7996360ef026',
                  clone_date='2026-09-24', clone_status='ok',
                  paper_url='https://arxiv.org/pdf/2312.03031',
                  paper_path='papers/bev_planner.pdf', paper_status='ok',
                  code_grade='A', branch='master'))
repos.append(dict(board='hugsim', method='LTF', slug='ltf',
                  repo_url='https://github.com/hyzhou404/NAVSIM',
                  repo_path='repos/hugsim__ltf',
                  commit='ca0ca7e4368646d8f7b86fb1fdaa1862c946176f',
                  clone_date='2026-09-24', clone_status='ok',
                  paper_url='https://arxiv.org/pdf/2406.15349',
                  paper_path='papers/ltf.pdf', paper_status='ok',
                  code_grade='B', branch='main'))
write_csv(out / 'repos.csv', rf, repos)

index_path = out / 'INDEX.md'
index = index_path.read_text()
index = index.replace('| nuscenes | Senna | agent_real | in-progress | [审计页](audits/nuscenes__senna.md) |\n', '')
index = index.replace('| hugsim | DrivoR | root | todo | [审计页](audits/hugsim__drivor.md) |\n', '')
index += '| nuscenes | BEV-Planner | agent_real | in-progress | [审计页](audits/nuscenes__bev_planner.md) |\n'
index += '| hugsim | LTF | root | in-progress | [审计页](audits/hugsim__ltf.md) |\n'
index = index.replace('| hugsim | WA-JEPA | root | todo |', '| hugsim | WA-JEPA | root | in-progress |')
index = index.replace('| hugsim | UniAD | root | todo |', '| hugsim | UniAD | root | in-progress |')
index_path.write_text(index)

with (out / 'progress.md').open('a') as f:
    f.write('\n- 阶段 0 复核修正：Senna、HUGSIM DrivoR 缺对应榜单实现降为 C，分别以 BEV-Planner、LTF 递补；BLUE、AutoVLA 降为 B；SparseDriveV2 切到 bench2drive 分支；TF++ track 标签待核。\n')
