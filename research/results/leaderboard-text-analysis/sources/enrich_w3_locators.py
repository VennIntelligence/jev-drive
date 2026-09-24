"""Add source locators to issue excerpts after independent review."""

import csv
from pathlib import Path

path = Path('/data/hack_audit/out2/w3_issues.csv')
with path.open(newline='', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = list(reader)

locators = {
    ('AFARI-Research/WA-JEPA', '1'): 'WA-JEPA Table 2, PDF p.7; HUGSIM score is a separate board',
    ('autonomousvision/carla_garage', '11'): 'TF++ original Table 6, PDF p.8',
    ('autonomousvision/carla_garage', '46'): 'TF++ original v2 Table 6, PDF p.8; Table 13, PDF p.16',
    ('autonomousvision/carla_garage', '47'): 'TF++ original Table 6, PDF p.8',
    ('autonomousvision/carla_garage', '73'): 'No comparable paper number; author comment in issue',
    ('autonomousvision/carla_garage', '103'): 'Hidden Biases Table 4, PDF p.6',
    ('autonomousvision/carla_garage', '120'): 'Hidden Biases Table 5, PDF p.8',
    ('autonomousvision/navsim', '35'): 'NAVSIM Table 1, PDF p.7; Hydra-MDP Table 1, PDF p.4',
    ('autonomousvision/navsim', '62'): 'Hydra-MDP Table 1, PDF p.4; NAVSIM Table 1, PDF p.7',
    ('E2E-AD/AD-MLP', '4'): 'AD-MLP Table 1, PDF p.3',
    ('E2E-AD/AD-MLP', '5'): 'AD-MLP Table 1, PDF p.3',
    ('George-Ling3/BLUE', '4'): 'BLUE arXiv v1 Table 1, PDF p.4 (original Think2Drive); current Table 1, PDF p.4 (corrected PDM-Lite)',
    ('George-Ling3/BLUE', '5'): 'BLUE Table 6, PDF p.6',
    ('hustvl/Senna', '11'): 'Senna Table II, PDF p.6; Senna* has ego status input',
    ('hustvl/Senna', '26'): 'Senna Table II, PDF p.6; Senna* has ego status input',
    ('kesai-labs/lead', '89'): 'TFv6 Table 5, PDF p.7; issue quotes 95.28, a different value from paper 95.2±0.3',
    ('kesai-labs/lead', '90'): 'TFv6 Table 5, PDF p.7; issue #90 clarifies estimate',
    ('kesai-labs/lead', '99'): 'TFv6 Table 5, PDF p.7; different sensor/data configuration',
    ('RenzKa/simlingo', '25'): 'SimLingo Section 3.4, PDF p.5',
    ('RenzKa/simlingo', '43'): 'SimLingo Table 2, PDF p.7',
    ('RenzKa/simlingo', '49'): 'SimLingo Section 3.4, PDF p.5; Appendix Table 7, PDF p.14',
    ('RenzKa/simlingo', '66'): 'SimLingo Section 3.4, PDF p.5',
    ('RenzKa/simlingo', '72'): 'SimLingo Table 2, PDF p.7',
    ('swc-17/SparseDriveV2', '16'): 'SparseDriveV2 Table 4, PDF p.13',
    ('Tsinghua-MARS-Lab/DriveMA', '2'): 'DriveMA Table 3, PDF p.7; value rounded from 7.893',
    ('Tsinghua-MARS-Lab/DriveMA', '3'): 'DriveMA Appendix training details, PDF p.14; code value is issue claim',
    ('ucla-mobility/AutoVLA', '48'): 'AutoVLA Table 1, PDF p.7',
    ('ucla-mobility/AutoVLA', '47'): 'AutoVLA Table S4, PDF p.30; issue cites rows 3/4, no score transcribed',
    ('valeoai/DrivoR', '4'): 'DrivoR Table 1, PDF p.5',
    ('valeoai/DrivoR', '31'): 'DrivoR Table 3, PDF p.6',
    ('valeoai/DrivoR', '34'): 'DrivoR Table 2, PDF p.5; 345-scene description, PDF p.6',
    ('valeoai/DrivoR', '40'): 'DrivoR Table 1, PDF p.5; scaling description, PDF p.6',
    ('valeoai/DrivoR', '41'): 'DrivoR Table 1, PDF p.5; Table 4a, PDF p.7',
    ('valeoai/DrivoR', '47'): 'DrivoR Table 3, PDF p.6',
    ('valeoai/DrivoR', '54'): 'DrivoR Table 2, PDF p.5; 345-scene description, PDF p.6',
}

for row in rows:
    issue = row['issue_url'].rsplit('/', 1)[-1]
    key = (row['repo'], issue)
    row['paper_locator'] = locators.get(key, 'N/A; issue URL is the source for user and author statements')
    if key == ('autonomousvision/carla_garage', '11'):
        row['paper_url'] = 'https://arxiv.org/pdf/2306.07957v1'
    if key in {('autonomousvision/carla_garage', '46'), ('autonomousvision/carla_garage', '47')}:
        row['paper_url'] = 'https://arxiv.org/pdf/2306.07957v2'
    if key == ('George-Ling3/BLUE', '4'):
        row['paper_url'] = 'https://arxiv.org/pdf/2606.08684v1; https://arxiv.org/pdf/2606.08684'
    if key == ('autonomousvision/carla_garage', '73'):
        row['paper_numbers'] = '该 issue 无可比论文数字；作者仅称用户 Longest6v2 DS 25 在方差内'
        row['paper_url'] = 'N/A（无对应论文数字）'
    if key == ('autonomousvision/navsim', '35'):
        row['paper_url'] = 'https://arxiv.org/pdf/2406.15349; https://arxiv.org/pdf/2406.06978'
        row['paper_numbers'] = 'NAVSIM Table 1 TransFuser PDMS 84.0；Hydra-MDP Table 1 TransFuser Score 78.0（DDC 省略）；用户引用两篇已发表数字'
    if key == ('autonomousvision/navsim', '62'):
        row['category'] = 'evaluation_config'
        row['related_methods'] = 'TransFuser (issue subject); PDM-Closed[A,excluded] (repo association only)'
        row['paper_numbers'] = 'Hydra-MDP Table 1 TransFuser Score 78.0（PDM score，§3.1 说明 DDC 因实现问题省略）；NAVSIM Table 1 TransFuser PDMS 84.0；两篇实现口径未核同'
    if key in {('autonomousvision/navsim', '158'), ('autonomousvision/navsim', '172')}:
        row['paper_numbers'] = '该 issue 无可比论文数字；零分来自用户评测显示'
        row['paper_url'] = 'N/A（用户报告，见 issue_url）'
    if key == ('kesai-labs/lead', '89'):
        row['paper_numbers'] = 'TFv6 论文 Table 5 最佳配置 Bench2Drive DS 95.2±0.3；用户在 issue 引用 95.28，来源/精度未核同'
    if key == ('kesai-labs/lead', '90'):
        row['paper_numbers'] = 'TFv6 Table 5 LEAD expert Bench2Drive DS 96.8、SR 96.6；作者 issue 回复说明这是估计，未直接在 Bench2Drive 评测'
    if key == ('kesai-labs/lead', '99'):
        row['paper_numbers'] = 'TFv6 论文 Table 5 最佳配置 Bench2Drive DS 95.2±0.3；该帖未报可比 DS'
    if key == ('RenzKa/simlingo', '49'):
        row['paper_numbers'] = '论文 §3.4 写每 epoch 650,000 样本；附录训练设置表写 14 epochs'
    if key in {('hustvl/Senna', '11'), ('hustvl/Senna', '26')}:
        row['paper_numbers'] = 'Senna Table II 的 Senna*（输入 ego status）平均 L2 0.22 m；无星 Senna 为 0.59 m；与用户所报指标不同'
    if key == ('valeoai/DrivoR', '41'):
        row['user_reported_numbers'] = '未报告本人复现实测；提问仅引用论文主结果 93.1 与消融约 90'
    if key == ('swc-17/SparseDriveV2', '9'):
        note = '主崩溃有明确修法；--agent-config 格式与 SAVE_PATH 子问未逐一答复。'
        if note not in row['case_summary']:
            row['case_summary'] += ' ' + note

fields = [key for key in fields if key != 'paper_locator'] + ['paper_locator']
with path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f'Enriched {len(rows)} issue rows; {len(locators)} explicit paper locators')
