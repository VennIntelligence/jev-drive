"""阶段 0 的可复核抽样元数据；只生成 CSV，不执行仓库代码。"""
import csv
from pathlib import Path

ROOT = Path('/data/hack_audit')
OUT = ROOT / 'out'
DATE = '2026-09-24'
B2D = 'https://github.com/autonomousvision/Bench2Drive-Leaderboard/blob/main/README.md'
CARLA = 'https://www.sota2.com/research/sota/autonomous-driving-on-carla-leaderboard-2-0-official-leaderboard'
NAV1 = 'https://www.sota2.com/research/sota/trajectory-planning-on-navsim-v1-test'
NAV2 = 'https://www.sota2.com/research/sota/autonomous-driving-planning-on-navsim-navhard-two-stage-v2'
NUSC = 'https://www.sota2.com/research/sota/open-loop-planning-on-nuscenes-v1-0-test'
WOD = 'https://www.sota2.com/research/sota/end-to-end-driving-on-waymo-e2e-driving-challenge-leaderboard'
HUG = 'https://arxiv.org/abs/2608.20974'  # 同一协议重测表 2，436 个场景

# board, rank, method, slug, venue/year, score, source, repo URL, grade, selected, reason, paper URL, branch
rows = [
 ('bench2drive',1,'TFv6','tfv6','CVPR 2026','DS 95.28',B2D,'https://github.com/kesai-labs/lead','A',1,'','https://arxiv.org/pdf/2512.20563','cvpr2026'),
 ('bench2drive',2,'LinkVLA','linkvla','2026','DS 91.01',B2D,'','C',0,'未找到该模型可审计的官方代码仓库','https://arxiv.org/pdf/2603.01441',''),
 ('bench2drive',3,'SteerVLA','steervla','2026','DS 91.00',B2D,'','C',0,'仅项目页，未找到公开训练或 agent 代码','https://arxiv.org/pdf/2602.08440',''),
 ('bench2drive',4,'FIVE-VLA','five_vla','2026','DS 90.95',B2D,'','C',0,'新论文未找到公开代码','https://arxiv.org/pdf/2609.18623',''),
 ('bench2drive',5,'BLUE','blue','EMNLP 2026','DS 90.58',B2D,'https://github.com/George-Ling3/BLUE','A',1,'','https://arxiv.org/pdf/2606.08684','main'),
 ('bench2drive',6,'RoG-DAgger','rog_dagger','2026','DS 90.34',B2D,'','C',0,'未找到官方代码','https://arxiv.org/pdf/2608.24525',''),
 ('bench2drive',7,'SparseDriveV2','sparsedrivev2','ECCV 2026','DS 89.15',B2D,'https://github.com/swc-17/SparseDriveV2','A',1,'','https://arxiv.org/pdf/2603.29163','main'),

 ('carla_lb2',1,'CarLLaVA','carllava','2024','official DS 6.87',CARLA,'https://github.com/RenzKa/simlingo','B',1,'同仓库另有 SimLingo-BASE；逐方法审计','https://arxiv.org/pdf/2406.10165','main'),
 ('carla_lb2',2,'SimLingo-BASE','simlingo_base','CVPR 2025','official DS 6.25',CARLA,'https://github.com/RenzKa/simlingo','A',1,'','https://arxiv.org/pdf/2503.09594','main'),
 ('carla_lb2',3,'TF++','tfpp','2024','official DS 5.56 MAP',CARLA,'https://github.com/autonomousvision/carla_garage','A',1,'','https://arxiv.org/pdf/2412.09602','leaderboard_2'),
 ('carla_lb2',4,'Kyber-E2E','kyber_e2e','2024','official DS 5.47',CARLA,'','C',0,'找到论文和报告，未找到该提交的公开 agent 代码','https://arxiv.org/pdf/2405.01394',''),

 ('navsim_v1',1,'TOAD+DrivoR','toad','2026','navtest PDMS 94.7','https://arxiv.org/abs/2606.07170','https://github.com/valeoai/TOAD','A',1,'','https://arxiv.org/pdf/2606.07170','main'),
 ('navsim_v1',2,'DriveVLA-M0','drivevla_m0','2026','navtest PDMS 94.1','https://arxiv.org/abs/2608.10413','https://github.com/ZebinX/DriveVLA-M0','A',1,'','https://arxiv.org/pdf/2608.10413','main'),
 ('navsim_v1',3,'RAP-DINO','rap','ICLR 2026','navtest PDMS 93.8','https://openreview.net/forum?id=a9bOgeqbdB','https://github.com/vita-epfl/RAP','A',1,'','https://arxiv.org/pdf/2510.04333','main'),
 ('navsim_v1',4,'DrivoR','drivor','CVPR 2026','navtest PDMS 93.7',NAV1,'https://github.com/valeoai/DrivoR','A',1,'','https://arxiv.org/pdf/2601.05083','main'),
 ('navsim_v1',5,'PDM-Closed','pdm_closed','2024','navtest PDMS 89.1','https://openreview.net/forum?id=a9bOgeqbdB','https://github.com/autonomousvision/navsim','A',0,'privileged 规则 baseline，仅参照','https://arxiv.org/pdf/2406.15349',''),

 ('navsim_v2',1,'PDM-Closed','pdm_closed','2024','navhard EPDMS 56.6',NAV2,'https://github.com/autonomousvision/navsim','A',0,'privileged 规则 baseline，仅参照','https://arxiv.org/pdf/2406.15349',''),
 ('navsim_v2',2,'DrivoR+TOAD','toad','2026','navhard EPDMS 56.3',NAV2,'https://github.com/valeoai/TOAD','A',1,'','https://arxiv.org/pdf/2606.07170','main'),
 ('navsim_v2',3,'DriveFuture','drivefuture','2026','navhard EPDMS 55.5','https://arxiv.org/abs/2605.09701','','C',0,'论文声称上榜但未找到公开可审计代码','https://arxiv.org/pdf/2605.09701',''),
 ('navsim_v2',4,'DrivoR','drivor','CVPR 2026','navhard EPDMS 54.6',NAV2,'https://github.com/valeoai/DrivoR','A',1,'','https://arxiv.org/pdf/2601.05083','main'),
 ('navsim_v2',5,'GTRS','gtrs','2025','navhard EPDMS 45.4',NAV2,'https://github.com/NVlabs/GTRS','A',1,'高于它的同表条目是 TOAD 对既有模型的组合；保留独立代码模型','https://arxiv.org/pdf/2506.06664','master'),

 ('nuscenes',1,'Senna','senna','2024','avg L2 0.22',NUSC,'https://github.com/hustvl/Senna','A',1,'','https://arxiv.org/pdf/2410.22313','main'),
 ('nuscenes',2,'SparseOccVLA','sparseoccvla','2026','avg L2 0.23',NUSC,'https://github.com/MSunDYY/SparseOccVLA','A',1,'','https://arxiv.org/pdf/2601.06474','main'),
 ('nuscenes',3,'OmniSpace','omnispace','2026','avg L2 0.28',NUSC,'','C',0,'未找到官方公开训练或规划推理代码','https://arxiv.org/pdf/2606.22617',''),
 ('nuscenes',4,'LVLDrive','lvldrive','2026','avg L2 0.29',NUSC,'','C',0,'未找到官方公开代码','https://arxiv.org/pdf/2512.24331',''),
 ('nuscenes',5,'AD-MLP','ad_mlp','2023','avg L2 0.35',NUSC,'https://github.com/E2E-AD/AD-MLP','A',1,'公开代码中的 ego-only 参照；注意排行协议与其他方法未必一致','https://arxiv.org/pdf/2305.10430','main'),

 ('wod_e2e',1,'DriveMA-4B','drivema','CoRL 2026','RFS 8.079','https://github.com/Tsinghua-MARS-Lab/DriveMA#results','https://github.com/Tsinghua-MARS-Lab/DriveMA','A',1,'','https://arxiv.org/pdf/2605.31271','main'),
 ('wod_e2e',2,'NTR','ntr','2026','RFS 8.0461','https://arxiv.org/abs/2605.31116','','C',0,'未找到官方代码','https://arxiv.org/pdf/2605.31116',''),
 ('wod_e2e',3,'RAP-DINO','rap','ICLR 2026','RFS 8.043',WOD,'https://github.com/vita-epfl/RAP','A',1,'','https://arxiv.org/pdf/2510.04333','main'),
 ('wod_e2e',4,'Poutine','poutine','2025','RFS 7.986',WOD,'','C',0,'技术报告公开，未找到对应模型代码','https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%20Special%20Mention%20-%20Poutine.pdf',''),
 ('wod_e2e',5,'AutoVLA','autovla','NeurIPS 2025','RFS 7.5566',WOD,'https://github.com/ucla-mobility/AutoVLA','A',1,'更高分的多个队伍无可审计代码；保留公开完整实现','https://arxiv.org/pdf/2506.13757','main'),

 ('hugsim',1,'WA-JEPA','wa_jepa','2026','HDS 0.4462',HUG,'https://github.com/AFARI-Research/WA-JEPA','A',1,'同一 436 场景协议重测','https://arxiv.org/pdf/2608.20974','main'),
 ('hugsim',2,'DrivoR','drivor','CVPR 2026','HDS 0.3252',HUG,'https://github.com/valeoai/DrivoR','A',1,'同一 436 场景协议重测','https://arxiv.org/pdf/2601.05083','main'),
 ('hugsim',3,'UniAD','uniad','CVPR 2023','HDS 0.3124',HUG,'https://github.com/hyzhou404/UniAD_SIM','B',1,'HUGSIM 适配推理代码；分数来自 WA-JEPA 同协议重测','https://arxiv.org/pdf/2212.10156','main'),
 ('hugsim',4,'LTF','ltf','2024','HDS 0.2310',HUG,'','C',0,'未找到该 HUGSIM 提交的独立可审计代码','https://arxiv.org/pdf/2406.15349',''),
]

sampling_fields = ['board','rank','method','venue_year','score','ranking_source_url','snapshot_date','repo_url','code_grade','selected','not_selected_reason','split_protocol_note']
with (OUT / 'sampling.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, sampling_fields)
    w.writeheader()
    for b,r,m,s,v,score,src,url,grade,sel,reason,paper,branch in rows:
        note = {'carla_lb2':'官方网页表格为空；SOTA2 转录官方提交。SENSORS/MAP 不直接比较。',
                'navsim_v1':'navtest，PDMS。', 'navsim_v2':'navhard 双阶段，EPDMS。',
                'nuscenes':'SOTA2 的 avg L2；不同原论文的碰撞计算和 ego 输入协议可能不同。',
                'wod_e2e':'WOD-E2E test，RFS。','hugsim':'WA-JEPA 表 2；436 场景、同一控制器及评分实现。',
                'bench2drive':'Bench2Drive 0.0.3，自报 DS。'}[b]
        w.writerow(dict(board=b,rank=r,method=m,venue_year=v,score=score,ranking_source_url=src,
                        snapshot_date=DATE,repo_url=url,code_grade=grade,selected='yes' if sel else 'no',
                        not_selected_reason=reason if not sel else '',split_protocol_note=note))

repo_fields = ['board','method','slug','repo_url','repo_path','commit','clone_date','clone_status','paper_url','paper_path','paper_status','code_grade','branch']
with (OUT / 'repos.csv').open('w', newline='') as f:
    w=csv.DictWriter(f,repo_fields);w.writeheader()
    for b,r,m,s,v,score,src,url,grade,sel,reason,paper,branch in rows:
        if sel:
            w.writerow(dict(board=b,method=m,slug=s,repo_url=url,repo_path=f'repos/{b}__{s}',commit='',clone_date=DATE,
                            clone_status='pending',paper_url=paper,paper_path=f'papers/{s}.pdf',paper_status='pending',code_grade=grade,branch=branch))

index = ['# 审计任务板','',f'抽样日期：{DATE}。先完成所有代码和论文备料，再启动阶段 1。','',
         '| 榜单 | 方法 | 负责代理 | 状态 | 审计页 |','| --- | --- | --- | --- | --- |']
for b,r,m,s,v,score,src,url,grade,sel,reason,paper,branch in rows:
    if sel:
        index.append(f'| {b} | {m} | pending | todo | [审计页](audits/{b}__{s}.md) |')
(OUT/'INDEX.md').write_text('\n'.join(index)+'\n')
(OUT/'questions.md').write_text('# 待决问题与已知限制\n\n- CARLA 官方网页当前表格为空，抽样依赖 SOTA2 对官方榜单的转录；SENSORS 和 MAP 成绩不可直接比较。\n- NAVSIM 官方 Hugging Face 提交数据返回 401，排名用公开论文对比表及 SOTA2；navtest 与 navhard 分开处理。\n- nuScenes 开环规划论文存在不同 L2、碰撞协议，横向名次只用来抽样，不把数值差当能力差。\n')
