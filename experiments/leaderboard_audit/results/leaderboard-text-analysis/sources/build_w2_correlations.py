#!/usr/bin/env python3
"""Mechanical, tie-aware Spearman summaries for W2."""
import csv
from math import sqrt
from pathlib import Path

ROOT=Path('/data/hack_audit/out2')
with (ROOT/'w2_cross_board.csv').open(newline='') as f: rows=list(csv.DictReader(f))
index={(r['method'],r['board'],r['protocol/split'],r['metric']):float(r['score']) for r in rows}

def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); out=[0.0]*len(values);j=0
    while j<len(order):
        k=j+1
        while k<len(order) and values[order[k]]==values[order[j]]:k+=1
        rank=(j+1+k)/2
        for t in order[j:k]:out[t]=rank
        j=k
    return out

def rho(x,y):
    a,b=ranks(x),ranks(y); ma=sum(a)/len(a);mb=sum(b)/len(b)
    numerator=sum((u-ma)*(v-mb) for u,v in zip(a,b))
    denominator=sqrt(sum((u-ma)**2 for u in a)*sum((v-mb)**2 for v in b))
    return numerator/denominator if denominator else float('nan')

def get(board,protocol,metric):
    return {m:score for (m,b,p,k),score in index.items() if (b,p,k)==(board,protocol,metric)}

BD='base-set closed-loop 220 routes';L6='Longest6 v2, CARLA LB2 local'
NV1='navtest, NAVSIM v1';NV2H='navhard-two-stage, NAVSIM v2, corrected EPDMS'
NV2T='navtest, NAVSIM v2, corrected EPDMS'
open_b2d='base-set open-loop 2s@2Hz';closed_b2d='base-set closed-loop 220 routes'

def match(a,b,exclusions=()):
    names=sorted((a.keys() & b.keys())-set(exclusions))
    return names,[a[n] for n in names],[b[n] for n in names]

lines=['# W2 跨榜/跨协议机械相关','',
'数据：[`w2_cross_board.csv`](w2_cross_board.csv)。`papers/` 相对于 `/data/hack_audit/`；`PDF p.` 指 PDF 文件页序，不是印刷页码。长表记录已发表分数，没有运行模型。',
'',
'计算：对两个协议与指标列按 `method` 精确匹配；每列按原始数值升序赋平均秩（并列取平均），再计算两列秩的 Pearson 相关，即 Spearman ρ。L2 数值越低越好，其余所列主指标越高越好；这里没有倒转 L2 的符号。`n < 5` 只列配对数值。',
'',
'## n ≥ 5 的配对','',
'| 列 A | 列 B | n | Spearman ρ | 方法与数值（A → B） |','|---|---|---:|---:|---|']
computed=[]
def calc(label_a,a,label_b,b,exclusions=()):
    names,x,y=match(a,b,exclusions)
    if len(names)<5: raise ValueError(f'need n>=5: {label_a}/{label_b} n={len(names)}')
    v=rho(x,y)
    vals='；'.join(f'{m} {aa:g}→{bb:g}' for m,aa,bb in zip(names,x,y))
    lines.append(f'| {label_a} | {label_b} | {len(names)} | {v:+.4f} | {vals} |')
    computed.append((label_a,label_b,len(names),v,names))

seed_l2=get('bench2drive',open_b2d,'Avg L2 (m)')
for metric in ['Driving Score','Success Rate (%)','Efficiency','Comfortness']:
    calc('Bench2Drive base-set 开环 Avg L2',seed_l2,f'Bench2Drive base-set 闭环 {metric}',get('bench2drive',closed_b2d,metric))

v1=get('navsim_v1',NV1,'PDMS');v2=get('navsim_v2',NV2H,'EPDMS')
calc('NAVSIM v1 navtest PDMS',v1,'NAVSIM v2 navhard two-stage 修复后 EPDMS',v2,exclusions=['DrivoR (+134k SimScale)'])

bds=get('bench2drive',BD,'Driving Score');bsr=get('bench2drive',BD,'Success Rate (%)')
lds=get('carla_lb2',L6,'Driving Score');lrc=get('carla_lb2',L6,'Route Completion (%)')
for la,a in [('Bench2Drive DS',bds),('Bench2Drive SR',bsr)]:
    for lb,b in [('CARLA Longest6 v2 DS',lds),('CARLA Longest6 v2 RC',lrc)]:
        calc(la,a,lb,b)

wod=get('wod_e2e','test, vision-based E2E','RFS Overall')
calc('NAVSIM v1 navtest PDMS',v1,'WOD-E2E test RFS Overall',wod)

lines += ['',
'`DrivoR (+134k SimScale)` 与 TOAD 表中的 `DrivoR` 在两榜上均为 94.6/54.6；相关计算只计一次，原始两种出处仍保存在长表。Bench2Drive ↔ Longest6 的 DS 对 DS 相关含 TFv6 Table 5 的多个同论文传感器/骨干变体（ρ 约 +0.9296），并非独立方法观测。该表中的 LEAD 为论文估计的特权 expert：Bench2Drive DS 96.8 是估计值，不是统一公开提交。TOAD 的基座与搜索变体、DriveMA 的两个模型规模也分别计入，观察值并非统计独立。配对允许不同训练数据或 checkpoint；`same_checkpoint` 列记录证据，ρ 本身只描述表内排序。',
'',
'## n < 5 的配对数据','',
'以下协议各自单列，没有把 NAVSIM v2 navtest 与 navhard、修复前与修复后、CARLA 官方 MAP 与 SENSORS、或不同 nuScenes 指标实现合并。',
'',
'| 列 A | 列 B | n | 方法与数值（A → B） |','|---|---|---:|---|']
def small(label_a,a,label_b,b,exclusions=()):
    names,x,y=match(a,b,exclusions)
    if len(names)>=5: raise ValueError(f'expected n<5: {label_a}/{label_b} n={len(names)}')
    vals='；'.join(f'{m} {aa:g}→{bb:g}' for m,aa,bb in zip(names,x,y)) or '无共同方法'
    lines.append(f'| {label_a} | {label_b} | {len(names)} | {vals} |')

small('NAVSIM v1 navtest PDMS',v1,'NAVSIM v2 navtest 修复后 EPDMS',get('navsim_v2',NV2T,'EPDMS'))
small('NAVSIM v1 navtest PDMS',v1,'HUGSIM 436 场景 HD-Score',get('hugsim','436 scenarios, common controller, zero-shot','HD-Score'))
small('NAVSIM v2 navhard 修复后 EPDMS',v2,'HUGSIM 436 场景 HD-Score',get('hugsim','436 scenarios, common controller, zero-shot','HD-Score'))
for track in ['MAP','SENSORS']:
    small('Bench2Drive DS',bds,f'CARLA LB2 official {track} DS',get('carla_lb2',f'official test, {track} track','Driving Score'))
small('NAVSIM v1 navtest PDMS',v1,'WOD-E2E validation RFS',get('wod_e2e','validation','RFS Overall'))
small('Bench2Drive DS',bds,'NAVSIM v1 navtest PDMS',v1)
small('Bench2Drive DS',bds,'NAVSIM v2 navtest 修复后 EPDMS',get('navsim_v2',NV2T,'EPDMS'))
small('Bench2Drive DS',bds,'HUGSIM 436 场景 HD-Score',get('hugsim','436 scenarios, common controller, zero-shot','HD-Score'))
small('Bench2Drive DS',bds,'WOD-E2E test RFS',wod)
small('NAVSIM v2 navhard 修复后 EPDMS',v2,'WOD-E2E test RFS',wod)
small('NAVSIM v2 navhard 修复后 EPDMS',v2,'WOD-E2E validation RFS',get('wod_e2e','validation','RFS Overall'))
for ns_protocol in ['val, ST-P3 metric, 1/2/3s mean','val, OmniSpace Table 1 metric, 1/2/3s mean','val, 1/2/3s mean']:
    small('Bench2Drive DS',bds,f'nuScenes {ns_protocol} Avg L2',get('nuscenes',ns_protocol,'Avg L2 (m)'))
small('nuScenes ST-P3 Avg L2',get('nuscenes','val, ST-P3 metric, 1/2/3s mean','Avg L2 (m)'),'NAVSIM v1 navtest PDMS',v1)
for split in ['KITTI360','nuScenes','PandaSet','Waymo']:
    small('NAVSIM v1 navtest PDMS',v1,f'HUGSIM zero-shot {split} HDS (%)',get('hugsim',f'zero-shot {split}, TOAD Table 4','HD-Score (%)'))

lines += ['',
'## 榜单官方论文自身的开环/闭环分析','',
'- **Bench2Drive 官方论文**：Table 3 同时列出 2 秒、2 Hz 的开环 Avg. L2 与 220 条路线的闭环 DS/SR 等。正文称 UniAD-Base 的 L2 低于 VAD、闭环表现较差；同表中 UniAD-Base 的 DS/SR 高于 VAD，而 Efficiency/Comfortness 低于 VAD，原句没有指明所说的闭环指标。原文没有报告总体相关系数。出处：`papers/bench2drive_benchmark.pdf`，Table 3 与其后讨论，PDF p.9。',
'- **NAVSIM v1 官方论文**：Section 4.1 在 396 个 navmini 场景上比较 37 个规则式与 114 个学习式 planner 的 nuPlan closed-loop score（CLS）对 PDMS 和 nuPlan open-loop score（OLS）的关系。Fig. 3/4 同时给 Spearman 与 Pearson 图示；正文称 PDMS 对 CLS 的相关性在五类 planner 中均高于 OLS。图中没有印出每个系数的精确数值，本表不从图像估读。出处：`papers/ltf.pdf`，Fig. 3，PDF p.6；Fig. 4，PDF p.7。',
'- **NAVSIM v2 官方论文**：Section 4.1 在 83 个 planner、244 个 Stage 1 与 4164 个 Stage 2 观察上比较 EPDMS、ADE、OLS 与 8 秒 nuPlan CLS；这里用于相关性实验的是删去 TLC/LK/EC 的简化 EPDMS。文中给两阶段 2×4 秒与单阶段基线的 Pearson `r=0.89`（`R²=0.8`）和 `r=0.83`（`R²=0.7`）；Fig. 4 还展示 Spearman 排序相关，但没有印出精确系数。出处：`papers/navsim_v2_benchmark.pdf`，Fig. 3 与 Section 4.1，PDF p.6；Fig. 4 与续文，PDF p.7。',
'',
'## 已知边界','',
'1. `NAVSIM v2 navtest` 是单阶段旧测试集上的扩展指标；`navhard-two-stage` 是官方两阶段榜单。SparseDriveV2、DriveFuture、WA-JEPA 在 navtest 表中明确区分修复前的 `EPDMS*` 与修复后的 `EPDMS`。NTR 的 navtest EPDMS 90.9 未注明修复状态，故独列。DrivoR Table 14 也分别列修复前后数值；这里用后者。',
'2. 不同论文重报的同名方法可有版本或四舍五入差异。长表只选一个可定位来源作为每个键的主记录；例如 LTF v2 navhard 用官方修复后快照 25.1，RAP 早期表的修复前 23.12 不混入。',
'3. Bench2Drive 官方开环/闭环是同一训练集下基准作者重评；跨榜值多为不同模型训练、独立提交或未知 checkpoint。尤其 `RAP-DINO` 的 Bench2Drive 结果实际为 `RAP-ResNet`，保留分名而不配对。',
'4. HUGSIM 的 WA-JEPA Table 2 是统一 436 场景，TOAD Table 4 是分 KITTI360/nuScenes/PandaSet/Waymo 的另一协议；nuScenes L2 也有不同实现和输入协议。没有跨这些不同列凑足样本数。',
'5. 这些相关性是方法级已发表数字的描述统计，含多种模型变体和来源，不附显著性检验，也不作因果或能力解读。',
'']
(ROOT/'w2_correlations.md').write_text('\n'.join(lines))
print(f'wrote {len(computed)} correlations: '+', '.join(f'{n}/{v:+.3f}' for _,_,n,v,_ in computed))
