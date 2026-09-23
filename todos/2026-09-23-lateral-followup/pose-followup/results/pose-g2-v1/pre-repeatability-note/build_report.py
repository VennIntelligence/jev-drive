"""Presentation wrapper: copies frozen decisions; never changes acceptance."""
from pathlib import Path
import csv,json,hashlib,shutil,collections
BASE=Path('/home/ujs/mycode/jev-drive');D=BASE/'todos/2026-09-23-lateral-followup/pose-followup';O=Path(__file__).resolve().parent
R=Path('/data/runs/b2d/controller/lateral-followup/pose-g2-v1');A=Path('/data/runs/b2d/controller/lateral-followup/pose-g2-v1-analysis-v1');F=Path('/data/runs/b2d/controller/lateral-followup/pose-g2-v1-figures-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def write(name,rows):
 with (O/name).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(v):return '缺失' if v is None or v=='' else f'{float(v):.6f}'
for n in ['cases.csv','metrics.csv','recovery.csv','coverage.json','required-conditions.json','control-reasons.json','pose-contract.json','manifest.json']:
 p=O/('analysis-manifest.json' if n=='manifest.json' else n);assert not p.exists();shutil.copy2(A/n,p)
for p in F.glob('*.png'):shutil.copy2(p,O/p.name)
shutil.copy2(F/'manifest.json',O/'figures-manifest.json');shutil.copy2(F/'plot-notes.json',O/'plot-notes.json')
cs=read(A/'cases.csv');ms=read(A/'metrics.csv');fs=read(A/'frames.csv');q=json.loads((A/'required-conditions.json').read_text());events=[json.loads(s) for s in (R/'events.jsonl').read_text().splitlines()]
assert events[-1]['kind']=='end' and events[-1]['status']=='completed' and sum(e['kind']=='route_end' for e in events)==6
counts=collections.Counter(c['status'] for c in q['required_conditions']);failed=[c for c in q['required_conditions'] if c['status']!='pass']
assert counts==dict(pass_=155,fail=1) if False else counts=={'pass':155,'fail':1}
write('all-conditions.csv',[{k:c.get(k) for k in ['condition','status','field','band','baseline','candidate','comparison','limit']} for c in q['required_conditions']])
compact=[];reason_totals=collections.Counter();pose_totals=collections.Counter()
for c in cs:
 v=json.loads(c['validation_json']);rr=[r for r in fs if r['route']==c['route'] and r['variant']==c['variant']];reasons=collections.Counter(r['reason'] for r in rr);statuses=collections.Counter(json.loads(r['pose_status_json'])['lateral_reason'] for r in rr);reason_totals.update(reasons);pose_totals.update(statuses)
 compact.append(dict(route=c['route'],variant=c['variant'],ticks=int(c['source_rows']),truth_rows=int(c['truth_rows']),g2_pass=v['gate_pass'],cte_rms_m=v['cross_track_rms_m'],cte_p95_m=v['cross_track_p95_m'],speed_rms_mps=v['speed_rms_mps'],pose_p90_m=v['pose_p90_m'],pose_heading_p90_deg=v['pose_heading_deg']['p90'],wall_s=v['wall_s'],pose_contract_errors=int(c['pose_contract_error_count']),invalid_control=int(c['invalid_control_count']),unknown_reason=int(c['unknown_reason_count']),mode_error=sum(int(c[k]) for k in ['aim_mode_missing_ticks','aim_mode_mismatch_ticks','aim_used_or_fallback_error_ticks']),reasons=json.dumps(reasons,sort_keys=True),pose_statuses=json.dumps(statuses,sort_keys=True)))
write('case-summary.csv',compact)
wm={(r['route'],r['segment'],r['variant']):r for r in ms if r['subset']=='all' and r['band']=='window'};windows=[('26966','1','右急弯'),('24240','1','左弯'),('17563','1','S1'),('17563','2','S2')]
comparison=[]
fields=['cte_m_rms','cte_m_p95_abs','cte_m_max_abs','heading_deg_p95_abs','speed_error_mps_rms','actual_speed_mean_mps','steer_rate_per_s_p95_abs','lateral_accel_mps2_p95_abs','lateral_jerk_mps3_p95_abs','pose_position_error_m_rms','pose_position_error_m_p90_abs','pose_position_error_m_p95_abs','pose_position_error_m_max_abs','pose_left_error_m_mean','pose_forward_error_m_mean','pose_heading_error_deg_p95_abs']
for route,seg,label in windows:
 b,c=(wm[route,seg,v] for v in ['baseline-zero','candidate-fixed-k'])
 for field in fields:
  bv,cv=float(b[field]),float(c[field]);comparison.append(dict(route=route,segment=seg,metric=field,baseline=bv,candidate=cv,delta=cv-bv,percent_change=(cv/bv-1)*100 if bv else None,baseline_count=b['count'],candidate_count=c['count']))
write('window-comparison.csv',comparison)
# Preserve every phase, including approach/gap guards and terminal hold; no result-dependent phase selection.
phasefields=['route','variant','segment','kind','band','subset','count','pose_position_error_m_rms','pose_position_error_m_p90_abs','pose_position_error_m_p95_abs','pose_position_error_m_max_abs','pose_left_error_m_mean','pose_forward_error_m_mean','pose_heading_error_deg_p95_abs','cte_m_rms','heading_deg_p95_abs','lateral_jerk_mps3_p95_abs']
write('pose-phase-summary.csv',[{k:r.get(k) for k in phasefields} for r in ms if r['subset']=='all'])
wall=events[-1]['t']-next(e['t'] for e in events if e['kind']=='start');cost=dict(campaign_start_to_end_s=wall,case_wall_s_sum=sum(c['wall_s'] for c in compact),cases=6,ticks=sum(c['ticks'] for c in compact),boundary='campaign end precedes final server.stop; case validation.wall_s includes actor setup, cleanup, telemetry parsing and metrics; no analysis/development time included');(O/'cost.json').write_text(json.dumps(cost,indent=2)+'\n')
text='''# 固定 k 后轴位姿补偿：局部收益明确，冻结验收未通过

2026-09-23。`pose-g2-v1` 的六例已闭合；六例全部通过基础 G2、无碰撞。冻结验收 **156 项：155 通过、1 失败、0 证据不足，总体 FAIL**。唯一失败是右急弯真实车身航向跟踪 P95 超过原 +1° 余量。**不触发反序六例确认，不追加新候选，不升级默认。**

本轮共同基线为 pursuit、linear aim、max .5、PI .5/.25；唯一变化是后轴侧向传播系数 0→0.010659832 s²/m，在原 GNSS 融合之前加入补偿。系数不重新拟合，运行时不读真值。Hermite 与旧 short .375 的失败及协议仍各自保留。本轮不借用其他阶段的基线：尤其当前 S1 基线 CTE RMS 为 .225250m，旧 Hermite 轮为 .235236m；只能使用当前相邻配对，不能拼接挑优。

## 唯一失败与主目标

'''
text+=table(['必要条件','基线','固定 k','冻结上限'],[[c['condition'],fmt(c['baseline']),fmt(c['candidate']),fmt(c['limit'])] for c in failed])
right=next(r for r in comparison if r['route']=='26966' and r['metric']=='cte_m_rms')
text+=f'\n\n右急弯 heading abs P95 从 4.271547° 升到 5.449551°，增加 1.178004°，超允许上限约 0.178004°。该量是 **truth 车身 yaw 相对参考线 5m 居中弦方向**，不是 pose 的估计航向误差，也不是实际速度方向 course 的误差。后续只读 body/course 分解属于解释性附录，不能替换原门槛或改判通过。\n\n该窗 CTE RMS .558694→.429021m，下降 **{-right["percent_change"]:.3f}%**，达到本轮预登记 15% 主目标；CTE P95 .849224→.712170m、post10m RMS .317045→.101648m 也改善。这些局部收益是真实本轮闭环观测，但不能抵消另一项必要条件失败。\n\n## 四个固定窗口：收益与代价同时保留\n\n所有值使用完整 pad 窗口全部帧，不按低速或 gyro 删除样本。表内 B→K 为本轮基线→固定 k；P95 除有符号均值外均为绝对值分位数。所有精确数值及增量见 [window-comparison.csv](window-comparison.csv)。\n\n'
groups=[('路径与车身航向',[('cte_m_rms','CTE RMS m'),('cte_m_p95_abs','CTE P95 m'),('cte_m_max_abs','CTE peak m'),('heading_deg_p95_abs','heading P95 °')]),('速度与动作',[('speed_error_mps_rms','speed RMS m/s'),('actual_speed_mean_mps','mean speed m/s'),('steer_rate_per_s_p95_abs','steer-rate P95 /s')]),('物理横向运动',[('lateral_accel_mps2_p95_abs','accel P95 m/s²'),('lateral_jerk_mps3_p95_abs','jerk P95 m/s³')]),('定位位置范数',[('pose_position_error_m_rms','RMS m'),('pose_position_error_m_p90_abs','P90 m'),('pose_position_error_m_p95_abs','P95 m'),('pose_position_error_m_max_abs','max m')]),('定位有符号偏差与估计航向',[('pose_left_error_m_mean','body-left mean m'),('pose_forward_error_m_mean','body-forward mean m'),('pose_heading_error_deg_p95_abs','pose heading P95 °')])]
for title,columns in groups:
 text+='### '+title+'\n\n'+table(['窗口']+[label for _,label in columns],[[route+' '+label]+[fmt(wm[route,seg,'baseline-zero'][field])+'→'+fmt(wm[route,seg,'candidate-fixed-k'][field]) for field,_ in columns] for route,seg,label in windows])+'\n\n'
text+='''四窗 CTE RMS 均下降，但不能写成“所有量更好”。右急弯、左弯及 S1 横向加速度 P95 上升；右急弯和 S1 jerk P95 上升；S2 航向 P95 小幅上升。全部数值照常列出。emitted-rate/横向加速度仍满足原保护余量；jerk 本轮是完整报告项，没有另加事后门槛，也不是官方 DS 的加分项。

**左弯定位退化不能隐藏：** pose position RMS .094703→.135926m，P95 .154144→.258880m，而路径 CTE RMS .107807→.078434m。定位估计和车辆跟踪是不同量；当前全 G2 pose 门槛仍通过，本轮没有预设逐窗 pose 非恶化门槛，不能事后新增或把这项代价改叫改善。

所有 signed pose 数值为 **估计后轴−真值后轴** 在 **truth body left/forward** 上的投影：left=[sin(truth yaw),−cos(truth yaw)]。这与路径 CTE 的 reference normal 不同，也与部分旧离线 shadow 的 reference-left 不能直接拼接。物理 jerk 先对 world acceleration 按实际 dt 求导，再投当前 body；steer-rate 是归一化命令变化率，二者不能互相替代为舒适性结论。

## 恢复、覆盖与边界

右急弯基线恢复 censored；候选在原冻结观察窗内记录到恢复，时间 .45s。基线未恢复不能填0、不能构造有限“改善百分比”，也不能把本轮恢复诊断自动变成额外 pass。S1 .05→.05s、S2 .70→.60s、左弯 0→0s；原始恢复表全部保留。本轮恢复/jerk 沿用诊断地位，不移植 Hermite 阶段的新门槛。

四窗样本 B/K：右急弯70/70、左弯117/117、S1 70/70、S2 71/72，总657帧；不同帧数来自各自按 truth station 进入/退出固定窗口，不能重采样成相等。六例完整遥测与 truth 共2760帧，连续且逐帧对齐，未剔除起步、停止、低速或瞬态。窗口均达到至少20帧并完整进出，原独立覆盖检查通过。

[pose-phase-summary.csv](pose-phase-summary.csv) 保留全部 full_route、entry/core/exit/post10m、approach/gap straight guards、endpoint_last5m 与 terminal_hold 的定位/CTE/航向/jerk，不能因窗口外退化而省略。完整原始 [metrics.csv](metrics.csv) 还包含 moving 辅助统计、幅值和 slew 限制比例、raw 动作等，不用于替代全帧主判定。

## 六例基础 G2 与契约

'''
text+=table(['case','ticks','G2','全程CTE RMS m','全程CTE P95 m','巡航speed RMS m/s','pose P90 m','case wall s'],[[c['route']+'/'+c['variant'],c['ticks'],'PASS',fmt(c['cte_rms_m']),fmt(c['cte_p95_m']),fmt(c['speed_rms_mps']),fmt(c['pose_p90_m']),fmt(c['wall_s'])] for c in compact])
text+='\n\n六例均 0 collision、0 invalid_control、0 unknown reason、0 pose contract error、0 模式缺失/错误/无理由 fallback；同一 case 内 raw motion/control 时间与传感器帧一致，独立原始传感器 replay 通过。两臂显式 linear aim，公共配置审计确认仅固定系数不同。原 G2 停车保持、终点、pose heading、速度及全程CTE全部通过；这些不取代逐窗条件。\n\n'
text+=f'合法状态仍全保留：{dict(reason_totals)}。补偿状态总计 {dict(pose_totals)}；首帧 no_interval 不虚构已应用补偿，k=0 的 disabled 与候选 applied 明确分开。逐例计数在 [case-summary.csv](case-summary.csv)，逐帧 sensor 重算原始表在 [{A}/pose-recomputed.csv]({A}/pose-recomputed.csv)。真实错误计数为0不等于删除合法安全制动。\n\n'
text+='''## 图与数据

下列文章图来自首例前冻结 plot_pose.py 和共享样式，全部8组 PNG/PDF 保留；完整路线展示启动到停车全过程，四窗分别展示，不只挑右弯。两臂使用同一坐标轴，时间按各自首选中样本对齐，不做轨迹时间插值或平滑。速度点线为独立参考；pose 面板明确 truth body left。图形只解释原判定，不创造门槛。

'''
for stem,caption in [('four-window-summary','四窗的路径、动作、物理运动与定位分别比较：CTE 均下降，但左弯定位范数增加，部分 accel/jerk 上升。'),('route-26966-window-1','右急弯：路径外侧残差减少，真实车身航向误差 P95 增加并触发唯一失败；位姿误差下降不能替代航向守护。'),('route-24240-window-1','左弯：路径 CTE 下降而定位范数上升，必须保留这一方向相反的代价。'),('route-17563-window-1','S1：本轮相邻共同基线与候选，保留正负曲率切换和全部瞬态。'),('route-17563-window-2','S2：保留限速率段、符号切换和不同进入/退出帧数，不合并两个 S 抵消代价。')]:
 text+=f'![{stem}]({stem}.png)\n\n{caption} [PDF]({F}/{stem}.pdf)\n\n'
for route in ['26966','24240','17563']:text+=f'完整路线 {route}：[PNG](route-{route}-full.png) · [PDF]({F}/route-{route}-full.pdf)。\n\n'
text+=f'全程绘图数据 [{F}/plot-data.csv]({F}/plot-data.csv)，四窗原样数据 [window-plot-data.csv]({F}/window-plot-data.csv)，汇总 [summary-plot-data.csv]({F}/summary-plot-data.csv)。原始全部列/行保留，无平滑、无缺帧填0；绘图中的 NaN 只断线。\n\n## 时间成本、来源与结论范围\n\n'
text+=f'成功6例 campaign start→end **{wall:.6f}s**；6例 validation.wall_s 合计 **{cost["case_wall_s_sum"]:.6f}s**。前者包含server/map启动但 end早于最终server.stop；后者包括actor setup、cleanup、遥测解析和指标处理，不是纯tick时间；均未包含实现、测试、离线分析和之前驱动恢复时间。本轮没有额外失败attempt或反序确认；此前 Hermite 的0例基础设施失败成本在其旧报告单独保留，不混入本候选数值。\n\n'
text+=f'冻结协议：[protocol.md](../../protocol.md)。运行源/配置/环境完整快照：[{R}/provenance]({R}/provenance)；原始[{R}]({R})，验收[{A}]({A})，绘图[{F}]({F})。图文 wrapper 源、输入/输出 SHA 与验证详见 manifest.json；冻结分析/绘图及共享样式字节未修改。\n\n'
text+='''本轮支持“固定经验补偿在当前 MKZ、6/8m/s、三路线 oracle 适配链上能减少部分定位与路径残差”，同时显示左弯定位代价与右急弯车身航向代价。它仍是开发筛选失败，不是默认资格、跨车标定、真实 TCP 模型提升、NPC 安全证明或全220榜单提升。body/course 等后续只读解释另附证据，不改变此处冻结结论。
'''
(O/'README.md').write_text(text)
# All source/output evidence under original closed manifests must still match.
verified=[]
for mf in [A/'manifest.json',F/'manifest.json']:
 j=json.loads(mf.read_text())
 for section in ['inputs','outputs']:
  for key,value in j[section].items():
   p=Path(key);p=p if p.is_absolute() else mf.parent/p;expected=value['sha256'] if isinstance(value,dict) else value
   assert p.exists() and sha(p)==expected,(p,expected);verified.append(dict(path=str(p),sha256=expected))
assert read(A/'frames.csv')==read(F/'plot-data.csv')
(O/'verification.json').write_text(json.dumps(dict(hash_checks=len(verified),checks=verified,plot_rows_preserved=2760,conditions_copied_not_recomputed=True),indent=2)+'\n')
manifest=dict(raw_root=str(R),analysis_root=str(A),figures_root=str(F),status=q['status'],conditions=dict(counts),sources={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in [A/'manifest.json',F/'manifest.json',R/'events.jsonl']},outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(O.iterdir()) if p.is_file() and p.name!='manifest.json'})
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(out=str(O),conditions=dict(counts),cost=cost,reasons=dict(reason_totals),pose_status=dict(pose_totals),hash_checks=len(verified)),indent=2))
