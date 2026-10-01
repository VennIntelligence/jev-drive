"""Post-run presentation only; never changes frozen acceptance or its output."""
from pathlib import Path
import csv,json,ast,hashlib,shutil,collections
import numpy as np
BASE=Path('/home/ujs/mycode/jev-drive'); D=BASE/'todos/2026-09-23-lateral-followup'; O=Path(__file__).resolve().parent
A=Path('/data/runs/b2d/controller/lateral-followup/aim-g2-v2-analysis-root-v1'); F=Path('/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1'); R=Path('/data/runs/b2d/controller/lateral-followup/aim-g2-v2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(name,rows):
 with (O/name).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def f(v):return f'{float(v):.6f}'
for n in ['cases.csv','metrics.csv','recovery.csv','coverage.json','required-conditions.json','control-reasons.json','manifest.json']:
 target=O/('analysis-manifest.json' if n=='manifest.json' else n);assert not target.exists();shutil.copy2(A/n,target)
shutil.copy2(F/'plot-source-manifest.json',O/'plot-source-manifest.json')
cs=list(csv.DictReader((A/'cases.csv').open())); ms=list(csv.DictReader((A/'metrics.csv').open())); fs=list(csv.DictReader((A/'frames.csv').open())); q=json.loads((A/'required-conditions.json').read_text()); es=[json.loads(s) for s in (R/'events.jsonl').read_text().splitlines()]; assert es[-1]['kind']=='end' and es[-1]['status']=='completed'; assert sum(e['kind']=='route_end' for e in es)==6
counts=collections.Counter(c['status'] for c in q['required_conditions']); failures=[c for c in q['required_conditions'] if c['status']!='pass']
compact=[]
for c in cs:
 v=json.loads(c['validation_json']);compact.append(dict(route=c['route'],variant=c['variant'],ticks=int(c['source_rows']),truth_rows=int(c['truth_rows']),g2_pass=v['gate_pass'],cte_rms_m=v['cross_track_rms_m'],cte_p95_m=v['cross_track_p95_m'],cruise_speed_rms_mps=v['speed_rms_mps'],pose_p90_m=v['pose_p90_m'],case_wall_s=v['wall_s'],aim_requested=c['aim_requested_counts'],aim_used=c['aim_used_counts'],fallback=c['aim_fallback_counts'],reasons=c['reason_counts'],invalid_control=int(c['invalid_control_count']),unknown_reason=int(c['unknown_reason_count']),mode_errors=sum(int(c[x]) for x in ['aim_mode_missing_ticks','aim_mode_mismatch_ticks','aim_used_or_fallback_error_ticks'])))
writecsv('case-summary.csv',compact)
windows=[('26966','1','右急弯'),('24240','1','左弯'),('17563','1','S1'),('17563','2','S2')];wm={(r['route'],r['segment'],r['variant']):r for r in ms if r['band']=='window' and r['subset']=='all'}
updates=[]
for route,seg,label in windows:
 for variant in ['baseline-linear','candidate-hermite']:
  m=wm[route,seg,variant]; rows=[r for r in fs if r['route']==route and r['variant']==variant and int(m['first_frame'])<=int(r['frame'])<=int(m['last_frame'])];assert len(rows)==int(m['count'])
  for subset in ['all','update','nonupdate','unknown_update']:
   rs=[r for r in rows if subset=='all' or (subset=='update' and r['trajectory_updated']=='True') or (subset=='nonupdate' and r['trajectory_updated']=='False') or (subset=='unknown_update' and r['trajectory_updated'] not in ['True','False'])]
   d=dict(route=route,segment=seg,variant=variant,subset=subset,n=len(rs))
   for field in ['rate','raw_rate','lateral_jerk']:
    vals=[float(r[field]) for r in rs if r[field] and np.isfinite(float(r[field]))];d[field+'_n']=len(vals);d[field+'_rms']=float(np.sqrt(np.mean(np.square(vals)))) if vals else None;d[field+'_p95_abs']=float(np.percentile(np.abs(vals),95)) if vals else None
   updates.append(d)
writecsv('update-phase-supplement.csv',updates)
writecsv('all-conditions.csv',[{k:c.get(k) for k in ['condition','status','field','band','baseline','candidate','comparison','limit']} for c in q['required_conditions']])
wall=es[-1]['t']-next(e['t'] for e in es if e['kind']=='start'); cost={'successful_run_start_to_end_s':wall,'case_validation_wall_s_sum':sum(c['case_wall_s'] for c in compact),'prior_zero_case_startup_failure_s':182.18247413635254,'combined_recorded_run_attempt_s':wall+182.18247413635254,'successful_run_cases':6,'prior_run_cases':0,'boundary':'Campaign start/end includes server/map setup but end precedes final server.stop; validation.wall_s spans case actor setup to cleanup and metric serialization. Neither includes analysis/driver repair time.'};(O/'cost.json').write_text(json.dumps(cost,indent=2)+'\n')
intro='''# Hermite 六例冻结验收：未通过\n\n2026-09-23。唯一变化是 pursuit aim 的 linear→local chord-length Hermite 插值，max 前视 .5、PI .5/.25、速度、适配器和限幅不变。实际成功运行是 `aim-g2-v2`，六例原 G2 均通过、无碰撞；**147 项必要条件中 143 通过、4 失败、0 证据不足，总体 FAIL**。本轮不将 Hermite 升级为合格候选或默认控制器，不触发“筛选全通过后”的确认。\n\n此前 short .375 的旧 15% CTE 主目标失败仍保持原判。本轮是插值纹波的新机制试验，不能以基础 G2 或部分 jerk 下降追认旧目标。这里是无背景交通的导航 oracle 闭环，不是真实 TCP 模型、交互驾驶或 220 路榜单结果。\n\n## 必要条件失败\n\n'''
intro+=table(['条件','baseline','Hermite','冻结上限'],[[x['condition'],f(x['baseline']),f(x['candidate']),f(x['limit'])] for x in failures])
intro+='\n\n右急弯 CTE RMS 超余量约 0.000435m、P95 超余量约 0.004284m；即使差距较小，也按原精度判失败，不四舍五入放行。两个目标窗 emitted-rate P95 均上升，未达到各降至少 20% 的主要求。\n\n## 四窗口完整指标\n\n以下全部为预登记 pad 窗口、全帧统计，左右正号和物理量定义保持冻结协议。`B→H` 表示本轮 linear/.5 基线到 Hermite；不使用窗口平均抵消失败。\n\n'
fields=[('cte_m_rms','CTE RMS m'),('cte_m_p95_abs','CTE abs P95 m'),('cte_m_max_abs','CTE abs peak m'),('steer_rate_per_s_p95_abs','emitted-rate abs P95 /s'),('lateral_jerk_mps3_p95_abs','lat jerk abs P95 m/s³'),('lateral_accel_mps2_p95_abs','lat accel abs P95 m/s²'),('heading_deg_p95_abs','heading abs P95 °'),('speed_error_mps_rms','speed error RMS m/s'),('actual_speed_mean_mps','mean speed m/s')]
for key,title in fields:
 intro+='### '+title+'\n\n'+table(['窗口','B','H','H−B'],[[route+' '+label,f(wm[route,seg,'baseline-linear'][key]),f(wm[route,seg,'candidate-hermite'][key]),f(float(wm[route,seg,'candidate-hermite'][key])-float(wm[route,seg,'baseline-linear'][key]))] for route,seg,label in windows])+'\n\n'
intro+='''四窗物理横向 jerk P95 均下降，但这个分位数不能代表完整舒适性、安全或所有瞬态。物理 jerk 是 world acceleration 先按实际时间求导、再投当前 body；它与归一化 steer-rate 是不同量，不可互相替代。四窗 CTE RMS 均上升；S1/S2 emitted-rate P95 仍为 2/s，不能描述为完全消除动作跳变。\n\n## 覆盖、模式、故障与恢复\n\n'''
intro+=table(['case','全帧/truth','requested','used','fallback'],[[c['route']+'/'+c['variant'],str(c['ticks'])+'/'+str(c['truth_rows']),c['aim_requested'],c['aim_used'],c['fallback']] for c in compact])
intro+='\n\n六例共 2760 个控制/truth 帧，全部连续对齐；0 invalid_control、0 unknown reason、0 requested-mode 缺失/错误或 used/fallback 契约错误。Hermite 共 1380 帧：1073 帧使用 Hermite、2 帧有理由地回退 linear、305 帧没有求 aim；基线 1380 帧：1078 帧 linear、302 帧没有求 aim。无 aim 的 null 不当作 Hermite 成功。两次回退均为 `two_or_fewer_distinct_knots`；逐帧位置保留在原 frames.csv。\n\n窗口样本为右急弯 70/71、左弯 117/117、S1 70/70、S2 71/71，共 657 帧；各窗全部保留，均超过 20 帧门槛，没有低速排除。合法停车原因也不删除：trajectory_behind 共 11 帧，均在参考终点附近；stationary_trajectory 共 596 帧；stop_hold 共 3 帧。详情见 case-summary.csv 与 control-reasons.json。\n\n右急弯两臂恢复均 censored：不能称 recovery pass，也不能证明恢复不恶化。固定 post10m CTE RMS 0.317045→0.302212m，单独的残差保护条件通过。另三窗恢复时间基线/Hermite 相同：左弯 0s、S1 0.1s、S2 0.7s；没有新增截尾，满足原时间余量。完整路线图保留未恢复区域，未为候选延长观察窗。\n\n所有 147 项原始判定见 [all-conditions.csv](all-conditions.csv) 与 [required-conditions.json](required-conditions.json)。[update-phase-supplement.csv](update-phase-supplement.csv) 额外列出 update/non-update/unknown 与全窗 raw/emitted rate、jerk 分布；这是按已登记字段的补充展示，不增加或替换验收条件，同期差不能当作更新的独立因果效果。\n\n## 完整路线与图\n\n'''
intro+=table(['case','G2','ticks','全程CTE RMS m','全程CTE P95 m','巡航speed RMS m/s','case wall s'],[[c['route']+'/'+c['variant'],'PASS',c['ticks'],f(c['cte_rms_m']),f(c['cte_p95_m']),f(c['cruise_speed_rms_mps']),f(c['case_wall_s'])] for c in compact])
intro+='\n\n基础 G2 全程含停车阶段，而逐窗使用固定 station 窗；两个范围不能混用。下面为首例前冻结 plot_aim 原样输出，未为结果改绘图脚本或截取有利片段。每图同列两臂，共 7 PNG + 7 PDF，3 份全路线逐帧绘图 CSV。\n\n'
for stem in ['route-26966-window-1','route-24240-window-1','route-17563-window-1','route-17563-window-2','route-26966-full','route-24240-full','route-17563-full']:
 intro+=f'[{stem} PNG]({F}/{stem}.png) · [PDF]({F}/{stem}.pdf)\n\n'
intro+='''窗口图包括等比例 world XY、CTE、raw/emitted steer、实际/参考速度、横向加速度/jerk、steer-rate、heading 及更新/插值事件；full 图覆盖起步、转弯、停车、恢复观察区。原始脉冲和终点安全制动均保留，没有平滑。\n\n## 运行成本与来源\n\n'''
intro+=f'成功六例 campaign start→end 为 **{wall:.6f}s**；六例 validation.wall_s 合计 **{cost["case_validation_wall_s_sum"]:.6f}s**。前者包括 CARLA/map 启动开销，但 end 在最终 server.stop 之前；后者是每例 actor setup 至 cleanup/指标写入的计时，并不是只计 simulation step。\n\n此前 aim-g2-v1 因 CARLA 端口启动失败，**0 例**，另记 182.182474s 基础设施成本；不拼入六例控制表现或剔除成本。两次已记录 start→end 合计 {cost["combined_recorded_run_attempt_s"]:.6f}s，不包含驱动恢复准备及本次离线分析时间，也不作 harness 加速因果结论。\n\n'
intro+=f'原始闭合运行：[{R}]({R})。冻结分析：[{A}]({A})。图/全帧 CSV：[{F}]({F})。运行源码 18d6790；协议 SHA 624a72c286282efec63538e1ddaa26c05438f98cc9aa665110fd165ea30e6143，验收器 c940883c4753a1a8409649272bd28be31da61f9313df6e58949ec0479745fef3，plotter ac4add41911b8199d03cfda8fc77cdc781e7c188810e3fb33a6d1d2a236fdd42。原始和图文版本完整保留，manifest 列出输入与输出 SHA256。\n\n'
intro+='理想圆轨迹插值诊断与历史真实 state shadow 的相反现象均保持原样。本轮闭环结果只否定该候选满足预登记收益/保护条件，不能据此证明所有几何插值无用，也不能证明侧向运动模型已是唯一主因。后续其他机制若获授权，必须保持独立协议和结果，不能在本轮调阈值补判。\n'
(O/'README.md').write_text(intro)
# Verify every frozen analyzer/plot input and output hash, without modifying either source.
verification=[]
for manifest in [A/'manifest.json',F/'plot-source-manifest.json']:
 j=json.loads(manifest.read_text())
 for section in ['inputs','outputs']:
  for path,info in j.get(section,{}).items():
   p=Path(path); p=p if p.is_absolute() else manifest.parent/p
   expected=info['sha256'] if isinstance(info,dict) else info
   assert p.exists() and sha(p)==expected,(p,expected)
   verification.append({'path':str(p),'sha256':expected})
(O/'source-verification.json').write_text(json.dumps({'verified':verification,'count':len(verification)},indent=2)+'\n')
manifest={'raw_root':str(R),'analysis_root':str(A),'figures_root':str(F),'conditions':dict(counts),'status':q['status'],'sources':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [A/'manifest.json',F/'plot-source-manifest.json',R/'events.jsonl']},'outputs':{str(p.relative_to(O)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(O.iterdir()) if p.is_file() and p.name!='manifest.json'}}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'out':str(O),'conditions':dict(counts),'cost':cost,'hash_checks':len(verification)},indent=2))
