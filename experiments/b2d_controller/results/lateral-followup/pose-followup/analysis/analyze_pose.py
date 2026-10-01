#!/usr/bin/env python3
"""分析固定四个转弯窗口；读取独立 run-root，不导入 CARLA 或运行控制器。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from pose_contract import audit_pose, pose_features, FIXED_K, KEY

ROUTES = ('26966', '17563', '24240')
VARIANTS = ('baseline-zero', 'candidate-fixed-k')
BASE = Path(__file__).resolve().parent


def clocks_agree(records, trace):
    """GameTime and world clock may differ by a constant case-local offset."""
    if not records or len(records)!=len(trace):return False
    offsets=np.asarray([a.get('sim_time',np.nan)-b.get('sim_time',np.nan) for a,b in zip(records,trace)])
    return bool(np.isfinite(offsets).all() and np.ptp(offsets)<=1e-8)


def common_config(config):
    expected=dict(lookahead='max',max_lookahead_time_s=.5,longitudinal_mode='pi',pi_kp=.5,pi_ki=.25,
                  aim_interpolation='linear',speed_window='near',steer_rate=2.,max_steer=.8)
    defaults=dict(speed_window='near',steer_rate=2.,max_steer=.8)
    return all(config.get(k,defaults.get(k))==v for k,v in expected.items())


def invalid_control(row, config):
    """非法输出或明确输入故障；trajectory_behind 的合法安全制动不等同非法控制。"""
    values=[row.get(k,np.nan) for k in ('steer','throttle','brake')]
    if not np.isfinite(values).all():return True
    steer,throttle,brake=values
    if abs(steer)>config.get('max_steer',.8)+1e-8:return True
    if not -1e-8<=throttle<=config.get('max_throttle',.75)+1e-8:return True
    if not -1e-8<=brake<=config.get('max_brake',1.)+1e-8:return True
    if throttle>1e-8 and brake>1e-8:return True
    reason=str(row.get('reason',''))
    if reason in ('motion_gap','time_regression','future_trajectory','trajectory_outside_history','duplicate_tick'):
        return True
    return any(token in reason for token in ('invalid','stale','timestamp_discontinuity',
        'compass_dropout_timeout','compass_uninitialized'))


def aim_mode_evidence(records, expected):
    fields=('aim_interpolation','aim_interpolation_used','aim_interpolation_fallback')
    missing=sum(any(k not in r for k in fields) for r in records)
    mismatch=sum(r.get('aim_interpolation')!=expected for r in records)
    used_errors=0
    for r in records:
        used,fallback=r.get('aim_interpolation_used'),r.get('aim_interpolation_fallback')
        has_aim=r.get('aim_xy') is not None
        if used is None:
            used_errors+=bool(has_aim or fallback is not None)
        elif used not in ('linear','hermite'):
            used_errors+=1
        elif expected=='linear':
            used_errors+=bool(used!='linear' or fallback is not None)
        elif used=='linear':
            used_errors+=not (isinstance(fallback,str) and bool(fallback))
        else:
            used_errors+=fallback is not None
    return dict(aim_mode_missing_ticks=missing,aim_mode_mismatch_ticks=mismatch,
                aim_used_or_fallback_error_ticks=int(used_errors))


def project_rows(records, xy, steer_limit, steer_rate=2.):
    """CTE 左正；heading 是 CARLA 右正 yaw 减道路 yaw。"""
    xy = np.asarray(xy, float)
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
    d = np.diff(xy, axis=0)
    length = np.linalg.norm(d, axis=1)
    station = np.r_[0., np.cumsum(length)]
    rows, previous = [], None
    for r in records:
        rate = np.nan
        updated = None
        raw_rate = np.nan
        slew_clipped = amplitude_clipped = None
        if previous is not None:
            dt = r['sim_time'] - previous['sim_time']
            if dt > 0 and r['frame'] == previous['frame'] + 1:
                rate = (r['steer'] - previous['steer']) / dt
                updated = r.get('trajectory_frame') != previous.get('trajectory_frame') if r.get('trajectory_frame') is not None and previous.get('trajectory_frame') is not None else None
                raw_rate = (r.get('raw_steer', np.nan)-previous.get('raw_steer', np.nan))/dt
                raw = r.get('raw_steer', np.nan)
                if np.isfinite(raw):
                    slew = np.clip(raw, previous['steer']-steer_rate*dt, previous['steer']+steer_rate*dt)
                    slew_clipped = bool(abs(raw-slew)>1e-8)
                    amplitude_clipped = bool(abs(slew-np.clip(slew,-steer_limit,steer_limit))>1e-8)
        previous = r
        if r.get('truth_xy') is None:
            continue
        p = np.asarray(r['truth_xy'], float)
        if not np.isfinite(p).all():
            continue
        u = np.clip(np.sum((p-xy[:-1])*d, axis=1)/(length*length), 0, 1)
        foot = xy[:-1]+u[:, None]*d
        i = int(np.argmin(np.sum((foot-p)**2, axis=1)))
        s = station[i]+u[i]*length[i]
        delta = p-foot[i]
        cte = (d[i, 1]*delta[0]-d[i, 0]*delta[1])/length[i]
        interp = lambda q: np.array([np.interp(q, station, xy[:, j]) for j in range(2)])
        tangent = interp(s+2.5)-interp(s-2.5)
        yaw = r.get('truth_yaw', np.nan)
        heading = np.degrees((yaw-np.arctan2(tangent[1], tangent[0])+np.pi) % (2*np.pi)-np.pi)
        rows.append(dict(frame=r['frame'], time=r['sim_time'], progress=s, cte=cte,
            heading=heading, speed=r['speed_mps'], steer=r['steer'], rate=rate,
            saturated=abs(r['steer']) >= steer_limit-1e-6,
            limited=bool(r.get('steer_limited', False)), reason=r.get('reason'),
            raw_steer=r.get('raw_steer', np.nan), raw_rate=raw_rate,
            trajectory_frame=r.get('trajectory_frame'),trajectory_updated=updated,
            aim_interpolation=r.get('aim_interpolation'),aim_interpolation_used=r.get('aim_interpolation_used'),
            aim_interpolation_fallback=r.get('aim_interpolation_fallback'),
            aim_fields_json=json.dumps({k:v for k,v in r.items() if k.startswith('aim') or 'lookahead' in k},sort_keys=True),
            slew_clipped=slew_clipped, amplitude_clipped=amplitude_clipped,
            rejoin_concern=r.get('route_rejoin', {}).get('curvature_bound_satisfied') is False))
    return rows, float(station[-1])


def plant_features(trace):
    """世界加速度先求时间导数，再投到当前右向量；不对旋转标量求差。"""
    result, previous = {}, None
    for row in trace:
        k = row.get('plant_kinematics', row)
        acceleration = np.asarray(k.get('acceleration_mps2', [np.nan]*3),float)
        right = np.asarray(k.get('right_vector', [np.nan]*3),float)
        forward = np.asarray(k.get('forward_vector', [np.nan]*3),float)
        jerk = np.full(3,np.nan)
        if previous is not None:
            dt=row['sim_time']-previous['sim_time']
            if dt>0 and row['frame']==previous['frame']+1:
                prior=previous.get('plant_kinematics',previous)
                jerk=(acceleration-np.asarray(prior.get('acceleration_mps2',[np.nan]*3),float))/dt
        result[row['frame']]=dict(actual_speed=row.get('speed',np.nan),
            reference_speed=row.get('reference_speed_mps',np.nan),
            speed_error=row.get('speed',np.nan)-row.get('reference_speed_mps',np.nan),
            lateral_accel=float(acceleration@right), longitudinal_accel=float(acceleration@forward),
            lateral_jerk=float(jerk@right), longitudinal_jerk=float(jerk@forward))
        previous=row
    return result


def required_conditions(metrics, cases, coverage, recovery):
    """阈值取冻结协议；缺失、非有限、覆盖不足不自动通过。"""
    checks=[]
    def add(name, passed, **evidence):
        checks.append(dict(condition=name,status='insufficient' if passed is None else ('pass' if passed else 'fail'),**evidence))
    for case in cases:
        ident=case['route']+'/'+case['variant']
        gates=case.get('g2_gates',{})
        names=('completed','no_collision','full_lateral_rms','full_lateral_p95','cruise_speed','pose','heading',
               'endpoint','hold_duration','hold_displacement','hold_speed','telemetry_complete')
        for name in names:
            value=gates.get(name)
            add('G2/'+ident+'/'+name,value if isinstance(value,bool) else None,source='validation.json:gates.'+name)
        add('case/'+ident+'/no_invalid_control',case.get('invalid_control_count',0)==0 if 'invalid_control_count' in case else None)
        add('case/'+ident+'/complete_frames_and_truth',case.get('complete_frames_and_truth'))
        add('case/'+ident+'/reason_evidence',True if case.get('unknown_reason_count')==0 else None,unknown_count=case.get('unknown_reason_count'))
        if not case.get('legacy_fixture'):
            add('case/'+ident+'/common_config_contract',case.get('common_config_contract'))
            add('case/'+ident+'/pose_compensation_contract',case.get('pose_contract_complete'),error_count=case.get('pose_contract_error_count'))
            add('case/'+ident+'/pose_fields_complete',case.get('pose_fields_complete'))
            add('case/'+ident+'/aim_mode_evidence',None if case.get('aim_mode_missing_ticks',1) else case.get('aim_mode_mismatch_ticks',1)==0 and case.get('aim_used_or_fallback_error_ticks',1)==0,missing_ticks=case.get('aim_mode_missing_ticks'),mismatch_ticks=case.get('aim_mode_mismatch_ticks'),used_or_fallback_errors=case.get('aim_used_or_fallback_error_ticks'))
    recovery_index={(r['route'],r['variant'],r['segment']):r for r in recovery}
    recovery_notes=[]
    index={(m['route'],m['variant'],m['segment'],m['band']):m for m in metrics if m['subset']=='all'}
    for route in ROUTES:
        for segment in ([1,2] if route=='17563' else [1]):
            for variant in VARIANTS:
                c=coverage.get((route,variant,segment))
                add(f'coverage/{route}/{variant}/{segment}',c.get('complete') if c else None,evidence=c)
            def compare(name,field,band='window',delta=None,ratio=None,minimum=None,reverse=False,strict=False):
                b=index.get((route,'baseline-zero',segment,band),{})
                c=index.get((route,'candidate-fixed-k',segment,band),{})
                bv,cv=b.get(field),c.get(field)
                covered=all(coverage.get((route,v,segment),{}).get('complete') is True for v in VARIANTS)
                enough=b.get('count',0)>=20 and c.get('count',0)>=20
                valid=covered and enough and all(isinstance(v,(float,int)) and np.isfinite(v) for v in (bv,cv))
                if minimum is not None and valid and bv<=minimum:valid=False
                limit=bv*ratio if valid and ratio is not None else (bv+delta if valid else None)
                passed=(cv<limit-1e-9 if strict else (cv>=limit if reverse else cv<=limit)) if valid else None
                add(f'turn/{route}/{segment}/{name}',passed,field=field,band=band,baseline=bv,candidate=cv,
                    comparison='<' if strict else ('>=' if reverse else '<='),limit=limit-1e-9 if strict and limit is not None else limit,near_zero_baseline_threshold=minimum)
            compare('cte_p95_delta','cte_m_p95_abs',delta=.05)
            compare('cte_peak_delta','cte_m_max_abs',delta=.10)
            compare('heading_p95_delta','heading_deg_p95_abs',delta=1.)
            compare('speed_error_rms_delta','speed_error_mps_rms',delta=.10)
            compare('actual_speed_mean_drop','actual_speed_mean_mps',delta=-.20,reverse=True)
            compare('lateral_accel_p95_ratio','lateral_accel_mps2_p95_abs',ratio=1.10,minimum=.01)
            compare('emitted_rate_p95_ratio','steer_rate_per_s_p95_abs',ratio=1.20,minimum=.01)
            if route=='26966':
                compare('primary_cte_rms_15percent','cte_m_rms',ratio=.85)
                compare('primary_cte_p95_no_increase','cte_m_p95_abs',delta=0.)
                compare('post10m_rms_no_increase','cte_m_rms',band='post10m',delta=0.)
            else:
                compare('cte_rms_delta','cte_m_rms',delta=.03)
            recovery_notes.append(dict(route=route,segment=segment,status='diagnostic_only',
                baseline=recovery_index.get((route,VARIANTS[0],segment)),
                candidate=recovery_index.get((route,VARIANTS[1],segment)),
                reason='Censor retained; recovery and jerk have no added acceptance gate.'))

    statuses=[c['status'] for c in checks]
    return dict(status='fail' if 'fail' in statuses else ('insufficient' if 'insufficient' in statuses else 'pass'),
        required_conditions=checks,minimum_window_samples=20,
        near_zero_ratio_baseline=dict(lateral_accel_mps2=.01,emitted_steer_rate_per_s=.01),
        original_g2_thresholds=dict(completed=True,no_collision=True,full_lateral_rms_m=.5,
            full_lateral_p95_m=1.,cruise_speed_rms_after5s_mps=.5,pose_p90_m=.5,
            pose_heading_p90_deg=1.,endpoint_m=1.,hold_duration_min_s=5.,
            hold_displacement_max_m=.1,hold_speed_strict_max_mps=.1,telemetry_complete=True),
        recovery_notes=recovery_notes,scope='后轴传播固定k开发筛选；急右弯15%CTE收益及原四窗保护；jerk/recovery仅诊断；不代表默认或真实TCP资格')


def recover(rows, window):
    exits = [r for r in rows if r['progress'] >= window['core_end_m']]
    result = dict(core_exited=bool(exits), recovery_time_s=None, recovery_censored=True)
    if not exits:
        return result
    begin = exits[0]['time']
    start, previous = None, None
    for r in rows:
        if r['time'] < begin or r['progress'] > window['end_m']+10:
            continue
        good = abs(r['cte']) <= .25 and abs(r['heading']) <= 5
        if not good or (previous is not None and r['frame'] != previous['frame']+1):
            start = None
        if good and start is None:
            start = r['time']
        if good and r['time']-start >= .5-1e-7:
            result.update(recovery_time_s=start-begin, recovery_censored=False)
            break
        previous = r
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows', type=Path, default=BASE/'lateral-evidence-v1/turn-windows.csv')
    ap.add_argument('--reference-index', type=Path, default=BASE/'lateral-evidence-v1/routes.csv')
    ap.add_argument('--protocol', type=Path, help='冻结协议路径；提供时记录其哈希')
    ap.add_argument('--legacy-fixture',action='store_true',help='Historical-input test only; missing new mode fields explicitly marked, never a live qualification')
    ap.add_argument('--allow-incomplete', action='store_true', help='仅供阶段性检查；缺失 case 显式记录')
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('拒绝写入已有输出目录，请使用新版本目录。')
    helper = BASE/'lateral-evidence-v2/analyze.py'
    legacy = runpy.run_path(str(helper))
    stats, csv_write = legacy['stats'], legacy['csv_write']
    hashes = {}
    def record(p):
        p = Path(p)
        hashes[str(p.resolve())] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
    for p in (__file__, BASE/'pose_contract.py', helper, args.windows, args.reference_index):
        record(p)
    if args.protocol:record(args.protocol)
    windows = {route: [] for route in ROUTES}
    for r in csv.DictReader(args.windows.open()):
        if r['dataset'] == 'G2-v4' and r['route'] in windows:
            windows[r['route']].append({k: (v if k in ('dataset', 'route', 'kind') else float(v)) for k,v in r.items()})
    if [len(windows[r]) for r in ROUTES] != [1, 2, 1]:
        raise SystemExit('需要固定的 26966/17563/24240 四个窗口。')
    refs = {r['route']: Path(r['source']) for r in csv.DictReader(args.reference_index.open())
            if r['dataset'] == 'G2-v4' and r['route'] in windows}
    missing = []
    for route in ROUTES:
        for variant in VARIANTS:
            case = args.run_root/route/variant/'pursuit'
            for name in ('control.jsonl', 'route_reference.json', 'agent_config.json','validation_trace.json','validation.json','motion.jsonl'):
                if not (case/name).is_file():
                    missing.append(str(case/name))
    if missing and not args.allow_incomplete:
        raise SystemExit('六个 case 输入尚未完整；未创建输出目录：\n'+'\n'.join(missing))
    metrics, recovery, cases, coverage, reason_events, full_frames = [], [], [], {}, [], []
    pose_audits, pose_recomputed = [], []
    for route in ROUTES:
        record(refs[route])
        canonical = np.asarray(json.loads(refs[route].read_text())['world_xy'], float)
        for variant in VARIANTS:
            directory = args.run_root/route/variant/'pursuit'
            identity = dict(route=route, variant=variant, preset='pursuit')
            if any(not (directory/n).is_file() for n in ('control.jsonl','route_reference.json','agent_config.json','validation_trace.json','validation.json','motion.jsonl')):
                cases.append(dict(**identity, status='missing_inputs', directory=str(directory)))
                continue
            for name in ('control.jsonl','route_reference.json','agent_config.json','validation_trace.json','validation.json','motion.jsonl'):
                record(directory/name)
            reference = np.asarray(json.loads((directory/'route_reference.json').read_text())['world_xy'], float)
            if reference.shape != canonical.shape or not np.allclose(reference, canonical, atol=.01, rtol=0):
                raise SystemExit(f'{directory}: 原始路线几何与固定参考不一致，不能直接复用 station 窗口。')
            agent_config = json.loads((directory/'agent_config.json').read_text())
            configpath = Path(agent_config['controller_config'])
            record(configpath)
            config = json.loads(configpath.read_text())
            limit = config.get('max_steer', .8)
            records = [json.loads(line) for line in (directory/'control.jsonl').read_text().splitlines() if line]
            rows, total = project_rows(records, canonical, limit,config.get('steer_rate',2.))
            motion=[json.loads(line) for line in (directory/'motion.jsonl').read_text().splitlines() if line]
            pose_audit=audit_pose(records,motion,config,variant,reference=json.loads((directory/'route_reference.json').read_text()))
            pose_audits.append(dict(**identity,**{k:v for k,v in pose_audit.items() if k!='frames'}))
            pose_recomputed.extend(dict(**identity,**r) for r in pose_audit['frames'])
            original_byframe={r['frame']:r for r in records}
            for row in rows:
                original=original_byframe[row['frame']]
                row.update(pose_features(original))
                row['pose_status_json']=json.dumps(original.get('pose_status'),sort_keys=True)
                row['route_terminal_hold']=original.get('route_terminal_hold')
            trace=json.loads((directory/'validation_trace.json').read_text())
            if not isinstance(trace,list):raise SystemExit('validation_trace 必须为 JSON 数组。')
            features=plant_features(trace)
            for row in rows:
                row.update(features.get(row['frame'],{k:np.nan for k in ('actual_speed','reference_speed','speed_error',
                    'lateral_accel','longitudinal_accel','lateral_jerk','longitudinal_jerk')}))
            full_frames.extend(dict(**identity,**r) for r in rows)
            # 用独立 validation truth 帧定义预计覆盖，不能因 control 缺帧而缩小分母。
            truth_records=[dict(r,steer=0.,speed_mps=r.get('speed',np.nan)) for r in trace]
            expected_rows,_=project_rows(truth_records,canonical,limit)
            frames=[r['frame'] for r in records];trace_frames=[r['frame'] for r in trace]
            consecutive=lambda ff:len(ff)>0 and all(b==a+1 for a,b in zip(ff,ff[1:]))
            complete_frames=(consecutive(frames) and consecutive(trace_frames) and frames==trace_frames
                             and len(rows)==len(records) and len(expected_rows)==len(trace)
                             and all(r.get('truth_frame')==r['frame'] for r in records)
                             and clocks_agree(records,trace))
            invalid_count=sum(invalid_control(r,config) for r in records)
            unknown_reason_count=sum(r.get('reason') not in ('tracking','stop_hold','stationary_trajectory','trajectory_behind') and not invalid_control(r,config) for r in records)
            expected_mode='linear'
            mode_counts={str(v):sum(r.get('aim_interpolation')==v for r in records) for v in set(r.get('aim_interpolation') for r in records)}
            used_counts={str(v):sum(r.get('aim_interpolation_used')==v for r in records) for v in set(r.get('aim_interpolation_used') for r in records)}
            fallback_counts={str(v):sum(r.get('aim_interpolation_fallback')==v for r in records) for v in set(r.get('aim_interpolation_fallback') for r in records)}
            mode_evidence=aim_mode_evidence(records,expected_mode)
            byframe={r['frame']:r for r in rows}
            reason_counts={}
            for r in records:
                reason=str(r.get('reason','missing'))
                reason_counts[reason]=reason_counts.get(reason,0)+1
                if reason not in ('tracking','stop_hold','stationary_trajectory'):
                    p=byframe.get(r['frame'],{}).get('progress')
                    reason_events.append(dict(**identity,frame=r['frame'],sim_time=r['sim_time'],reason=reason,
                        invalid_control=invalid_control(r,config),station_m=p,
                        remaining_reference_m=total-p if p is not None else None,
                        near_reference_end_5m=bool(total-p<=5) if p is not None else None,
                        throttle=r.get('throttle'),brake=r.get('brake'),steer=r.get('steer')))
            case_row = dict(**identity, status='read', directory=str(directory), source_rows=len(records),
                truth_rows=len(rows), steer_limit=limit, config_json=json.dumps(config, sort_keys=True),
                cruise_mps=agent_config.get('cruise_mps'),complete_frames_and_truth=complete_frames,
                aim_requested_counts=mode_counts,aim_used_counts=used_counts,aim_fallback_counts=fallback_counts,
                **mode_evidence,legacy_fixture=args.legacy_fixture,
                common_config_contract=common_config(config) and agent_config.get('cruise_mps')==(6. if route=='17563' else 8.),
                pose_contract_complete=pose_audit['complete'],pose_contract_error_count=len(pose_audit['errors']),
                pose_fields_complete=all(np.isfinite([r[k] for k in ('pose_position_error','raw_pose_position_error','pose_heading_error_deg','pose_left_error','pose_forward_error')]).all() for r in rows) and bool(rows),
                invalid_control_count=invalid_count,unknown_reason_count=unknown_reason_count,validation_trace_count=len(trace),reason_counts=reason_counts)
            validation = directory/'validation.json'
            if validation.exists():
                record(validation)
                case_row['validation_json'] = json.dumps(json.loads(validation.read_text()), sort_keys=True)
                case_row['g2_gates']=json.loads(validation.read_text()).get('gates',{})
            cases.append(case_row)
            for w in windows[route]:
                ident = dict(**identity, segment=int(w['segment']), kind=w['kind'])
                selections = {
                    'window': [r for r in rows if w['start_m'] <= r['progress'] <= w['end_m']],
                    'entry': [r for r in rows if max(0,w['start_m']-5) <= r['progress'] < w['start_m']],
                    'core': [r for r in rows if w['core_start_m'] <= r['progress'] <= w['core_end_m']],
                    'exit': [r for r in rows if w['core_end_m'] < r['progress'] <= w['end_m']],
                    'post10m': [r for r in rows if w['end_m'] < r['progress'] <= min(total,w['end_m']+10)],
                }
                if w is windows[route][0]:
                    selections['full_route']=rows
                    selections['endpoint_last5m']=[r for r in rows if r['progress']>=total-5]
                    selections['terminal_hold']=[r for r in rows if r['route_terminal_hold'] is True]
                    guards={'26966':[(0,19)],'24240':[(0,13)],'17563':[(0,22.5),(53.5,68.5)]}[route]
                    for i,(start,end) in enumerate(guards,1):
                        selections[f'straight_guard_{i}']=[r for r in rows if start<=r['progress']<=end]
                expected=[r for r in expected_rows if w['start_m']<=r['progress']<=w['end_m']]
                selected=selections['window']
                required_fields=('cte','heading','rate','raw_steer','actual_speed','reference_speed','speed_error',
                                 'lateral_accel','lateral_jerk')
                missing_fields={field:sum(not np.isfinite(r.get(field,np.nan)) for r in selected) for field in required_fields}
                covered=dict(expected_count=len(expected),observed_count=len(selected),minimum_count=20,
                    missing_counts=missing_fields,entered=any(r['progress']<w['start_m'] for r in expected_rows),
                    exited=any(r['progress']>w['end_m'] for r in expected_rows),
                    complete_frames_and_truth=complete_frames)
                covered['complete']=bool(complete_frames and len(selected)>=20 and
                    [r['frame'] for r in selected]==[r['frame'] for r in expected] and
                    not any(missing_fields.values()) and covered['entered'] and covered['exited'])
                coverage[(route,variant,int(w['segment']))]=covered
                for band, selected in selections.items():
                    for subset in ('all', 'moving_ge_2mps'):
                        rr = selected if subset == 'all' else [r for r in selected if r['actual_speed'] >= 2]
                        m = dict(**ident, band=band, subset=subset, count=len(rr),
                            low_speed_count=sum(r['speed'] < 2 for r in rr),
                            saturated_fraction=float(np.mean([r['saturated'] for r in rr])) if rr else None,
                            limited_fraction=float(np.mean([r['limited'] for r in rr])) if rr else None)
                        for field, unit in (('cte','cte_m'),('heading','heading_deg'),('rate','steer_rate_per_s'),
                            ('steer','steer'),('speed','sensor_speed_mps'),('raw_steer','raw_steer'),('raw_rate','raw_steer_rate_per_s'),
                            ('actual_speed','actual_speed_mps'),('reference_speed','reference_speed_mps'),
                            ('speed_error','speed_error_mps'),('lateral_accel','lateral_accel_mps2'),
                            ('longitudinal_accel','longitudinal_accel_mps2'),('lateral_jerk','lateral_jerk_mps3'),
                            ('longitudinal_jerk','longitudinal_jerk_mps3'),
                            ('pose_position_error','pose_position_error_m'),('raw_pose_position_error','raw_pose_position_error_m'),
                            ('pose_heading_error_deg','pose_heading_error_deg'),('pose_world_dx','pose_world_dx_m'),
                            ('pose_world_dy','pose_world_dy_m'),('pose_left_error','pose_left_error_m'),('pose_forward_error','pose_forward_error_m')):
                            values=np.asarray([r[field] for r in rr],float)
                            m.update(stats(values, unit))
                            available=values[np.isfinite(values)]
                            m[unit+'_p90_abs']=float(np.percentile(abs(available),90)) if len(available) else None
                            m[unit+'_mean']=float(np.mean(available)) if len(available) else None
                            m[unit+'_missing_count']=int(len(values)-len(available))
                        for field in ('actual_speed','reference_speed'):
                            values=[r[field] for r in rr]
                            m[field+'_mean_mps']=float(np.mean(values)) if values and np.isfinite(values).all() else None
                        for field in ('slew_clipped','amplitude_clipped'):
                            available=[r[field] for r in rr if r[field] is not None]
                            m[field+'_count']=sum(available);m[field+'_known_count']=len(available)
                            m[field+'_fraction']=float(np.mean(available)) if available else None
                        m['rejoin_concern_count']=sum(r['rejoin_concern'] for r in rr)
                        m['trajectory_update_count']=sum(r['trajectory_updated'] is True for r in rr)
                        m['aim_mode_missing_ticks']=sum(r['aim_interpolation'] is None for r in rr)
                        m['aim_not_evaluated_ticks']=sum(r['aim_interpolation'] is not None and r['aim_interpolation_used'] is None for r in rr)
                        m['aim_hermite_used_ticks']=sum(r['aim_interpolation_used']=='hermite' for r in rr)
                        m['aim_fallback_counts_json']=json.dumps({str(v):sum(r['aim_interpolation_fallback']==v for r in rr) for v in set(r['aim_interpolation_fallback'] for r in rr)},sort_keys=True)
                        m['coverage_complete']=covered['complete']
                        m['cte_left_positive_mean_m'] = float(np.mean([r['cte'] for r in rr])) if rr else None
                        m['heading_right_positive_mean_deg'] = float(np.mean([r['heading'] for r in rr])) if rr else None
                        if rr:
                            peak = max(rr, key=lambda r:abs(r['cte']))
                            m.update(peak_cte_left_positive_m=peak['cte'], peak_cte_frame=peak['frame'],
                                peak_cte_station_m=peak['progress'], first_frame=rr[0]['frame'], last_frame=rr[-1]['frame'])
                        metrics.append(m)
                recovery.append(dict(**ident, **recover(rows,w), padded_window_exited=bool(selections['post10m'])))
    for route in ROUTES:
        pair=[c for c in cases if c['route']==route and c.get('status')=='read']
        configs=[{k:v for k,v in json.loads(c['config_json']).items() if k!=KEY} for c in pair]
        identical=len(pair)==2 and configs[0]==configs[1]
        for case in pair:
            case['common_config_contract']=bool(case['common_config_contract'] and identical)
            case['pair_config_only_coefficient_differs']=identical
    args.out.mkdir(parents=True)
    for name, rows in [('metrics.csv',metrics),('recovery.csv',recovery),('cases.csv',cases),('frames.csv',full_frames),('pose-recomputed.csv',pose_recomputed)]:
        csv_write(args.out/name,rows)
    conditions=required_conditions(metrics,cases,coverage,recovery)
    conditions['legacy_fixture']=args.legacy_fixture
    conditions['live_qualification_allowed']=not args.legacy_fixture
    (args.out/'required-conditions.json').write_text(json.dumps(conditions,ensure_ascii=False,indent=2)+'\n')
    (args.out/'coverage.json').write_text(json.dumps({'/'.join(map(str,k)):v for k,v in coverage.items()},indent=2)+'\n')
    (args.out/'pose-contract.json').write_text(json.dumps(pose_audits,ensure_ascii=False,indent=2)+'\n')
    (args.out/'pose-contract-source.py').write_bytes((BASE/'pose_contract.py').read_bytes())
    (args.out/'control-reasons.json').write_text(json.dumps(reason_events,indent=2)+'\n')
    for source in (Path(__file__),helper):
        (args.out/('analysis-source.py' if source==Path(__file__) else 'stats-source.py')).write_bytes(source.read_bytes())
    manifest = dict(legacy_fixture=args.legacy_fixture,run_root=str(args.run_root.resolve()), inputs=hashes,
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()},
        complete_input_matrix=not missing, missing_inputs=missing, routes=list(ROUTES), variants=list(VARIANTS),
        protocol=dict(windows='Frozen v1: 26966 one,17563 two,24240 one; no error-based reselection',
            cte='LEFT positive, opposite the historical independent v2 perpendicular sign; RMS/p95abs/maxabs unchanged',
            heading='CARLA right-positive vehicle yaw minus centered5m road tangent, degrees',
            rate='Emitted normalized steer difference / actual sim_time difference; only consecutive frames',
            physical='Join validation_trace JSON array by frame; actual speed minus reference; accel dot right/forward; world acceleration finite difference divided by trace sim_time dt then dot CURRENT right/forward for jerk. No smoothing.',
            limits='raw->slew clamp using previous emitted and actualdt->amplitude clamp. Diagnostic requested clamp stages; safe fallback may use a different branch. Same-frame applied_control is previous command, never treated as actuator error.',
            invalid_control='Nonfinite/out-of-config-bounds output, pedal conflict, explicit invalid/stale/input/pose fault. trajectory_behind legal safe brake is separately indexed with station and proximity to endpoint, all its samples retained.',
            bands='entry/core/exit/post10m; full_route; endpoint_last5m; terminal_hold; straight guards26966[0,19],24240[0,13],17563[0,22.5]and[53.5,68.5]' ,
            retention='All frames incl reversal/stall; moving>=2m/s auxiliary; no pooled cross-route passing claim',
            recovery='First core exit to first >=.5s consecutive absCTE<=.25m and absheading<=5deg interval before padded_end+10m; absent censored',
            geometry_guard='Each case original world_xy must match canonical reference pointwise within.01m; no resampling new trajectories',
            candidate='baseline-zero vs candidate-fixed-k=.010659832; both linear aim max(3m,.5s*v); original four-window guards; no Hermite action-benefit or recovery gates.'))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases), metrics=len(metrics), windows=len(recovery), out=str(args.out))))


if __name__ == '__main__':
    main()
