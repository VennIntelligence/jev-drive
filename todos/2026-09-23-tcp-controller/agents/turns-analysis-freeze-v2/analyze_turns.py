#!/usr/bin/env python3
"""分析固定四个转弯窗口；读取独立 run-root，不导入 CARLA 或运行控制器。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

ROUTES = ('26966', '17563', '24240')
VARIANTS = ('baseline-max', 'short-max')
BASE = Path(__file__).resolve().parent


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
    return any(token in reason for token in ('invalid','stale','timestamp_discontinuity',
        'compass_dropout_timeout','compass_uninitialized'))


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
        raw_rate = np.nan
        slew_clipped = amplitude_clipped = None
        if previous is not None:
            dt = r['sim_time'] - previous['sim_time']
            if dt > 0 and r['frame'] == previous['frame'] + 1:
                rate = (r['steer'] - previous['steer']) / dt
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


def required_conditions(metrics, cases, coverage):
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
    index={(m['route'],m['variant'],m['segment'],m['band']):m for m in metrics if m['subset']=='all'}
    for route in ROUTES:
        for segment in ([1,2] if route=='17563' else [1]):
            for variant in VARIANTS:
                c=coverage.get((route,variant,segment))
                add(f'coverage/{route}/{variant}/{segment}',c.get('complete') if c else None,evidence=c)
            def compare(name,field,band='window',delta=None,ratio=None,minimum=None,reverse=False):
                b=index.get((route,'baseline-max',segment,band),{})
                c=index.get((route,'short-max',segment,band),{})
                bv,cv=b.get(field),c.get(field)
                covered=all(coverage.get((route,v,segment),{}).get('complete') is True for v in VARIANTS)
                enough=b.get('count',0)>=20 and c.get('count',0)>=20
                valid=covered and enough and all(isinstance(v,(float,int)) and np.isfinite(v) for v in (bv,cv))
                if minimum is not None and valid and bv<=minimum:valid=False
                limit=bv*ratio if valid and ratio is not None else (bv+delta if valid else None)
                passed=(cv>=limit if reverse else cv<=limit) if valid else None
                add(f'turn/{route}/{segment}/{name}',passed,field=field,band=band,baseline=bv,candidate=cv,
                    comparison='>=' if reverse else '<=',limit=limit,near_zero_baseline_threshold=minimum)
            compare('cte_p95_delta','cte_m_p95_abs',delta=.05)
            compare('cte_peak_delta','cte_m_max_abs',delta=.10)
            compare('heading_p95_delta','heading_deg_p95_abs',delta=1.)
            compare('speed_error_rms_delta','speed_error_mps_rms',delta=.10)
            compare('actual_speed_mean_drop','actual_speed_mean_mps',delta=-.20,reverse=True)
            compare('lateral_accel_p95_ratio','lateral_accel_mps2_p95_abs',ratio=1.10,minimum=.01)
            compare('emitted_rate_p95_ratio','steer_rate_per_s_p95_abs',ratio=1.20,minimum=.01)
            if route=='26966':
                compare('primary_rms_15percent','cte_m_rms',ratio=.85)
                compare('primary_p95_no_increase','cte_m_p95_abs',delta=0.)
                compare('post10m_rms_no_increase','cte_m_rms',band='post10m',delta=0.)
            else:compare('other_window_rms_delta','cte_m_rms',delta=.03)
    statuses=[c['status'] for c in checks]
    return dict(status='fail' if 'fail' in statuses else ('insufficient' if 'insufficient' in statuses else 'pass'),
        required_conditions=checks,minimum_window_samples=20,
        near_zero_ratio_baseline=dict(lateral_accel_mps2=.01,emitted_steer_rate_per_s=.01),
        original_g2_thresholds=dict(completed=True,no_collision=True,full_lateral_rms_m=.5,
            full_lateral_p95_m=1.,cruise_speed_rms_after5s_mps=.5,pose_p90_m=.5,
            pose_heading_p90_deg=1.,endpoint_m=1.,hold_duration_min_s=5.,
            hold_displacement_max_m=.1,hold_speed_strict_max_mps=.1,telemetry_complete=True),
        scope='开发必要条件；jerk/恢复仅报告；不代表默认控制器资格或真实TCP收益')


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
    for p in (__file__, helper, args.windows, args.reference_index):
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
            for name in ('control.jsonl', 'route_reference.json', 'agent_config.json','validation_trace.json','validation.json'):
                if not (case/name).is_file():
                    missing.append(str(case/name))
    if missing and not args.allow_incomplete:
        raise SystemExit('六个 case 输入尚未完整；未创建输出目录：\n'+'\n'.join(missing))
    metrics, recovery, cases, coverage, reason_events = [], [], [], {}, []
    for route in ROUTES:
        record(refs[route])
        canonical = np.asarray(json.loads(refs[route].read_text())['world_xy'], float)
        for variant in VARIANTS:
            directory = args.run_root/route/variant/'pursuit'
            identity = dict(route=route, variant=variant, preset='pursuit')
            if any(not (directory/n).is_file() for n in ('control.jsonl','route_reference.json','agent_config.json','validation_trace.json','validation.json')):
                cases.append(dict(**identity, status='missing_inputs', directory=str(directory)))
                continue
            for name in ('control.jsonl','route_reference.json','agent_config.json','validation_trace.json','validation.json'):
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
            trace=json.loads((directory/'validation_trace.json').read_text())
            if not isinstance(trace,list):raise SystemExit('validation_trace 必须为 JSON 数组。')
            features=plant_features(trace)
            for row in rows:
                row.update(features.get(row['frame'],{k:np.nan for k in ('actual_speed','reference_speed','speed_error',
                    'lateral_accel','longitudinal_accel','lateral_jerk','longitudinal_jerk')}))
            # 用独立 validation truth 帧定义预计覆盖，不能因 control 缺帧而缩小分母。
            truth_records=[dict(r,steer=0.,speed_mps=r.get('speed',np.nan)) for r in trace]
            expected_rows,_=project_rows(truth_records,canonical,limit)
            frames=[r['frame'] for r in records];trace_frames=[r['frame'] for r in trace]
            consecutive=lambda ff:len(ff)>0 and all(b==a+1 for a,b in zip(ff,ff[1:]))
            complete_frames=(consecutive(frames) and consecutive(trace_frames) and frames==trace_frames
                             and len(rows)==len(records) and len(expected_rows)==len(trace))
            invalid_count=sum(invalid_control(r,config) for r in records)
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
                invalid_control_count=invalid_count,validation_trace_count=len(trace),reason_counts=reason_counts)
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
                            ('longitudinal_jerk','longitudinal_jerk_mps3')):
                            m.update(stats([r[field] for r in rr], unit))
                        for field in ('actual_speed','reference_speed'):
                            values=[r[field] for r in rr]
                            m[field+'_mean_mps']=float(np.mean(values)) if values and np.isfinite(values).all() else None
                        for field in ('slew_clipped','amplitude_clipped'):
                            available=[r[field] for r in rr if r[field] is not None]
                            m[field+'_count']=sum(available);m[field+'_known_count']=len(available)
                            m[field+'_fraction']=float(np.mean(available)) if available else None
                        m['rejoin_concern_count']=sum(r['rejoin_concern'] for r in rr)
                        m['coverage_complete']=covered['complete']
                        m['cte_left_positive_mean_m'] = float(np.mean([r['cte'] for r in rr])) if rr else None
                        m['heading_right_positive_mean_deg'] = float(np.mean([r['heading'] for r in rr])) if rr else None
                        if rr:
                            peak = max(rr, key=lambda r:abs(r['cte']))
                            m.update(peak_cte_left_positive_m=peak['cte'], peak_cte_frame=peak['frame'],
                                peak_cte_station_m=peak['progress'], first_frame=rr[0]['frame'], last_frame=rr[-1]['frame'])
                        metrics.append(m)
                recovery.append(dict(**ident, **recover(rows,w), padded_window_exited=bool(selections['post10m'])))
    args.out.mkdir(parents=True)
    for name, rows in [('metrics.csv',metrics),('recovery.csv',recovery),('cases.csv',cases)]:
        csv_write(args.out/name,rows)
    conditions=required_conditions(metrics,cases,coverage)
    (args.out/'required-conditions.json').write_text(json.dumps(conditions,ensure_ascii=False,indent=2)+'\n')
    (args.out/'coverage.json').write_text(json.dumps({'/'.join(map(str,k)):v for k,v in coverage.items()},indent=2)+'\n')
    (args.out/'control-reasons.json').write_text(json.dumps(reason_events,indent=2)+'\n')
    manifest = dict(run_root=str(args.run_root.resolve()), inputs=hashes,
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()},
        complete_input_matrix=not missing, missing_inputs=missing, routes=list(ROUTES), variants=list(VARIANTS),
        protocol=dict(windows='Frozen v1: 26966 one,17563 two,24240 one; no error-based reselection',
            cte='LEFT positive, opposite the historical independent v2 perpendicular sign; RMS/p95abs/maxabs unchanged',
            heading='CARLA right-positive vehicle yaw minus centered5m road tangent, degrees',
            rate='Emitted normalized steer difference / actual sim_time difference; only consecutive frames',
            physical='Join validation_trace JSON array by frame; actual speed minus reference; accel dot right/forward; world acceleration finite difference divided by trace sim_time dt then dot CURRENT right/forward for jerk. No smoothing.',
            limits='raw->slew clamp using previous emitted and actualdt->amplitude clamp. Diagnostic requested clamp stages; safe fallback may use a different branch. Same-frame applied_control is previous command, never treated as actuator error.',
            invalid_control='Nonfinite/out-of-config-bounds output, pedal conflict, explicit invalid/stale/input/pose fault. trajectory_behind legal safe brake is separately indexed with station and proximity to endpoint, all its samples retained.',
            bands='entry padded_start-5..padded_start;core;exit core_end..padded_end;post10m padded_end..padded_end+10',
            retention='All frames incl reversal/stall; moving>=2m/s auxiliary; no pooled cross-route passing claim',
            recovery='First core exit to first >=.5s consecutive absCTE<=.25m and absheading<=5deg interval before padded_end+10m; absent censored',
            geometry_guard='Each case original world_xy must match canonical reference pointwise within.01m; no resampling new trajectories',
            candidate='Expected max(3m,.375s*v) vs max(3m,.5s*v); actual config bytes archived in input hashes and cases.csv, no parameter enforcement'))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases), metrics=len(metrics), windows=len(recovery), out=str(args.out))))


if __name__ == '__main__':
    main()
