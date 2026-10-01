#!/usr/bin/env python3
"""只检验固定窗口内转向变化与5Hz轨迹更新的同期关系，不作因果验收。"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--figures',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise SystemExit('拒绝覆盖旧诊断。')
    base=Path(__file__).resolve().parent;windowfile=base/'lateral-evidence-v1/turn-windows.csv'
    sources={}
    def record(p):sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    record(Path(__file__));record(windowfile)
    windows={}
    for w in csv.DictReader(windowfile.open()):
        if w['dataset']=='G2-v4' and w['route'] in ('26966','17563','24240'):windows.setdefault(w['route'],[]).append(w)
    summaries=[];peaks=[]
    for route,ws in windows.items():
        samples=args.figures/f'route-{route}-samples.csv';record(samples)
        sample=list(csv.DictReader(samples.open()))
        for variant in ('baseline-max','short-max'):
            log=args.run_root/route/variant/'pursuit/control.jsonl';record(log)
            rows=[json.loads(line) for line in log.read_text().splitlines() if line]
            previous={r['frame']+1:r for r in rows};lookup={r['frame']:r for r in rows}
            for w in ws:
                group=[]
                for sample_row in sample:
                    if sample_row['variant']!=variant or not float(w['start_m'])<=float(sample_row['progress'])<=float(w['end_m']):continue
                    r=lookup[int(sample_row['frame'])];p=previous.get(r['frame'])
                    if p is None:continue
                    dt=r['sim_time']-p['sim_time']
                    group.append(dict(route=route,variant=variant,segment=int(w['segment']),frame=r['frame'],
                        station_m=float(sample_row['progress']),sim_time=r['sim_time'],
                        trajectory_updated=r['trajectory_frame']!=p['trajectory_frame'],
                        emitted_rate_abs=abs((r['steer']-p['steer'])/dt),
                        raw_rate_abs=abs((r.get('raw_steer',np.nan)-p.get('raw_steer',np.nan))/dt),
                        steer=r['steer'],raw_steer=r.get('raw_steer'),limited=r.get('steer_limited'),
                        pose_heading_error_deg=np.degrees(r.get('pose_heading_error_rad',np.nan)),
                        pose_error_m=r.get('pose_error_m'),rejoin_concern=r.get('route_rejoin',{}).get('curvature_bound_satisfied') is False))
                threshold=float(np.percentile([r['emitted_rate_abs'] for r in group],90))
                high=[r for r in group if r['emitted_rate_abs']>=threshold]
                summaries.append(dict(route=route,variant=variant,segment=int(w['segment']),count=len(group),
                    trajectory_update_count=sum(r['trajectory_updated'] for r in group),
                    emitted_rate_p90=threshold,high_rate_count=len(high),
                    high_rate_on_update_count=sum(r['trajectory_updated'] for r in high),
                    rejoin_concern_count=sum(r['rejoin_concern'] for r in group)))
                peaks.extend(sorted(group,key=lambda r:r['emitted_rate_abs'],reverse=True)[:5])
    args.out.mkdir(parents=True)
    for name,rows in [('per-window-replan.csv',summaries),('top5-rate-events.csv',peaks)]:
        with (args.out/name).open('w') as f:
            writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(inputs=sources,outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*.csv')},
        protocol='固定窗口全帧；轨迹更新=trajectory_frame与连续上一frame不同；高变化率=本窗abs emitted rate>=p90（保留并列）；列top5只作定位，不用于改变验收。',
        limitation='同期关联不能区分rejoin几何、导航进度、定位噪声和控制纠偏；不以truth诊断反推已证明原因。未修改阈值或运行控制。')
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
