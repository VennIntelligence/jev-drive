"""Aggregate the offline steer replay and plot with the shared paper style."""
from pathlib import Path
import csv
import glob
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import plot_style
from b2d_tfv6_steer_diagnosis import OUT, FULL, dense_station, turn_distance, write_csv


def data():
    rows=list(csv.DictReader(open(FULL/'ticks.csv')))
    for r in rows:
        for k,v in list(r.items()):
            if k not in ('source','route','c_reason') and v not in ('','None'):
                try: r[k]=float(v)
                except ValueError: pass
    return rows


def summary(rows):
    b=np.array([abs(x['b_executed']) for x in rows]); c=np.array([abs(x['c_shadow']) for x in rows]);d=np.array([abs(x['d_shadow']) for x in rows])
    gap=b-c
    return dict(ticks=len(rows),runs=len(set((x['source'],x['route'],x['seed']) for x in rows)),
                b_abs_mean=float(b.mean()),c_abs_mean=float(c.mean()),d_abs_mean=float(d.mean()),
                c_gap_mean=float(gap.mean()),c_gap_median=float(np.median(gap)),
                d_gap_mean=float((b-d).mean()),c_weak_over_005=float(np.mean(gap>.05)),
                opposite_sign=float(np.mean([np.sign(x['b_executed'])!=np.sign(x['c_shadow']) for x in rows])),
                rate_clamped=float(np.mean([x['rate_clamped'] for x in rows])),
                max_clamped=float(np.mean([x['max_clamped'] for x in rows])),
                horizon_capped=float(np.mean([x['horizon_capped'] for x in rows])),
                horizon_m_median=float(np.median([x['horizon_m'] for x in rows])),
                c_lookahead_m_median=float(np.median([x['lookahead_m'] for x in rows])),
                b_aim_m_median=float(np.median([x['b_aim_actual_m'] for x in rows])))


def speed_entry():
    out=[]
    for source in ('D3b','W2'):
        for route,bend in (('2084',33),('27529',16)):
            dense=dense_station(route)
            for seed in (0,1,2):
                for arm in ('B','C'):
                    base=('/data/runs/b2d/tfv6-d3/factorial/cases/2' if source=='D3b' else
                          '/data/runs/b2d/tfv6-w2/formal/level2/cases/2')
                    pp=glob.glob(f'{base}/route-{route}/seed-{seed}/{arm}/attempt-*/frames.jsonl')
                    if not pp: continue
                    points=[]
                    for line in open(pp[-1]):
                        f=json.loads(line)
                        if not f.get('truth'):continue
                        dist=turn_distance(dense,bend,f['truth']['location'][:2])
                        points.append((dist,float(f['truth']['speed_mps']),float(f['controller_speed_mps'])))
                    for marker in (10.,5.,0.):
                        near=min(points,key=lambda z:abs(z[0]-marker))
                        out.append(dict(source=source,route=route,seed=seed,arm=arm,
                                        marker_m=marker,actual_distance_m=near[0],
                                        truth_speed_mps=near[1],controller_speed_mps=near[2]))
    write_csv(OUT/'speed_entry.csv',out)
    return out


def main():
    rows=data()
    turn=[x for x in rows if abs(x['plan_curvature'])>=.03 and x['speed']>=1. and abs(x['b_executed'])>=.05]
    bins=[]
    specs=[('curvature',[(0,.01),(.01,.03),(.03,.06),(.06,.12),(.12,1e6)],rows),
           ('speed',[(1,2),(2,4),(4,6),(6,8),(8,1e6)],turn),
           ('distance',[(-1e6,-10),(-10,-5),(-5,0),(0,5),(5,10),(10,20),(20,1e6)],turn)]
    for name,edges,subset in specs:
        for lo,hi in edges:
            part=[x for x in subset if x['speed']>=1 and abs(x['b_executed'])>=.05 and
                  lo <= (abs(x['plan_curvature']) if name=='curvature' else
                         x['speed'] if name=='speed' else x['dense_turn_distance_m']) < hi]
            if part:bins.append(dict(group=name,lower=lo,upper=hi,**summary(part)))
    write_csv(OUT/'bins.csv',bins)
    base=[];sens=[]
    variants={'B aim, C law':'aim_b','B PID, C aim':'b_pid_on_c_aim','no rate limit':'no_rate',
              'no max limit':'no_max','lookahead 0.3 s':'time_0.3','lookahead 0.4 s':'time_0.4',
              'lookahead 0.6 s':'time_0.6','lookahead 0.7 s':'time_0.7','lookahead 1.0 s':'time_1.0'}
    for source in ('D3b','W2','all'):
        part=[x for x in turn if source=='all' or x['source']==source]
        base.append(dict(source=source,**summary(part)))
        b=np.array([abs(x['b_executed']) for x in part]);c=np.array([abs(x['c_shadow']) for x in part])
        for name,key in variants.items():
            z=np.array([abs(x[key]) for x in part])
            sens.append(dict(source=source,variant=name,ticks=len(part),
                             mean_abs_steer=float(z.mean()),gap_mean=float((b-z).mean()),
                             gap_closed=float((z-c).mean()),
                             gap_closed_fraction=float((z-c).mean()/(b-c).mean())))
    write_csv(OUT/'turn_summary.csv',base)
    write_csv(OUT/'sensitivity.csv',sens)
    entries=speed_entry()
    plot_style.apply()
    fig,ax=plt.subplots(1,2,figsize=(plot_style.DOUBLE_COLUMN_IN,2.35))
    cur=[x for x in bins if x['group']=='curvature']
    ax[0].bar(range(len(cur)),[x['c_gap_mean'] for x in cur],color=plot_style.PALETTE['blue'])
    ax[0].set_xticks(range(len(cur)),['<.01','.01–.03','.03–.06','.06–.12','≥.12'])
    ax[0].set_xlabel('Absolute plan curvature (1/m)');ax[0].set_ylabel('Mean |B| − |C| steer')
    ax[0].set_title('Same B trajectory, all runs')
    names=['B aim, C law','B PID, C aim','no rate limit','lookahead 0.7 s','lookahead 1.0 s']
    z=[next(x for x in sens if x['source']=='all' and x['variant']==name) for name in names]
    ax[1].barh(range(len(z)),[x['gap_closed'] for x in z],color=plot_style.PALETTE['orange'])
    ax[1].set_yticks(range(len(z)),names);ax[1].invert_yaxis()
    ax[1].set_xlabel('Mean gap closed (steer)');ax[1].set_title('One-factor open-loop changes')
    fig.tight_layout();plot_style.save(fig,OUT/'steer_gap');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(plot_style.DOUBLE_COLUMN_IN,2.35))
    distance=[x for x in bins if x['group']=='distance']
    def distance_label(x):
        lo,hi=x['lower'],x['upper']
        return f'{int(lo)}:{int(hi)}' if abs(lo)<1e5 and abs(hi)<1e5 else (f'<{int(hi)}' if lo<0 else f'≥{int(lo)}')
    ax[0].bar(range(len(distance)),[x['c_gap_mean'] for x in distance],color=plot_style.PALETTE['blue'])
    ax[0].set_xticks(range(len(distance)),[distance_label(x) for x in distance],rotation=25)
    ax[0].set_xlabel('Distance to bend (m; positive before)');ax[0].set_ylabel('Mean |B| − |C| steer')
    ax[0].set_title('Dense 2084/27529 only')
    for source,col in [('D3b',plot_style.PALETTE['orange']),('W2',plot_style.PALETTE['green'])]:
        for arm,style in [('B','o'),('C','s')]:
            vals=[]
            for marker in (10.,5.,0.):
                rr=[x['truth_speed_mps'] for x in entries if x['source']==source and x['arm']==arm and x['marker_m']==marker]
                vals.append(float(np.median(rr)))
            ax[1].plot([10,5,0],vals,marker=style,color=col,label=f'{source} {arm}')
    ax[1].set_xlabel('Distance before bend (m)');ax[1].set_ylabel('Median truth speed (m/s)')
    ax[1].legend(ncol=2,fontsize=6);ax[1].set_title('Actual paired B/C trajectories')
    fig.tight_layout();plot_style.save(fig,OUT/'bend_speed');plt.close(fig)


if __name__=='__main__':main()
