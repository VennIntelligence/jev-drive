"""Offline TFv6 B-trajectory controller diagnosis; no simulator dependency."""
from collections import deque
from pathlib import Path
import csv
import glob
import json
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2d_controller import Controller

OUT = Path('results/diagnosis/steer')
FULL = Path('/data/runs/b2d/tfv6-w2/diagnosis/steer')
DENSE = Path('/data/runs/b2d/tfv6-d3/dense')
COMMON = dict(longitudinal_mode='pi', lookahead='max', pi_kp=.5, pi_ki=.25,
              max_lookahead_time_s=.5)


def paths():
    d3 = glob.glob('/data/runs/b2d/tfv6-d3/factorial/cases/2/route-*/seed-*/B/attempt-1/frames.jsonl')
    d3 = [p for p in d3 if '/route-2084/' in p or '/route-27529/' in p]
    formal = []
    for p in glob.glob('/data/runs/b2d/tfv6-w2/formal/**/B/done.json', recursive=True):
        if '/aborted/' in p:
            continue
        done = json.loads(Path(p).read_text())
        q = Path(p).parent / ('attempt-' + str(done['attempt'])) / 'frames.jsonl'
        if q.exists():
            formal.append(str(q))
    return [('D3b', p) for p in sorted(d3)] + [('W2', p) for p in sorted(formal)]


def route_seed(p):
    parts = Path(p).parts
    return next(x[6:] for x in parts if x.startswith('route-')), int(next(x[5:] for x in parts if x.startswith('seed-')))


def curvature(w):
    # Signed circumcircle curvature of waypoint triples, median over the first four triples.
    p = np.asarray(w, float)
    values = []
    for a, b, c in zip(p[:4], p[1:5], p[2:6]):
        ab, bc, ac = b-a, c-b, c-a
        den = np.linalg.norm(ab)*np.linalg.norm(bc)*np.linalg.norm(ac)
        if den > 1e-8:
            values.append(2*(ab[0]*ac[1]-ab[1]*ac[0])/den)
    return float(np.median(values)) if values else 0.


def dense_station(route):
    p = DENSE / ('2-' + route + '.json')
    if not p.exists():
        return None
    points = np.array([z['xyz'][:2] for z in json.loads(p.read_text())['dense']], float)
    seg = np.diff(points, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    arc = np.r_[0., np.cumsum(lens)]
    return points, seg, lens, arc


def turn_distance(dense, bend, xy):
    if dense is None:
        return float('nan')
    p, seg, lens, arc = dense
    v = np.asarray(xy) - p[:-1]
    u = np.clip(np.sum(v*seg, axis=1)/np.maximum(lens*lens, 1e-12), 0, 1)
    error = np.sum((v-u[:,None]*seg)**2, axis=1)
    i = int(np.argmin(error))
    return float(arc[bend] - (arc[i] + u[i]*lens[i]))


def steer_from_aim(aim, speed, c):
    k = 2*aim[1]/max(float(aim@aim), 1e-8)
    angle = math.atan(c.wheelbase*k)
    scale = float(np.interp(speed*3.6, c.steering_curve[:,0], c.steering_curve[:,1]))
    return -angle/(c.max_steer_rad*scale), k


def clamp(v, prev, elapsed, c):
    rate = float(np.clip(v, prev-c.steer_rate*elapsed, prev+c.steer_rate*elapsed))
    return float(np.clip(rate, -c.max_steer, c.max_steer)), rate


def author_pid(w, speed, history):
    p = np.asarray(w, float)
    desired = float(2*np.linalg.norm(p[1]-p[3]))
    brake = desired < .4 or speed/desired > 1.1 if desired > 0 else True
    target = 2.25 if desired < 5.5 else 3.
    ids = np.flatnonzero(np.linalg.norm(p, axis=1) >= target)
    ix = int(ids[0]) if len(ids) else len(p)-1
    aim = p[ix]
    error = float(np.degrees(np.arctan2(aim[1],aim[0]))/90.)
    if speed < .01 or brake:
        error = 0.
    previous = history[-1]
    history.append(error)
    proportional = 1.25*error
    integral = .75*sum(history)/len(history)
    derivative = .3*(error-previous)
    return dict(b_aim_index=ix, b_aim_x=float(aim[0]), b_aim_y=float(aim[1]),
                b_aim_target_m=target, b_aim_actual_m=float(np.linalg.norm(aim)),
                b_angle=error, b_p=proportional, b_i=integral, b_d=derivative,
                b_pid=float(np.clip(proportional+integral+derivative,-1,1)),
                b_desired_speed=desired, b_brake=int(brake))


def process(source, path):
    route, seed = route_seed(path)
    c = Controller(preset='pursuit', **COMMON)
    hist = deque([0.]*20,maxlen=20)
    dense = dense_station(route)
    bend = {'2084':33,'27529':16}.get(route)
    rows=[]
    for line in open(path):
        f=json.loads(line)
        if not f.get('rear_waypoint') or not f.get('raw_control'):
            continue
        t=f['sim_time']; speed=f['controller_speed_mps']; signed=f['raw_signed_speed_mps']
        prev=c._last_steer; elapsed=c.dt if c._last_time is None else t-c._last_time
        c.update(f['rear_waypoint'],t,trajectory_dt=.25)
        th,steer,br=c.step(t,speed,0.)
        replay=0. if br>0 and signed<.01 else steer
        diag=c.diagnostics
        b=author_pid(f['waypoint'],signed,hist)
        q=f['raw_control']; executed=f['executed_control']
        if not q.get('C') or not q.get('D') or not executed:
            continue
        p=np.asarray(f['rear_waypoint'],float)
        horizon=float(c._arc[-1]) if c._arc is not None else float('nan')
        station=float(c._geometry(0.)[1][0]) if c._points is not None and c._geometry(0.)[1] is not None else float('nan')
        aim=np.asarray(diag['aim_xy'],float) if diag.get('aim_xy') is not None else None
        raw=float(diag.get('raw_steer',float('nan')))
        k=float('nan') if aim is None else steer_from_aim(aim,speed,c)[1]
        rate=float(np.clip(raw,prev-c.steer_rate*elapsed,prev+c.steer_rate*elapsed)) if math.isfinite(raw) else float('nan')
        b_aim=p[b['b_aim_index']]
        aim_b_raw=steer_from_aim(b_aim,speed,c)[0]
        aim_b=clamp(aim_b_raw,prev,elapsed,c)[0]
        if aim is not None:
            # Corresponding point on the author actor-origin polyline at C's rear-plan station.
            actor_points=np.vstack((np.zeros(2),np.asarray(f['waypoint'],float)))
            aim_station=min(station+diag['lookahead_m'],horizon)
            actor_aim=np.array([np.interp(aim_station,c._arc,actor_points[:,j]) for j in (0,1)])
            c_angle=math.degrees(math.atan2(float(actor_aim[1]),float(actor_aim[0])))/90.
            if signed < .01 or b['b_brake']:
                c_angle=0.
            b_on_c_raw=(1.25*c_angle + .75*(sum(hist)-b['b_angle']+c_angle)/len(hist)
                        + .3*(c_angle-hist[-2]))
            b_on_c=clamp(b_on_c_raw,prev,elapsed,c)[0]
        else:
            b_on_c=float('nan')
        no_rate=float(np.clip(raw,-c.max_steer,c.max_steer))
        no_max=rate
        # Alternate lookahead times use the same anchored trajectory and limiter history.
        alt={}
        if math.isfinite(station):
            points,geo=c._geometry(0.)
            for sec in (.3,.4,.6,.7,1.):
                dist=max(3.,sec*speed)
                alt_aim=c._lateral_aim(points,min(station+dist,horizon))
                alt[str(sec)]=clamp(steer_from_aim(alt_aim,speed,c)[0],prev,elapsed,c)[0]
        xy=f['truth']['location'][:2] if f.get('truth') else None
        dist=turn_distance(dense,bend,xy) if bend is not None and xy is not None else float('nan')
        row=dict(source=source,route=route,seed=seed,step=f['step'],sim_time=t,
                 speed=speed,raw_speed=signed,plan_curvature=curvature(f['waypoint']),
                 dense_turn_distance_m=dist, b_executed=executed['steer'],
                 b_raw=q['B']['steer'],c_shadow=q['C']['steer'],d_shadow=q['D']['steer'],
                 c_replay=replay,c_replay_diff=abs(replay-q['C']['steer']),
                 b_replay_diff=abs((0. if b['b_brake'] and signed<.01 else b['b_pid'])-q['B']['steer']),
                 horizon_m=horizon,station_m=station,lookahead_m=diag.get('lookahead_m'),
                 lookahead_s=diag.get('lookahead_m')/max(speed,.01) if diag.get('lookahead_m') is not None else float('nan'),
                 aim_x=float(aim[0]) if aim is not None else float('nan'),
                 aim_y=float(aim[1]) if aim is not None else float('nan'),
                 pursuit_curvature=k,c_nominal=raw,c_after_rate=rate,c_after_max=steer,
                 rate_clamped=int(math.isfinite(raw) and abs(rate-raw)>1e-8),
                 max_clamped=int(math.isfinite(rate) and abs(steer-rate)>1e-8),
                 horizon_capped=int(math.isfinite(station) and diag.get('lookahead_m') is not None and station+diag['lookahead_m']>horizon+1e-8),
                 aim_b=aim_b,b_pid_on_c_aim=b_on_c,no_rate=no_rate,no_max=no_max,
                 c_target_speed=diag.get('target_speed_mps'),c_reason=diag.get('reason'),
                 **{'time_'+key:val for key,val in alt.items()},**b)
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    FULL.mkdir(parents=True,exist_ok=True)
    rows=[]; runs=[]
    for source,p in paths():
        rr=process(source,p)
        route,seed=route_seed(p)
        runs.append(dict(source=source,route=route,seed=seed,path=p,ticks=len(rr),
                         max_c_replay_diff=max((r['c_replay_diff'] for r in rr),default=float('nan')),
                         max_b_replay_diff=max((r['b_replay_diff'] for r in rr),default=float('nan'))))
        rows+=rr
    write_csv(FULL/'ticks.csv',rows)
    turn=[r for r in rows if abs(r['plan_curvature'])>=.03 and r['speed']>=1.
          and abs(r['b_executed'])>=.05]
    write_csv(OUT/'turn_ticks.csv',turn)
    write_csv(OUT/'runs.csv',runs)
    print(len(runs),len(rows),'max C',max(r['c_replay_diff'] for r in rows),'max B',max(r['b_replay_diff'] for r in rows))


if __name__=='__main__': main()
