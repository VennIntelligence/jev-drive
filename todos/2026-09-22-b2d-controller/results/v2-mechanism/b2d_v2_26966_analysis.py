import json,hashlib,math
from pathlib import Path
import numpy as np
BASE=Path('/data/runs/b2d/controller/development3/26966')
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def stats(a):
 a=np.array(a,float)
 return dict(n=len(a),rms=float(np.sqrt(np.mean(a*a))),p95=float(np.percentile(np.abs(a),95)),max=float(np.max(np.abs(a)))) if len(a) else dict(n=0)
world=np.array(load(BASE/'pursuit/route_reference.json')['world_xy'])
world=world[np.r_[True, np.linalg.norm(np.diff(world,axis=0),axis=1)>1e-8]]
d=np.diff(world,axis=0);ds=np.linalg.norm(d,axis=1);s=np.r_[0,np.cumsum(ds)]
h=np.unwrap(np.arctan2(d[:,1],d[:,0]));k=np.diff(h)/((ds[:-1]+ds[1:])/2)
print('ROUTE length/headings/curvature',s[-1],np.rad2deg(h[0]),np.rad2deg(h[-1]),np.min(k),np.max(k))
print('CURVATURE CENTERS',[(round(float(s[i+1]),2),round(float(k[i]),3)) for i in range(len(k)) if abs(k[i])>.02])
summary={}
for preset in ('carla','pursuit'):
 root=BASE/preset; v=load(root/'validation_trace.json');c=lines(root/'control.jsonl');cs={r['frame']:r for r in c}
 joined=[]
 for row in v:
  ctl=cs[row['frame']]
  yaw=ctl['truth_yaw'];heading=(yaw-row['path_heading_rad']+math.pi)%(2*math.pi)-math.pi
  joined.append(dict(row,ctl=ctl,truth_heading_error=heading))
 print('\nPRESET',preset)
 peak=max(joined,key=lambda r:abs(r['cross_track_m']))
 print('PEAK', {k:peak[k] for k in ('tick','elapsed_s','progress_m','cross_track_m','truth_heading_error','speed')})
 print('PEAKCTL', {k:peak['ctl'].get(k) for k in ('steer','raw_steer','steer_limited','lookahead_m','target_speed_mps','pose_error_m','route_cross_track_m','trajectory_age_s')})
 print('REASONS', {reason:sum(r['reason']==reason for r in c) for reason in set(r['reason'] for r in c)})
 print('LIMITS',sum(r.get('steer_limited',False) for r in c), max(abs(r['steer']) for r in c),max(abs(r.get('raw_steer') or 0) for r in c))
 total=sum(r['cross_track_m']**2 for r in joined)
 bins=[]
 for lo,hi in ((0,20),(20,28),(28,34),(34,40),(40,48),(48,60),(60,81)):
  subset=[r for r in joined if lo<=r['progress_m']<hi]
  stat=stats([r['cross_track_m'] for r in subset]);stat.update(lo=lo,hi=hi,sse_fraction=sum(r['cross_track_m']**2 for r in subset)/total)
  if subset:
   stat.update(time=[subset[0]['elapsed_s'],subset[-1]['elapsed_s']],speed_mean=float(np.mean([r['speed'] for r in subset])),cte_mean=float(np.mean([r['cross_track_m'] for r in subset])),heading_mean_deg=float(np.rad2deg(np.mean([r['truth_heading_error'] for r in subset]))),pose_p90=float(np.percentile([r['ctl']['pose_error_m'] for r in subset],90)),steer_max=max(abs(r['ctl']['steer']) for r in subset))
  bins.append(stat);print('BIN',stat)
 print('MOVING',stats([r['cross_track_m'] for r in joined if r['speed']>=.5]))
 print('SAMPLES near bend')
 for progress in (20,24,28,30,32,34,36,38,40,42,44,48,52):
  r=min(joined,key=lambda r:abs(r['progress_m']-progress));z=r['ctl']
  effective=math.atan(2.86047149*(-z['yaw_rate_rps'])/max(z['speed_mps'],.1))/(math.radians(70)*np.interp(z['speed_mps']*3.6,[0,20,60,120],[1,.9,.8,.7]))
  print(progress, 't %.2f s %.2f cte %.3f head %.2f v %.2f u %.3f equiv %.3f LA %.2f pose %.3f estcte %.3f'%(r['elapsed_s'],r['progress_m'],r['cross_track_m'],math.degrees(r['truth_heading_error']),z['speed_mps'],z['steer'],effective,z.get('lookahead_m',0),z['pose_error_m'],z['route_cross_track_m']))
 summary[preset]=dict(full=stats([r['cross_track_m'] for r in joined]),moving=stats([r['cross_track_m'] for r in joined if r['speed']>=.5]),bins=bins,peak={k:peak[k] for k in ('tick','elapsed_s','progress_m','cross_track_m','truth_heading_error','speed')},steer_limited_ticks=sum(r.get('steer_limited',False) for r in c),max_abs_steer=max(abs(r['steer']) for r in c),source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'control.jsonl',root/'validation_trace.json',root/'trajectories.jsonl',root/'route_reference.json')})
Path('/tmp/b2d_v2_26966_analysis.json').write_text(json.dumps(summary,indent=2))
