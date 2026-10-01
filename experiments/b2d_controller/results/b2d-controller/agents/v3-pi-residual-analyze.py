"""Read-only PI residual diagnostics. Same-state alternatives are not plant predictions."""
import json, math, hashlib
from pathlib import Path
from collections import Counter
import numpy as np

BASE=Path('/data/runs/b2d/controller/development-v3')
OUT=Path(__file__).with_name('v3-pi-residual-data.json')

def stats(a):
 a=np.asarray(a,dtype=float)
 if not len(a):return {'count':0}
 return dict(count=len(a),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),min=float(a.min()),p05=float(np.percentile(a,5)),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))

def analyze(row):
 p=BASE/row['route_id']/row['variant']/row['preset'];truth=json.loads((p/'validation_trace.json').read_text());controls={c['frame']:c for c in map(json.loads,(p/'control.jsonl').read_text().splitlines())}
 pairs=[(t,controls[t['frame']]) for t in truth];cruise=row['cruise_mps'];gate=[(t,c) for t,c in pairs if t['elapsed_s']>=5 and t['reference_speed_mps']>=cruise-.1];e=np.array([t['speed']-cruise for t,c in gate]);sq=float(sum(e*e));assert abs(stats(e)['rms']-row['speed_rms_mps'])<1e-12
 def subset(gs):
  ts=[t for t,c in gs];cs=[c for t,c in gs]
  errors=[t['speed']-cruise for t,c in gs]
  return dict(error=stats(errors),squared_error_share=sum(x*x for x in errors)/sq if sq else 0,target=stats([c['target_speed_mps'] for c in cs]),integral_effort=stats([c['longitudinal_integral_effort'] for c in cs]),integration_limited_fraction=float(np.mean([c['longitudinal_integration_limited'] for c in cs])) if cs else None,throttle=stats([c['throttle'] for c in cs]),brake=stats([c['brake'] for c in cs]),gear_counts=dict(Counter(str(t['applied_control']['gear']) for t in ts)))
 bins={('%s-%s'%(lo,hi)):subset([(t,c) for t,c in gate if lo<=t['elapsed_s']<hi]) for lo,hi in [(5,10),(10,15),(15,20),(20,25),(25,1000)]}
 states={name:subset([(t,c) for t,c in gate if fn(t,c)]) for name,fn in [('integration_frozen',lambda t,c:c['longitudinal_integration_limited']),('integration_active',lambda t,c:not c['longitudinal_integration_limited']),('underspeed_over_05',lambda t,c:t['speed']<cruise-.5),('overspeed',lambda t,c:t['speed']>cruise)]}
 # Hold observed state and I fixed. This is algebraic sensitivity, not replay of unseen closed loop.
 same=[]
 for t,c in gate:
  err=c['target_speed_mps']-c['speed_mps'];inte=c['longitudinal_integral_effort'];newraw=.5*err+inte;newcandidate=inte+.25*err*.05
  same.append(dict(t=t['elapsed_s'],error=err,I=inte,actual_effort=c['longitudinal_effort'],half_kp_same_I_effort=float(np.clip(newraw,-1,.75)),was_limited=c['longitudinal_integration_limited'],half_kp_candidate_limited=(.5*err+newcandidate>.75 and err>0) or (.5*err+newcandidate< -1 and err<0)))
 shifts=[]
 for i,(t,c) in enumerate(gate):
  if i and t['applied_control']['gear']!=gate[i-1][0]['applied_control']['gear']:shifts.append(dict(t=t['elapsed_s'],before=gate[i-1][0]['applied_control']['gear'],after=t['applied_control']['gear'],speed=t['speed'],integral=c['longitudinal_integral_effort']))
 freeze_intervals=[]
 for i,(t,c) in enumerate(gate):
  if not c['longitudinal_integration_limited'] or (i and gate[i-1][1]['longitudinal_integration_limited']):continue
  j=i
  while j+1<len(gate) and gate[j+1][1]['longitudinal_integration_limited']:j+=1
  freeze_intervals.append(dict(start=t['elapsed_s'],duration=(j-i+1)*.05,speed_start=t['speed'],speed_end=gate[j][0]['speed'],integral_start=c['longitudinal_integral_effort'],integral_end=gate[j][1]['longitudinal_integral_effort']))
 return dict(source=str(p),official_speed_rms=row['speed_rms_mps'],failed_gates=[k for k,v in row['gates'].items() if not v],gate=subset(gate),pre5_diagnostic=subset([(t,c) for t,c in pairs if t['elapsed_s']<5 and t['reference_speed_mps']>=cruise-.1]),time_bins=bins,post10_diagnostic=subset([(t,c) for t,c in gate if t['elapsed_s']>=10]),post15_diagnostic=subset([(t,c) for t,c in gate if t['elapsed_s']>=15]),states=states,gear_transitions=shifts,freeze_intervals=freeze_intervals,same_state_half_kp=dict(note='Observed target/speed/I fixed; not a simulated result or expected gate',frozen_rows=len([x for x in same if x['was_limited']]),newly_unfrozen_rows=len([x for x in same if x['was_limited'] and not x['half_kp_candidate_limited']]),candidate_limited_rows=len([x for x in same if x['half_kp_candidate_limited']]),effort=stats([x['half_kp_same_I_effort'] for x in same]),brake=stats([max(0,-x['half_kp_same_I_effort']) for x in same])),source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [p/'validation.json',p/'validation_trace.json',p/'control.jsonl']})

if __name__=='__main__':
 rows=json.loads((BASE/'summary.json').read_text());assert len(rows)==18 and all(r['status']=='completed' for r in rows)
 result={r['route_id']+'/'+r['variant']+'/'+r['preset']:analyze(r) for r in rows}
 with OUT.open('x') as f:json.dump(result,f,indent=2)
 for k,v in result.items():
  print(k,'RMS',round(v['official_speed_rms'],6),'post10',round(v['post10_diagnostic']['error'].get('rms',0),4),'post15',round(v['post15_diagnostic']['error'].get('rms',0),4),'freeze%',round(100*v['gate']['integration_limited_fraction'],1),'I',round(v['gate']['integral_effort']['median'],3),'half-unfreeze',v['same_state_half_kp']['newly_unfrozen_rows'],'/',v['same_state_half_kp']['frozen_rows'])
