"""Read-only analysis of the matched6m/s API-reported gear diagnostic."""
import importlib.util
import json
from collections import Counter
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pulse_analysis',here/'v2-longitudinal-analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
base=Path('/data/runs/b2d/controller/development-v2-speed-diagnostic')
rows=json.loads((base/'summary.json').read_text())
assert len(rows)==6 and all(r['status']=='completed' for r in rows)
result={}
for route in ('1773','17563'):
 for preset in ('carla','tcp','pursuit'):
  root=base/route/preset
  stats=module.analyze(root,6.)
  truth=json.loads((root/'validation_trace.json').read_text())
  controls={r['frame']:r for r in map(json.loads,(root/'control.jsonl').read_text().splitlines())}
  eligible=[i for i,r in enumerate(truth) if r['elapsed_s']>=5 and r['reference_speed_mps']>=5.9]
  ts=[truth[i] for i in eligible]
  counts=Counter(r['applied_control']['gear'] for r in ts)
  changes=[]
  for i in eligible:
   if i and truth[i]['applied_control']['gear']!=truth[i-1]['applied_control']['gear']:
    r=truth[i];c=controls[r['frame']]
    changes.append(dict(t=r['elapsed_s'],gear_before=truth[i-1]['applied_control']['gear'],gear_now=r['applied_control']['gear'],speed=r['speed'],reported_applied_control=r['applied_control'],current_command={k:c[k] for k in ('throttle','brake','steer','target_speed_mps')}))
  match={k:max(abs(truth[i]['applied_control'][k]-controls[truth[i-1]['frame']][k]) for i in eligible if i>0) for k in ('throttle','brake','steer')}
  gear_stats={}
  for gear in counts:
   ids=[i for i in eligible if truth[i]['applied_control']['gear']==gear]
   gear_stats[str(gear)]=dict(samples=len(ids),speed=module.distribution([truth[i]['speed'] for i in ids]),
          next_interval_acceleration=module.distribution([(truth[i+1]['speed']-truth[i]['speed'])/.05 for i in ids if i+1<len(truth)]),
          applied_throttle_mean=float(np.mean([truth[i]['applied_control']['throttle'] for i in ids])),
          applied_brake_mean=float(np.mean([truth[i]['applied_control']['brake'] for i in ids])))
  stats.update(api_reported_gear_counts=dict(counts),gear_transition_counts=dict(Counter('%s>%s'%(r['gear_before'],r['gear_now']) for r in changes)),
      gear_changes=changes,api_pedals_max_abs_difference_vs_previous_command=match,per_reported_gear=gear_stats,
      note='gear0 occurs in moving cruise; APIgear is observed, internal clutch/RPM/wheel slip unavailable')
  result[route+'/'+preset]=stats
output=here/'v2-longitudinal-gear-data.json'
with output.open('x') as stream:json.dump(result,stream,indent=2)
for key,r in result.items():
 print(key,'rms',round(r['gate_speed_error']['rms'],4),'gears',r['api_reported_gear_counts'],'changes',r['gear_transition_counts'],'pedaldiff',r['api_pedals_max_abs_difference_vs_previous_command'],'pulsemedian',r['pulse_duration']['median'],'releasedrop',r['release_to_minimum_drop']['median'])
