"""Read-only longitudinal pulse diagnostics on preserved controller traces."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

BASE = Path('/data/runs/b2d/controller')
CASES = [
    ('v1_s_carla', BASE/'development-s-v1/17563/carla', 6.),
    ('v1_s_pursuit', BASE/'development-s-v1/17563/pursuit', 6.),
    ('v2_s_carla', BASE/'development-v2/17563/baseline/carla', 6.),
    ('v2_s_pursuit', BASE/'development-v2/17563/baseline/pursuit', 6.),
    ('v2_s_pursuit_max', BASE/'development-v2/17563/max/pursuit', 6.),
    ('v2_straight8_carla', BASE/'development-v2/1773/baseline/carla', 8.),
    ('v2_straight8_pursuit', BASE/'development-v2/1773/baseline/pursuit', 8.),
]


def distribution(values):
    if not len(values):
        return {'n': 0}
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), rms=float(np.sqrt(np.mean(a*a))),
                minimum=float(a.min()), maximum=float(a.max()),
                p05=float(np.percentile(a,5)), median=float(np.median(a)), p95=float(np.percentile(a,95)))


def analyze(root, cruise):
    trace = json.loads((root/'validation_trace.json').read_text())
    controls = {r['frame']: r for r in map(json.loads, (root/'control.jsonl').read_text().splitlines())}
    paired = [(r, controls[r['frame']]) for r in trace]
    eligible = [i for i,(r,c) in enumerate(paired) if r['elapsed_s']>=5 and r['reference_speed_mps']>=cruise-.1]
    rows=[paired[i][0] for i in eligible];cs=[paired[i][1] for i in eligible]
    stable=[i for i in eligible if abs(paired[i][1]['target_speed_mps']-cruise)<.01]
    pulses=[]
    for i in eligible:
        if controls[trace[i]['frame']]['brake']<=0 or (i and controls[trace[i-1]['frame']]['brake']>0):
            continue
        end=i
        while end+1<len(trace) and controls[trace[end+1]['frame']]['brake']>0:
            end+=1
        stop=min(i+6,end+1)
        peak=max(range(i,stop+1),key=lambda k:trace[k]['speed'])
        release=end+1
        if release>=len(trace):
            continue
        next_brake=next((k for k in range(release+1,min(release+21,len(trace)))
                         if controls[trace[k]['frame']]['brake']>0), min(release+21,len(trace)))
        minimum=min(range(release,next_brake),key=lambda k:trace[k]['speed'])
        pulses.append(dict(start_t=trace[i]['elapsed_s'],duration_s=(end-i+1)*.05,
                           onset_speed=trace[i]['speed'],peak_after_brake_speed=trace[peak]['speed'],
                           onset_to_peak_s=(peak-i)*.05,release_t=trace[release]['elapsed_s'],
                           release_speed=trace[release]['speed'],minimum_after_release_speed=trace[minimum]['speed'],
                           release_to_minimum_s=(minimum-release)*.05,
                           release_to_minimum_drop_mps=trace[release]['speed']-trace[minimum]['speed']))
    throttle_plateaus=[]
    for i in eligible:
        c=paired[i][1]
        under=lambda j:paired[j][1]['throttle']>=.749 and paired[j][1]['brake']==0 and paired[j][0]['speed']<paired[j][1]['target_speed_mps']-.5
        if not under(i) or (i and under(i-1)):
            continue
        end=i
        while end+1<len(paired) and under(end+1):end+=1
        if (end-i+1)*.05>=.25:
            throttle_plateaus.append(dict(start_t=trace[i]['elapsed_s'],duration_s=(end-i+1)*.05,
                                          first_speed=trace[i]['speed'],last_speed=trace[end]['speed']))
    speed_error=[r['speed']-cruise for r in rows]
    actual_acceleration=[(trace[i+1]['speed']-trace[i]['speed'])/.05 for i in eligible if i+1<len(trace)]
    output=dict(cruise_mps=cruise,gate_speed_error=distribution(speed_error),
                stable_target_speed_error=distribution([paired[i][0]['speed']-cruise for i in stable]),
                target=distribution([c['target_speed_mps'] for c in cs]),
                pitch_deg=distribution([r['pitch_deg'] for r in rows]),
                max_abs_pitch_gravity_component_mps2=9.81*math.sin(math.radians(max(abs(r['pitch_deg']) for r in rows))),
                measured_acceleration_mps2=distribution(actual_acceleration),
                brake_fraction=float(np.mean([c['brake']>0 for c in cs])),
                full_brake_fraction=float(np.mean([c['brake']>=.999 for c in cs])),
                full_throttle_fraction=float(np.mean([c['throttle']>=.749 for c in cs])),
                cruise_reasons=sorted(set(c['reason'] for c in cs)),
                gear_recorded=any('gear' in c for c in cs),brake_pulses=pulses,
                pulse_duration=distribution([p['duration_s'] for p in pulses]),
                onset_to_peak=distribution([p['onset_to_peak_s'] for p in pulses]),
                release_to_minimum=distribution([p['release_to_minimum_s'] for p in pulses]),
                release_to_minimum_drop=distribution([p['release_to_minimum_drop_mps'] for p in pulses]),
                full_throttle_under_target_plateaus=throttle_plateaus,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (root/'validation_trace.json',root/'control.jsonl',root/'validation.json')})
    expected=json.loads((root/'validation.json').read_text())['speed_rms_mps']
    assert abs(output['gate_speed_error']['rms']-expected)<1e-12
    return output


if __name__=='__main__':
    destination=Path(__file__).with_name('v2-longitudinal-data.json')
    if destination.exists():raise FileExistsError(destination)
    result={label:analyze(root,cruise) for label,root,cruise in CASES}
    with destination.open('x') as stream:json.dump(result,stream,indent=2)
    for label,r in result.items():
        print(label, 'gate_rms',r['gate_speed_error']['rms'],'stable_target_rms',r['stable_target_speed_error']['rms'],
              'pulses',len(r['brake_pulses']),'duration_med',r['pulse_duration'].get('median'),
              'onsetpeak_med',r['onset_to_peak'].get('median'),'release_min_med',r['release_to_minimum'].get('median'),
              'release_drop_med',r['release_to_minimum_drop'].get('median'),'plateaus',len(r['full_throttle_under_target_plateaus']))
