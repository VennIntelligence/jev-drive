"""D3b 2091 launch and hold telemetry, including scene context."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from b2d_tfv6_d3a import OUT, speed_options
from b2d_tfv6_d3_analyze import ROOT


def main():
    rows=[]
    for seed in range(3):
        for arm in 'BCEF':
            case=ROOT/'factorial/cases/1/route-2091'/f'seed-{seed}'/arm
            done=json.loads((case/'done.json').read_text())
            attempt=case/f"attempt-{done['attempt']}"
            semantic={x['step']:x for x in (json.loads(line) for line in (attempt/'d3_nav_semantic.jsonl').open())}
            for frame in (json.loads(line) for line in (attempt/'frames.jsonl').open()):
                if frame['step']>1200:break
                speeds=speed_options(frame) or {}
                raw=frame.get('raw_control') or {}
                actual=frame.get('executed_control') or {}
                final=frame.get('final_control') or {}
                nav=frame.get('d3_nav') or {}
                actors=frame.get('nearby_actors') or []
                ahead=[a for a in actors if a['type'].startswith(('vehicle.','walker.')) and
                       a['relative_xy_m'][0]>0 and abs(a['relative_xy_m'][1])<3]
                nearest=min((a['distance_m'] for a in ahead),default=None)
                chosen=raw.get(arm) or {}
                processed=final.get(arm) or {}
                row={'seed':seed,'arm':arm,'official_status':done['official_status'],
                     'step':frame['step'],'sim_time_s':frame['sim_time'],
                     'ego_speed_mps':frame['truth']['forward_speed_mps'],
                     'target_dense_index':(semantic.get(frame['step']) or {}).get('target_dense_index'),
                     'selected_pop_distance_m':nav.get('selected_pop_distance'),
                     'first_segment_mps':speeds.get('first'),'author_mps':speeds.get('author'),
                     'mean8_mps':speeds.get('mean8'),'tfv6_target_mps':frame.get('target_speed'),
                     'c_reason':(frame.get('controller_reason') or {}).get('C'),
                     'b_shadow_throttle':(raw.get('B') or {}).get('throttle'),
                     'b_shadow_brake':(raw.get('B') or {}).get('brake'),
                     'c_shadow_throttle':(raw.get('C') or {}).get('throttle'),
                     'c_shadow_brake':(raw.get('C') or {}).get('brake'),
                     'arm_raw_throttle':chosen.get('throttle'),'arm_raw_brake':chosen.get('brake'),
                     'actual_throttle':actual.get('throttle'),'actual_brake':actual.get('brake'),
                     'post_longitudinal_changed':bool(chosen and processed and
                         any(abs(chosen[k]-processed[k])>1e-5 for k in ('throttle','brake'))),
                     'creep_active':(frame.get('heuristic') or {}).get('creep_active'),
                     'stop_sign_active':(frame.get('heuristic') or {}).get('stop_sign_active'),
                     'light_state':(frame.get('d3_traffic_light') or {}).get('state'),
                     'at_light':(frame.get('d3_traffic_light') or {}).get('at_light'),
                     'nearest_front_vehicle_or_walker_m':nearest}
                rows.append(row)
    path=OUT/'d3b-2091-launch.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print({'rows':len(rows),'path':str(path)})


if __name__=='__main__':main()
