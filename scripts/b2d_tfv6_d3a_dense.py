"""Re-score D3a branch preference using evaluator-dense route geometry."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from b2d_tfv6_d3a import OUT, polyline_distance

ROOT=Path('/data/runs/b2d/tfv6-d3/dense')
BEND={'2084':33,'27529':16}


def main():
    packages={r:json.loads((ROOT/f'2-{r}.json').read_text()) for r in BEND}
    geometry={}
    for route,package in packages.items():
        dense=np.asarray([item['xyz'][:2] for item in package['dense']],dtype=float)
        i=BEND[route]
        # Continue the inbound lane direction from the beginning of the turn.
        direction=dense[i]-dense[i-5];direction/=np.linalg.norm(direction)
        wrong=np.asarray([dense[i]+direction*t for t in np.linspace(0,70,141)])
        geometry[route]=(dense[i:],wrong,dense[i])
    rows=[];first={}
    with (OUT/'route-timeline.csv').open() as stream:
        for row in csv.DictReader(stream):
            route=row['route']
            if route not in geometry:continue
            correct,wrong,bend=geometry[route]
            ego=np.asarray([float(row['x']),float(row['y'])])
            distance_to_bend=float(np.linalg.norm(ego-bend))
            for name in ('route','waypoint'):
                world=np.asarray(json.loads(row[f'{name}_world_xy_json']),dtype=float)
                endpoint=world[-1:] if len(world) else world
                dc=polyline_distance(endpoint,correct)
                dw=polyline_distance(endpoint,wrong)
                preference=bool(dc is not None and dw is not None and distance_to_bend<35 and dw+2<dc)
                result={'level':row['level'],'route':route,'seed':row['seed'],'arm':row['arm'],
                        'window_kind':row['window_kind'],'step':row['step'],'sim_time_s':row['sim_time_s'],
                        'ego_x':row['x'],'ego_y':row['y'],'distance_to_dense_bend_m':distance_to_bend,
                        'prediction':name,'endpoint_correct_dense_m':dc,'endpoint_wrong_straight_m':dw,
                        'wrong_preferred_margin_2m':preference}
                rows.append(result)
                key=(row['level'],route,row['seed'],row['arm'],row['window_kind'],name)
                if preference and key not in first:first[key]=result
    path=OUT/'d3a-dense-branch.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    summary=[]
    for key,result in first.items():
        summary.append(dict(zip(('level','route','seed','arm','window_kind','prediction'),key),
                            first_wrong_step=result['step'],first_wrong_s=result['sim_time_s'],
                            endpoint_correct_dense_m=result['endpoint_correct_dense_m'],
                            endpoint_wrong_straight_m=result['endpoint_wrong_straight_m']))
    path=OUT/'d3a-dense-first-wrong.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summary)
    print({'rows':len(rows),'first_wrong':len(summary)})


if __name__=='__main__':main()
