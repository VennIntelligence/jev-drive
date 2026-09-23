#!/usr/bin/env python3
"""Publication files from a closed TCP analysis edition; CPU matplotlib only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_tcp import basis,finite,vector


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_frames(path):
    rows=list(csv.DictReader(path.open()))
    for row in rows:
        for key,val in list(row.items()):
            if val in ('','None'):row[key]=None
            else:
                try:row[key]=float(val)
                except ValueError:pass
    return rows

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=250,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():p.error('Use a new figure edition')
    data=json.loads((args.analysis/'comparison.json').read_text());args.out.mkdir(parents=True)
    cases=data['cases'];sources={str((args.analysis/'comparison.json').resolve()):sha(args.analysis/'comparison.json')};records=[]
    for route in ('24211','1711','1773'):
        active=[(arm,cases[route+':'+arm]) for arm in ('native_common','pi_common') if route+':'+arm in cases]
        if not active:continue
        fig,axs=plt.subplots(4,len(active),figsize=(7*len(active),11),squeeze=False,sharex='col')
        rawsets=[]
        for col,(arm,case) in enumerate(active):
            csvpath=args.analysis/(route+'-'+arm+'-frames.csv');sources[str(csvpath.resolve())]=sha(csvpath);rows=load_frames(csvpath)
            origin=rows[0]['timestamp'];time=[r['timestamp']-origin for r in rows]
            def plot(ax,key,label,**kw):ax.plot(time,[r.get(key) if finite(r.get(key)) else float('nan') for r in rows],label=label,**kw)
            plot(axs[0,col],'speed_truth_mps','Truth longitudinal speed',color='black');plot(axs[0,col],'desired_speed_mps','Model desired speed',color='#d95f02',alpha=.85)
            axs[0,col].set_ylabel('Speed (m/s)');axs[0,col].set_title(route+' / '+arm)
            plot(axs[1,col],'accel_lon_mps2','Longitudinal',color='#1b9e77');plot(axs[1,col],'accel_lat_mps2','Lateral (right+)',color='#7570b3');axs[1,col].set_ylabel('Acceleration (m/s²)')
            plot(axs[2,col],'jerk_lon_mps3','Longitudinal',color='#1b9e77');plot(axs[2,col],'jerk_lat_mps3','Lateral (right+)',color='#7570b3');axs[2,col].set_ylabel('World-derived jerk (m/s³)')
            plot(axs[3,col],'steer','Selected steer',color='#386cb0');axs[3,col].set_ylim(-1.05,1.05);axs[3,col].axhline(1,color='gray',lw=.5);axs[3,col].axhline(-1,color='gray',lw=.5);axs[3,col].set_ylabel('Steer (right+)');axs[3,col].set_xlabel('Simulation time from first controlled tick (s)')
            first=case.get('first_collision_frame');contact=next((r['timestamp']-origin for r in rows if first is not None and r['frame']>=first),None)
            for ax in axs[:,col]:
                ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
                if contact is not None:ax.axvline(contact,color='red',ls=':',label='First contact')
            rawpath=Path(case['attempt_path'])/'tcp-control.jsonl';expected=case['source_sha256'][str(rawpath.resolve())];assert sha(rawpath)==expected
            sources[str(rawpath.resolve())]=expected;rawsets.append((arm,case,[json.loads(x) for x in rawpath.read_text().splitlines()]))
        save(fig,args.out,route+'-behavior')
        allxy=[r['truth']['xyz'] for _,_,raw in rawsets for r in raw if vector((r.get('truth') or {}).get('xyz'))]
        xrange=max(p[0] for p in allxy)-min(p[0] for p in allxy) if allxy else 1
        yrange=max(p[1] for p in allxy)-min(p[1] for p in allxy) if allxy else 1
        horizontal,vertical=(0,1) if xrange>=yrange else (1,0)
        fig,axs=plt.subplots(len(active),1,figsize=(15,1.3*len(active)+.5),squeeze=False)
        shown=[]
        for col,(arm,case,raw) in enumerate(rawsets):
            ax=axs[col,0];valid=[r for r in raw if vector((r.get('truth') or {}).get('xyz'))];xy=[r['truth']['xyz'] for r in valid]
            if xy:ax.plot([p[horizontal] for p in xy],[p[vertical] for p in xy],color='black',lw=1.5,label='Actual ego actor position')
            shown.extend(xy)
            navpath=Path(case['attempt_path'])/'tcp-route-reference.json'
            if navpath.exists():
                nav=json.loads(navpath.read_text()).get('world_route') or [];sources[str(navpath.resolve())]=sha(navpath)
                if nav:ax.plot([p[horizontal] for p in nav],[p[vertical] for p in nav],color='gray',ls=':',lw=1,label='Navigation reference')
            next_sample=None;sampled=0
            for r in valid:
                pts=(r.get('prediction') or {}).get('raw_waypoints');tf=r['truth'];t=r.get('timestamp')
                if not finite(t) or not vector(tf.get('rotation_deg')) or not isinstance(pts,list) or len(pts)!=4 or not all(vector(p,2) for p in pts):continue
                if next_sample is not None and t+1e-9<next_sample:continue
                next_sample=t+2.;sampled+=1;fwd,right=basis(tf['rotation_deg']);origin=tf['xyz']
                for offset,color,style,label in [(0.,'#e66101','-','Predictions: assumed actor origin'),(-1.4,'#5e3c99','--','Predictions: assumed GNSS origin (-1.4 m)')]:
                    world=[[origin[k]+fwd[k]*(point[0]+offset)+right[k]*point[1] for k in range(3)] for point in pts]
                    shown.extend(world)
                    ax.plot([p[horizontal] for p in world],[p[vertical] for p in world],color=color,ls=style,alpha=.45,lw=.8,label=label if sampled==1 else None)
                    for i,(local,w) in enumerate(zip(pts,world)):
                        records.append(dict(route_id=route,arm=arm,frame=r['frame'],timestamp=t,origin_offset_m=offset,point_index=i,prediction_time_s=(i+1)*.5,raw_forward_m=local[0],raw_right_m=local[1],world_x_m=w[0],world_y_m=w[1],world_z_m=w[2]))
            ax.set_title(route+' / '+arm,fontsize=9,loc='left');ax.set_xlabel('CARLA world '+('x' if horizontal==0 else 'y')+' (m)');ax.set_ylabel('world '+('x' if vertical==0 else 'y')+' (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        if shown:
            hlo,hhi=min(p[horizontal] for p in shown)-1,max(p[horizontal] for p in shown)+1
            vlo,vhi=min(p[vertical] for p in shown)-.25,max(p[vertical] for p in shown)+.25
            for ax in axs[:,0]:
                ax.set_xlim(hlo,hhi);ax.set_ylim(vhi,vlo)
            panelheight=max(1.6,15*(vhi-vlo)/max(hhi-hlo,1)+1.1)
            fig.set_size_inches(15,panelheight*len(active)+.5)
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=2,fontsize=8)
        fig.suptitle('Equal scale, shared limits; predictions every 2 s; origin hypotheses only',fontsize=10)
        save(fig,args.out,route+'-world-predictions')
    if records:
        with (args.out/'prediction-samples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    for source in [Path(__file__),Path(__file__).with_name('analyze_tcp.py')]:
        sources[str(source.resolve())]=sha(source);(args.out/source.name).write_bytes(source.read_bytes())
    (args.out/'README.md').write_text('''# TCP behavior figures

Full controlled route, including startup, stopping, faults and post-contact behavior. No smoothing or warmup removal. Time origin is per attempt; equal displayed times do not imply equal world/NPC states. Numerical frame CSVs are in the linked analysis edition recorded by manifest.json.

Behavior panels show truth longitudinal speed/model desired, longitudinal/lateral acceleration, primary world-acceleration-derived jerk projected into current body axes, and selected steer. Missing samples stay gaps. Steering differences around1e-9 can arise from float32 CARLA control storage, not algorithm changes; raw differences remain in analysis.

World panels are stacked with shared tight limits and equal distance scale. World axes are reordered when needed to put route extent horizontally; axis labels disclose the order, without anisotropic stretching. World panels use actual actor position and navigation reference, with raw model points sampled every2s. Orange assumes actor origin, purple assumes GNSS offset−1.4m. Both are anchored on contemporaneous truth orientation for visualization only. Neither validates the training physical origin or sensor pose, and neither is a supervised model-execution error metric. Predictions are finite2s horizons, never extended. Full raw predictions and truth remain in source logs.

A/B official historical speed differences are not PI or harness gains: both present arms remove the official low-speed throttle cap. Six-case behavior acceptance is separate from an individual plot looking smoother; this report does not establish a new default or leaderboard gain.
''')
    (args.out/'manifest.json').write_text(json.dumps(dict(analysis_edition=str(args.analysis.resolve()),sources=sources,outputs={p.name:sha(p) for p in args.out.iterdir() if p.is_file()},prediction_sampling_s=2.,prediction_origin_hypotheses_m=[0.,-1.4],body_orientation='Contemporaneous truth rotation, visualization hypothesis only'),indent=2))
    print(json.dumps(dict(figure_files=len(list(args.out.glob('*.png'))),out=str(args.out))))

if __name__=='__main__':main()
