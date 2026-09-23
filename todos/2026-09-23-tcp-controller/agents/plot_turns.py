#!/usr/bin/env python3
"""固定窗口和全路线的英文图；读取冻结 helper，不修改任何运行输入。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent
COLORS={'baseline-max':'#2563a6','short-max':'#d35f20'}
LABELS={'baseline-max':'Baseline: max(3 m, 0.5 s × speed)',
        'short-max':'Candidate: max(3 m, 0.375 s × speed)'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--analysis',type=Path,required=True,help='冻结分析输出目录')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--frozen-helper',type=Path,default=BASE/'turns-analysis-freeze-v2/analyze_turns.py')
    ap.add_argument('--windows',type=Path,default=BASE/'lateral-evidence-v1/turn-windows.csv')
    args=ap.parse_args()
    if args.out.exists():raise SystemExit('拒绝覆盖已有图版；请选择新目录。')
    ns=runpy.run_path(str(args.frozen_helper))
    sources={}
    def record(p):
        p=Path(p);sources[str(p.resolve())]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    for p in (__file__,args.frozen_helper,args.windows,args.analysis/'manifest.json',args.analysis/'required-conditions.json'):
        record(p)
    analysis_manifest=json.loads((args.analysis/'manifest.json').read_text())
    if not analysis_manifest.get('complete_input_matrix'):raise SystemExit('分析的六例输入不完整，不生成最终图版。')
    for path,meta in analysis_manifest['inputs'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=meta['sha256']:
            raise SystemExit('分析来源已改变：'+path)
    windows={r:[] for r in ns['ROUTES']}
    for r in csv.DictReader(args.windows.open()):
        if r['dataset']=='G2-v4' and r['route'] in windows:
            windows[r['route']].append({k:v if k in ('route','dataset','kind') else float(v) for k,v in r.items()})
    data={};references={}
    for route in ns['ROUTES']:
        for variant in ns['VARIANTS']:
            folder=args.run_root/route/variant/'pursuit'
            for name in ('control.jsonl','validation_trace.json','agent_config.json','route_reference.json'):
                record(folder/name)
            cfg=json.loads((folder/'agent_config.json').read_text());configpath=Path(cfg['controller_config']);record(configpath)
            controller=json.loads(configpath.read_text())
            reference=np.asarray(json.loads((folder/'route_reference.json').read_text())['world_xy'],float)
            references[route]=reference
            controls=[json.loads(line) for line in (folder/'control.jsonl').read_text().splitlines() if line]
            trace=json.loads((folder/'validation_trace.json').read_text())
            rows,_=ns['project_rows'](controls,reference,controller.get('max_steer',.8),controller.get('steer_rate',2.))
            feature=ns['plant_features'](trace);byframe={r['frame']:r for r in controls}
            for row in rows:
                row.update(feature.get(row['frame'],{}))
                raw=byframe[row['frame']];row['x'],row['y']=raw['truth_xy']
                row['elapsed_s']=row['time']-controls[0]['sim_time']
            data[(route,variant)]=rows
    args.out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'grid.alpha':.2,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
    generated=[]
    def save(fig,name):
        fig.savefig(args.out/(name+'.png'),bbox_inches='tight')
        fig.savefig(args.out/(name+'.pdf'),bbox_inches='tight')
        generated.extend([name+'.png',name+'.pdf']);plt.close(fig)
    def draw_xy(ax,route,selections,window=None):
        reference=references[route];origin=reference[0]
        if window is not None:
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
            mask=(distance>=max(0,window['start_m']-3))&(distance<=window['end_m']+3)
            ref=reference[mask]
        else:ref=reference
        ax.plot(ref[:,0]-origin[0],ref[:,1]-origin[1],'--',color='#555555',lw=1.4,label='Original route')
        for variant,rows in selections.items():
            ax.plot([r['x']-origin[0] for r in rows],[r['y']-origin[1] for r in rows],
                color=COLORS[variant],lw=1.4,label=LABELS[variant])
        ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis()
        ax.set_xlabel('World x from route start (m)');ax.set_ylabel('World y from route start (m; CARLA)')
    for route in ns['ROUTES']:
        selections={v:data[(route,v)] for v in ns['VARIANTS']}
        fig,axes=plt.subplots(1,3,figsize=(15,4.1),gridspec_kw={'width_ratios':[1,1.1,1.1]})
        draw_xy(axes[0],route,selections)
        for variant,rows in selections.items():
            t=[r['elapsed_s'] for r in rows]
            axes[1].plot(t,[r['cte'] for r in rows],color=COLORS[variant],label=LABELS[variant])
            axes[2].plot(t,[r['steer'] for r in rows],color=COLORS[variant],label=LABELS[variant])
        axes[1].set(xlabel='Time from first control frame (s)',ylabel='CTE (m, LEFT positive)')
        axes[2].set(xlabel='Time from first control frame (s)',ylabel='Emitted steer (normalized, RIGHT positive)')
        for ax in axes[1:]:ax.axhline(0,color='gray',lw=.6)
        axes[0].legend(fontsize=7,loc='best')
        fig.suptitle(f'Route {route}: complete trajectory and all recorded frames\nDiagnostic route oracle | policy none | no frame exclusion')
        fig.tight_layout();save(fig,f'route-{route}-full')
        # 每个曲线点的精确输入，包含停车与首帧缺失导数；不隐藏样本。
        rows=[dict(route=route,variant=v,**r) for v in ns['VARIANTS'] for r in selections[v]]
        keys=list(dict.fromkeys(k for row in rows for k in row))
        name=f'route-{route}-samples.csv'
        with (args.out/name).open('w') as f:
            writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
        generated.append(name)
        for window in windows[route]:
            part={v:[r for r in selections[v] if window['start_m']<=r['progress']<=window['end_m']] for v in ns['VARIANTS']}
            fig,axes=plt.subplots(2,3,figsize=(14,7.8))
            draw_xy(axes[0,0],route,part,window)
            axes[0,0].legend(fontsize=7,loc='best')
            for variant,rows in part.items():
                if not rows:continue
                t=np.array([r['time'] for r in rows])-rows[0]['time'];color=COLORS[variant]
                axes[0,1].plot(t,[r['cte'] for r in rows],color=color,label=LABELS[variant])
                axes[0,2].plot(t,[r['steer'] for r in rows],color=color,label='Emitted '+variant)
                axes[0,2].plot(t,[r['raw_steer'] for r in rows],color=color,ls=':',alpha=.65,label='Raw '+variant)
                axes[1,0].plot(t,[r['actual_speed'] for r in rows],color=color,label='Actual '+variant)
                axes[1,0].plot(t,[r['reference_speed'] for r in rows],color=color,ls=':',alpha=.65)
                axes[1,1].plot(t,[r['lateral_accel'] for r in rows],color=color)
                axes[1,2].plot(t,[r['lateral_jerk'] for r in rows],color=color)
            labels={(0,1):'CTE (m, LEFT positive)',(0,2):'Steer (normalized, RIGHT positive)',
                    (1,0):'Speed (m/s; dotted = reference)',(1,1):'Lateral acceleration (m/s², RIGHT positive)',
                    (1,2):'World-derivative lateral jerk (m/s³, RIGHT positive)'}
            for ij,label in labels.items():
                axes[ij].set(xlabel='Time from window entry (s; each run aligned)',ylabel=label)
                axes[ij].axhline(0,color='gray',lw=.6)
            axes[0,2].legend(fontsize=7)
            fig.suptitle(f"Route {route}, fixed window {int(window['segment'])}: {window['start_m']:g}–{window['end_m']:g} m\n"
                         'Diagnostic route oracle | policy none | all window frames | raw, unfiltered signals')
            fig.tight_layout();save(fig,f"route-{route}-window-{int(window['segment'])}")
    manifest=dict(inputs=sources,outputs={name:hashlib.sha256((args.out/name).read_bytes()).hexdigest() for name in generated},
        analysis=str(args.analysis.resolve()),run_root=str(args.run_root.resolve()),
        protocol='Frozen four station windows; all complete-route rows and all window rows. No moving-only plot. Each run window-entry time aligned independently. CTE LEFT positive, emitted steer/accel RIGHT positive. Jerk=world acceleration derivative projected on current right vector.',
        caveats='Signals are unfiltered. Same-frame applied_control is previous command and is not plotted as tracking error. Full CSV preserves frame/time and missing derivative NaN. No model or official-score claim.')
    (args.out/'plot-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(figures=7,formats=['png','pdf'],sample_csvs=3,out=str(args.out))))


if __name__=='__main__':main()
