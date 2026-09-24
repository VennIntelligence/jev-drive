"""D3b factorial outcome grid using the paper plot style."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from research import plot_style

OUT=REPO/'todos/2026-09-23-tfv6-controller/results/diagnosis/d3'
ROOT=Path('/data/runs/b2d/tfv6-d3')
GROUPS=[('2084',seed) for seed in range(3)]+[('27529',seed) for seed in range(3)]+[('2091',seed) for seed in range(3)]
ARMS='BCEF'


def category(status):
    lower=status.lower()
    if status=='Completed':return 'completed'
    if 'deviated from the route' in lower:return 'deviation'
    if 'got blocked' in lower:return 'blocked'
    if 'tickruntime' in lower:return 'tick limit'
    return 'other'


def main():
    rows=list(csv.DictReader((OUT/'d3b-cases.csv').open()))
    data={(r['route'],int(r['seed']),r['arm']):r for r in rows if r['phase']=='factorial'}
    if len(data)!=36:raise RuntimeError(f'Expected 36 factorial cases, found {len(data)}')
    colors={'completed':plot_style.PALETTE['green'],
            'deviation':plot_style.PALETTE['vermillion'],
            'blocked':plot_style.PALETTE['orange'],
            'tick limit':plot_style.PALETTE['purple'],
            'other':plot_style.BASELINE}
    plot_style.apply()
    fig,ax=plt.subplots(figsize=(plot_style.DOUBLE_COLUMN_IN,4.2))
    for yi,(route,seed) in enumerate(GROUPS):
        for xi,arm in enumerate(ARMS):
            item=data[(route,seed,arm)]
            kind=category(item['official_status'])
            ax.add_patch(Rectangle((xi-.47,yi-.44),.94,.88,facecolor=colors[kind],alpha=.19,
                                   edgecolor=colors[kind],linewidth=.8))
            ax.text(xi,yi-.09,f"DS {float(item['ds']):.1f}",ha='center',va='center',fontsize=8,color='#222222')
            ax.text(xi,yi+.17,kind,ha='center',va='center',fontsize=6.6,color=colors[kind])
    ax.set_xlim(-.5,3.5);ax.set_ylim(8.5,-.5)
    ax.set_xticks(range(4),list(ARMS));ax.xaxis.tick_top()
    ax.set_yticks(range(9),[f'{r} / seed {s}' for r,s in GROUPS])
    ax.tick_params(length=0);ax.grid(False)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title('D3b factorial: official outcome and driving score',pad=22)
    handles=[Patch(facecolor=v,alpha=.25,edgecolor=v,label=k) for k,v in colors.items()]
    ax.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.11),fontsize=7)
    fig.subplots_adjust(left=.18,right=.99,top=.84,bottom=.15)
    metadata=plot_style.save(fig,OUT/'d3b-factorial-outcomes')
    print(metadata)
    plt.close(fig)
    for route in ('2084','27529'):
        fig,axes=plt.subplots(1,3,figsize=(plot_style.DOUBLE_COLUMN_IN,2.65),sharex=True,sharey=True)
        package=json.loads((ROOT/'dense'/f'2-{route}.json').read_text())
        dense=np.asarray([p['xyz'][:2] for p in package['dense']])
        arm_colors={'B':plot_style.PALETTE['blue'],'C':plot_style.PALETTE['vermillion'],
                    'E':plot_style.PALETTE['orange'],'F':plot_style.PALETTE['green']}
        for seed,ax in enumerate(axes):
            ax.plot(dense[:,0],dense[:,1],color=plot_style.BASELINE,linestyle='--',
                    linewidth=1.2,label='evaluator dense route',zorder=1)
            for arm in ARMS:
                item=data[(route,seed,arm)]
                case=ROOT/'factorial/cases/2'/f'route-{route}'/f'seed-{seed}'/arm
                done=json.loads((case/'done.json').read_text())
                attempt=case/f"attempt-{done['attempt']}"
                truth=[json.loads(s)['truth']['location'][:2] for s in (attempt/'frames.jsonl').open()]
                xy=np.asarray(truth)
                ax.plot(xy[:,0],xy[:,1],color=arm_colors[arm],linewidth=1.1,
                        label=arm,zorder=2)
                off=item['first_ego_offroute_3m_step']
                if off:
                    ix=int(off)
                    ax.scatter(xy[ix,0],xy[ix,1],s=17,facecolors='none',
                               edgecolors=arm_colors[arm],linewidths=.9,zorder=3)
            ax.set_aspect('equal',adjustable='box')
            ax.set_title(f'seed {seed}')
            ax.set_xlabel('CARLA x (m)')
            if seed==0:ax.set_ylabel('CARLA y (m)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=5,fontsize=6.7)
        fig.suptitle(f'Route {route}: factorial ego paths and first >3 m departure',y=.99)
        fig.subplots_adjust(left=.08,right=.99,top=.84,bottom=.23,wspace=.16)
        metadata=plot_style.save(fig,OUT/f'd3b-{route}-paths')
        print(route,metadata)
        plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(plot_style.DOUBLE_COLUMN_IN,2.55),sharex=True,sharey=True)
    arm_colors={'B':plot_style.PALETTE['blue'],'C':plot_style.PALETTE['vermillion'],
                'E':plot_style.PALETTE['orange'],'F':plot_style.PALETTE['green']}
    for seed,ax in enumerate(axes):
        for arm in ARMS:
            case=ROOT/'factorial/cases/1/route-2091'/f'seed-{seed}'/arm
            done=json.loads((case/'done.json').read_text())
            attempt=case/f"attempt-{done['attempt']}"
            series=[]
            for line in (attempt/'frames.jsonl').open():
                frame=json.loads(line)
                if frame['sim_time']>60:break
                if frame['step']%5==0 and frame.get('truth'):
                    series.append((frame['sim_time'],frame['truth']['forward_speed_mps']))
            if series:
                xy=np.asarray(series)
                ax.plot(xy[:,0],xy[:,1],color=arm_colors[arm],linewidth=.8,label=arm)
        ax.set_xlim(0,60);ax.set_ylim(-.2,11)
        ax.set_title(f'seed {seed}')
        ax.set_xlabel('simulation time (s)')
        if seed==0:ax.set_ylabel('ego speed (m/s)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=4)
    fig.suptitle('Route 2091: launch and subsequent stops',y=.99)
    fig.subplots_adjust(left=.08,right=.99,top=.83,bottom=.26,wspace=.13)
    metadata=plot_style.save(fig,OUT/'d3b-2091-launch-speed')
    print('2091',metadata)
    plt.close(fig)


if __name__=='__main__':main()
