"""Paper-style figures for the registered privileged-ceiling readouts (English labels)."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from research.plot_style import apply, PALETTE, DOUBLE_COLUMN_IN, save
from b2d_privileged_geometry import ARMS


def figures(root,diagnostic=False):
    apply();out=root/'summary';r=pd.read_csv(out/'routes.csv');e=pd.read_csv(out/'events.csv');v=pd.read_csv(out/'visibility.csv')
    p=pd.read_csv(out/'paired.csv');colors=list(PALETTE.values())
    fig,axs=plt.subplots(1,2,figsize=(DOUBLE_COLUMN_IN,2.7),layout='constrained')
    y=np.arange(len(p));axs[0].barh(y,p.dDS,color=colors[:len(p)])
    axs[0].errorbar(p.dDS,y,xerr=np.stack([p.dDS-p.DS_lo,p.DS_hi-p.dDS]),fmt='none',ecolor='#222222',lw=.7,capsize=2)
    axs[0].set(yticks=y,yticklabels=p.arm,xlabel='Paired DS change',title='(a) Low-score diagnostic routes' if diagnostic else '(a) Registered skill cohorts')
    axs[0].axvline(0,color='#777777',lw=.6)
    axs[1].barh(y,100*p.fail_delta,color=colors[:len(p)])
    axs[1].errorbar(100*p.fail_delta,y,xerr=100*np.stack([p.fail_delta-p.fail_lo,p.fail_hi-p.fail_delta]),fmt='none',ecolor='#222222',lw=.7,capsize=2)
    axs[1].set(yticks=y,yticklabels=p.arm,xlabel='Failure-rate change (pp)',title='(b) Route-cluster 95% intervals')
    axs[1].axvline(0,color='#777777',lw=.6);save(fig,out/'paired_effects');plt.close(fig)
    fig,ax=plt.subplots(figsize=(DOUBLE_COLUMN_IN,2.7),layout='constrained')
    keys=['recover_vehicle','recover_red','recover_route_completion','recover_layout','recover_pedestrian']
    labels=['Vehicle contact','Red light','Incomplete route','Static contact','Pedestrian contact']
    means=r.groupby('arm')[keys].mean().reindex(ARMS);x=np.arange(len(ARMS));width=.15
    for j,(key,label) in enumerate(zip(keys,labels)):
        ax.bar(x+(j-2)*width,means[key],width,label=label,color=colors[j])
    ax.set(xticks=x,xticklabels=ARMS,ylabel='Recoverable DS (algebraic)',title='Penalty removal with other losses held fixed')
    ax.legend(ncol=3,loc='upper left');save(fig,out/'loss_decomposition');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(DOUBLE_COLUMN_IN,2.7),layout='constrained')
    for ax,group in zip(axs,('junction','obstacle')):
        for j,camera in enumerate(('road','wide','union')):
            vv=v[(v['group']==group)&(v.kind==group)&(v.camera==camera)]
            values=np.sort(vv.lead_s.to_numpy())
            ax.step(values,np.arange(1,len(values)+1)/max(1,len(values)),where='post',label=camera,color=colors[j])
        ax.axvline(2,color='#777777',lw=.7,ls='--');ax.set(xlabel='Observed visibility lead (s)',ylabel='Object ECDF',title=group.capitalize())
        ax.set_ylim(0,1);ax.legend(loc='lower right')
    save(fig,out/'visibility_upper_bound');plt.close(fig)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    figures(p.parse_args().root)
