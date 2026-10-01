#!/usr/bin/env python3
"""Publication plots of descriptive rear lateral-motion regression; never fit or run control."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WINDOWS=[('26966',1,24.,51.,'Right sharp'),('24240',1,18.,64.,'Left turn'),('17563',1,27.5,48.5,'S1'),('17563',2,73.5,95.,'S2')]
VARIANTS=['baseline-max','short-max']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):return list(csv.DictReader(Path(p).open()))
def csvwrite(p,rs):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--propagation',type=Path,required=True);p.add_argument('--regression',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('A fresh figure edition is required')
    sources={}
    def record(path,expected=None):
        actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Source changed: '+str(path))
        sources[str(Path(path).resolve())]=actual
    for folder in (a.propagation,a.regression):
        for name,digest in json.loads((folder/'outputs-sha256.json').read_text()).items():record(folder/name,digest)
        record(folder/'outputs-sha256.json')
    info=json.loads((a.regression/'summary.json').read_text());k=info['coefficient']
    for path,digest in {**json.loads((a.propagation/'inputs-sha256.json').read_text()),**info['inputs']}.items():record(path,digest)
    record(a.propagation/'inputs-sha256.json');record(Path(__file__))
    propagation={(r['route'],r['variant'],int(r['frame'])):r for r in csvread(a.propagation/'frames.csv')}
    raw=csvread(a.regression/'frames.csv');rows=[]
    for r in raw:
        route,variant,frame=r['route'],r['variant'],int(r['frame']);q=propagation[(route,variant,frame)]
        y=float(r['model_minus_rear_right_mps']);prediction=float(r['global_prediction_mps']);x=float(r['v2_omega']);error=float(r['global_error_mps'])
        assert np.isfinite([y,prediction,x,error,float(r['station_m'])]).all()
        assert abs(y-float(q['true_heading_model_minus_rear_right_mps']))<1e-12
        assert abs(prediction-k*x)<1e-12 and abs(error-(y-prediction))<1e-12
        rows.append(dict(route=route,variant=variant,segment=int(r['segment']),frame=frame,station_m=float(r['station_m']),dt_s=float(q['dt_s']),split='baseline_fit' if variant=='baseline-max' else 'short_arm_held_out',
            coefficient=k,interval_sensor_speed_mps=float(r['interval_speed_mps']),interval_world_gyro_rps=float(r['world_gyro_rps']),v2_omega=x,
            model_minus_rear_right_mps=y,sensor_feature_prediction_mps=prediction,residual_after_offline_subtraction_mps=error,
            actual_pose_heading_model_minus_rear_right_mps=float(q['prediction_minus_rear_right_mps']),true_rear_interval_right_mps=float(q['rear_interval_right_mps'])))
    assert len(rows)==657 and sum(r['split']=='baseline_fit' for r in rows)==328
    baseline=[r for r in rows if r['split']=='baseline_fit'];xx=np.array([r['v2_omega'] for r in baseline]);yy=np.array([r['model_minus_rear_right_mps'] for r in baseline]);assert abs(k-float(xx@yy/(xx@xx)))<1e-12
    windows=csvread(a.regression/'windows.csv');summary=[]
    for w in windows:
        rs=[r for r in rows if r['route']==w['route'] and r['variant']==w['variant'] and r['segment']==int(w['segment'])];assert len(rs)==int(w['n'])
        indices=[r['frame'] for r in rs];assert all(y==x+1 for x,y in zip(indices,indices[1:]))
        target_rms=float(np.sqrt(np.mean([r['model_minus_rear_right_mps']**2 for r in rs])));residual_rms=float(np.sqrt(np.mean([r['residual_after_offline_subtraction_mps']**2 for r in rs])))
        assert abs(target_rms-float(w['target_rms_mps']))<1e-12 and abs(residual_rms-float(w['residual_rms_mps']))<1e-12
        summary.append(dict(route=w['route'],segment=int(w['segment']),variant=w['variant'],split=rs[0]['split'],n=len(rs),first_frame=rs[0]['frame'],last_frame=rs[-1]['frame'],target_rms_mps=target_rms,residual_rms_mps=residual_rms,rms_reduction_fraction=1-residual_rms/target_rms))
    a.out.mkdir(parents=True);csvwrite(a.out/'plot-data.csv',rows);csvwrite(a.out/'rms-plot-data.csv',summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42,'savefig.dpi':250})
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),bbox_inches='tight');fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(13,12),sharey='row',layout='constrained')
    for i,(route,segment,lo,hi,label) in enumerate(WINDOWS):
        for j,variant in enumerate(VARIANTS):
            rs=[r for r in rows if (r['route'],r['segment'],r['variant'])==(route,segment,variant)];ax=axs[i,j]
            ax.plot([r['station_m'] for r in rs],[r['model_minus_rear_right_mps'] for r in rs],color='#263648',lw=1.7,label='Model minus rear truth (truth-heading label)')
            ax.plot([r['station_m'] for r in rs],[r['sensor_feature_prediction_mps'] for r in rs],color='#da6b24',ls='--',lw=1.45,label='Sensor features: k v² omega')
            ax.axhline(0,color='#777777',lw=.65);ax.set_xlim(lo,hi);ax.set_xlabel('Route station (m)')
            if j==0:ax.set_ylabel('Lateral velocity difference\n(m/s, RIGHT positive)')
            split='BASELINE: fit data' if j==0 else 'SHORT ARM: held out from fit'
            ax.set_title(f'{route} {label} | {split} | n={len(rs)}',loc='left')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle(f'One baseline-fitted coefficient across four fixed windows: k = {k:.8f} s²/m\nAll 328 fit + 329 held-out samples; no filtering, lag search or transient removal',fontsize=13)
    save(fig,'rear-lateral-velocity-four-windows')
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),sharey=True,layout='constrained')
    for j,variant in enumerate(VARIANTS):
        rs=[next(r for r in summary if (r['route'],r['segment'],r['variant'])==(route,segment,variant)) for route,segment,_,_,_ in WINDOWS];x=np.arange(4);width=.34
        b1=axs[j].bar(x-width/2,[r['target_rms_mps'] for r in rs],width,color='#263648',label='Unexplained label RMS')
        b2=axs[j].bar(x+width/2,[r['residual_rms_mps'] for r in rs],width,color='#da6b24',label='Residual after offline subtraction')
        for bars in (b1,b2):axs[j].bar_label(bars,labels=[f'{v.get_height():.3f}' for v in bars],padding=3,fontsize=9)
        axs[j].set_xticks(x,[f'{route}\n{label}' for route,segment,lo,hi,label in WINDOWS]);axs[j].set_title('BASELINE fit data (n=328)' if j==0 else 'SHORT arm held out (n=329)');axs[j].set_ylim(0,.46)
    axs[0].set_ylabel('RMS lateral velocity difference (m/s)');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Descriptive residual reduction; no localization or controller change deployed',fontsize=13);save(fig,'rear-lateral-residual-rms')
    (a.out/'plot_pose_regression.py').write_bytes(Path(__file__).read_bytes())
    for label,source in [('propagation-recompute.py',a.propagation/'recompute.py'),('regression-recompute.py',a.regression/'recompute.py')]: (a.out/label).write_bytes(source.read_bytes())
    (a.out/'fit-summary.json').write_bytes((a.regression/'summary.json').read_bytes())
    (a.out/'verification.json').write_text(json.dumps(dict(samples=657,baseline_fit_samples=328,short_held_out_samples=329,all_window_frames_retained=True,within_window_frames_consecutive=True,target_matches_propagation_within=1e-12,prediction_and_residual_match_within=1e-12,baseline_only_coefficient_reproduced=True,window_rms_matches_archived_summary_within=1e-12,raw_source_hashes_verified=len(sources)),indent=2))
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=sources,outputs={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},coefficient=k,fit_scope=info['fit_definition'],target_definition=info['y'],predictor_definition=info['x'],limitations=info['limits']),indent=2))
    print(json.dumps(dict(out=str(a.out),png=2,pdf=2,plotted_samples=len(rows))))
if __name__=='__main__':main()
