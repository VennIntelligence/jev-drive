#!/usr/bin/env python
"""Fixed 18-case PI acceptance and matched vendor comparisons; read-only analysis."""
import argparse
from collections import Counter
import hashlib
import json
import shlex
from pathlib import Path

import numpy as np
from b2d_controller_matrix_plot import Sources, extract_case, stats, write_csv

ROUTES = ('1773', '1773-speed6', '17563', '24240', '26966', '25854')
GROUPS = (('pi', 'carla', 'baseline'), ('pi', 'pursuit', 'baseline'), ('pi-max', 'pursuit', 'max'))


def complete(root):
    summary = json.loads((root/'summary.json').read_text())
    keys = [(str(r['route_id']), r['variant'], r['preset']) for r in summary]
    expected = {(r, v, p) for r in ROUTES for v, p, _ in GROUPS}
    ends = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()
            if line.strip() and json.loads(line).get('kind') == 'end']
    if len(keys) != 18 or set(keys) != expected or not ends or ends[-1].get('status') != 'completed' or ends[-1].get('cases') != 18:
        raise RuntimeError('Require all 18 declared cases and a completed campaign end event')


def waveform(case, sources):
    trace, row = case['trace'], case['row']
    control = sources.read(Path(row['source_dir'])/'control.jsonl', jsonl=True)
    by_frame = {r['frame']: r for r in control}
    selected = [r for r in trace if r['elapsed_s'] >= 5. and r['reference_speed_mps'] >= row['cruise_mps']-.1]
    speed = np.asarray([r['signed_speed'] for r in selected])
    gears = [r.get('applied_control', {}).get('gear') for r in selected]
    observed_gears = [g for g in gears if g is not None]
    gains = sorted(set((r.get('longitudinal_kp'), r.get('longitudinal_ki')) for r in control
                       if r.get('longitudinal_kp') is not None and r.get('longitudinal_ki') is not None))
    row.update(longitudinal_gains_observed=json.dumps(gains), cruise_samples=len(selected),
               cruise_signed_speed_std_mps=float(np.std(speed)) if len(speed) else None,
               cruise_signed_speed_p95_minus_p5_mps=float(np.percentile(speed,95)-np.percentile(speed,5)) if len(speed) else None,
               cruise_signed_speed_min_mps=float(np.min(speed)) if len(speed) else None,
               cruise_signed_speed_max_mps=float(np.max(speed)) if len(speed) else None,
               cruise_under_by_half_mps_fraction=float(np.mean(speed < row['cruise_mps']-.5)) if len(speed) else None,
               gear_observed_cruise_samples=len(observed_gears),
               gear_observed_values=json.dumps(sorted(set(observed_gears))),
               gear_changes=sum(a is not None and b is not None and a!=b for a,b in zip(gears,gears[1:])) if observed_gears else None)
    # Each episode is a contiguous interval with true speed >0.5 m/s below cruise.
    # This records recurrent deficits without claiming a transmission cause.
    episodes, active = [], []
    for r in selected:
        if r['signed_speed'] < row['cruise_mps']-.5:
            if active and r['frame'] != active[-1]['frame']+1:
                episodes.append(active);active=[]
            active.append(r)
        elif active:
            episodes.append(active);active=[]
    if active:episodes.append(active)
    row['underspeed_episodes'] = len(episodes)
    row['underspeed_episodes_at_least_025s'] = sum(e[-1]['sim_time']-e[0]['sim_time'] >= .25-1e-6 for e in episodes)
    episode_rows = [dict(route_id=row['route_id'], variant=row.get('variant'), preset=row['preset'],
                         role=row['role'], source_dir=row['source_dir'], start_frame=e[0]['frame'],
                         end_frame=e[-1]['frame'], duration_s=e[-1]['sim_time']-e[0]['sim_time'],
                         minimum_speed_mps=min(r['signed_speed'] for r in e)) for e in episodes]
    frame_rows = []
    for r in trace:
        c = by_frame.get(r['frame'], {})
        frame_rows.append(dict(route_id=row['route_id'], variant=row.get('variant'), preset=row['preset'], role=row['role'],
                               source_dir=row['source_dir'], frame=r['frame'], sim_time=r['sim_time'], elapsed_s=r['elapsed_s'],
                               signed_speed_mps=r['signed_speed'], reference_speed_mps=r['reference_speed_mps'],
                               cross_track_m=r['cross_track_m'], endpoint_error_m=r['endpoint_error_m'],
                               gear=r.get('applied_control', {}).get('gear'),
                               throttle=c.get('throttle'), brake=c.get('brake'), steer=c.get('steer'),
                               controller_reason=c.get('reason'), integral_effort=c.get('longitudinal_integral_effort'),
                               integration_limited=c.get('longitudinal_integration_limited'),
                               longitudinal_kp=c.get('longitudinal_kp'), longitudinal_ki=c.get('longitudinal_ki')))
    return frame_rows, episode_rows


def plots(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'pdf.fonttype':42})
    outputs=[]
    def save(fig,name,note):
        fig.text(.01,.008,'Privileged map-route diagnostic; all failures retained. '+note,fontsize=8)
        for ext in ('png','pdf'):
            p=out/(name+'.'+ext);fig.savefig(p,dpi=300,bbox_inches='tight');outputs.append(p)
        plt.close(fig)
    for mode in ('speed-gear','cte'):
        fig,axes=plt.subplots(3,6,figsize=(19,9))
        for i,(variant,preset,_) in enumerate(GROUPS):
            for j,route in enumerate(ROUTES):
                item=data[route,variant,preset];ax=axes[i,j]
                for key,color,style in [('vendor','#777777','--'),('pi','#0072B2','-')]:
                    case=item.get(key)
                    if case is None:continue
                    trace=case['trace'];t=[r['elapsed_s'] for r in trace]
                    field='signed_speed' if mode=='speed-gear' else 'cross_track_m'
                    ax.plot(t,[r[field] for r in trace],color=color,linestyle=style,lw=1)
                if mode=='speed-gear':
                    twin=ax.twinx()
                    for key,color,style in [('pi','#D55E00','-'),('vendor_gear','#CC79A7',':')]:
                        case=item.get(key)
                        if case is None:continue
                        trace=[r for r in case['trace'] if r.get('applied_control',{}).get('gear') is not None]
                        if trace:twin.step([r['elapsed_s'] for r in trace],
                                          [r['applied_control']['gear'] for r in trace],where='post',
                                          color=color,linestyle=style,lw=.8,alpha=.6)
                    twin.set_ylabel('Reported gear field' if j==5 else '',color='#D55E00')
                    twin.tick_params(axis='y',colors='#D55E00',labelsize=7)
                    ax.axhline(item['pi']['row']['cruise_mps'],color='#aaaaaa',lw=.6,linestyle=':')
                ax.set_title('%s | %s %s'%(route,preset,'max' if variant=='pi-max' else 'additive'))
                if not item.get('vendor'):ax.text(.03,.95,'No matched vendor run',transform=ax.transAxes,va='top',fontsize=7)
                ax.set_xlabel('Elapsed time (s)');ax.grid(alpha=.15)
                if j==0:ax.set_ylabel('True signed speed (m/s)' if mode=='speed-gear' else 'True CTE (m, left +)')
        legend=[Line2D([0],[0],color='#777777',linestyle='--',label='Matched vendor speed / CTE'),
                Line2D([0],[0],color='#0072B2',label='PI speed / CTE')]
        if mode=='speed-gear':legend += [Line2D([0],[0],color='#D55E00',label='PI reported gear'),
                                         Line2D([0],[0],color='#CC79A7',linestyle=':',label='Vendor gear (separate replicate on S)')]
        fig.legend(handles=legend,loc='upper center',ncol=len(legend),frameon=False)
        fig.tight_layout(rect=[0,.035,1,.965])
        save(fig,'six-route-'+mode,'Gear is the CARLA control API field, not proof of internal transmission state; vendor S gear uses a separately logged replicate.' if mode=='speed-gear' else 'Complete traces include startup and stop; no favorable interval selection.')
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for j,(variant,preset,_) in enumerate(GROUPS):
        for i,(metric,label,gate) in enumerate([('full_cte_rms_m','Full-route CTE RMS (m)',.5),
                                                ('cruise_speed_rms_mps','Cruise speed RMS error (m/s)',.5)]):
            ax=axes[i,j]
            for key,offset,color in [('vendor',-.16,'#777777'),('pi',.16,'#0072B2')]:
                for x,route in enumerate(ROUTES):
                    case=data[route,variant,preset].get(key)
                    if case is None:continue
                    value=case['row'][metric]
                    if value is None:continue
                    ax.bar(x+offset,value,width=.29,color=color)
                    if not case['row']['gate_pass']:ax.scatter(x+offset,value,marker='x',color='black',s=24,zorder=3)
            ax.axhline(gate,color='#999999',linestyle=':',lw=1)
            ax.set_xticks(np.arange(6),ROUTES,rotation=20)
            ax.set_ylabel(label);ax.set_title(preset+' '+variant);ax.grid(axis='y',alpha=.15)
    fig.legend(handles=[Line2D([0],[0],color='#777777',lw=6,label='Matched vendor'),Line2D([0],[0],color='#0072B2',lw=6,label='PI')],loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.04,1,.955]);save(fig,'six-route-gates','Original gates unchanged. x marks any failed gate; blank vendor max straight6 has no matched observation.')
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pi',type=Path,default=Path('/data/runs/b2d/controller/development-v3'))
    p.add_argument('--vendor',type=Path,default=Path('/data/runs/b2d/controller/development-v2'))
    p.add_argument('--speed-diagnostic',type=Path,default=Path('/data/runs/b2d/controller/development-v2-speed-diagnostic'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();complete(args.pi)
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Choose a fresh output folder')
    sources=Sources();data={};cases=[];frames=[];episodes=[];aux=[]
    for root in (args.pi,args.vendor,args.speed_diagnostic):
        for path in list((root/'inputs').glob('*'))+[root/'provenance/manifest.json',root/'summary.json',root/'events.jsonl']:
            if path.is_file():sources.register(path)
    sources.register(Path(__file__));sources.register(Path(__file__).with_name('b2d_controller_matrix_plot.py'))
    matrix=sources.read(args.pi/'inputs/matrix.json');parameters={}
    for case in matrix['cases']:
        variant=case['variant'];path=args.pi/'inputs'/('controller-'+variant+'.json')
        cfg=sources.read(path)
        if sources.files[str(path.resolve())]['sha256']!=case['controller_config_sha256']:
            raise ValueError('Archived PI configuration hash mismatch')
        if cfg.get('longitudinal_mode')!='pi':raise ValueError('Expected PI longitudinal mode')
        parameters[variant]=dict(kp=cfg.get('pi_kp',1.),ki=cfg.get('pi_ki',.25),
                                 config_source=str(path.resolve()),sha256=case['controller_config_sha256'])
    def read(path,role,route,preset,variant,source_route=None):
        case=extract_case(path,role,source_route or route,preset,sources)
        case['row'].update(route_id=route,variant=variant,role=role)
        f,e=waveform(case,sources);frames.extend(f);episodes.extend(e)
        return case
    for route in ROUTES:
        for variant,preset,vendor_variant in GROUPS:
            candidate=read(args.pi/route/variant/preset,'pi',route,preset,variant)
            observed=json.loads(candidate['row']['longitudinal_gains_observed'])
            expected=[parameters[variant]['kp'],parameters[variant]['ki']]
            if observed and observed!=[expected]:raise ValueError('PI telemetry gains differ from archived configuration')
            vendor=None;gear=None
            if route!='1773-speed6':vendor=read(args.vendor/route/vendor_variant/preset,'vendor',route,preset,vendor_variant)
            elif vendor_variant=='baseline':vendor=read(args.speed_diagnostic/'1773'/preset,'vendor',route,preset,'baseline',source_route='1773')
            if vendor:
                a=np.asarray(candidate['reference']['world_xy']);b=np.asarray(vendor['reference']['world_xy'])
                if a.shape!=b.shape or not np.allclose(a,b,atol=1e-6,rtol=0):raise ValueError('Unmatched reference geometry')
                if candidate['row']['cruise_mps']!=vendor['row']['cruise_mps']:raise ValueError('Unmatched cruise')
                cases.append(vendor)
                if any('applied_control' in r for r in vendor['trace']):gear=vendor
            if route=='17563' and vendor_variant=='baseline':
                gear=read(args.speed_diagnostic/'17563'/preset,'gear_replicate',route,preset,vendor_variant)
                aux.append(gear)
            data[route,variant,preset]=dict(pi=candidate,vendor=vendor,vendor_gear=gear)
            cases.append(candidate)
    matched=[]
    for (route,variant,preset),item in data.items():
        a=item['pi']['row'];b=item['vendor']['row'] if item['vendor'] else None
        row=dict(route_id=route,variant=variant,preset=preset,cruise_mps=a['cruise_mps'],pi_pass=a['gate_pass'],
                 vendor_pass=b['gate_pass'] if b else None,paired=b is not None,pi_failed_gates=a['failed_gates'],
                 vendor_failed_gates=b['failed_gates'] if b else None,
                 newly_failed_gates=','.join(k for k,v in item['pi']['summary']['gates'].items()
                                             if not v and item['vendor']['summary']['gates'].get(k)) if b else None)
        for metric in ('full_cte_rms_m','full_cte_p95_m','moving_cte_rms_m','cruise_speed_rms_mps','endpoint_error_m',
                       'stop_hold_displacement_m','cruise_signed_speed_std_mps','cruise_signed_speed_p95_minus_p5_mps',
                       'cruise_under_by_half_mps_fraction','underspeed_episodes_at_least_025s'):
            row[metric+'_pi']=a[metric];row[metric+'_vendor']=b[metric] if b else None
            row[metric+'_delta']=a[metric]-b[metric] if b and a[metric] is not None and b[metric] is not None else None
        matched.append(row)
    args.out.mkdir(parents=True,exist_ok=True)
    csvs=[('cases.csv',[c['row'] for c in cases]),('matched-comparisons.csv',matched),('frames.csv',frames),
          ('underspeed-episodes.csv',episodes),('gear-replicates.csv',[c['row'] for c in aux])]
    for name,rows in csvs:write_csv(args.out/name,rows)
    outputs=plots(data,args.out)
    report=dict(required_pi_cases=18,matched_vendor_cases=17,pi_passed=sum(c['row']['gate_pass'] for c in cases if c['row']['role']=='pi'),
                comparisons=matched,selection={'routes':ROUTES,'groups':GROUPS,'parameters':parameters,'parameter_policy':'declared archived configuration, verified against observed telemetry; no tuning by analysis'},
                definitions={'gates':'unaltered per-case validation gates; all CTE RMS/p95 reproduced from independent trace',
                             'cruise_mask':'elapsed>=5s and independent reference>=cruise-.1m/s; same as acceptance',
                             'cycling':'descriptive true speed spread and recurring >=.25s deficit episodes >.5m/s below cruise; no causal transmission claim',
                             'gear':'observed actor.get_control().gear before next command; control API field only',
                             'pairing':'same v2 adapter, preset/lookahead, original reference and 6/8mps; straight6 max unpaired',
                             'gear_replicates':'S vendor gear uses development-v2-speed-diagnostic/17563; metrics pair against original v2 matrix'},
                limitations=['One run per case; no uncertainty or leaderboard generalization.',
                             'PI includes a >.2s motion-gap guard absent from vendor mode.',
                             'Different exposure on failing/short traces can affect descriptive cycle counts.',
                             'Gear fields do not establish torque interruption or internal gear-shift causation.'],
                sources=list(sources.files.values()),outputs=[])
    for path in [args.out/name for name,_ in csvs]+outputs:
        report['outputs'].append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    command='PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python scripts/b2d_controller_pi_plot.py '+ \
            ' '.join(shlex.quote(x) for x in ['--pi',str(args.pi),'--vendor',str(args.vendor),
                                              '--speed-diagnostic',str(args.speed_diagnostic),'--out',str(args.out)+'-reproduced'])
    readme=['# Fixed PI development comparison', '',
            '%d of 18 declared PI cases pass every original gate. All cases and failures are retained; '
            '17 vendor cases are matched, with vendor/max straight6 unmeasured.' % report['pi_passed'], '',
            'Archived PI parameters: '+json.dumps({k:{n:v[n] for n in ('kp','ki')} for k,v in parameters.items()})+'.', '']
    for name,title,note in [
        ('six-route-speed-gear','Speed and reported gear','Full traces show remaining speed variation. The gear field is an API observation; vendor S gear uses a separate replicate and does not establish transmission causation.'),
        ('six-route-cte','Independent lateral error','Full-route true CTE includes startup and stopping. Consult every original gate and paired change in matched-comparisons.csv before accepting a candidate.'),
        ('six-route-gates','Original acceptance gates','Black crosses mark any failed gate, including metrics not plotted. The vendor/max straight6 bar is absent because no matched observation exists.')]:
        readme += ['## '+title,'','!['+title+']('+name+'.png)','','[Vector PDF]('+name+'.pdf)','',note,'']
    readme += ['## Data and provenance','',
               '- [cases.csv](cases.csv): every case/gate, lateral/speed/stop/timing/rejoin metrics.',
               '- [matched-comparisons.csv](matched-comparisons.csv): exact matched values and differences; unpaired max straight6 explicit.',
               '- [frames.csv](frames.csv): raw frame IDs, timestamps, signals, source paths and observation roles.',
               '- [underspeed-episodes.csv](underspeed-episodes.csv): deficit interval start/end frames and durations.',
               '- [gear-replicates.csv](gear-replicates.csv): separate vendor S gear observations.',
               '- [report.json](report.json): raw source paths/SHA-256, parameters, selection, definitions and limitations.','',
               'The original five routes pair against development-v2. Straight6 CARLA/pursuit pair against '
               'development-v2-speed-diagnostic/1773. Vendor S gear comes from development-v2-speed-diagnostic/17563 '
               'as a separate replicate; acceptance metrics still pair against the original v2 matrix. '
               'Cruise statistics use elapsed>=5s and independent reference>=cruise-0.1m/s. '
               'Moving CTE uses |true signed speed|>=0.5m/s and does not replace full-route gates.','',
               '## Reproduce','','Run from the v2 worktree; choose a fresh output folder.','','```bash',command,'```','']
    (args.out/'README.md').write_text('\n'.join(readme))
    print(json.dumps({'out':str(args.out),'pi_passed':report['pi_passed'],'total':18},indent=2))


if __name__=='__main__':main()
