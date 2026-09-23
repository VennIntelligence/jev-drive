#!/usr/bin/env python3
"""Read-only article figures from frozen pose analysis; all samples, no new gates."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VARIANTS = ('baseline-zero', 'candidate-fixed-k')
WINDOWS = (('26966', 1, 'Right', 24., 51.), ('24240', 1, 'Left', 18., 64.),
           ('17563', 1, 'S1', 27.5, 48.5), ('17563', 2, 'S2', 73.5, 95.))
FIELDS = (('cte', 'CTE (m; left +)'), ('actual_speed', 'Speed (m/s)'),
          ('rate', 'Emitted steer rate (/s)'), ('lateral_accel', 'Lateral acceleration (m/s²)'),
          ('lateral_jerk', 'Physical lateral jerk (m/s³)'),
          ('pose_position_error', 'Pose position error (m)'),
          ('pose_left_error', 'Pose error (m; truth body left +)'), ('heading', 'Tracking heading (°; right +)'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def csv_read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def load_style(path):
    spec = importlib.util.spec_from_file_location('pose_publication_style', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply()
    return module

def broken_line(rows, field):
    """Insert only NaN separators: never connect absent/nonconsecutive samples."""
    x, y = [], []
    origin = numeric(rows[0]['time']) if rows else np.nan
    previous = None
    for row in rows:
        time, frame = numeric(row['time']), int(row['frame'])
        if previous is not None and (frame != previous[0]+1 or not time > previous[1] or time-previous[1] > .051):
            x.append(np.nan); y.append(np.nan)
        x.append(time-origin); y.append(numeric(row.get(field)))
        previous = frame, time
    return x, y

def curve_figure(rows, identity, stem, style, exports, fixture=False):
    fig, axes = plt.subplots(4, 2, figsize=(style.DOUBLE_COLUMN_IN, 5.95), layout='constrained', sharex=True)
    for index, (ax, (field, label)) in enumerate(zip(axes.flat, FIELDS)):
        for variant, color, dash in zip(VARIANTS, (style.BASELINE, style.PREDICTION), ('-', '--')):
            selected = [r for r in rows if r['variant'] == variant]
            x, y = broken_line(selected, field)
            ax.plot(x, y, color=color, linestyle=dash, label='Baseline k = 0' if variant == VARIANTS[0] else 'Fixed k = 0.010659832')
            if field == 'actual_speed':
                xx, yy = broken_line(selected, 'reference_speed')
                ax.plot(xx, yy, color=color, linestyle=':', linewidth=.75)
        style.zero_line(ax)
        style.panel(ax, f'({chr(97+index)}) '+(identity+' · ' if index == 0 else '')+label)
        # Units in short panel labels; omit duplicate y labels to retain usable plot width.
    for ax in axes[-1]:
        ax.set_xlabel('Time from first selected sample (s)')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2,
               title='Fixture only · no live qualification' if fixture else None)
    exports[stem.name] = style.save(fig, stem)
    plt.close(fig)

def summary_figure(metrics, out, style, exports, fixture=False):
    descriptions = (('cte_m_rms','CTE RMS (m)'), ('steer_rate_per_s_p95_abs','Steer-rate abs P95 (/s)'),
                    ('lateral_accel_mps2_p95_abs','Lateral accel. abs P95 (m/s²)'),
                    ('lateral_jerk_mps3_p95_abs','Lateral jerk abs P95 (m/s³)'),
                    ('pose_position_error_m_rms','Pose position RMS (m)'),
                    ('pose_left_error_m_mean','Mean pose error (m; body left +)'))
    index = {(r['route'],int(r['segment']),r['variant']):r for r in metrics if r['band']=='window' and r['subset']=='all'}
    plot_rows=[]
    fig, axes=plt.subplots(3,2,figsize=(style.DOUBLE_COLUMN_IN,4.65),layout='constrained')
    for i,(ax,(field,label)) in enumerate(zip(axes.flat,descriptions)):
        for j,variant in enumerate(VARIANTS):
            values=[]
            for route, segment, name, start, end in WINDOWS:
                row=index.get((route,segment,variant),{})
                value=numeric(row.get(field));values.append(value)
                plot_rows.append(dict(route=route,segment=segment,variant=variant,metric=field,value=value,
                                      sample_count=row.get('count'),coverage_complete=row.get('coverage_complete')))
            positions=np.arange(4)+(j-.5)*.35
            ax.bar(positions,values,width=.34,color=(style.BASELINE,style.PREDICTION)[j],label=('Baseline k = 0','Fixed k')[j])
            for position,value in zip(positions,values):
                if not np.isfinite(value):ax.text(position,0,'NA',ha='center',va='bottom',fontsize=7)
        style.panel(ax,f'({chr(97+i)}) {label}')
        ax.set_xticks(range(4),['Right','Left','S1','S2']);style.bars(ax);style.zero_line(ax)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=2,
               title='Fixture only · no live qualification' if fixture else None)
    exports['four-window-summary']=style.save(fig,out/'four-window-summary');plt.close(fig)
    csv_write(out/'summary-plot-data.csv',plot_rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--style',type=Path)
    args=parser.parse_args()
    if args.out.exists():raise SystemExit('Output must be a fresh directory; prior editions are immutable.')
    style_path=args.style
    if style_path is None:
        local=HERE/'plot_style.py'
        style_path=local if local.exists() else HERE.parents[3]/'research/plot_style.py'
    # Resolve project style from repository root without assuming the invocation cwd.
    if not style_path.exists():
        style_path=next((p/'research/plot_style.py' for p in HERE.parents if (p/'research/plot_style.py').exists()),style_path)
    required=('frames.csv','metrics.csv','cases.csv','coverage.json','required-conditions.json','manifest.json')
    for name in required:
        if not (args.analysis/name).is_file():raise SystemExit(f'Missing analysis input: {name}')
    rows=csv_read(args.analysis/'frames.csv');metrics=csv_read(args.analysis/'metrics.csv')
    expected={(route,variant) for route in ('26966','24240','17563') for variant in VARIANTS}
    observed={(r['route'],r['variant']) for r in rows}
    if not observed<=expected:raise SystemExit(f'Unexpected route/variant identity: {observed-expected}')
    for row in rows:
        for key in ('frame','time','progress'):
            if not np.isfinite(numeric(row.get(key))):raise SystemExit(f'Unplotable identity {key}: {row}')
    verdict=json.loads((args.analysis/'required-conditions.json').read_text())
    fixture=bool(verdict.get('legacy_fixture'))
    style=load_style(style_path)
    args.out.mkdir(parents=True)
    csv_write(args.out/'plot-data.csv',rows)  # Preserves every input field and input row order.
    exports={};coverage=[];window_rows=[]
    for route in ('26966','24240','17563'):
        selected=[r for r in rows if r['route']==route]
        curve_figure(selected,route,args.out/f'route-{route}-full',style,exports,fixture)
    for route,segment,name,start,end in WINDOWS:
        selected=[r for r in rows if r['route']==route and start<=numeric(r['progress'])<=end]
        curve_figure(selected,f'{route} {name}',args.out/f'route-{route}-window-{segment}',style,exports,fixture)
        window_rows.extend(dict(r,window_segment=segment,window_start_m=start,window_end_m=end) for r in selected)
        for variant in VARIANTS:
            rr=[r for r in selected if r['variant']==variant]
            coverage.append(dict(route=route,segment=segment,variant=variant,n=len(rr),
                nonfinite={field:sum(not np.isfinite(numeric(r.get(field))) for r in rr) for field,_ in FIELDS}))
    csv_write(args.out/'window-plot-data.csv',window_rows)
    summary_figure(metrics,args.out,style,exports,fixture)
    notes=dict(analysis_status=verdict.get('status'),legacy_fixture=verdict.get('legacy_fixture'),
        qualification='No decision recomputed; use frozen required-conditions.json.',
        sample_count=len(rows),expected_cases=6,observed_cases=len(observed),missing_cases=sorted(expected-observed),
        curves='All original rows including startup, stationary tail, failures, and transients. No smoothing/resampling. Missing samples become gaps, never zero.',
        alignment='Per-arm elapsed time from first selected sample. Window selection uses frozen independent truth station, inclusive boundaries.',
        signs='CTE left-positive. Pose left error=(estimated rear XY − truth rear XY) dot truth-yaw left. Acceleration/jerk and heading right-positive.',
        physics='Speed/acceleration/jerk copied from independent analysis; jerk differentiates world acceleration before projection. Speed reference dotted.',
        summary='Frozen metrics.csv values copied; bars are descriptive, not extra gates. NA means missing, never passing.',
        window_coverage=coverage,figures=exports)
    (args.out/'plot-notes.json').write_text(json.dumps(notes,indent=2)+'\n')
    shutil.copy2(__file__,args.out/'plot-source.py');shutil.copy2(style_path,args.out/'plot_style.py')
    inputs=[Path(__file__).resolve(),style_path.resolve()]+[args.analysis/n for n in required]
    manifest=dict(inputs={str(p.resolve()):dict(sha256=digest(p),bytes=p.stat().st_size) for p in inputs},
        outputs={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(args.out.iterdir()) if p.is_file()},
        retention='No deletion; original analysis/source manifests linked by hash. Plot and style bytes archived.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out),figures=len(exports),rows=len(rows),analysis_status=verdict.get('status'))))

if __name__=='__main__':main()
