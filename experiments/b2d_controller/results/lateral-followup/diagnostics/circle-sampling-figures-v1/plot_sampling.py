"""Plot the fixed noiseless circle diagnosis; no empirical driving claim."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
rows = list(csv.DictReader((source/'frames.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig, axes = plt.subplots(2, 2, figsize=(10,6), constrained_layout=True)
used=[]
for col, radius in enumerate((8.,20.)):
    rr=[r for r in rows if float(r['radius'])==radius and float(r['speed'])==8 and float(r['sign'])==1 and float(r['coefficient'])==.5 and 20<=int(r['tick'])<40]
    used.extend(rr)
    t=[(int(r['tick'])-20)*.05 for r in rr]
    for key,label,color in [('linear','Linear aim','#2563a6'),('hermite','Hermite aim','#d35f20')]:
        axes[0,col].plot(t,[float(r[key])-float(r['analytic']) for r in rr],label=label,color=color)
        axes[1,col].plot(t,[float(r[key+'_rate']) for r in rr],color=color)
    axes[0,col].set_title(f'Radius {radius:g} m; speed 8 m/s; lookahead 4 m')
    axes[0,col].set_ylabel('Raw steer error vs analytic (normalized)')
    axes[1,col].set_ylabel('Raw steer rate (1/s)')
    axes[1,col].set_xlabel('Time within fixed steady interval (s)')
    for ax in axes[:,col]:ax.grid(alpha=.2)
axes[0,0].legend()
fig.suptitle('Prescribed exact circular motion: sampling ripple without noise or rejoining')
for ext in ('png','pdf'):fig.savefig(out/('circle-sampling.'+ext),dpi=200)
plt.close(fig)
with (out/'plotted-samples.csv').open('x') as f:
    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(used)
shutil.copy2(__file__,out/'plot_sampling.py')
(out/'manifest.json').write_text(json.dumps(dict(source=str(source),inputs={str(source/'frames.csv'):hashlib.sha256((source/'frames.csv').read_bytes()).hexdigest()},outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},scope='Fixed ticks20..39 of synthetic prescribed motion; no filtered values, physical controller loop or CARLA result.'),indent=2))
