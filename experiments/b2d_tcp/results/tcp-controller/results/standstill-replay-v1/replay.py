from pathlib import Path
import importlib.util,json,csv,hashlib,subprocess,sys
root=Path('/home/ujs/mycode/jev-drive');sys.path.insert(0,str(root/'scripts'));out=Path(__file__).parent
old=out/'old-control.py';old.write_bytes(subprocess.check_output(['git','-C',str(root),'show','d8bfffb:scripts/b2d_tcp_control.py']))
new=out/'new-control.py';new.write_bytes((root/'scripts/b2d_tcp_control.py').read_bytes())
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.TCPControlComparison()
a,b=load(old,'old'),load(new,'new');rows=[]
for raw in (out/'captured-motion-prefix.jsonl').read_text().splitlines():
 r=json.loads(raw);p=r.get('prediction')
 if not p:continue
 args=(r['timestamp'],r['raw_speed_mps'],p['metadata']['desired_speed'],*p['native_control'])
 x,y=a.step(*args),b.step(*args)
 rows.append(dict(frame=r['frame'],timestamp=r['timestamp'],raw_speed_mps=r['raw_speed_mps'],desired=p['metadata']['desired_speed'],native_throttle=p['native_control'][0],native_brake=p['native_control'][2],old_reason=x['diagnostics']['reason'],new_reason=y['diagnostics']['reason'],old_native_throttle=x['native_common_control'][0],old_native_brake=x['native_common_control'][2],new_native_throttle=y['native_common_control'][0],new_native_brake=y['native_common_control'][2],old_pi_throttle=x['pi_common_control'][0],old_pi_brake=x['pi_common_control'][2],new_pi_throttle=y['pi_common_control'][0],new_pi_brake=y['pi_common_control'][2]))
with (out/'replay.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(samples=len(rows),old_reverse_faults=sum(r['old_reason']=='reverse_motion' for r in rows),new_reverse_faults=sum(r['new_reason']=='reverse_motion' for r in rows),native_restart_released=sum(r['old_native_throttle']==0 and r['old_native_brake']==1 and r['new_native_throttle']>0 and r['new_native_brake']==0 for r in rows),interpretation='Recorded-input replay proves guard change, not driving recovery. Full six cases will rerun.',sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (old,new,out/'captured-motion-prefix.jsonl',out/'replay.csv',Path(__file__))})
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
