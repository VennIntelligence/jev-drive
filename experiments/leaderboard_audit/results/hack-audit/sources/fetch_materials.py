"""只下载浅克隆源码和论文；不下载权重、数据集，不运行仓库代码。"""
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/data/hack_audit')
CSV = ROOT/'out/repos.csv'
rows = list(csv.DictReader(CSV.open()))
groups = {}
for row in rows:
    groups.setdefault((row['repo_url'], row['branch']), []).append(row)

def clone_group(group):
    first = group[0]
    target = ROOT/first['repo_path']
    if not (target/'.git').exists():
        env = os.environ.copy(); env['GIT_LFS_SKIP_SMUDGE']='1'
        cmd=['git','clone','--depth','1','--single-branch','--branch',first['branch'],first['repo_url'],str(target)]
        for attempt in range(2):
            try:
                subprocess.run(cmd,check=True,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
                break
            except Exception as exc:
                if attempt==1: return first['repo_url'],'failed: '+str(exc)[:100]
                if target.exists() and not (target/'.git').exists():
                    subprocess.run(['rm','-rf',str(target)],check=False)
    commit=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    for row in group[1:]:
        alias=ROOT/row['repo_path']
        if not alias.exists(): alias.symlink_to(target, target_is_directory=True)
    return first['repo_url'],commit

clone_results={}
with ThreadPoolExecutor(max_workers=5) as pool:
    futures=[pool.submit(clone_group,g) for g in groups.values()]
    for future in as_completed(futures):
        url,result=future.result();clone_results[url]=result;print('clone',url,result,flush=True)

papers={}
for row in rows: papers.setdefault(row['paper_path'],row['paper_url'])
papers['papers/hugsim_benchmark.pdf']='https://arxiv.org/pdf/2412.01718'

def fetch_paper(item):
    path,url=item;target=ROOT/path
    if target.exists() and target.read_bytes()[:4]==b'%PDF': return path,'ok'
    for attempt in range(3):
        try:
            subprocess.run(['curl','-fLsS','--retry','2','--connect-timeout','15','--max-time','90',url,'-o',str(target)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=110)
            if target.read_bytes()[:4]!=b'%PDF': return path,'failed: non-PDF response'
            subprocess.run(['pdftotext','-layout',str(target),str(target.with_suffix('.txt'))],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            return path,'ok'
        except Exception as exc:
            if attempt==2: return path,'failed: '+str(exc)[:100]

paper_results={}
with ThreadPoolExecutor(max_workers=6) as pool:
    futures=[pool.submit(fetch_paper,item) for item in papers.items()]
    for future in as_completed(futures):
        path,result=future.result();paper_results[path]=result;print('paper',path,result,flush=True)

fields=list(rows[0])
for row in rows:
    result=clone_results.get(row['repo_url'],'failed: missing group')
    row['commit']=result if len(result)==40 else ''
    row['clone_status']='ok' if row['commit'] else result
    row['paper_status']=paper_results.get(row['paper_path'],'failed: missing result')
with CSV.open('w',newline='') as f:
    w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
