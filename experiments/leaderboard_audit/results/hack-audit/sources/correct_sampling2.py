"""第二次源码核验：严格区分方法对应的闭环入口。"""
import csv
from pathlib import Path

out = Path('/data/hack_audit/out')

def table(name):
    with (out/name).open(newline='') as f:
        r=csv.DictReader(f)
        return r.fieldnames,list(r)

def save(name,fields,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

fields, rows=table('sampling.csv')
for r in rows:
    if r['board']=='carla_lb2' and r['method'] in ('CarLLaVA','SimLingo-BASE'):
        r['code_grade']='C'
        r['selected']='no'
        r['not_selected_reason']='公开 main 无可对应该旧方法及官方分数的闭环推理/评测入口；仅 full SimLingo agent，Base 有训练模块'
    if r['board']=='nuscenes' and r['method']=='AD-MLP':
        r['score']='SOTA2 avg L2 0.35; corrected paper/README 0.29'
        r['split_protocol_note']+=' SOTA2 的 0.35 对应无高层命令消融；论文/更新后 README 完整模型是 0.29。'
save('sampling.csv',fields,rows)

fields, rows=table('repos.csv')
rows=[r for r in rows if (r['board'],r['method']) not in {('carla_lb2','CarLLaVA'),('carla_lb2','SimLingo-BASE')}]
save('repos.csv',fields,rows)

index=out/'INDEX.md'
lines=[line for line in index.read_text().splitlines()
       if not (line.startswith('| carla_lb2 | CarLLaVA |') or line.startswith('| carla_lb2 | SimLingo-BASE |'))]
index.write_text('\n'.join(lines)+'\n')
with (out/'progress.md').open('a') as f:
    f.write('- CARLA-LB2 复核：CarLLaVA/SimLingo-BASE 固定公开提交无对应闭环入口，降 C 不入选；该榜仅 TF++ 满足入选条件，按规则如实报告不足 3 项。AD-MLP 的 SOTA2 0.35 与纠错后论文/README 0.29 分开记录。\n')
