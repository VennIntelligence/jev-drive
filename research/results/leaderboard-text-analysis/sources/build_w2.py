#!/usr/bin/env python3
"""Transcribe W2 cross-board score rows from local sampled/official PDFs."""
import csv
from pathlib import Path

ROOT = Path('/data/hack_audit/out2')
FIELDS = ['method','board','protocol/split','metric','score','source','is_author_reported','same_checkpoint']
rows = []

with (ROOT/'sources/w2_seed_bench2drive.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        row['same_checkpoint']='unknown; Table 3 pairs metrics but does not explicitly identify checkpoint'
        rows.append({k:row[k] for k in FIELDS})

def add(method, board, protocol, metric, score, paper, table, page, author='yes', checkpoint='unknown'):
    source = f'papers/{paper}.pdf ({table}, PDF p.{page})'
    rows.append(dict(zip(FIELDS, [method,board,protocol,metric,str(score),source,author,checkpoint])))

NV1='navtest, NAVSIM v1'
NV2H='navhard-two-stage, NAVSIM v2, corrected EPDMS'
NV2T='navtest, NAVSIM v2, corrected EPDMS'
BD='base-set closed-loop 220 routes'
L6='Longest6 v2, CARLA LB2 local'
H436='436 scenarios, common controller, zero-shot'

# NAVSIM v1 / v2: the paired tables supply the broadest same-protocol cohort.
cross = [
    # name, v1, v2, v1 paper/table/page, v2 paper/table/page, checkpoint evidence
    ('PDM-Closed',89.1,56.6,'ltf','Table 1',7,'toad','Table 6',15,'not applicable; rule-based planner, no learned checkpoint'),
    ('TransFuser',84.0,23.1,'ltf','Table 1',7,'toad','Table 6',15,'unknown; different benchmark re-evaluations'),
    ('LTF',83.8,25.1,'ltf','Table 1',7,'navsim_v2_benchmark','Table 2',8,'unknown; official v2 corrected snapshot'),
    ('DiffusionDrive',88.1,24.2,'toad','Table 5',14,'toad','Table 6',15,'unknown; comparison checkpoint identity not stated'),
    ('World4Drive',85.1,34.9,'toad','Table 5',14,'toad','Table 6',15,'unknown; comparison checkpoint identity not stated'),
    ('iPad',91.7,34.7,'toad','Table 5',14,'toad','Table 6',15,'unknown; public checkpoint used, cross-version identity not stated'),
    ('RAP-DINO',93.8,39.6,'rap','Table 1',6,'toad','Table 6',15,'yes; RAP Section 4 says trained model directly evaluated on v1/v2; v2 rescored after metric fix'),
    ('Hydra-MDP',90.9,40.9,'toad','Table 5',14,'toad','Table 6',15,'unknown; public checkpoint used, cross-version identity not stated'),
    ('GTRS (V2-99)',90.4,45.4,'toad','Table 5',14,'toad','Table 6',15,'unknown; public checkpoint used, cross-version identity not stated'),
    ('ZTRS (V2-99)',86.9,48.1,'toad','Table 5',14,'toad','Table 6',15,'unknown; public checkpoint used, cross-version identity not stated'),
    ('DrivoR (+134k SimScale)',94.6,54.6,'drivor','Table 13',15,'drivor','Table 14 (after fix)',16,'unknown; both +134k rows, cross-version checkpoint identity not explicit'),
    ('DrivoR',94.6,54.6,'toad','Table 5',14,'toad','Table 6',15,'unknown; TOAD public checkpoint, v1/v2 identity not explicit'),
]
for m,a,b,pa,ta,na,pb,tb,nb,ck in cross:
    add(m,'navsim_v1',NV1,'PDMS',a,pa,ta,na, 'yes' if pa=='drivor' else 'no (comparison/official table)',ck)
    add(m,'navsim_v2',NV2H,'EPDMS',b,pb,tb,nb, 'yes' if pb=='drivor' else 'no (comparison/official table)',ck)
add('UniAD','navsim_v1',NV1,'PDMS',83.4,'ltf','Table 1',7,'no (benchmark authors re-evaluated)','unknown; HUGSIM UniAD checkpoint identity not stated')

for m,a,b in [
    ('ZTRS (V2-99) + TOAD',89.0,49.2),('GTRS (V2-99) + TOAD',90.9,51.7),
    ('Hydra-MDP + TOAD',91.4,49.7),('iPad + TOAD',93.4,49.8),
    ('RAP-DINO + TOAD',93.9,49.0),('DrivoR + TOAD',94.7,56.3)]:
    ck='unknown; same public base planner family, no weight updates, v1/v2 checkpoint identity unstated (TOAD p.5)'
    add(m,'navsim_v1',NV1,'PDMS',a,'toad','Table 5',14,'yes',ck)
    add(m,'navsim_v2',NV2H,'EPDMS',b,'toad','Table 6',15,'yes',ck)

# NAVSIM v2 navtest is a separate evaluation: do not pool with navhard two-stage.
for m,v1,v2,fp,tp,pp,ck in [
    ('SparseDriveV2',92.0,90.1,'sparsedrivev2','Table 2 / Table 3',11,'unknown; v1/v2 use 20/10 second-layer velocity anchors; checkpoint identity not explicitly stated'),
    ('DriveFuture',90.7,89.9,'drivefuture','Table 3 / Table 2',8,'unknown; checkpoint identity across protocols not stated'),
    ('WA-JEPA',91.8,91.7,'wa_jepa','Table 3 / Table 1',7,'unknown; checkpoint identity across protocols not stated'),
]:
    p1,t1,p1n = ('sparsedrivev2','Table 2',11) if m=='SparseDriveV2' else ('drivefuture','Table 3',8) if m=='DriveFuture' else ('wa_jepa','Table 3',7)
    p2,t2,p2n = ('sparsedrivev2','Table 3',12) if m=='SparseDriveV2' else ('drivefuture','Table 2',7) if m=='DriveFuture' else ('wa_jepa','Table 1',6)
    add(m,'navsim_v1',NV1,'PDMS',v1,p1,t1,p1n,'yes',ck)
    add(m,'navsim_v2',NV2T,'EPDMS',v2,p2,t2,p2n,'yes',ck)
add('NTR','navsim_v1',NV1,'PDMS',94.1,'ntr','Table 2',7,'yes','unknown; Waymo and NAVSIM trained separately')
add('NTR','navsim_v2','navtest, NAVSIM v2, correction status not stated','EPDMS',90.9,'ntr','Table 3',7,'yes','unknown; v1/v2 checkpoint identity not stated')
add('DriveFuture','navsim_v2',NV2H,'EPDMS',55.5,'drivefuture','Table 1',7,'yes','unknown; navhard versus navtest checkpoint identity not stated')
add('SparseDriveV2','bench2drive',BD,'Driving Score',89.15,'sparsedrivev2','Table 4',13,'yes','no; separate NAVSIM and Bench2Drive training in Sections 4.2-4.3')
add('SparseDriveV2','bench2drive',BD,'Success Rate (%)',70.00,'sparsedrivev2','Table 4',13,'yes','no; separate NAVSIM and Bench2Drive training in Sections 4.2-4.3')

# RAP changes both backbone and training data for CARLA and fine-tunes on WOD.
add('RAP-DINO','wod_e2e','test, vision-based E2E','RFS Overall',8.04,'rap','Table 3',7,'yes','no; Waymo fine-tuned; later NTR Table 1 labels the WOD submission an ensemble')
add('RAP-DINO','wod_e2e','test, vision-based E2E','ADE@5s (m)',2.65,'rap','Table 3',7,'yes','no; Waymo fine-tuned; later NTR Table 1 labels the WOD submission an ensemble')
add('RAP-ResNet','bench2drive',BD,'Driving Score',66.42,'rap','Table 4',8,'yes','no; RAP Section 4 says different ResNet34 backbone and CARLA-specific training')
add('RAP-ResNet','bench2drive',BD,'Success Rate (%)',37.27,'rap','Table 4',8,'yes','no; RAP Section 4 says different ResNet34 backbone and CARLA-specific training')

# WOD test + NAVSIM v1 method families (cross-dataset fine-tuning/weights not established).
for m,rfs,pdms in [('DriveMA-2B',8.060,90.5),('DriveMA-4B',8.079,91.2)]:
    add(m,'wod_e2e','test, vision-based E2E','RFS Overall',rfs,'drivema','Table 1',6,'yes','unknown; NAVSIM and WOD training/checkpoint identity not established')
    add(m,'navsim_v1',NV1,'PDMS',pdms,'drivema','Table 2',6,'yes','unknown; NAVSIM and WOD training/checkpoint identity not established')
add('NTR','wod_e2e','test, vision-based E2E','RFS Overall',7.9982,'ntr','Table 1 (single model)',7,'yes','no; benchmark-specific training in Section 4')
add('NTR (Ensemble)','wod_e2e','test, vision-based E2E','RFS Overall',8.0461,'ntr','Table 1 (ensemble)',7,'yes','no; ensemble is distinct from NAVSIM single planner')
add('AutoVLA','navsim_v1',NV1,'PDMS',89.11,'autovla','Table 1 (Post-RFT)',7,'yes','unknown; different benchmark fine-tuning described, checkpoint identity not stated')
add('AutoVLA','wod_e2e','test, vision-based E2E','RFS Overall',7.5566,'ntr','Table 1',7,'no (comparison table)','no; Waymo fine-tuning, see AutoVLA p.9')
add('AutoVLA','bench2drive',BD,'Driving Score',78.84,'autovla','Table 3',9,'yes','no; CARLA SFT model versus real-data variants, p.9')
add('AutoVLA','bench2drive',BD,'Success Rate (%)',57.73,'autovla','Table 3',9,'yes','no; CARLA SFT model versus real-data variants, p.9')
add('AutoVLA','nuscenes','val, ST-P3 metric, 1/2/3s mean','Avg L2 (m)',0.40,'autovla','Table S2 (w/ CoT)',26,'yes','unknown; nuScenes checkpoint versus CARLA/NAVSIM not stated')

# Official CARLA LB2 test and Bench2Drive; SimLingo explicitly calls the CARLA model the B2D model.
for m, bds, bsr, mp, se in [
    ('SimLingo-BASE',85.94,66.82,6.25,6.87),
    ('TF++',84.21,67.27,5.56,5.18)]:
    p,t,pn=('simlingo_base','Table 2',7) if m=='SimLingo-BASE' else ('tfpp','Table 4',6)
    ck='unknown; B2D row labelled LB2.0 model, but MAP/SENSORS submission checkpoint not identified' if m=='SimLingo-BASE' else 'unknown; LB2 and B2D checkpoint identity not stated'
    add(m,'bench2drive',BD,'Driving Score',bds,p,t,pn,'yes',ck)
    add(m,'bench2drive',BD,'Success Rate (%)',bsr,p,t,pn,'yes',ck)
    for track,ds in [('MAP',mp),('SENSORS',se)]:
        add(m,'carla_lb2',f'official test, {track} track','Driving Score',ds,'simlingo_base','Table 1',6,'yes' if m=='SimLingo-BASE' else 'no (comparison table)',ck)

# Same rows of TFv6 Table 5 give B2D and the CARLA Longest6 v2 protocol.
for m,bds,bsr,lds,lrc in [
    ('HiP-AD',86.8,69.1,7,56),('SimLingo',85.1,67.2,22,70),('TFv5 (RegNetY-032, 110C+L)',83.5,67.3,23,70),
    ('TFv6 (ResNet34, 360C)',91.6,79.5,43,85),('TFv6 (ResNet34, 360C+L)',94.7,85.6,52,88),
    ('TFv6 (ResNet34, 360C+R)',94.2,85.3,52,88),('TFv6 (ResNet34, 360C+L+R)',95.0,84.3,54,89),
    ('TFv6 (ResNet34, 140C+L+R)',94.7,82.1,57,99),('TFv6 (RegNetY-032, 140C+L+R)',95.2,86.8,62,91),
    ('PDM-Lite',97.0,92.3,73,100),('LEAD',96.8,96.6,73,93)]:
    author='yes' if m.startswith(('TFv5','TFv6','LEAD')) else 'no (comparison table)'
    ck=('not applicable; rule-based privileged expert, no learned checkpoint' if m=='PDM-Lite'
        else 'unknown; one configuration row in TFv6 Table 5, but seed/checkpoint pairing not stated' if m.startswith(('TFv5','TFv6'))
        else 'unknown; compared across protocols in one table')
    for board,protocol,metric,score in [('bench2drive',BD,'Driving Score',bds),('bench2drive',BD,'Success Rate (%)',bsr),('carla_lb2',L6,'Driving Score',lds),('carla_lb2',L6,'Route Completion (%)',lrc)]:
        row_ck=('not applicable; privileged LEAD expert; Bench2Drive score estimated from standard leaderboard (issue #90)' if board=='bench2drive' else 'not applicable; privileged LEAD expert; Longest6 result reported in Table 5') if m=='LEAD' else ck
        add(m,board,protocol,metric,score,'tfv6','Table 5',7,author,row_ck)
for m,bds,bsr,lds,lrc in [('TF++',84.21,67.27,23,70),('RoG-DAgger',90.34,73.51,44,88)]:
    ck='unknown; two protocols in RoG-DAgger Table 1, checkpoint identity not explicit'
    for board,protocol,metric,score in [('bench2drive',BD,'Driving Score',bds),('bench2drive',BD,'Success Rate (%)',bsr),('carla_lb2',L6,'Driving Score',lds),('carla_lb2',L6,'Route Completion (%)',lrc)]:
        # TF++ B2D already sourced from its paper above.
        if m=='TF++' and board=='bench2drive': continue
        add(m,board,protocol,metric,score,'rog_dagger','Table 1',6,'yes' if m=='RoG-DAgger' else 'no (comparison table)',ck)
for m,ds in [('TFv5',1.08),('TFv6 (RegNetY-032, 140C+L+R)',3.52),('PDM-Lite',36.30)]:
    ck='not applicable; rule-based privileged expert, no learned checkpoint' if m=='PDM-Lite' else 'unknown; Town13 versus other evaluation checkpoints not explicit'
    add(m,'carla_lb2','Town13 val, CARLA LB2 local','Driving Score',ds,'tfv6','Table 4',6,'yes' if m.startswith('TFv') else 'no (comparison table)',ck)

# HUGSIM common 436-scenario evaluation and TOAD four-source-dataset evaluation are separate protocols.
for m,hds in [('WA-JEPA',0.4462),('LTF',0.2310),('DrivoR',0.3252),('UniAD',0.3124),('VAD',0.1393)]:
    add(m,'hugsim',H436,'HD-Score',hds,'wa_jepa','Table 2',7,'yes' if m=='WA-JEPA' else 'no (benchmark comparison)','unknown; source-model checkpoints not linked to other boards')
for m,vals in [
    ('DrivoR',[17.7,39.7,35.1,44.2]),('DrivoR + TOAD',[18.4,49.9,38.0,42.8])]:
    for split,hds in zip(['KITTI360','nuScenes','PandaSet','Waymo'],vals):
        add(m,'hugsim',f'zero-shot {split}, TOAD Table 4','HD-Score (%)',hds,'toad','Table 4',6,'yes' if m.endswith('TOAD') else 'no (comparison table)','unknown; public DrivoR base used without weight update, board checkpoint link unverified')

# nuScenes/CARLA cross-dataset reports: retain the metric family in the protocol name.
add('AD-MLP','nuscenes','val, ST-P3 metric, 1/2/3s mean','Avg L2 (m)',0.29,'ad_mlp','Table 1 (full model)',3,'yes','no; Bench2Drive official authors retrained under base-set')
add('VAD-Base','nuscenes','val, OmniSpace Table 1 metric, 1/2/3s mean','Avg L2 (m)',0.37,'omnispace','Table 1',6,'no (comparison table)','unknown; Bench2Drive checkpoint not linked')
add('VAD-Base','bench2drive',BD,'Driving Score',42.35,'omnispace','Table 2',7,'no (comparison table)','unknown; nuScenes checkpoint not linked')
for m,ns,bds,bsr in [('OmniSpace (Qwen2.5-VL 7B)',0.28,79.65,60.15),('OmniSpace (Qwen3-VL 4B)',0.28,81.40,65.52)]:
    add(m,'nuscenes','val, OmniSpace Table 1 metric, 1/2/3s mean','Avg L2 (m)',ns,'omnispace','Table 1',6,'yes','no; trained/evaluated for each benchmark separately (Section 4.1)')
    add(m,'bench2drive',BD,'Driving Score',bds,'omnispace','Table 2',7,'yes','no; trained/evaluated for each benchmark separately (Section 4.1)')
    add(m,'bench2drive',BD,'Success Rate (%)',bsr,'omnispace','Table 2',7,'yes','no; trained/evaluated for each benchmark separately (Section 4.1)')
add('SteerVLA','nuscenes','val, 1/2/3s mean','Avg L2 (m)',0.40,'steervla','Table 5',22,'yes','no; p.22 explicitly says stronger backbones than CARLA experiments')
add('SteerVLA','bench2drive',BD,'Driving Score',90.71,'steervla','Table 3',21,'yes','no; p.22 explicitly says stronger nuScenes backbones')
add('SteerVLA','bench2drive',BD,'Success Rate (%)',73.64,'steervla','Table 3',21,'yes','no; p.22 explicitly says stronger nuScenes backbones')

# LTFv6 real-world experiment: NAVSIM v1/v2 same architecture, Waymo is separately fine-tuned.
for m,v1,v2,wod in [('LTFv6',85.4,28.3,7.51),('LTFv6 + LEAD',86.4,31.4,7.76)]:
    add(m,'navsim_v1',NV1,'PDMS',v1,'tfv6','Table 6',8,'yes','unknown; NAVSIM v1/v2 checkpoint identity not stated')
    add(m,'navsim_v2',NV2H,'EPDMS',v2,'tfv6','Table 6',8,'yes','unknown; NAVSIM v1/v2 checkpoint identity not stated')
    add(m,'wod_e2e','validation','RFS Overall',wod,'tfv6','Table 6',8,'yes','no; WOD model separately fine-tuned (p.7)')

# Check duplicates: equal key/score from different source is omitted; contradictory scores are never silently merged.
seen={}
for row in rows:
    key=tuple(row[k] for k in FIELDS[:4])
    if key in seen:
        if row['score']!=seen[key]['score']:
            raise SystemExit(f'contradictory duplicate {key}: {seen[key]["score"]} vs {row["score"]}')
    else: seen[key]=row
rows=list(seen.values())
rows.sort(key=lambda r:(r['board'],r['protocol/split'],r['method'],r['metric']))
with (ROOT/'w2_cross_board.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,FIELDS);w.writeheader();w.writerows(rows)
print(f'wrote {len(rows)} score rows')
