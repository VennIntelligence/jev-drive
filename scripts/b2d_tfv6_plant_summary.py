"""Aggregate logged plant curves, normative steer, and opposite-sign turns."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import plot_style
from b2d_tfv6_plant_diagnosis import OUT,FULL,L,TRACK,MAX,scale
from b2d_tfv6_steer_diagnosis import write_csv

SPEED_BINS=[(2,4),(4,6),(6,8),(8,12)]


def load_runs():
    groups=defaultdict(list)
    for r in csv.DictReader(open(FULL/'ticks.csv')):
        for k,v in r.items():
            if k not in ('source','route','arm','reason'):
                r[k]=float(v) if v not in ('','None') else float('nan')
        groups[(r['source'],r['route'],int(r['seed']),r['arm'])].append(r)
    return groups


def arrays(rr):
    keys=('t','speed','steer','b_shadow_steer','plan_k_right','k_cmd_right',
          'k_ach_right','k_nom_right','k_ack_right','k_unscaled_right',
          'world_x','world_y','yaw_deg','yaw_rate_right')
    return {k:np.array([z[k] for z in rr],float) for k in keys}


def turn_mask(a,lag,lo=2,hi=12):
    n=len(a['t'])-lag
    if n<=0:return np.zeros(0,bool)
    return ((a['speed'][:n]>=lo)&(a['speed'][:n]<hi)&
            (a['speed'][lag:]>=lo)&(a['speed'][lag:]<hi)&
            (np.abs(a['plan_k_right'][:n])>=.03)&
            (np.abs(a['b_shadow_steer'][:n])>=.05)&
            (np.abs(a['t'][lag:]-a['t'][:n]-lag*.05)<.015)&
            np.isfinite(a['k_ach_right'][lag:]))


def gain(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if len(x)<2 or float(x@x)<1e-12:return float('nan')
    return float(x@y/(x@x))


def corr(x,y):
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return float('nan')
    return float(np.corrcoef(x,y)[0,1])


def lag_analysis(groups):
    out=[]
    for arm in ('A','B','C','D','E','F'):
        for predictor in ('k_ack_right','k_cmd_right'):
            if predictor=='k_cmd_right' and arm not in ('C','D'):continue
            for lag in range(9):
                x_all=[];y_all=[];dx_all=[];dy_all=[]
                for key,rr in groups.items():
                    if key[3]!=arm:continue
                    a=arrays(rr);m=turn_mask(a,lag)
                    if m.sum()<5:continue
                    x=a[predictor][:len(m)][m];y=a['k_ach_right'][lag:][m]
                    good=np.isfinite(x)&np.isfinite(y)
                    if good.sum()<5:continue
                    x=x[good];y=y[good]
                    x_all.extend(x-x.mean());y_all.extend(y-y.mean())
                    # Consecutive selected ticks only, to avoid differencing across gaps.
                    ii=np.where(m)[0]
                    adjacent=np.diff(ii)==1
                    if len(adjacent):
                        xx=a[predictor];yy=a['k_ach_right']
                        dx_all.extend(np.diff(xx[ii])[adjacent]);dy_all.extend(np.diff(yy[ii+lag])[adjacent])
                out.append(dict(arm=arm,predictor=predictor,lag_ticks=lag,lag_s=lag*.05,
                                n=len(x_all),corr_centered=corr(x_all,y_all),
                                corr_change=corr(dx_all,dy_all)))
    write_csv(OUT/'lag.csv',out)
    return out


def plant_map(groups):
    out=[]
    for lo,hi in SPEED_BINS:
        for subset in ('active','quasi_steady'):
            for arm in ('all','all_ex_B','A','B','C','D','E','F'):
                nom=[];ack=[];unscaled=[];ach=[];runs=set()
                for key,rr in groups.items():
                    if arm=='all_ex_B' and key[3]=='B':continue
                    if arm not in ('all','all_ex_B') and key[3]!=arm:continue
                    a=arrays(rr);n=len(rr)-1
                    if n<5:continue
                    v=a['speed'];s=a['steer'];t=a['t'];kn=a['k_nom_right']
                    m=((v[:n]>=lo)&(v[:n]<hi)&(v[1:]>=lo)&(v[1:]<hi)&
                       (np.abs(s[:n])>=.05)&np.isfinite(a['k_ach_right'][1:])&
                       (np.abs(t[1:]-t[:n]-.05)<.015))
                    if subset=='quasi_steady':
                        dk=np.r_[np.full(4,np.nan),np.abs(kn[4:]-kn[:-4])]
                        dv=np.r_[np.full(4,np.nan),np.abs(v[4:]-v[:-4])]
                        m &= (dk[:n]<.015)&(dv[:n]<.75)
                    if m.any():runs.add(key)
                    nom.extend(a['k_nom_right'][:n][m]);ack.extend(a['k_ack_right'][:n][m])
                    unscaled.extend(a['k_unscaled_right'][:n][m]);ach.extend(a['k_ach_right'][1:][m])
                x=np.array(nom);z=np.array(ack);u=np.array(unscaled);y=np.array(ach)
                if len(y)<15:continue
                out.append(dict(speed_lo=lo,speed_hi=hi,subset=subset,arm=arm,n=len(y),runs=len(runs),
                                gain_nom=gain(x,y),gain_ack=gain(z,y),gain_unscaled=gain(u,y),
                                rmse_nom=float(np.sqrt(np.mean((y-x)**2))),
                                rmse_ack=float(np.sqrt(np.mean((y-z)**2))),
                                rmse_unscaled=float(np.sqrt(np.mean((y-u)**2))),
                                mean_abs_ach=float(np.mean(abs(y)))))
    write_csv(OUT/'plant_map.csv',out)
    return out


def command_gain(groups):
    out=[]
    for arm in ('B','C','D'):
        for source in ('W2','D3b','all'):
            for lo,hi in SPEED_BINS:
                cmd=[];ach=[];nom=[];ack=[];runs=set();lag=2 if arm in ('C','D') else 1
                for key,rr in groups.items():
                    if key[3]!=arm or (source!='all' and key[0]!=source):continue
                    a=arrays(rr);m=turn_mask(a,lag,lo,hi)
                    if not len(m):continue
                    x=a['k_cmd_right'] if arm in ('C','D') else a['k_nom_right']
                    m &= np.isfinite(x[:len(m)])&(np.abs(x[:len(m)])>=.01)
                    if m.any():runs.add(key)
                    cmd.extend(x[:len(m)][m]);ach.extend(a['k_ach_right'][lag:][m])
                    nom.extend(a['k_nom_right'][:len(m)][m]);ack.extend(a['k_ack_right'][:len(m)][m])
                x=np.array(cmd);y=np.array(ach);n=np.array(nom);a=np.array(ack)
                if len(x)<10:continue
                out.append(dict(arm=arm,source=source,speed_lo=lo,speed_hi=hi,n=len(x),runs=len(runs),
                                gain_command=gain(x,y),gain_nom=gain(n,y),gain_ack=gain(a,y),
                                ack_to_command=gain(x,a),
                                rmse_command=float(np.sqrt(np.mean((y-x)**2)))))
    write_csv(OUT/'command_gain.csv',out)
    return out


def rear_path_validation(groups):
    out=[]
    for key,rr in groups.items():
        a=arrays(rr);n=len(rr)
        if n<5:continue
        yaw=np.deg2rad(a['yaw_deg']);p=np.c_[a['world_x']-1.389*np.cos(yaw),
                                        a['world_y']-1.389*np.sin(yaw)]
        v1=p[2:n-2]-p[:n-4];v2=p[4:]-p[2:n-2];whole=p[4:]-p[:n-4]
        den=np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)*np.linalg.norm(whole,axis=1)
        k=2*(v1[:,0]*whole[:,1]-v1[:,1]*whole[:,0])/np.maximum(den,1e-9)
        y=a['k_ach_right'][2:n-2]
        m=((a['speed'][2:n-2]>=2)&(np.abs(a['steer'][2:n-2])>=.05)&np.isfinite(y)&
           (np.abs(a['t'][4:]-a['t'][:n-4]-.2)<.02)&(np.abs(k)<1.))
        if m.sum()<5:continue
        out.append(dict(source=key[0],route=key[1],seed=key[2],arm=key[3],n=int(m.sum()),
                        corr=corr(k[m],y[m]),bias=float(np.mean(k[m]-y[m])),
                        mae=float(np.mean(abs(k[m]-y[m])))))
    write_csv(OUT/'rear_path_check.csv',out)
    return out


def rear_slip_check(groups):
    out=[]
    for lo,hi in SPEED_BINS:
        for subset in ('active','quasi_steady'):
            measured=[];pred11=[];pred14=[];runs=set()
            for key,rr in groups.items():
                a=arrays(rr);n=len(rr)
                if n<7:continue
                yaw=np.deg2rad(a['yaw_deg'])
                p=np.c_[a['world_x']-1.389*np.cos(yaw),a['world_y']-1.389*np.sin(yaw)]
                course=np.arctan2((p[4:]-p[:-4])[:,1],(p[4:]-p[:-4])[:,0])
                beta=(yaw[2:-2]-course+np.pi)%(2*np.pi)-np.pi
                v=a['speed'][2:-2];w=a['yaw_rate_right'][2:-2]
                m=((v>=lo)&(v<hi)&(np.abs(a['plan_k_right'][2:-2])>=.03)&
                   (np.abs(a['b_shadow_steer'][2:-2])>=.05)&(np.abs(w)>.02)&
                   (np.abs(a['t'][4:]-a['t'][:-4]-.2)<.02)&(np.abs(beta)<.2))
                if subset=='quasi_steady':
                    kn=a['k_nom_right'];vv=a['speed']
                    dk=np.r_[np.full(4,np.nan),np.abs(kn[4:]-kn[:-4])]
                    dv=np.r_[np.full(4,np.nan),np.abs(vv[4:]-vv[:-4])]
                    m &= (dk[2:-2]<.015)&(dv[2:-2]<.75)
                if m.any():runs.add(key)
                measured.extend(beta[m]);pred11.extend(np.arctan((v[m]+1.)*w[m]/(11.*9.81)))
                pred14.extend(np.arctan((v[m]+1.)*w[m]/(14.07*9.81)))
            y=np.array(measured);x=np.array(pred11);z=np.array(pred14)
            if len(y)<10:continue
            out.append(dict(speed_lo=lo,speed_hi=hi,subset=subset,n=len(y),runs=len(runs),
                            gain_c11=gain(x,y),mean_abs_beta_deg=float(np.degrees(np.mean(abs(y)))),
                            rmse_c11_deg=float(np.degrees(np.sqrt(np.mean((y-x)**2)))),
                            rmse_c1407_deg=float(np.degrees(np.sqrt(np.mean((y-z)**2))))))
    write_csv(OUT/'rear_slip.csv',out)
    return out


def inverse_plant(k,speed,gamma):
    wanted=k/gamma
    if TRACK*abs(wanted)/2>=1:return float('nan')
    k_nom=wanted/(1-TRACK*abs(wanted)/2)
    return math.atan(L*k_nom)/(MAX*scale(speed))


def triple_curves(w):
    p=np.asarray(w,float);vals=[]
    for x,y,z in zip(p[:-2],p[1:-1],p[2:]):
        a=y-x;b=z-x;c=z-y;den=np.linalg.norm(a)*np.linalg.norm(b)*np.linalg.norm(c)
        vals.append(2*(a[0]*b[1]-a[1]*b[0])/den if den>1e-8 else 0.)
    return np.array(vals)


def plan_at_aim(w,station):
    p=np.asarray(w,float);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.vstack((np.zeros(2),p)),axis=0),axis=1))]
    vals=triple_curves(p)
    return -float(np.interp(station,arc[2:-1],vals))


def normative(map_rows):
    gammas={int(z['speed_lo']):z['gain_ack'] for z in map_rows if z['subset']=='quasi_steady' and z['arm']=='all'}
    tr=list(csv.DictReader(open('results/diagnosis/steer/turn_ticks.csv')))
    run_paths={(z['source'],z['route'],int(z['seed'])):z['path'] for z in csv.DictReader(open('results/diagnosis/steer/runs.csv'))}
    wanted=defaultdict(set)
    for z in tr:wanted[(z['source'],z['route'],int(z['seed']))].add(int(z['step']))
    frames={}
    for key,steps in wanted.items():
        for line in open(run_paths[key]):
            f=json.loads(line)
            if f['step'] in steps:frames[key,f['step']]=f
    out=[];opposite=[]
    for z in tr:
        key=(z['source'],z['route'],int(z['seed']));step=int(z['step']);f=frames[key,step]
        speed=float(z['speed']);lo=2 if speed<4 else 4 if speed<6 else 6 if speed<8 else 8
        gamma=gammas[lo]
        cmd=-float(z['pursuit_curvature'])
        station=float(z['station_m'])+float(z['lookahead_m'])
        plan=plan_at_aim(f['rear_waypoint'],station)
        norm=inverse_plant(cmd,speed,gamma);norm_ack=inverse_plant(cmd,speed,1.)
        plan_norm=inverse_plant(plan,speed,gamma)
        b=float(z['b_executed']);c=float(z['c_shadow'])
        tri=triple_curves(f['waypoint']);strong=tri[np.abs(tri)>=.02]
        s_shape=bool(len(strong)>1 and np.any(np.sign(strong[1:])!=np.sign(strong[:-1])))
        sign_opp=bool(np.sign(b)!=np.sign(c))
        row=dict(source=key[0],route=key[1],seed=key[2],step=step,speed=speed,
                 dense_turn_distance_m=float(z['dense_turn_distance_m']),
                 plan_k_near=float(z['plan_curvature']),plan_k_aim=plan,
                 c_cmd_k=cmd,b_steer=b,c_steer=c,c_plant_steer=norm,
                 c_ack_only_steer=norm_ack,plan_plant_steer=plan_norm,
                 gamma=gamma,opposite=int(sign_opp),s_shape=int(s_shape),
                 b_near_zero=int(abs(b)<.1),c_near_zero=int(abs(c)<.02),
                 b_matches_plan=int(np.sign(b)==np.sign(float(z['plan_curvature']))),
                 c_matches_plan=int(np.sign(c)==np.sign(float(z['plan_curvature']))),
                 b_aim_sign=int(np.sign(float(z['b_aim_y']))),
                 c_aim_sign=int(np.sign(-float(z['aim_y']))))
        out.append(row)
        if sign_opp:opposite.append(row)
    write_csv(OUT/'normative_ticks.csv',out)
    write_csv(OUT/'opposite_ticks.csv',opposite)
    summary=[]
    for source in ('all','D3b','W2'):
        src=[z for z in out if source=='all' or z['source']==source]
        for subset in ('all_finite','calibrated_feasible','plan_feasible'):
            if subset=='all_finite':part=[z for z in src if np.isfinite(z['c_plant_steer'])]
            elif subset=='calibrated_feasible':part=[z for z in src if z['speed']>=2 and z['speed']<12 and
                                                       np.isfinite(z['c_plant_steer']) and abs(z['c_plant_steer'])<=.8]
            else:part=[z for z in src if z['speed']>=2 and z['speed']<12 and
                             np.isfinite(z['plan_plant_steer']) and abs(z['plan_plant_steer'])<=.8 and
                             np.isfinite(z['c_plant_steer']) and abs(z['c_plant_steer'])<=.8]
            if not part:continue
            b=np.array([z['b_steer'] for z in part]);c=np.array([z['c_steer'] for z in part]);n=np.array([z['c_plant_steer'] for z in part]);p=np.array([z['plan_plant_steer'] for z in part])
            gap=float(np.mean(abs(b)-abs(c)))
            summary.append(dict(source=source,subset=subset,n=len(part),baseline_gap=gap,
                                b_mean_abs=float(np.mean(abs(b))),c_mean_abs=float(np.mean(abs(c))),
                                c_plant_mean_abs=float(np.mean(abs(n))),
                                closed_gap=float(np.mean(abs(n)-abs(c))),
                                closed_fraction=float(np.mean(abs(n)-abs(c))/gap),
                                b_closer_to_c_target=float(np.mean(abs(b-n)<abs(c-n))),
                                b_over_c_target=float(np.mean(np.sign(n)*(b-n)>.05)),
                                b_under_c_target=float(np.mean(np.sign(n)*(b-n)<-.05)),
                                plan_mean_abs=float(np.mean(abs(p[np.isfinite(p)]))) if np.isfinite(p).any() else float('nan'),
                                b_closer_to_plan_target=float(np.mean(abs(b-p)<abs(c-p))) if np.isfinite(p).all() else float('nan'),
                                b_under_plan_target=float(np.mean(np.sign(p)*(b-p)<-.05)) if np.isfinite(p).all() else float('nan')))
    write_csv(OUT/'normative_summary.csv',summary)
    # Opposite-sign runs: contiguous segments are a more useful noise check than isolated ticks.
    by_run=defaultdict(list)
    for z in opposite:by_run[(z['source'],z['route'],z['seed'])].append(z['step'])
    spans=[]
    for key,steps in by_run.items():
        steps=sorted(steps);start=previous=steps[0]
        for step in steps[1:]:
            if step!=previous+1:
                spans.append(dict(source=key[0],route=key[1],seed=key[2],start=start,end=previous,ticks=previous-start+1))
                start=step
            previous=step
        spans.append(dict(source=key[0],route=key[1],seed=key[2],start=start,end=previous,ticks=previous-start+1))
    write_csv(OUT/'opposite_spans.csv',spans)
    comparison=[]
    specs=[('all','all',lambda z:True),
           ('speed','1-2',lambda z:1<=z['speed']<2),('speed','2-4',lambda z:2<=z['speed']<4),
           ('speed','4-6',lambda z:4<=z['speed']<6),('speed','6-8',lambda z:6<=z['speed']<8),
           ('speed','8+',lambda z:z['speed']>=8),
           ('curvature','.03-.06',lambda z:.03<=abs(z['plan_k_near'])<.06),
           ('curvature','.06-.12',lambda z:.06<=abs(z['plan_k_near'])<.12),
           ('curvature','.12+',lambda z:abs(z['plan_k_near'])>=.12),
           ('dense','before 0-5',lambda z:0<=z['dense_turn_distance_m']<5),
           ('dense','after 0-5',lambda z:-5<=z['dense_turn_distance_m']<0),
           ('dense','after 5-10',lambda z:-10<=z['dense_turn_distance_m']<-5)]
    for group,label,test in specs:
        part=[z for z in out if test(z)]
        if not part:continue
        opp=[z for z in part if z['opposite']]
        comparison.append(dict(group=group,bin=label,n=len(part),opposite_n=len(opp),
                               opposite_rate=len(opp)/len(part),
                               opposite_s_shape=sum(z['s_shape'] for z in opp),
                               opposite_c_near_zero=sum(z['c_near_zero'] for z in opp),
                               opposite_b_near_zero=sum(z['b_near_zero'] for z in opp),
                               opposite_b_plan_sign=sum(z['b_matches_plan'] for z in opp)))
    write_csv(OUT/'opposite_summary.csv',comparison)
    return out,opposite


def plots(lags,map_rows,command,norm,opposite):
    plot_style.apply()
    fig,ax=plt.subplots(1,2,figsize=(plot_style.DOUBLE_COLUMN_IN,2.35))
    for arm,color in [('B',plot_style.PALETTE['orange']),('C',plot_style.PALETTE['blue']),('D',plot_style.PALETTE['green'])]:
        q=[r for r in lags if r['arm']==arm and r['predictor']=='k_ack_right']
        ax[0].plot([r['lag_s'] for r in q],[r['corr_centered'] for r in q],marker='o',color=color,label=arm)
    ax[0].set_xlabel('Steer lead over truth curvature (s)');ax[0].set_ylabel('Within-run correlation')
    ax[0].legend();ax[0].set_title('Turn ticks, moving ≥2 m/s')
    q=[r for r in map_rows if r['subset']=='quasi_steady' and r['arm']=='all']
    xx=[(r['speed_lo']+r['speed_hi'])/2 for r in q]
    ax[1].plot(xx,[r['gain_nom'] for r in q],marker='o',label='nominal gain')
    ax[1].plot(xx,[r['gain_ack'] for r in q],marker='s',label='Ackermann gain')
    ax[1].axhline(1,color='gray',linewidth=.6)
    ax[1].set_xlabel('Speed (m/s)');ax[1].set_ylabel('Fit achieved / predicted')
    ax[1].set_ylim(.65,1.07);ax[1].legend();ax[1].set_title('All arms, quasi-steady')
    fig.tight_layout();plot_style.save(fig,OUT/'plant_map');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(plot_style.DOUBLE_COLUMN_IN,2.35))
    for arm,color in [('C',plot_style.PALETTE['blue']),('D',plot_style.PALETTE['green'])]:
        q=[r for r in command if r['arm']==arm and r['source']=='W2']
        ax[0].plot([(r['speed_lo']+r['speed_hi'])/2 for r in q],[r['gain_command'] for r in q],
                   marker='o',color=color,label=arm)
    ax[0].axhline(1,color='gray',linewidth=.6);ax[0].set_ylim(.65,1.1)
    ax[0].set_xlabel('Speed (m/s)');ax[0].set_ylabel('Achieved / pursuit curvature')
    ax[0].legend();ax[0].set_title('Own turn trajectories, 0.10 s lead')
    shared=[r for r in norm if 2<=r['speed']<12 and np.isfinite(r['c_plant_steer']) and
            np.isfinite(r['plan_plant_steer']) and abs(r['c_plant_steer'])<=.8 and abs(r['plan_plant_steer'])<=.8]
    names=['B executed','C shadow','C + Ackermann','C + empirical map','Plan + empirical map']
    values=[]
    for key in ('b_steer','c_steer','c_ack_only_steer','c_plant_steer','plan_plant_steer'):
        values.append(float(np.mean([abs(r[key]) for r in shared])))
    ax[1].barh(range(len(names)),values,color=[plot_style.PALETTE['orange'],plot_style.PALETTE['blue'],
                                              plot_style.PALETTE['sky_blue'],plot_style.PALETTE['green'],
                                              plot_style.PALETTE['purple']])
    ax[1].set_yticks(range(len(names)),names);ax[1].invert_yaxis()
    ax[1].set_xlabel('Mean absolute steer');ax[1].set_title(f'Same B turn ticks, feasible n={len(shared)}')
    fig.tight_layout();plot_style.save(fig,OUT/'normative');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(plot_style.DOUBLE_COLUMN_IN,2.2))
    speed=[(1,2),(2,4),(4,6),(6,8),(8,100)]
    for j,(lo,hi) in enumerate(speed):
        part=[z for z in norm if lo<=z['speed']<hi]
        ax[0].bar(j,sum(z['opposite'] for z in part)/len(part),color=plot_style.PALETTE['blue'])
    ax[0].set_xticks(range(5),['1–2','2–4','4–6','6–8','≥8'])
    ax[0].set_xlabel('Speed (m/s)');ax[0].set_ylabel('Opposite-sign fraction')
    ax[0].set_ylim(0,.7);ax[0].set_title('All B turn ticks')
    regions=[('0–5 before',lambda z:0<=z['dense_turn_distance_m']<5),
             ('0–5 after',lambda z:-5<=z['dense_turn_distance_m']<0),
             ('5–10 after',lambda z:-10<=z['dense_turn_distance_m']<-5)]
    for j,(name,test) in enumerate(regions):
        part=[z for z in norm if test(z)]
        ax[1].bar(j,sum(z['opposite'] for z in part)/len(part),color=plot_style.PALETTE['orange'])
    ax[1].set_xticks(range(3),[x[0] for x in regions],rotation=20)
    ax[1].set_ylabel('Opposite-sign fraction');ax[1].set_ylim(0,.7)
    ax[1].set_title('Dense 2084/27529 only')
    fig.tight_layout();plot_style.save(fig,OUT/'opposite_sign');plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    groups=load_runs()
    lags=lag_analysis(groups);maps=plant_map(groups);commands=command_gain(groups)
    rear_path_validation(groups);rear_slip_check(groups)
    norm,opposite=normative(maps)
    plots(lags,maps,commands,norm,opposite)
    print('norm',len(norm),'opposite',len(opposite))


if __name__=='__main__':main()
