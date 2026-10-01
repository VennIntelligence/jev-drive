"""Locked validation readouts, segment bootstrap, frontier and paper figures for gate/curve.

Commands: gate-val (AUC first), curve-read, frontier, figures. Results stay in the box run directory;
only the explicitly selected small tables and PNG figures are downloaded to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_l as L  # noqa: E402
from jevdrive import op_adapt_r2 as R  # noqa: E402
from jevdrive import waymo as W  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from op_adapt_l_gate_curve import ROOT, MAIN_TAGS, Gate, grant_check  # noqa: E402
from op_adapt_l_gate_curve_prep import dump  # noqa: E402
from op_adapt_l_train import Log  # noqa: E402


class Bootstrap:
    """One shared segment resample matrix for all paired statistics of a row set."""
    def __init__(self, groups):
        self.gi, self.n, self.weights = L.cluster_boot(np.asarray(groups), B=2000, seed=0)
        self.counts = np.bincount(self.gi, minlength=self.n).astype(float)
        self.denominator = self.weights @ self.counts

    def draws(self, x):
        total = np.bincount(self.gi, weights=np.asarray(x, float), minlength=self.n)
        return (self.weights @ total) / self.denominator

    def paired(self, a, o):
        a, o = np.asarray(a, float), np.asarray(o, float)
        d = self.draws(a-o)
        lo, hi = np.nanpercentile(d, [2.5,97.5])
        return {"n": len(a), "clusters": self.n, "adapt": float(a.mean()), "orig": float(o.mean()),
                "delta": float((a-o).mean()), "lo": float(lo), "hi": float(hi)}

    def quantile(self, x, q):
        order = np.argsort(x)
        values, code = np.asarray(x)[order], self.gi[order]
        boot = []
        for row in self.weights:
            cumulative = np.cumsum(row[code])
            pos = (cumulative[-1]-1)*q
            left = np.searchsorted(cumulative, np.floor(pos)+1, side="left")
            right = np.searchsorted(cumulative, np.ceil(pos)+1, side="left")
            boot.append(values[left]+(pos-np.floor(pos))*(values[right]-values[left]))
        lo, hi = np.percentile(boot, [2.5,97.5])
        return {"n": len(x), "clusters": self.n, "adapt": float(np.quantile(x,q)), "lo":float(lo), "hi":float(hi)}


def old_eval(tag, which):
    return dict(np.load(L.lroot("readout",tag,"eval") / f"{which}.npz", allow_pickle=True))


def stationary(tab):
    return np.maximum.reduce([tab[k] for k in ("v0","vm05","vm1")]) <= .5


def indicators(tab, rows, plan):
    return L.row_metrics(plan, tab, rows, L.CAM_X["wod"])


def plan_metrics(tab, rows, plans, po, tag):
    groups = tab["seq"][rows]
    orig = indicators(tab, rows, po)
    per = [indicators(tab,rows,p) for p in plans]
    masks = {s: tab[f"s_{s}"][rows] for s in L.SLICE3+L.CONTRAST}
    masks["other"] = ~np.logical_or.reduce([masks[s] for s in L.SLICE3])
    masks["all"] = np.ones(len(rows),bool)
    result = []
    for seed in tqdm(list(range(len(plans)))+["mean"], desc=f"{tag} paired metrics", mininterval=5):
        m = per[seed] if seed != "mean" else {k:np.mean([p[k] for p in per],axis=0) for k in orig}
        for s, mask in masks.items():
            if not mask.any():
                continue
            boot = Bootstrap(groups[mask])
            keys = ["ade4","slow","fast"]
            if s in L.SLICE3:
                keys += [f"cap_{s}"]
            if s=="stay": keys += ["false_start"]
            if s=="control": keys += ["false_stop","false_turn"]
            if s=="straight_int": keys += ["false_turn"]
            for k in keys:
                result.append({"model":tag,"seed":seed,"slice":s,"metric":k,"xscale":1.0,
                               **boot.paired(m[k][mask],orig[k][mask])})
        dr = (A.plan_drift(plans[seed],po) if seed!="mean" else
              np.mean([A.plan_drift(p,po) for p in plans],axis=0))
        mask = masks["other"]
        boot = Bootstrap(groups[mask])
        for k,q in (("drift_median",.5),("drift_p95",.95)):
            result.append({"model":tag,"seed":seed,"slice":"other","metric":k,"xscale":1.0,
                           **boot.quantile(dr[mask],q)})
    return result


def rater_data(names):
    sets = Z.load_sets()["rater"]
    pos = pd.Series(np.arange(len(sets["name"])),index=sets["name"].astype(str))
    idx = pos.reindex(names).to_numpy().astype(int)
    calib = json.loads((Z.root()/"op_calib.json").read_text())
    xy = np.stack([np.asarray(calib[n.rsplit("-",1)[0]]["1"]["extrinsic"]).reshape(4,4)[:2,3] for n in names])
    return {"traj":sets["traj"][idx],"scores":sets["scores"][idx],"speed":W.init_speed(sets["past"][idx]),
            "category":sets["cluster"][idx].astype(str), "xy":xy,
            "segment":np.asarray([n.rsplit("-",1)[0] for n in names])}


def rfs(plan, data, scale):
    wp = np.stack([Z.openpilot_to_wod(plan[i,:,:3],plan[i,:,11],A.T_IDXS,data["xy"][i]) for i in range(len(plan))])[...,:2]
    wp[...,0] *= scale
    return np.asarray(W.rater_feedback_score(wp,data["traj"],data["scores"],data["speed"]),float)


def rfs_metrics(names, plans, po, tag, out):
    data = rater_data(names)
    result = []
    boot = Bootstrap(data["segment"])
    cats = np.unique(data["category"])
    for scale in (1.,1.06):
        orig = rfs(po,data,scale)
        per = [rfs(p,data,scale) for p in plans]
        np.savez(out/f"rfs_frames_x{scale}.npz",orig=orig,adapt=np.asarray(per),names=names,category=data["category"])
        for seed in list(range(len(plans)))+["mean"]:
            a = per[seed] if seed!="mean" else np.mean(per,axis=0)
            draws, amean, omean = [], [], []
            for cat in cats:
                mask = data["category"]==cat
                sums = np.bincount(boot.gi[mask],weights=(a-orig)[mask],minlength=boot.n)
                counts = np.bincount(boot.gi[mask],minlength=boot.n)
                den = boot.weights @ counts
                with np.errstate(invalid="ignore",divide="ignore"):
                    draws.append((boot.weights@sums)/den)
                amean.append(a[mask].mean()); omean.append(orig[mask].mean())
                result.append({"model":tag,"seed":seed,"slice":f"cat:{cat}","metric":"rfs","xscale":scale,
                               **Bootstrap(data["segment"][mask]).paired(a[mask],orig[mask])})
            dd = np.mean(draws,axis=0)
            lo,hi = np.nanpercentile(dd,[2.5,97.5])
            official = W.rfs_by_cluster(a,data["category"])[0]
            official_o = W.rfs_by_cluster(orig,data["category"])[0]
            assert abs(official-np.mean(amean))<1e-9 and abs(official_o-np.mean(omean))<1e-9
            result.append({"model":tag,"seed":seed,"slice":"all","metric":"rfs","xscale":scale,"n":len(names),
                           "clusters":boot.n,"adapt":official,"orig":official_o,"delta":official-official_o,
                           "lo":lo,"hi":hi,"undefined_bootstraps":int((~np.isfinite(dd)).sum())})
            result.append({"model":tag,"seed":seed,"slice":"all","metric":"rfs_frame_mean","xscale":scale,
                           **boot.paired(a,orig)})
    return result


def gate_val(out, log, args):
    lock = json.loads((ROOT/"gate-lock/lock.json").read_text())
    assert (ROOT/"gate-lock/DONE").exists()
    for p in (Path(lock["head"]),ROOT/"gate_norm.npz"):
        assert hashlib.sha256(p.read_bytes()).hexdigest()==lock["sha256"][p.name], "Gate lock checksum mismatch"
    dev = torch.device("cuda")
    ck = torch.load(lock["head"],map_location="cpu",weights_only=False)
    head = Gate(ck["features"],ck["hidden"]).to(dev).eval()
    head.load_state_dict(ck["head"])
    D = L.Data(("wodval",),hstore=True)
    orig = old_eval("O","wodval")
    rows, tab = orig["rows"], D.tab["wodval"]
    eligible = stationary(tab)[rows]
    rr = rows[eligible]
    prob = np.zeros(len(rows),np.float32)
    with torch.no_grad():
        for offset in tqdm(range(0,len(rr),256),desc="Locked val AUC",mininterval=5):
            batch = D.hs["wodval"].gather(D.dom["wodval"].ctx[rr[offset:offset+256]]).reshape(-1,ck["features"])
            prob[np.flatnonzero(eligible)[offset:offset+256]] = head(torch.as_tensor(batch,device=dev)).sigmoid().cpu().numpy()
    # AUC is deliberately the first validation outcome written to disk or logs.
    aucs = []
    for name,mask in (("all_stationary",eligible),("start_stay",eligible&(tab["s_start"][rows]|tab["s_stay"][rows]))):
        labels = tab["s_start"][rows[mask]].astype(int)
        b = Bootstrap(tab["seq"][rows[mask]])
        scores = prob[mask]
        auc = roc_auc_score(labels,scores)
        dd = [roc_auc_score(labels,scores,sample_weight=w[b.gi]) for w in tqdm(b.weights,desc=f"AUC bootstrap {name}",mininterval=5)]
        lo,hi = np.percentile(dd,[2.5,97.5])
        aucs.append({"split":"wodval","population":name,"n":int(mask.sum()),"segments":b.n,"positive":int(labels.sum()),
                     "auc":float(auc),"lo":float(lo),"hi":float(hi),"G0_pass":bool(auc>=.75 and lo>.5)})
        log.info(f"FIRST VAL READOUT AUC {name}: {auc:.4f} [{lo:.4f}, {hi:.4f}]")
        log.scalar(f"val/auc/{name}",auc,0)
    pd.DataFrame(aucs).to_csv(out/"auc.csv",index=False)
    np.savez(out/"prob.npz",rows=rows,prob=prob,eligible=eligible,tau=lock["tau"])
    plans=[]
    for seed,tag in enumerate(tqdm(MAIN_TAGS,desc="Gated cached plans",mininterval=5)):
        z=old_eval(tag,"wodval")
        assert np.array_equal(z["rows"],rows)
        use_o=eligible&(prob<=lock["tau"])
        p=z["plan"].copy(); p[use_o]=orig["plan"][use_o]
        assert np.array_equal(p[~eligible],z["plan"][~eligible])
        plans.append(p)
        np.savez(out/f"wodval-s{seed}.npz",rows=rows,plan=p)
    result=plan_metrics(tab,rows,plans,orig["plan"],"gate")
    # Only rater frames require a new O feature pass; all main plans remain cached.
    import op_adapt_l_prep as P
    rz=old_eval("O","rater"); names=rz["names"].astype(str)
    kin=P.wod_kin(names)
    mask=stationary(kin)
    rd=R.Domain("rater",L.r2t())
    pos=pd.Series(np.arange(len(names)),index=names).reindex(rd.col("key").astype(str)).to_numpy().astype(int)
    assert np.array_equal(names[pos],rd.col("key").astype(str))
    rp=np.zeros(len(names),np.float32)
    om=L.LModel().to(dev).eval()
    with torch.no_grad():
        for i in tqdm(range(0,len(rd),64),desc="Rater O gate features",mininterval=5):
            indices=np.arange(i,min(i+64,len(rd)))
            selected=indices[mask[pos[indices]]]
            if not len(selected): continue
            x,valid=rd.gather(selected)
            h=om.stage4(torch.as_tensor(x,device=dev)).reshape(-1,ck["features"])
            rp[pos[selected]]=head(h).sigmoid().cpu().numpy()
    rplans=[]
    for seed,tag in enumerate(MAIN_TAGS):
        z=old_eval(tag,"rater")
        assert np.array_equal(z["names"].astype(str),names)
        p=z["plan"].copy(); choose=mask&(rp<=lock["tau"]); p[choose]=rz["plan"][choose]
        rplans.append(p)
        np.savez(out/f"rater-s{seed}.npz",names=names,plan=p)
    result+=rfs_metrics(names,rplans,rz["plan"],"gate",out)
    pd.DataFrame(result).to_csv(out/"metrics.csv",index=False)
    log.event("end",tau=lock["tau"],auc=aucs[0]["auc"])


def curve_read(out, log, args):
    z=[dict(np.load(ROOT/f"eval-{args.size}-s{s}/wodval.npz")) for s in (0,1)]
    o=old_eval("O","wodval"); rows=o["rows"]
    assert all(np.array_equal(x["rows"],rows) for x in z)
    tab=dict(np.load(L.lroot("prep")/"wodval.npz",allow_pickle=True))
    result=plan_metrics(tab,rows,[x["plan"] for x in z],o["plan"],f"curve-{args.size}")
    rz=[dict(np.load(ROOT/f"eval-{args.size}-s{s}/rater.npz",allow_pickle=True)) for s in (0,1)]
    ro=old_eval("O","rater")
    assert all(np.array_equal(x["names"],ro["names"]) for x in rz)
    result+=rfs_metrics(ro["names"].astype(str),[x["plan"] for x in rz],ro["plan"],f"curve-{args.size}",out)
    pd.DataFrame(result).to_csv(out/"metrics.csv",index=False)
    log.event("end",size=args.size)


def hull(points):
    p=np.asarray(sorted(points),float)
    upper=[]
    for x,y in p:
        if upper and x==upper[-1][0]:
            if y<=upper[-1][1]: continue
            upper.pop()
        while len(upper)>=2:
            a,b=upper[-2:]
            cross=(b[0]-a[0])*(y-b[1])-(b[1]-a[1])*(x-b[0])
            if cross>=0: upper.pop()
            else: break
        upper.append((x,y))
    return np.asarray(upper)


def frontier(out,log,args):
    orig=old_eval("O","wodval");rows=orig["rows"]
    tab=dict(np.load(L.lroot("prep")/"wodval.npz",allow_pickle=True))
    mo=indicators(tab,rows,orig["plan"])
    masks={s:tab[f"s_{s}"][rows] for s in ("start","stay")}
    gi,n,weights=L.cluster_boot(tab["seq"][rows],2000,0)
    draws={};points=[]
    for arm in tqdm(("dw03","dw1","main","dw10","stayheavy","gate"),desc="Paired frontier",mininterval=5):
        if arm=="gate":
            plans=[np.load(ROOT/"gate-val"/f"wodval-s{s}.npz")["plan"] for s in range(3)]
        else:
            plans=[old_eval(f"{arm}-s{s}","wodval")["plan"] for s in range(3)]
        mm=[indicators(tab,rows,p) for p in plans]
        ds={};point={"model":arm}
        for s,k in (("start","cap_start"),("stay","false_start")):
            mask=masks[s]
            d=np.mean([m[k] for m in mm],axis=0)-mo[k]
            totals=np.bincount(gi[mask],weights=d[mask],minlength=n)
            counts=np.bincount(gi[mask],minlength=n)
            ds[s]=(weights@totals)/(weights@counts)
            point[s]=float(d[mask].mean())
            point[f"{s}_lo"],point[f"{s}_hi"]=np.percentile(ds[s],[2.5,97.5])
        draws[arm]=ds;points.append(point)
    baseline=[p for p in points if p["model"]!="gate"]
    gate=next(p for p in points if p["model"]=="gate")
    upper=hull([(0,0)]+[(p["stay"],p["start"]) for p in baseline])
    delta=gate["start"]-np.interp(gate["stay"],upper[:,0],upper[:,1]) if upper[0,0]<=gate["stay"]<=upper[-1,0] else np.nan
    dd=[]
    for j in tqdm(range(2000),desc="Frontier segment bootstrap",mininterval=5):
        h=hull([(0,0)]+[(draws[p["model"]]["stay"][j],draws[p["model"]]["start"][j]) for p in baseline])
        x,y=draws["gate"]["stay"][j],draws["gate"]["start"][j]
        dd.append(y-np.interp(x,h[:,0],h[:,1]) if h[0,0]<=x<=h[-1,0] else np.nan)
    finite = np.asarray(dd)[np.isfinite(dd)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    optional = lambda x: float(x) if np.isfinite(x) else None
    verdict={"delta":optional(delta),"lo":optional(lo),"hi":optional(hi),"undefined_bootstraps":int((~np.isfinite(dd)).sum()),
             "G6_pass":bool(np.isfinite(delta) and delta>0 and lo>0),"hull":upper.tolist()}
    dump(out/"frontier.json",verdict)
    pd.DataFrame(points).to_csv(out/"frontier.csv",index=False)
    log.event("end",**verdict)


def figures(out,log,args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
    import plot_style as S
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply()
    df=pd.read_csv(ROOT/"frontier/frontier.csv")
    fig,axes=plt.subplots(1,2,figsize=(S.DOUBLE_COLUMN_IN,2.75))
    offsets={"gate":(-20,16),"dw10":(4,-15),"stayheavy":(4,9),"main":(4,8),"dw1":(4,0),"dw03":(-38,-15)}
    labels={"dw03":"dw 0.3","dw1":"dw 1","main":"dw 3","dw10":"dw 10","stayheavy":"Stay-heavy","gate":"Gate"}
    h=np.asarray(json.loads((ROOT/"frontier/frontier.json").read_text())["hull"])
    for panel,ax in enumerate(axes):
        for _,p in df.iterrows():
            color=S.PALETTE["blue"] if p.model=="gate" else S.BASELINE
            ax.errorbar(100*p.stay,p.start,xerr=[[100*(p.stay-p.stay_lo)],[100*(p.stay_hi-p.stay)]],
                        yerr=[[p.start-p.start_lo],[p.start_hi-p.start]],fmt="o",color=color,markersize=3,capsize=2)
            if (panel==0 and p.model in ("dw03","dw1")) or (panel==1 and p.model not in ("dw03","dw1")):
                ax.annotate(labels[p.model],(100*p.stay,p.start),xytext=offsets[p.model],textcoords="offset points",
                            fontsize=7.5,color=color)
        ax.plot(100*h[:,0],h[:,1],color=S.BASELINE,linestyle="--")
        ax.axvline(2,color=S.PALETTE["vermillion"],ls=":",lw=.7)
        ax.axhline(.07,color=S.PALETTE["vermillion"],ls=":",lw=.7)
        ax.set(xlabel="False-start increase (pp)",ylabel="Start capture gain")
        S.panel(ax,"Full frontier" if panel==0 else "Near the registered limits")
    axes[1].set(xlim=(-.1,3.4),ylim=(-.01,.14))
    fig.subplots_adjust(left=.08,bottom=.20,right=.98,top=.88,wspace=.32)
    S.save(fig,out/"gate-frontier");plt.close(fig)
    frames=[]
    for size in ("25","110","300","all"):
        p=ROOT/f"curve-read-{size}/metrics.csv"
        frames.append(pd.read_csv(p))
    metrics=pd.concat(frames,ignore_index=True)
    counts=pd.read_csv(ROOT/"prep-v3/counts.csv").astype({"size":str})
    for axis in ("segments","events"):
        fig,axs=plt.subplots(1,3,figsize=(S.DOUBLE_COLUMN_IN,2.4))
        for ax,s,color,line in zip(axs,L.SLICE3,("blue","orange","green"),(.05,.15,.06)):
            d=metrics[(metrics.seed.astype(str)=="mean")&(metrics["slice"]==s)&(metrics.metric==f"cap_{s}")].copy()
            d["size"]=d.model.str.replace("curve-","",regex=False)
            d=d.merge(counts[counts["slice"]==s],on="size").sort_values(axis)
            ax.errorbar(d[axis],d.delta,yerr=np.stack([d.delta-d.lo,d.hi-d.delta]),fmt="o-",color=S.PALETTE[color],markersize=3,capsize=2)
            for seed in ("0","1"):
                q=metrics[(metrics.seed.astype(str)==seed)&(metrics["slice"]==s)&(metrics.metric==f"cap_{s}")].copy()
                q["size"]=q.model.str.replace("curve-","",regex=False)
                q=q.merge(counts[counts["slice"]==s],on="size").sort_values(axis)
                ax.plot(q[axis],q.delta,color=S.PALETTE[color],alpha=.3,lw=.6)
            ax.axhline(line,color=S.BASELINE,ls=":",lw=.7)
            ax.set(xlabel="Training segments" if axis=="segments" else "Training events",ylabel="Capture gain" if s=="start" else "")
            S.panel(ax,{"start":"Start","stop":"Stop","turn_onset":"Turn onset"}[s])
        fig.subplots_adjust(left=.075,bottom=.22,right=.98,top=.87,wspace=.3)
        S.save(fig,out/f"curve-{axis}");plt.close(fig)
    log.event("end",figures=3)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("gate-val","curve-read","frontier","figures"))
    ap.add_argument("--size",choices=("25","110","300","all"),default="25")
    args=ap.parse_args()
    name=f"curve-read-{args.size}" if args.command=="curve-read" else args.command
    out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
    if (out/"DONE").exists() or (out/"ERROR").exists(): raise SystemExit(f"Existing sentinel: {out}")
    log=Log(out)
    assert log.tb is not None
    try:
        grant_check(gpu=args.command=="gate-val")
        log.event("start",command=args.command,size=args.size)
        {"gate-val":gate_val,"curve-read":curve_read,"frontier":frontier,"figures":figures}[args.command](out,log,args)
        (out/"DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    except BaseException as e:
        (out/"ERROR").write_text(traceback.format_exc());log.event("error",error=str(e));raise
    finally: log.tb.close()


if __name__=="__main__": main()
