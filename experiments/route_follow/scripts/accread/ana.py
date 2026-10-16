"""Mac step 4: per flipped scene and label: first zero event, speed / station / lateral there, the human's speed at that station, human stop
points, wait lines on the log path, ACC won / guard counts in the 3 s before the event. Reads the three JSON files from tmp/accread/."""
import json, numpy as np, collections
A=json.load(open("tmp/accread/accread.json")); S=json.load(open("tmp/accread/accread_ser.json")); L=json.load(open("tmp/accread/accread_logs.json"))
ex=set(open("experiments/alpasim/results/pai_exam/hard_exam.txt").read().split()); st=set(open("experiments/route_follow/results/lon_diag/route_end_scenes.txt").read().split())
def proj(LP,LS,p):
    d=np.hypot(*(LP-p).T); i=int(d.argmin()); return LS[i], d[i]
def info(k,lab):
    x=A[k][lab]; c=x["ctl"]; d=S[k][lab]; lg=L[k]
    LP=np.array(lg["P"]); LT=np.array(lg["T"])/1e6; LS=np.r_[0,np.cumsum(np.hypot(*np.diff(LP,axis=0).T))]; Lv=np.gradient(LS,LT-LT[0]+1e-9*np.arange(len(LT)))
    t=np.array(c["timestamp_us"]); t0=t[0]-1e5; P=np.c_[c["x"],c["y"]]; v=np.array(c["vx"])
    ev={}
    for n in ("collision_front","collision_lateral","offroad","left_corridor_laterally","collision_rear"):
        T,V=map(np.array,d[n]); i=np.flatnonzero(V>0); ev[n]=(T[i[0]]-t0)/1e6 if len(i) else None
    T,V=map(np.array,d["progress"]); i=np.flatnonzero(V>=0.999); tp=(T[i[0]]-t0)/1e6 if len(i) else None
    zs=[(tt,n) for n,tt in ev.items() if tt is not None and n!="collision_rear"]
    tev,nev=min(zs) if zs else (None,None)
    stop=np.flatnonzero(Lv<0.5); 
    # human stop points: start of runs below 0.5 m/s
    runs=[LS[j] for j in stop if j==0 or Lv[j-1]>=0.5]
    r=dict(tev=tev,nev=nev,tprog=tp,rear=ev["collision_rear"],loglen=LS[-1],logvend=float(np.mean(Lv[-5:])),hstops=[round(s) for s in runs])
    if tev is not None:
        j=int(np.searchsorted((t-t0)/1e6,tev)); j=min(j,len(t)-1)
        s,lat=proj(LP,LS,P[j]); r.update(v_ev=float(v[j]),s_ev=float(s),lat_ev=float(lat),vlog_ev=float(np.interp(s,LS,Lv)))
        dec=[q for q in x["dec"] if tev-3<=(q["now"]-t0)/1e6<=tev]
        r["won3"]=dict(collections.Counter(q["acc"]["won"] for q in dec)); r["guard3"]=dict(collections.Counter(q["acc"]["guard"] or "-" for q in dec))
        r["vmodel3"]=[round(q["acc"]["v_model"],1) if q["acc"]["v_model"] is not None else None for q in dec[::6]]
        r["v15_3"]=[round(q["acc"]["v15"],1) if q["acc"]["v15"] is not None else None for q in dec[::6]]
        r["vcmin3"]=min([q["acc"]["v_curve_min"] for q in dec if q["acc"]["v_curve_min"] is not None] or [np.nan])
    # wait lines: station along log of each line centre (within 4 m)
    wls=[]
    for w in lg["wl"]:
        q=np.array(w["pts"]).mean(0); s,lat=proj(LP,LS,q)
        if lat<4: wls.append((round(s),w["kind"].split(".")[-1]))
    r["wl"]=sorted(wls)
    return r
if __name__=="__main__":
    for k in sorted(A):
        if "acc" not in A[k] or A[k]["acc"]["ctl"] is None: continue
        sg,sa=A[k]["g10"]["score"],A[k]["acc"]["score"]
        tag="NEW" if sg and not sa else "FIX" if sa and not sg else None
        if not tag: continue
        print(f"== {tag} {k[7:15]} {'EXAM' if k in ex else ''}{' STOP17' if k in st else ''} g10={sg:.2f} acc={sa:.2f}")
        for lab in ("g10","acc"):
            r=info(k,lab); print("  ",lab,{a:(round(b,1) if isinstance(b,float) else b) for a,b in r.items()})
