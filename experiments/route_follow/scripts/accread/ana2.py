"""Mac step 5: both labels' state where the failing label had its event (speed, model plan 4 s yaw, ACC won / guard / v_curve)."""
import json, numpy as np, collections, sys
sys.path.insert(0, __import__("os").path.dirname(__file__)); from ana import A,S,L,info,proj
def station_state(k,lab,s_q,win=1.5):
    x=A[k][lab]; c=x["ctl"]; lg=L[k]; LP=np.array(lg["P"]); LS=np.r_[0,np.cumsum(np.hypot(*np.diff(LP,axis=0).T))]
    t=np.array(c["timestamp_us"]); t0=t[0]-1e5; P=np.c_[c["x"],c["y"]]; v=np.array(c["vx"])
    ss=np.array([proj(LP,LS,p)[0] for p in P]); j=np.flatnonzero(ss>=s_q-0.5)
    if not len(j): return f"{lab}: never reached s={s_q:.0f} (max {ss.max():.0f})"
    j=j[0]; tq=(t[j]-t0)/1e6
    dec=[q for q in x["dec"] if abs((q["now"]-t0)/1e6-tq)<=win]
    w=dict(collections.Counter(q["acc"]["won"] for q in dec)); g=dict(collections.Counter(q["acc"]["guard"] or "-" for q in dec))
    yp=np.degrees(np.mean([q["yaw_pm"] for q in dec])) if dec else np.nan
    vm=np.mean([q["arc_pm"]/4 for q in dec]) if dec else np.nan; vs=np.mean([q["arc_ps"]/4 for q in dec]) if dec else np.nan
    vc=[q["acc"]["v_curve"] for q in dec if q["acc"]["v_curve"] is not None]
    return f"{lab}: t={tq:.1f} v={v[j]:.1f} plan4s yaw_model={yp:.0f}deg arc/4 model={vm:.1f} served={vs:.1f} " + (f"won={w} guard={g} v_curve={min(vc):.1f}" if vc else "")
for k in sorted(A):
    if "acc" not in A[k] or A[k]["acc"]["ctl"] is None: continue
    sg,sa=A[k]["g10"]["score"],A[k]["acc"]["score"]
    if bool(sg)==bool(sa): continue
    bad="g10" if not sg else "acc"; r=info(k,bad)
    if r.get("s_ev") is None: print(k[7:15],"no event"); continue
    print(f"== {'FIX' if bad=='g10' else 'NEW'} {k[7:15]} event of {bad} at s={r['s_ev']:.0f}")
    for lab in ("g10","acc"): print("   ",station_state(k,lab,r["s_ev"]))
