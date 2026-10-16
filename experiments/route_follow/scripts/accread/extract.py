"""PAI ACC read (results/pai_acc/read.md), box step 1 (.venv): per-rollout summary, controller trace and driver decisions (fix.acc,
model / served plan) of g10 and acc for the scenes whose zero status flips plus the turn scenes -> /tmp/accread.json."""
import json, glob, os, csv, sys
import numpy as np
O = "/root/autodl-tmp/ujs/runs/route_follow/pai_g10"
J = json.load(open(f"{O}/acc.json"))
g, a = J["g10"], J["acc"]
want = set(s for s in g if bool(g[s]["kind"]) != bool(a[s]["kind"])) | set(J["_turns"]) | {"clipgt-054b5901-fdcb-4146-b125-eb2bb333cf02"}
out = {}
for lab in ("g10", "acc"):
    for sim in glob.glob(f"{O}/{lab}/runs/*/sim"):
        run = os.path.dirname(sim)
        roll = {}
        for r in json.load(open(f"{sim}/aggregate/results-summary.json"))["rollouts"]:
            if r["clipgt_id"] in want:
                roll[r["rollout_id"]] = r
        if not roll:
            continue
        dec = {}
        for line in open(f"{run}/driver/drive.jsonl"):
            if '"infer": true' not in line:
                continue
            x = json.loads(line)
            if x["session"] not in roll:
                continue
            f = x.get("fix") or {}
            ac = f.get("acc") or {}
            pm = np.array(x["poses_model"]); ps = np.array(x["poses"])
            dec.setdefault(x["session"], []).append(dict(now=x["now"], k=x["k"], cmd=x.get("cmd"), v0=f.get("v0"), acc={k: ac.get(k) for k in ("v_set", "v_hand", "guard", "v_curve", "v_curve_min", "v_model", "cover", "v_lead", "v15", "won", "near")},
                 lead_src=(f.get("lead") or {}).get("src"), yaw_pm=float(pm[-1, 2]), yaw_pm2=float(pm[3, 2]), lat_pm=float(pm[-1, 1]), arc_pm=float(np.hypot(*np.diff(np.vstack([[0, 0], pm[:, :2]]), axis=0).T).sum()),
                 yaw_ps=float(ps[-1, 2]), arc_ps=float(np.hypot(*np.diff(np.vstack([[0, 0], ps[:, :2]]), axis=0).T).sum())))
        for rid, r in roll.items():
            c = f"{sim}/controller/alpasim_controller_{rid}.csv"
            ctl = None
            if os.path.exists(c):
                rows = list(csv.DictReader(open(c)))
                ctl = {k: [float(x[k]) for x in rows] for k in ("timestamp_us", "x", "y", "vx", "wz", "acceleration")}
            m = {k: v for k, v in (r.get("metrics") or {}).items() if isinstance(v, (int, float))}
            out.setdefault(r["clipgt_id"], {})[lab] = dict(score=r["score"], metrics=m, err=bool((r.get("metrics") or {}).get("error")), ctl=ctl, dec=dec.get(rid, []))
json.dump(out, open("/tmp/accread.json", "w"))
print(len(out), "scenes")
