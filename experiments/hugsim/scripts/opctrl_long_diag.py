"""Diagnosis of the stuck runs of the opctrl arm (decision 118): per run, when the car stops, what the model plans while standing,
and whether it ever asks to go. Reads zs_steps.jsonl of the opctrl and same-day base3 arms. CPU only.
Usage: python opctrl_long_diag.py <closed_dir> [out.json] [arm_tag] [base_tag]   (defaults cinque-opctrl, cinque-fixed-base3)"""
import csv, json, sys, collections
import numpy as np

D = sys.argv[1]
ARM = sys.argv[3] if len(sys.argv) > 3 else "cinque-opctrl"
BASE = sys.argv[4] if len(sys.argv) > 4 else "cinque-fixed-base3"
R = collections.defaultdict(dict)
for r in csv.DictReader(open(f"{D}/results.csv")):
    R[r["scenario"]][r["tag"]] = r


def load(tag, row):
    p = f"{D}/{tag}/zs/{row['run_dir'].split('/')[-1]}/zs_steps.jsonl"
    return [r for r in map(json.loads, open(p)) if "step" in r]


def summ(S):
    v = np.array([s["v"] for s in S])
    stop = np.array([bool(s.get("stop")) for s in S])
    still = v < 0.3
    mv = np.array([s["model_v"][0] for s in S]) if "model_v" in S[0] else np.zeros(len(S))
    mv8 = np.array([s["model_v"][1] for s in S])          # model plan speed at index 8 (~1.2 s)
    mv24 = np.array([s["model_v"][3] for s in S])
    lp = np.array([s.get("lead_prob") or 0 for s in S])
    nobj = np.array([s["n_obj"] for s in S])
    pos = np.array([s["pos"] for s in S])
    out = dict(n=len(S), end_v=float(v[-1]), frac_still=float(still.mean()), frac_stop_flag=float(stop.mean()),
               n_launch=int(((v[1:] >= 0.3) & (v[:-1] < 0.3)).sum()))
    # first standstill after having moved, and what follows
    moved = np.where(v > 1.0)[0]
    if len(moved):
        k0 = moved[0]
        st = np.where(still[k0:])[0]
        if len(st):
            k = k0 + st[0]
            out.update(first_stop_step=int(k), first_stop_pos=pos[k].round(2).tolist(),
                       vmax_before=float(v[:k].max()), still_after=float(still[k:].mean()),
                       plan_v1p2_while_still=float(np.median(mv8[k:][still[k:]])) if still[k:].any() else None,
                       plan_v6_while_still=float(np.median(mv24[k:][still[k:]])) if still[k:].any() else None,
                       lead_prob_while_still=float(np.median(lp[k:][still[k:]])) if still[k:].any() else None,
                       n_obj_while_still=float(np.median(nobj[k:][still[k:]])) if still[k:].any() else None,
                       stop_flag_while_still=float(stop[k:][still[k:]].mean()) if still[k:].any() else None,
                       pos_drift_after=float(np.linalg.norm(pos[-1] - pos[k])))
    else:
        out.update(never_moved=True, plan_v1p2=float(np.median(mv8)), stop_flag=float(stop.mean()), lead_prob=float(np.median(lp)))
    return out


res = {}
for s, d in R.items():
    o, b = d.get(ARM), d.get(BASE)
    if not o or not b:
        continue
    if o["end"] == "max_steps":
        res[s] = dict(op_end=o["end"], base_end=b["end"], op=summ(load(ARM, o)), base=summ(load(BASE, b)))
json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout", "w"), indent=1)
