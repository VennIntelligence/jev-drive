"""op-adapt H: every readout of a chain's models in one place (op-train venv, CPU). Writes $DATA_DIR/runs/op_adapt_H/report/<chain>/
dev.csv, probe_<model>_vs_<ref>.csv, navtest.csv, capture.csv, navhard.csv, hugsim.csv and summary.md (copied into
experiments/op_adapt_h/results/<chain>/ on the Mac).

  python experiments/op_adapt_h/scripts/h_report.py <chain> <tag> [<tag> ...]      (first tag = the main arm, second = its control)
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive import stats  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

H = data_dir() / "runs" / "op_adapt_H"
LR = data_dir() / "runs" / "op_adapt_L"


def nav_tokens_logs(split):
    import pickle
    from jevdrive import navsim_zs as Z
    with open(Z.root("index") / f"{split}_slim.pkl", "rb") as f:
        return {e["token"]: e["log_name"] for e in pickle.load(f)}


def per_token(csv):
    df = pd.read_csv(csv)
    df = df[~df.token.astype(str).str.startswith("extended_pdm_score") & (df.token != "average")]
    return df.set_index("token").score.astype(float)


def main(chain, tags):
    out = H / "report" / chain
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# op-adapt H readouts: chain `{chain}`", ""]
    # dev
    rows = []
    for m in ["O"] + tags:
        p = H / "dev" / "O.json" if m == "O" else H / "runs" / m / "dev.json"
        if p.exists():
            for dom, r in json.loads(p.read_text()).items():
                rows += [dict(model=m, domain=dom, metric=k, value=v) for k, v in r.items()]
    if rows:
        dev = pd.DataFrame(rows)
        dev.to_csv(out / "dev.csv", index=False)
        piv = dev.pivot_table(index=["domain", "metric"], columns="model", values="value")[["O"] + [t for t in tags if t in set(dev.model)]]
        md += ["## Dev (pool dev splits)", "", piv.round(3).to_markdown(), ""]
    # probe
    for p in sorted((H / "probe").glob("*/table_vs_*.csv")):
        m, ref = p.parent.name, p.stem.replace("table_vs_", "")
        if m not in tags:
            continue
        t = pd.read_csv(p)
        shutil.copy(p, out / f"probe_{m}_vs_{ref}.csv")
        key = t[t.metric.isin(["G_deg", "single_dx3", "repeat_dx3", "lat3_normal"])]
        key = key.assign(ci=[f"{d:+.2f} [{l:+.2f}, {h:+.2f}]" for d, l, h in zip(key.delta, key.lo, key.hi)])
        md += [f"## Probe (a): {m} vs {ref}", "", key[["domain", "bin", "metric", "n", "ref", "model", "ci"]].round(3).to_markdown(index=False), ""]
    # navtest (paired over logs)
    rows = []
    logs = None
    for m in tags:
        p = LR / "readout" / f"H-{m}" / "navtest.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        if not (j.get("model") and j.get("port_O")):
            continue
        a, o = per_token(j["model"]["csv"]), per_token(j["port_O"]["csv"])
        ix = a.index.intersection(o.index)
        logs = logs or nav_tokens_logs("navtest")
        r = stats.paired(100 * a[ix].to_numpy(), 100 * o[ix].to_numpy(), groups=np.array([logs.get(t, t) for t in ix]))
        rows.append(dict(model=m, port_O=j["port_O"]["score"], score=j["model"]["score"], delta=r["mean"], lo=r["lo"], hi=r["hi"], n=len(ix)))
    if rows:
        nt = pd.DataFrame(rows)
        nt.to_csv(out / "navtest.csv", index=False)
        md += ["## navtest PDMS (b), paired over logs", "", nt.round(3).to_markdown(index=False), ""]
    # capture (e)
    rows = []
    for m in tags:
        p = LR / "readout" / f"H-{m}" / "metrics.csv"
        if p.exists():
            c = pd.read_csv(p)
            c = c[(c.set == "wodval") & (c.xscale == 1.0) & (((c.slice == "start") & (c.metric == "cap_start")) | ((c.slice == "stop") & (c.metric == "cap_stop"))
                                                          | ((c.slice == "turn_onset") & (c.metric == "cap_turn_onset")) | ((c.slice == "stay") & (c.metric == "false_start")))]
            rows.append(c.assign(model=m))
    if rows:
        cap = pd.concat(rows)
        cap.to_csv(out / "capture.csv", index=False)
        md += ["## WOD val capture (e)", "", cap[["model", "slice", "metric", "n", "orig", "adapt", "delta", "lo", "hi"]].round(3).to_markdown(index=False), ""]
    # navhard (c)
    rows = []
    for m in tags:
        p = H / "readout" / m / "navhard.json"
        if p.exists():
            j = json.loads(p.read_text())
            for who in ("port_O", "model"):
                x = j.get(who) or {}
                rows.append(dict(model=m, who=who, **{k: v for k, v in x.items() if k != "csv"}))
    if rows:
        nh = pd.DataFrame(rows)
        nh.to_csv(out / "navhard.csv", index=False)
        md += ["## navhard two-stage EPDMS (c)", "", nh.round(3).to_markdown(index=False), ""]
    # hugsim (d)
    rows = []
    for m in tags:
        p = H / "hugsim" / m / "summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            rows.append({k: v for k, v in s.items() if k not in ("scenarios", "spin10")} | {"model": m})
            shutil.copy(p.parent / "spins.csv", out / f"hugsim_{m}.csv")
    if rows:
        hs = pd.DataFrame(rows)
        hs.to_csv(out / "hugsim.csv", index=False)
        md += ["## HUGSIM spin set (d), no de-rotation rule", "", hs.round(3).to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
