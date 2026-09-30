"""op-adapt L follow-up checks (todos/2026-10-01-op-adapt-L-followup.md). Existing checkpoints only, no training.

  perm        intent permutation: re-run the WOD val policy pass of main s0-s2 and noint s0 with the intent input as given /
              shuffled within slice / shuffled globally / forced straight / flipped / none -> followup/perm_*.csv   (op-train, 1 GPU)
  stoprows    the WOD val stop-slice rows, human-future corridor and the image list for the detector                 (op-train, CPU)
  stoplabel   lift the YOLO detections onto the ground and label each stop row lead / vru / other                    (op-train, CPU)
  stopread    per cause: event counts and paired capture of O vs main (3 seeds)                                     (op-train, CPU)
  figure      one figure of both checks                                                                              (op-train, CPU)

  CUDA_VISIBLE_DEVICES=0 taskset -c 8-74 python scripts/op_adapt_l_followup.py perm
Detection itself (between stoprows and stoplabel):
  python -m jevdrive.fastperc detect --backend yolo:yolo26x-seg.pt:1280:half --keep 0.05 --full --list <list> --out <dir> (envs/ultralytics)
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import op_adapt_l as L  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
B = 2000
OUT = REPO / "research" / "results" / "op-adapt-L" / "followup"
MODELS = {"main-s0": "main-s0", "main-s1": "main-s1", "main-s2": "main-s2", "noint-s0": "noint-s0"}
CONDS = ["a_given", "b1_shuffle_slice", "b2_shuffle_global", "c_straight", "d_flip", "e_none"]
STRATA = ["turn_onset", "start", "stop", "stay", "control", "straight_int"]     # priority order; the rest = other
STOP_CAUSE_CAM_KEY = "1"                                                          # FRONT in op_calib.json


def fdir(*p) -> Path:
    return L.lroot("followup", *p)


def wod_rows(D):
    rows = D.rows("wodval", "val", need_future=True)
    return rows, D.tab["wodval"]


# ================================================================ 1. intent permutation
def strata_of(tab, rows) -> np.ndarray:
    lab = np.full(len(rows), len(STRATA), np.int64)
    for i in range(len(STRATA) - 1, -1, -1):
        lab[np.asarray(tab[f"s_{STRATA[i]}"])[rows]] = i
    return lab


def make_intents(tab, rows) -> dict:
    it = np.asarray(tab["intent"])[rows].astype(np.int64)
    lab = strata_of(tab, rows)
    b1 = it.copy()
    rng = np.random.default_rng(20261001)
    for s in range(len(STRATA) + 1):
        ix = np.flatnonzero(lab == s)
        if len(ix) > 1:
            b1[ix] = it[ix][rng.permutation(len(ix))]
    b2 = it[np.random.default_rng(20261002).permutation(len(it))]
    flip = it.copy()
    flip[it == 2], flip[it == 3] = 3, 2
    return {"a_given": it, "b1_shuffle_slice": b1, "b2_shuffle_global": b2, "c_straight": np.ones_like(it), "d_flip": flip,
            "e_none": np.zeros_like(it)}


def policy_pass(model, D, rows, dev, intents: dict) -> dict:
    """One stage-4 pass per unique context frame, then the policy once per intent condition (L.fwd_rows with the hidden reused)."""
    import torch
    from jevdrive import op_adapt as A
    d = D.dom["wodval"]
    pi = A.plan_index(model.net.slices)
    valid = d.ctx[rows] >= 0
    uniq = np.unique(d.ctx[rows][valid])
    H = L.hidden_of(model, D, "wodval", uniq, dev, threads=8)
    idx = np.where(valid, np.searchsorted(uniq, np.maximum(d.ctx[rows], 0)), 0)
    out = {}
    with torch.no_grad():
        for c, it in intents.items():
            acc = []
            for i in range(0, len(rows), 192):
                s = slice(i, i + 192)
                o = model.policy(H[torch.from_numpy(idx[s]).to(dev)], torch.from_numpy(valid[s]).to(dev),
                                 torch.from_numpy(d.tc[rows[s]]).to(dev), torch.from_numpy(it[s]).to(dev))
                acc.append(o["outputs"].float()[:, pi].view(-1, 33, 15).cpu().numpy())
            out[c] = np.concatenate(acc)
    return out


def cmd_perm(a):
    import torch
    dev = torch.device("cuda")
    D = L.Data(("wod", "wodval", "nus"), hstore=True)
    rows, tab = wod_rows(D)
    zo = np.load(L.lroot("readout", "O", "eval", "wodval.npz"))
    assert np.array_equal(zo["rows"], rows)
    intents = make_intents(tab, rows)
    lab = strata_of(tab, rows)
    it = intents["a_given"]
    meta = {"n_rows": int(len(rows)), "strata": {(STRATA + ["other"])[s]: int((lab == s).sum()) for s in range(len(STRATA) + 1)},
            "intent_counts": {c: np.bincount(v, minlength=4).tolist() for c, v in intents.items()},
            "turn_onset_intent_by_cond": {c: np.bincount(v[lab == 0], minlength=4).tolist() for c, v in intents.items()},
            "changed_frac": {c: float((v != it).mean()) for c, v in intents.items()}}
    (fdir() / "perm_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    for name, m in MODELS.items():
        t0 = time.time()
        if (fdir() / f"plans_{name}.npz").exists():
            continue
        model = L.load_model(L.lroot("runs", m) / "ckpt-final.pt", dev)
        pl = policy_pass(model, D, rows, dev, intents)
        np.savez(fdir() / f"plans_{name}.npz", rows=rows, **pl)
        del model
        torch.cuda.empty_cache()
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    perm_read(D, rows, tab, zo)


def indicators(plan, tab, rows, cam) -> dict:
    """Per-row indicator arrays (n,) of one plan set."""
    m = L.row_metrics(plan, tab, rows, cam)
    p8 = L.rear_np(plan, cam)
    f8 = np.asarray(tab["fut"])[rows].astype(np.float32)
    m["sign"] = (np.sign(p8[:, 7, 1]) == np.sign(f8[:, 7, 1]))
    m["fut_y"] = f8[:, 7, 1]
    return m


READ = [("turn_onset", "cap_turn_onset"), ("turn_onset", "sign"), ("start", "cap_start"), ("stop", "cap_stop"),
        ("control", "false_turn"), ("straight_int", "false_turn"), ("stay", "false_start")]


def perm_read(D, rows, tab, zo):
    cam = D.cam["wodval"]
    groups = np.asarray(tab["seq"])[rows]
    flags = {s: np.asarray(tab[f"s_{s}"])[rows] for s in STRATA}
    io = indicators(zo["plan"], tab, rows, cam)
    ind = {}                                           # (model-or-group, cond) -> indicator dict
    for name in MODELS:
        z = np.load(fdir() / f"plans_{name}.npz")
        for c in CONDS:
            ind[(name, c)] = indicators(z[c], tab, rows, cam)
    seeds = [f"main-s{i}" for i in range(3)]
    for c in CONDS:                                    # main3: per-row mean of the three seeds' indicators
        ind[("main3", c)] = {k: np.mean([ind[(s, c)][k].astype(float) for s in seeds], 0) for k in ind[("main-s0", c)]}

    def pd_(xa, xo, m):
        return L.paired_delta(xa[m], xo[m], groups[m], B=B)
    out, left_right = [], []
    for (name, c), ix in ind.items():
        for s, k in READ:
            m = flags[s]
            r = pd_(ix[k], io[k], m)
            out.append({"model": name, "cond": c, "slice": s, "metric": k, **r})
        for side, sm in (("left", ind[(name, c)]["fut_y"] > 0), ("right", ind[(name, c)]["fut_y"] < 0)):
            m = flags["turn_onset"] & sm
            out.append({"model": name, "cond": c, "slice": f"turn_onset_{side}", "metric": "sign", **pd_(ix["sign"], io["sign"], m)})
    pd.DataFrame(out).to_csv(OUT / "perm_vs_orig.csv", index=False)
    # contrasts
    rows2 = []
    for name in ("main3", "main-s0", "main-s1", "main-s2"):
        for c in CONDS[1:]:
            for s, k in READ:
                m = flags[s]
                rows2.append({"model": name, "cond": c, "vs": f"{name}/a_given", "slice": s, "metric": k,
                              **pd_(ind[(name, c)][k], ind[(name, "a_given")][k], m)})
        for c in CONDS:                                # against noint (a): G_c - G_noint
            for s, k in READ:
                m = flags[s]
                rows2.append({"model": name, "cond": c, "vs": "noint-s0/a_given", "slice": s, "metric": k,
                              **pd_(ind[(name, c)][k], ind[("noint-s0", "a_given")][k], m)})
    pd.DataFrame(rows2).to_csv(OUT / "perm_contrasts.csv", index=False)
    # self-checks
    chk = {"noint_invariant": bool(all(np.array_equal(np.load(fdir() / "plans_noint-s0.npz")[c], np.load(fdir() / "plans_noint-s0.npz")["a_given"])
                                       for c in CONDS))}
    for s in (0, 1, 2):
        z = np.load(fdir() / f"plans_main-s{s}.npz")
        za = np.load(L.lroot("readout", f"main-s{s}", "eval", "wodval.npz"))
        chk[f"main-s{s}_a_vs_stored_max_abs"] = float(np.abs(z["a_given"] - za["plan"]).max())
        unk = np.asarray(tab["intent"])[rows] == 0
        chk[f"main-s{s}_unknown_intent_frames_a_vs_e_max_abs"] = float(np.abs(z["a_given"] - z["e_none"])[unk].max()) if unk.any() else None
    (OUT / "perm_selfcheck.json").write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk, indent=1))
    t = pd.DataFrame(out)
    print(t[(t.model == "main3") & t.slice.isin(["turn_onset", "control", "straight_int"])]
          [["cond", "slice", "metric", "n", "adapt", "orig", "delta", "lo", "hi"]].round(3).to_string(index=False))


# ================================================================ 2. stop split by cause
def cmd_stoprows(a):
    from jevdrive import waymo as W
    D = L.Data(("wod", "wodval", "nus"))
    rows = D.rows("wodval", "val", "stop", need_future=True)
    tab = D.tab["wodval"]
    names = np.asarray(tab["name"])[rows]
    df = W.load_index()
    key = pd.Series(np.arange(len(df)), index=W.frame_names(df))
    r = key.reindex(names).to_numpy()
    assert not np.isnan(r).any()
    r = r.astype(int)
    x = df.iloc[r]
    lst = pd.DataFrame({"key": names, "path": "", "shard": [str(W.shard_dir() / s) for s in x.shard], "off": x.front_off.to_numpy(),
                        "len": x.front_len.to_numpy(), "sequence": x.sequence.to_numpy(), "rater": False})
    lst.to_parquet(fdir() / "stop_list.parquet", index=False)
    np.savez(fdir() / "stoprows.npz", rows=rows, names=names)
    print(len(rows), "stop rows,", lst.sequence.nunique(), "segments")


def corridor(fut8: np.ndarray, ext=8.0, step=0.25) -> np.ndarray:
    """Dense polyline (m, 2): origin -> fut[0..7] -> fut[7] + ext * unit(fut[7]) (x fwd if the 4 s displacement < 1 m)."""
    end = fut8[7]
    n = np.linalg.norm(end)
    u = end / n if n >= 1.0 else np.array([1.0, 0.0])
    pts = np.vstack([[0.0, 0.0], fut8, end + ext * u])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


def cmd_stoplabel(a):
    from jevdrive import fusion_q4 as Q
    from jevdrive.fastperc import COCO_MAP  # noqa: F401  (class mapping used by the detector backend)
    from jevdrive import wod_zeroshot as Z
    D = L.Data(("wod", "wodval", "nus"))
    z = np.load(fdir() / "stoprows.npz", allow_pickle=True)
    rows, names = z["rows"], z["names"].astype(str)
    tab = D.tab["wodval"]
    fut = np.asarray(tab["fut"])[rows].astype(np.float64)
    v0 = np.asarray(tab["v0"])[rows].astype(float)
    S = np.linalg.norm(fut[:, 7], axis=1)
    Dc = np.maximum(15.0, S + 8.0)
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    d = Q.load_dets(a.dets, score=0.5)
    d = d[d.key.isin(set(names))].reset_index(drop=True)
    seq = d.key.str.rsplit("-", n=1).str[0].to_numpy()
    cal = {s: calib[s][STOP_CAUSE_CAM_KEY] for s in set(seq)}
    d = Q.lift_dets(d, seq, cal)
    d = d[d.lift_ok].reset_index(drop=True)
    pos = pd.Series(np.arange(len(names)), index=names)
    ri = pos.reindex(d.key).to_numpy().astype(int)
    path = [corridor(f) for f in fut]
    dist = np.array([np.sqrt(((path[i] - np.array([gx, gy])) ** 2).sum(1)).min() for i, gx, gy in zip(ri, d.gx, d.gy)])
    d["d"], d["row"] = dist, ri
    ahead = (d.gx.to_numpy() > 0) & (d.gx.to_numpy() <= Dc[ri])
    veh = ahead & (d.prompt.to_numpy() == "vehicle") & (dist <= 1.75)
    vru = ahead & d.prompt.isin(["pedestrian", "cyclist"]).to_numpy() & (dist <= 3.0)
    lab = pd.DataFrame({"row_i": np.arange(len(names)), "name": names, "seq": np.asarray(tab["seq"])[rows], "v0": v0, "S": S, "Dc": Dc})
    lab["yolo_vehicle"] = np.bincount(ri[veh], minlength=len(names)) > 0
    lab["yolo_vru"] = np.bincount(ri[vru], minlength=len(names)) > 0
    lab["n_det"] = np.bincount(ri, minlength=len(names))
    # O's lead head
    zo = np.load(L.lroot("readout", "O", "eval", "wodval.npz"))
    sel = np.searchsorted(zo["rows"], rows)
    assert np.array_equal(zo["rows"][sel], rows)
    from jevdrive.nq4_k import lead_decode
    lx, lv, lp = lead_decode(zo["lead"][sel], zo["lead_prob"][sel])
    lab["head_lead"] = (lp > 0.5) & (lx > 0) & (lx <= Dc)
    lab["head_p"], lab["head_x"] = lp, lx
    lab["cause"] = np.where(lab.yolo_vehicle, "lead", np.where(lab.yolo_vru, "vru", "other"))
    lab["cause_strict"] = np.where(lab.cause == "other", np.where(lab.head_lead, "other_head_lead", "other_strict"), lab.cause)
    lab["cause_head"] = np.where(lab.head_lead, "lead", np.where(lab.yolo_vru, "vru", "other"))
    lab.to_csv(OUT / "stop_causes.csv", index=False)
    print(lab.cause.value_counts().to_string(), "\n", lab.cause_strict.value_counts().to_string(), "\n both:",
          int((lab.yolo_vehicle & lab.yolo_vru).sum()), " agree yolo-vehicle vs head:", float((lab.yolo_vehicle == lab.head_lead).mean()))


def cmd_stopread(a):
    D = L.Data(("wod", "wodval", "nus"))
    lab = pd.read_csv(OUT / "stop_causes.csv")
    z = np.load(fdir() / "stoprows.npz", allow_pickle=True)
    rows = z["rows"]
    tab = D.tab["wodval"]
    cam = D.cam["wodval"]

    def cap(path):
        zz = np.load(path)
        r = np.searchsorted(zz["rows"], rows)
        assert np.array_equal(zz["rows"][r], rows)
        return L.cap_stop(L.rear_np(zz["plan"][r], cam)).astype(float)
    xo = cap(L.lroot("readout", "O", "eval", "wodval.npz"))
    xs = [cap(L.lroot("readout", f"main-s{s}", "eval", "wodval.npz")) for s in range(3)]
    xm = np.mean(xs, 0)
    groups = lab.seq.to_numpy()
    sets = {"all": np.ones(len(lab), bool)}
    for col, vals in (("cause", ["lead", "vru", "other"]), ("cause_strict", ["lead", "vru", "other_strict", "other_head_lead"]),
                      ("cause_head", ["lead", "vru", "other"])):
        for v in vals:
            sets[f"{col}:{v}"] = (lab[col] == v).to_numpy()
    sets["both_vehicle_and_vru"] = (lab.yolo_vehicle & lab.yolo_vru).to_numpy()
    out = []
    for k, m in sets.items():
        if not m.any():
            continue
        r = L.paired_delta(xm[m], xo[m], groups[m], B=B)
        seeds = [float((x[m] - xo[m]).mean()) for x in xs]
        out.append({"set": k, **r, "n_segments": int(len(np.unique(groups[m]))), "delta_s0": seeds[0], "delta_s1": seeds[1], "delta_s2": seeds[2],
                    "v0_median": float(lab.v0[m].median()), "S_median": float(lab.S[m].median())})
    t = pd.DataFrame(out)
    t.to_csv(OUT / "stop_by_cause.csv", index=False)
    print(t.round(3).to_string(index=False))


# ================================================================ figure
def cmd_figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    p = pd.read_csv(OUT / "perm_vs_orig.csv")
    s = pd.read_csv(OUT / "stop_by_cause.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    cols = {"main3": "#1f4e9c", "noint-s0": "#9a9a9a"}
    for a_, met, ttl in ((ax[0], "cap_turn_onset", "Turn-onset capture gain vs original"), (ax[1], "sign", "Turn-onset sign agreement vs original")):
        for j, m in enumerate(("main3", "noint-s0")):
            t = p[(p.model == m) & (p.slice == "turn_onset") & (p.metric == met)].set_index("cond").reindex(CONDS)
            x = np.arange(len(CONDS)) + (j - 0.5) * 0.3
            a_.errorbar(x, t.delta, yerr=[t.delta - t.lo, t.hi - t.delta], fmt="o", color=cols[m], ms=4, capsize=2, label="main (3 seeds)" if m == "main3" else "noint")
        a_.axhline(0, color="k", lw=0.6)
        a_.set_xticks(range(len(CONDS)))
        a_.set_xticklabels([c.split("_", 1)[0] for c in CONDS])
        a_.set_title(ttl)
        a_.set_xlabel("intent condition (a given, b1/b2 shuffled, c straight, d flipped, e none)")
    ax[0].set_ylabel("paired difference (95% CI)")
    ax[0].legend(frameon=False, loc="best")
    ss = s[s.set.isin(["all", "cause_strict:lead", "cause_strict:vru", "cause_strict:other_strict", "cause_strict:other_head_lead"])]
    y = np.arange(len(ss))[::-1]
    ax[2].errorbar(ss.delta, y, xerr=[ss.delta - ss.lo, ss.hi - ss.delta], fmt="o", color="#1f4e9c", ms=4, capsize=2)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([f"{r.set.split(':')[-1]} (n={int(r.n)}, {int(r.n_segments)} seg)" for r in ss.itertuples()])
    ax[2].axvline(0, color="k", lw=0.6)
    ax[2].set_xlabel("stop capture gain, main - original")
    ax[2].set_title("WOD stop gain by cause")
    fig.tight_layout()
    f = REPO / "research" / "figs" / "op-adapt-L-followup.png"
    fig.savefig(f, dpi=200)
    print(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["perm", "stoprows", "stoplabel", "stopread", "figure"])
    ap.add_argument("--dets", default=str(data_dir() / "runs" / "op_adapt_L" / "followup" / "dets"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"perm": cmd_perm, "stoprows": cmd_stoprows, "stoplabel": cmd_stoplabel, "stopread": cmd_stopread, "figure": cmd_figure}[a.cmd](a)
