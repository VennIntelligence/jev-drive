#!/usr/bin/env python3
"""ROUND9 INTERP (plans/2026-10-24-interp.md): training-free weight interpolation between NTV0-L18-s0 and R8K-L18-s2 (BCWI-style baseline).

  merge  --a 0.5 --tag INTP50-NR       theta = (1 - a) NTV0 + a R8K over every tensor of ckpt-final.pt model.net / model.parity (adapter, heads,
                                       route-memory branch); non-tensors from R8K + an `interp` record; adapter.pt = merged parity; route_mem.json
                                       = R8K's (head config must equal NTV0's) + `interp`. Refuses differing key sets / shapes / dtypes.
  check  --tags T.. --chk0 X --chk1 Y  serve load (sh30_core.Core) of NTV0, R8K and the given tags; frozen (non-checkpoint) params bit-equal across all;
                                       4 seeded random forwards: chk0 == NTV0 and chk1 == R8K bit for bit, NTV0 reloaded == NTV0, merged points finite
                                       and distinct from both ends -> --out JSON (exit 1 on any failure)
  report --dir O                       PAI 281 ACC + GAP OFF, new scorer primary (O/sv2/new/rollouts.csv for the points, the scorer_v2 report for the
                                       references), old (pai_fix facts) reported: means, paired CIs vs NTV0-s0 / R8K-s2, zero classes, negative flips on
                                       NTV0's full-score scenes, GPU hours -> O/results/interp.{md,json}
Box, repo root: merge / check in envs/op-train, report in .venv.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DD = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUNS = DD / "runs/op_parity/runs"
A0, A1 = "NTV0-L18-s0", "R8K-L18-s2"
SIDE = "route_mem.json"
OK = 0.98


def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as x:
        for b in iter(lambda: x.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cmd_merge(a):
    import torch
    d = RUNS / a.tag
    assert not d.exists(), f"{d} exists"
    x, y = (torch.load(RUNS / t / "ckpt-final.pt", map_location="cpu", weights_only=False) for t in (A0, A1))
    sx, sy = (json.loads((RUNS / t / SIDE).read_text()) for t in (A0, A1))
    for k in ("head", "n_pt", "nf", "offs"):
        assert sx.get(k) == sy.get(k), f"sidecar {k} differs: {sx.get(k)} vs {sy.get(k)}"
    assert x["route_mem"] == y["route_mem"] and x["model"]["arm"] == y["model"]["arm"] and x["model"]["adapter"] is None is y["model"]["adapter"]
    assert set(x["model"]) == set(y["model"]) == {"net", "adapter", "parity", "arm"}, (set(x["model"]), set(y["model"]))
    w = float(a.a)
    out, n = {"net": {}, "parity": {}}, {}
    for part in ("net", "parity"):
        P, Q = x["model"][part], y["model"][part]
        assert set(P) == set(Q), f"{part} keys differ: {sorted(set(P) ^ set(Q))[:10]}"
        for k in P:
            p, q = P[k], Q[k]
            assert p.shape == q.shape and p.dtype == q.dtype == torch.float32, (part, k, p.shape, q.shape, p.dtype, q.dtype)
            out[part][k] = (1.0 - w) * p + w * q
        n[part] = dict(tensors=len(P), numel=sum(v.numel() for v in P.values()), differ=sum(not torch.equal(P[k], Q[k]) for k in P))
    for t in (A0, A1):                                       # adapter.pt is the parity state in both sources
        ad = torch.load(RUNS / t / "adapter.pt", map_location="cpu")
        src = (x if t == A0 else y)["model"]["parity"]
        assert set(ad) == set(src) and all(torch.equal(ad[k], src[k]) for k in ad), f"{t}: adapter.pt != ckpt parity"
    rec = dict(a=w, theta="(1 - a) * A + a * B", A=A0, B=A1, sha_A=sha(RUNS / A0 / "ckpt-final.pt"), sha_B=sha(RUNS / A1 / "ckpt-final.pt"),
               parts=n, nontensors=f"cfg / onpol / route_mem / arm from {A1}", code=os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip())
    ck = {k: v for k, v in y.items() if k != "model"} | {"model": {"net": out["net"], "adapter": None, "parity": out["parity"], "arm": y["model"]["arm"]},
                                                          "interp": rec}
    d.mkdir(parents=True)
    torch.save(ck, d / "ckpt-final.pt")
    torch.save(out["parity"], d / "adapter.pt")
    (d / SIDE).write_text(json.dumps(sy | {"interp": rec}, indent=1))
    if w in (0.0, 1.0):                                      # the end points reproduce their source tensors exactly
        src = x if w == 0.0 else y
        assert all(torch.equal(out[p][k], src["model"][p][k]) for p in out for k in out[p]), f"a={w}: not bit-equal to its source"
    print(json.dumps({"tag": a.tag, **rec}, indent=1))


def cmd_check(a):
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO / "experiments/alpasim/lib"))
    import sh30_core as C
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tags = [A0, A1, a.chk0, a.chk1] + a.tags
    res, bad = dict(dev=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None, tags=tags, checks={}), []

    def chk(name, ok, **kw):
        res["checks"][name] = dict(ok=bool(ok), **kw)
        if not ok:
            bad.append(name)

    rng = np.random.default_rng(0)
    nin = 4
    frames = [rng.integers(0, 256, (2, 4) + tuple(C.FRAME), dtype=np.uint8) for _ in range(nin)]
    outs, frozen, H0 = {}, {}, None
    for i, t in enumerate(tags + [A0]):                      # the last: NTV0 reloaded (determinism control)
        core = C.Core(t, dev)
        assert core.rm, f"{t}: no route-memory branch at serve"
        ck = set(torch.load(RUNS / t / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"]["net"])
        fz = {k: p.detach().cpu() for k, p in core.model.net.params.items() if k not in ck}
        if i == 0:
            frozen, nfz = fz, sum(v.numel() for v in fz.values())
            res["frozen"] = dict(tensors=len(fz), numel=int(nfz), trained_in_ckpt=len(ck))
        else:
            chk(f"frozen equal {t} #{i}", set(fz) == set(frozen) and all(torch.equal(fz[k], frozen[k]) for k in fz))
        o, H = [], []
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for f in frames:
                p, c = (torch.from_numpy(x).to(core.dev) for x in f)
                h = core.model.net.run_batched(core.A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(1, 4, *core.A.H_SHAPE)
                ego = torch.randn(1, 20 + core.rm_dim, generator=g).to(core.dev) * 0.5
                tc = torch.tensor([[1.0, 0.0]], device=core.dev)
                o.append(core.model(h, ego, tc).float().cpu())
                H.append(h.float().cpu())
        if i == 0:
            H0 = H
        else:
            chk(f"encoder equal {t} #{i}", all(torch.equal(u, v) for u, v in zip(H, H0)))
        outs[(t, i)] = torch.cat(o)
        del core
        if dev == "cuda":
            torch.cuda.empty_cache()
    O = {t: outs[(t, i)] for i, t in enumerate(tags)}
    rel = outs[(A0, len(tags))]
    chk("NTV0 reload == NTV0", torch.equal(rel, O[A0]), max_abs=float((rel - O[A0]).abs().max()))
    chk(f"{a.chk0} == {A0}", torch.equal(O[a.chk0], O[A0]), max_abs=float((O[a.chk0] - O[A0]).abs().max()))
    chk(f"{a.chk1} == {A1}", torch.equal(O[a.chk1], O[A1]), max_abs=float((O[a.chk1] - O[A1]).abs().max()))
    chk("NTV0 != R8K", not torch.equal(O[A0], O[A1]), max_abs=float((O[A0] - O[A1]).abs().max()))
    for t in a.tags:
        chk(f"{t} finite, differs from both ends", bool(torch.isfinite(O[t]).all()) and not torch.equal(O[t], O[A0]) and not torch.equal(O[t], O[A1]),
            max_abs_vs_A=float((O[t] - O[A0]).abs().max()), max_abs_vs_B=float((O[t] - O[A1]).abs().max()))
    res["ok"], res["failed"] = not bad, bad
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(1 if bad else 0)


def f3(d):
    return "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"])


def gpu_hours(o):
    from jevdrive.cl import pool as P
    J = P.snapshot()[0]["jobs"]
    tot, rd, n = 0.0, 0.0, 0
    for ln in (o / "jobs.txt").read_text().splitlines():
        j = J.get(ln.split()[0]) if ln.split() else None
        if not j or not j.get("t0") or j["spec"]["vram_gb"] <= 1:
            continue
        h = ((j.get("t1") or time.time()) - j["t0"]) / 3600
        tot += h
        n += 1
        rd += h if j["spec"]["name"].startswith("pai-") else 0.0
    return dict(gpu_job_h=tot, read_h=rd, gpu_jobs=n)


def cmd_report(a):
    import numpy as np
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    pts = a.points.split(",")
    labs = {"NTV0-s0": A0, "R8K-s2": A1} | {p: p for p in pts}
    F_ = DD / "runs/pai_fix/facts"
    OLD = {k: pd.read_csv(F_ / f"{t}-281ag.csv").set_index("scene") for k, t in labs.items()}
    PS = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    NEW = {"NTV0-s0": PS[f"{A0}-ag"], "R8K-s2": PS[f"{A1}-ag"]}
    R = pd.read_csv(o / "sv2/new/rollouts.csv")
    for p in pts:
        y = R[R.read == f"{p}-ag"]
        assert len(y), f"no new-scorer rollouts for {p}"
        NEW[p] = y.drop_duplicates("scene", keep="last").set_index("scene").score
    rpc = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    common = sorted(set.intersection(*[set(F.index[F.score.notna()]) for F in OLD.values()])
                    & set.intersection(*[set(S.dropna().index) for S in NEW.values()]) - rpc)
    n = len(common)
    so = {k: OLD[k].score.reindex(common).to_numpy(float) for k in labs}
    sn = {k: NEW[k].reindex(common).to_numpy(float) for k in labs}
    res = dict(scenes=n, rpc_excluded=len(rpc), reads={}, pairs={}, flips={})
    for k in labs:
        m = PF.metrics(OLD[k].reindex(common))
        res["reads"][k] = dict(new_mean=float(sn[k].mean()), old_mean=float(so[k].mean()), new_zeros=int((sn[k] == 0).sum()),
                               old_zeros=int((so[k] == 0).sum()), zcls={c: int(m.get(f"z_{c}", 0)) for c in PF.ZCLS},
                               pcls={c: int(m.get(f"p_{c}", 0)) for c in PF.PCLS})
    for x in pts + ["R8K-s2"]:
        for y in ("NTV0-s0", "R8K-s2"):
            if x != y:
                res["pairs"][f"{x} - {y}"] = dict(new=stats.paired(sn[x], sn[y]), old=stats.paired(so[x], so[y]),
                                                   zeros_old=int((so[x] == 0).sum() - (so[y] == 0).sum()))
    for sc, S in (("new", sn), ("old", so)):
        full = S["NTV0-s0"] >= OK
        r8_fix = (~full) & (S["R8K-s2"] >= OK)
        res["flips"][sc] = {k: dict(ntv0_full=int(full.sum()), neg=int((full & (S[k] < OK)).sum()), neg_zero=int((full & (S[k] == 0)).sum()),
                                    neg_pts=float((S["NTV0-s0"] - S[k])[full & (S[k] < OK)].sum()),
                                    fixed=int(((~full) & (S[k] >= OK)).sum()), keeps_r8k_fixes=int((r8_fix & (S[k] >= OK)).sum()),
                                    r8k_fixes=int(r8_fix.sum()),
                                    shares_r8k_negs=int((full & (S[k] < OK) & (S["R8K-s2"] < OK)).sum()))
                             for k in labs if k != "NTV0-s0"}
    res["cost"] = gpu_hours(o)
    res["check"] = json.loads((o / "results/check.json").read_text()) if (o / "results/check.json").exists() else None
    K = list(labs)
    L = ["# INTERP: NTV0-L18-s0 <-> R8K-L18-s2 weight interpolation, PAI 281 ACC + GAP, Harmonizer OFF", "",
         f"theta = (1 - a) NTV0-L18-s0 + a R8K-L18-s2 (INTP50-NR: a = 0.5, INTP75-NR: a = 0.75). {n} scenes scored by all four reads with no driver "
         f"RPC error in any ({len(rpc)} RPC scenes excluded). New scorer (scorer_v2 new) primary, old in brackets.", "",
         "## Reads", "", "| read | new mean | old mean | zeros new / old | " + " | ".join(PF.ZCLS) + " |", "|:--|--:|--:|--:|" + "--:|" * len(PF.ZCLS)]
    for k in K:
        r = res["reads"][k]
        L.append(f"| {k} | {r['new_mean']:.3f} | {r['old_mean']:.3f} | {r['new_zeros']} / {r['old_zeros']} | "
                 + " | ".join(str(r["zcls"][c]) for c in PF.ZCLS) + " |")
    L += ["", "Zero classes: old-scorer facts (fclass).", "", "## Paired differences (95 % bootstrap CI, scenes as units)", "",
          "| pair | new scorer | old scorer | zeros (old) |", "|:--|--:|--:|--:|"]
    for k, p in res["pairs"].items():
        L.append(f"| {k} | {f3(p['new'])} | {f3(p['old'])} | {p['zeros_old']:+d} |")
    for sc in ("new", "old"):
        fl = res["flips"][sc]
        L += ["", f"## Flips against NTV0-s0 ({sc} scorer; full = score >= {OK})", "",
              f"NTV0-s0 full-score scenes: {fl['R8K-s2']['ntv0_full']}; R8K-s2 fixes (NTV0 < {OK}, R8K >= {OK}): {fl['R8K-s2']['r8k_fixes']}.", "",
              "| read | negative flips | of them to 0 | points lost there | shared with R8K-s2's | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2 fixes kept |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
        for k, f in fl.items():
            L.append(f"| {k} | {f['neg']} | {f['neg_zero']} | {f['neg_pts']:.2f} | {f['shares_r8k_negs']} | {f['fixed']} | {f['keeps_r8k_fixes']} |")
    c = res["cost"]
    L += ["", f"Cost: {c['gpu_job_h']:.2f} GPU job-hours ({c['read_h']:.2f} in the read chunks, {c['gpu_jobs']} GPU jobs)."]
    if res["check"]:
        L += [f"Load / forward check: {'passed' if res['check']['ok'] else 'FAILED'} ({len(res['check']['checks'])} checks, {res['check']['dev']})."]
    (o / "results").mkdir(exist_ok=True)
    (o / "results/interp.md").write_text("\n".join(L) + "\n")
    (o / "results/interp.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    m = s.add_parser("merge")
    m.add_argument("--a", type=float, required=True)
    m.add_argument("--tag", required=True)
    c = s.add_parser("check")
    c.add_argument("--tags", nargs="+", required=True)
    c.add_argument("--chk0", required=True)
    c.add_argument("--chk1", required=True)
    c.add_argument("--out", required=True)
    r = s.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--points", default="INTP50-NR,INTP75-NR")
    a = ap.parse_args()
    {"merge": cmd_merge, "check": cmd_check, "report": cmd_report}[a.sub](a)


if __name__ == "__main__":
    main()
