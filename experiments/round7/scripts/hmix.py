"""HARM-MIX (plans/2026-10-22-harmonizer-mix.md): Harmonizer-ON twins of the PAI human-state rows (on1_PAIH / PAIH05 / PAIH15) for
renderer-style-invariant training. Box, repo root, op-train env (CPU; no torch).

  select   scenes.tsv (pai_rows) held False scenes of on1_PAIH, sampled per scene class (seeded, fractions FRAC) -> <out>/scenes.txt,
           <out>/chunks/cNNN.txt (c000 = 1 scene, c001 = 10, then 32 a chunk), <out>/offcheck.txt (OFFN scenes of the selection, re-rendered
           OFF for gate C) and <out>/select.json: per scene class and per row stratum (harm.py's rules on the pool rows' log future), the
           rows of each pool and the share the selection covers
  gates    A: every non-token field of every ON pool row equals the OFF pool row of the same name (tab, agent, road, route, route features;
              `served` and the teacher are functions of the tokens and are reported, not judged)
           B: every ON chunk's renderer logs show the Harmonizer enabled and its checkpoint loaded
           C: the OFF re-render pool (offcheck scenes) equals on1_PAIH bit for bit on every shared row (tokens, ego, label, teacher)
           D: ON vs OFF relative L2 / 1 - cos of the newest token pair (and the mean over the 8 slots), per pool and stratum, against
              decision 435 (median relative L2 0.29, p90 0.37)
           -> <out>/gates.json (+ the result tables, stdout)
  twin     the twin store on1_<NAME>: tab.npz (names = the ON pools' row names, which are the OFF names; src pool), build.json, and
           @warp/front.parts.json = the zero-copy concatenation of the three ON pools' token files (jevdrive.tokens.link_front)

  $DATA_DIR/envs/op-train/bin/python experiments/round7/scripts/hmix.py select --out $DATA_DIR/runs/round7/hmix
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R)]
from jevdrive.tokens import link_front, open_front  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, FEAT = D / "runs/op_parity/cache", D / "runs/route_mem/feat"
TSV = R / "experiments/pai_rows/results/scenes.tsv"
POOLS = ("PAIH", "PAIH05", "PAIH15")
STRATA = ("launch", "stop", "turn", "curve", "straight", "other")
FRAC = dict(launch=0.5, rest=0.4, bend=0.3, stop_nolead=0.3, turn_sharp=0.3)   # launches and straight cruise (rest) weigh most (d439)
OFFN = 6


def strata(fut, v0) -> np.ndarray:
    """harm.py's stratum rules on pool rows (already usable: 8 valid slots, inside the log): index into STRATA, 5 = other."""
    lf, v0 = fut.astype(np.float64), v0.astype(np.float64)
    yaw4, d4, lat4 = np.abs(lf[:, 7, 2]), np.hypot(lf[:, 7, 0], lf[:, 7, 1]), np.abs(lf[:, 7, 1])
    vend = np.hypot(*(lf[:, 7, :2] - lf[:, 6, :2]).T) / 0.5
    rules = [(v0 < 0.3) & (d4 > 2.0),
             (v0 > 3) & (vend < 1.0) & (yaw4 < np.radians(15)),
             (yaw4 > np.radians(35)) & (v0 > 1),
             (yaw4 > np.radians(6)) & (yaw4 < np.radians(20)) & (v0 > 5) & (vend > 0.7 * v0),
             (yaw4 < np.radians(1.7)) & (lat4 < 0.5) & (v0 > 5) & (vend > 0.7 * v0)]
    s = np.full(len(v0), 5)
    for i, r in reversed(list(enumerate(rules))):
        s[r] = i
    return s


def pool_strata(t, plain=None) -> np.ndarray:
    """Row strata of a pool tab: plain pools from their own log future; stretch pools (tab `base`) inherit their plain row's stratum."""
    if "base" not in t:
        return strata(t["fut"], t["speed"])
    pl = plain if plain is not None else np.load(CR / "on1_PAIH" / "tab.npz")
    sp = dict(zip(pl["names"].astype(str), strata(pl["fut"], pl["speed"])))
    return np.array([sp[x] for x in t["base"].astype(str)])


def scene_of(names) -> np.ndarray:
    return np.array([x.split(":")[2] for x in names.astype(str)])


def cmd_select(a):
    out = Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tsv = list(csv.DictReader(open(TSV), delimiter="\t"))
    T = {p: np.load(CR / f"on1_{p}" / "tab.npz") for p in POOLS}
    have = set(scene_of(T["PAIH"]["names"]).tolist())
    cls = {r["scene_id"]: r["cls"] for r in tsv if r["held"] == "False" and r["scene_id"] in have}
    pick = []
    for c in sorted(set(cls.values())):
        sc = sorted(s for s in cls if cls[s] == c)
        pick += rng.choice(sc, int(round(FRAC[c] * len(sc))), replace=False).tolist()
    pick = sorted(pick)
    sel = set(pick)
    rec = dict(frac=FRAC, scenes_eligible=len(cls), scenes=len(pick), per_class={}, pools={})
    for c in sorted(set(cls.values())):
        rec["per_class"][c] = dict(eligible=sum(v == c for v in cls.values()), picked=sum(cls[s] == c for s in pick))
    for p in POOLS:
        sc = scene_of(T[p]["names"])
        el = np.isin(sc, list(cls))
        ins = np.isin(sc, pick)
        st = pool_strata(T[p], T["PAIH"])
        rec["pools"][p] = dict(rows=int(len(sc)), rows_eligible=int(el.sum()), rows_picked=int(ins.sum()), share=float(ins.sum() / el.sum()),
                               strata={s: dict(eligible=int((el & (st == i)).sum()), picked=int((ins & (st == i)).sum()),
                                               share=float((ins & (st == i)).sum() / max(1, (el & (st == i)).sum()))) for i, s in enumerate(STRATA)})
    order = rng.permutation(pick).tolist()
    chunks, j = [order[:1], order[1:11]], 11
    while j < len(order):
        chunks.append(order[j:j + 32])
        j += 32
    for i, c in enumerate(chunks):
        (out / "chunks" / f"c{i:03d}.txt").write_text("\n".join(c) + "\n")
    off = sorted(rng.choice(pick, OFFN, replace=False).tolist())
    (out / "offcheck.txt").write_text("\n".join(off) + "\n")
    (out / "scenes.txt").write_text("\n".join(pick) + "\n")
    rec |= dict(chunks=len(chunks), offcheck=off)
    (out / "select.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "pools"}))
    for p, v in rec["pools"].items():
        print(p, v["rows_picked"], "/", v["rows_eligible"], f"{v['share']:.3f}", {s: f"{x['picked']}/{x['eligible']} {x['share']:.2f}" for s, x in v["strata"].items()})


# ---------------------------------------------------------------- gates
def _load(pool):
    t = np.load(CR / f"on1_{pool}" / "tab.npz")
    return {k: t[k] for k in t.files}


def _eq(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind == "f":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def gate_a(off, on) -> dict:
    """Every non-token field of the ON pool's rows against the OFF pool's rows of the same name."""
    A, B = _load(off), _load(on)
    ia = {x: i for i, x in enumerate(A["names"].astype(str))}
    nb = B["names"].astype(str)
    miss = [x for x in nb if x not in ia]
    j = np.array([ia[x] for x in nb if x in ia])
    keep = np.array([x in ia for x in nb])
    res = dict(off=off, on=on, rows_on=int(len(nb)), rows_matched=int(len(j)), missing_in_off=miss[:10], n_missing=len(miss), fields={})
    scenes = np.unique(scene_of(nb))
    # rows of the OFF pool in the ON scenes that the ON pool lacks (the ON build lost or dropped them)
    so = scene_of(A["names"])
    res["off_rows_in_on_scenes"] = int(np.isin(so, scenes).sum())
    tok_dep = {"served"}                                              # the driver's plan: a function of the rendered frames
    for k in sorted(set(A) | set(B)):
        if k not in A or k not in B:
            res["fields"][f"tab.{k}"] = "only in " + ("OFF" if k in A else "ON")
            continue
        if len(B[k]) != len(nb):
            res["fields"][f"tab.{k}"] = dict(equal=_eq(A[k], B[k]), per_pool=True)
            continue
        e = _eq(A[k][j], B[k][keep])
        res["fields"][f"tab.{k}"] = dict(equal=e, token_dependent=k in tok_dep)
    for lf in ("agent", "road", "route"):
        La, Lb = np.load(CR / f"on1_{off}" / f"{lf}.npz"), np.load(CR / f"on1_{on}" / f"{lf}.npz")
        for k in La.files:
            res["fields"][f"{lf}.{k}"] = dict(equal=_eq(La[k][j], Lb[k][keep]))
    Fa, Fb = np.load(FEAT / f"on1_{off}.npz"), np.load(FEAT / f"on1_{on}.npz")
    fa = {x: i for i, x in enumerate(Fa["names"].astype(str))}
    jf = np.array([fa[x] for x in Fb["names"].astype(str)])
    for k in Fa.files:
        res["fields"][f"feat.{k}"] = dict(equal=_eq(Fa[k][jf], Fb[k]))
    ta, tb = (np.load(CR / f"on1_{p}@warp" / "teacher.npz") for p in (off, on))
    d = np.abs(ta["out"][j] - tb["out"][keep])
    res["fields"]["teacher.out"] = dict(equal=bool((d == 0).all()), token_dependent=True, max_abs=float(d.max()), med_abs=float(np.median(d)))
    for k in ("di", "pi"):
        res["fields"][f"teacher.{k}"] = dict(equal=_eq(ta[k], tb[k]))
    judged = {k: v for k, v in res["fields"].items() if isinstance(v, dict) and not v.get("token_dependent")}
    res["pass"] = bool(not miss and len(j) == len(nb) and res["off_rows_in_on_scenes"] == len(nb) and all(v["equal"] for v in judged.values())
                       and not any(isinstance(v, str) for v in res["fields"].values()))
    res["unequal"] = [k for k, v in judged.items() if not v["equal"]]
    return res, j, keep


def tok_dist(fa, fb, ia, ib, bs=2048):
    """Per row: newest-pair (slot 7) relative L2 and 1 - cos, and the relative L2 averaged over the 8 slots."""
    rl7, cs7, rla, same = [], [], [], 0
    o = np.argsort(ia)
    for s in range(0, len(o), bs):
        q = o[s:s + bs]
        A, B = fa[ia[q]].astype(np.float32), fb[ib[q]].astype(np.float32)
        A, B = A.reshape(len(q), 8, -1), B.reshape(len(q), 8, -1)
        na, nb_ = np.linalg.norm(A, axis=-1), np.linalg.norm(B, axis=-1)
        r = np.linalg.norm(A - B, axis=-1) / (na + 1e-12)
        rl7.append(r[:, 7]), rla.append(r.mean(1)), cs7.append(1 - (A[:, 7] * B[:, 7]).sum(-1) / (na[:, 7] * nb_[:, 7] + 1e-12))
        same += int((A == B).all((1, 2)).sum())
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    return np.concatenate(rl7)[inv], np.concatenate(cs7)[inv], np.concatenate(rla)[inv], same


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else float("nan")


def cmd_gates(a):
    out = Path(a.out)
    res = dict(A={}, B={}, C={}, D={})
    # ---- A + D per pool pair
    for p in POOLS:
        on = f"{p.replace('PAIH', a.on, 1)}"
        rA, j, keep = gate_a(p, on)
        res["A"][p] = rA
        fa, fb = open_front(CR / f"on1_{p}@warp" / "front.npy"), open_front(CR / f"on1_{on}@warp" / "front.npy")
        ib = np.flatnonzero(keep)
        rl7, cs7, rla, same = tok_dist(fa, fb, j, ib)
        B_ = _load(on)
        st = pool_strata(B_, _load(on.replace("05", "").replace("15", "")) if "base" in B_ else None)[ib]
        dd = dict(rows=int(len(ib)), rows_identical=same)
        for g, m in [("all", np.ones(len(ib), bool))] + [(s, st == i) for i, s in enumerate(STRATA)]:
            dd[g] = dict(n=int(m.sum()), rl2_7_med=q(rl7[m], 0.5), rl2_7_p10=q(rl7[m], 0.1), rl2_7_p90=q(rl7[m], 0.9), cos_7_med=q(cs7[m], 0.5),
                         rl2_8_med=q(rla[m], 0.5))
        dd["pass"] = bool(same == 0 and 0.22 <= dd["all"]["rl2_7_med"] <= 0.36 and dd["all"]["rl2_7_p90"] <= 0.5 and dd["all"]["rl2_7_p10"] >= 0.1)
        res["D"][p] = dd
    # ---- B: renderer logs of every ON chunk
    for c in sorted(glob.glob(str(out / "roll/ON/*/runs/*"))):
        logs = sorted(glob.glob(f"{c}/sim/native-logs/renderer-*.log"))
        txt = "".join(open(f, errors="replace").read() for f in logs)
        res["B"][Path(c).name] = dict(logs=len(logs), enabled='"enable_harmonizer": true' in txt,
                                      loaded="checkpoint ready at" in txt and "harmonizer_nontemporal.pt" in txt)
    res["B_pass"] = bool(res["B"]) and all(v["logs"] > 0 and v["enabled"] and v["loaded"] for v in res["B"].values())
    for c in sorted(glob.glob(str(out / "roll/OFF/*/runs/*"))):                # the OFF re-render must NOT have it
        txt = "".join(open(f, errors="replace").read() for f in glob.glob(f"{c}/sim/native-logs/renderer-*.log"))
        res["B"]["OFF:" + Path(c).name] = dict(enabled='"enable_harmonizer": true' in txt)
    # ---- C: the OFF re-render against the stored OFF pool, bit for bit
    if (CR / f"on1_{a.offc}" / "tab.npz").exists():
        rC, j, keep = gate_a("PAIH", a.offc)
        fa, fb = open_front(CR / "on1_PAIH@warp" / "front.npy"), open_front(CR / f"on1_{a.offc}@warp" / "front.npy")
        ib = np.flatnonzero(keep)
        o = np.argsort(j)
        teq = sum(int((fa[j[o[s:s + 1024]]] == fb[ib[o[s:s + 1024]]]).all((1, 2, 3)).sum()) for s in range(0, len(o), 1024))
        rC |= dict(tokens_equal_rows=int(teq), scenes=sorted(set(scene_of(_load(a.offc)["names"]).tolist())))
        rC["pass"] = bool(rC["pass"] and teq == len(ib) and rC["fields"]["teacher.out"]["equal"] and rC["fields"]["tab.served"]["equal"])
        res["C"] = rC
    res["pass"] = dict(A=all(v["pass"] for v in res["A"].values()), B=res["B_pass"], C=bool(res["C"].get("pass")), D=all(v["pass"] for v in res["D"].values()))
    (out / "gates.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["pass"]))
    for p in POOLS:
        print(p, "A", res["A"][p]["pass"], res["A"][p]["rows_matched"], "/", res["A"][p]["rows_on"], "unequal", res["A"][p]["unequal"],
              "| D", {g: (v["n"], round(v["rl2_7_med"], 3), round(v["rl2_7_p90"], 3)) for g, v in res["D"][p].items() if isinstance(v, dict)})


def cmd_twin(a):
    """on1_<name>: names + src of the ON pools' rows, front = zero-copy concatenation of their token files (no tokens copied)."""
    g = json.loads((Path(a.out) / "gates.json").read_text())
    assert all(g["pass"].values()), f"gates not all passed: {g['pass']}"
    dst, warp = CR / f"on1_{a.name}", CR / f"on1_{a.name}@warp"
    assert not dst.exists() and not warp.exists(), f"{dst} exists: never overwritten"
    ons = [p.replace("PAIH", a.on, 1) for p in POOLS]
    names, src = [], []
    for p, on in zip(POOLS, ons):
        n = np.load(CR / f"on1_{on}" / "tab.npz")["names"].astype(str)
        names.append(n), src.append(np.full(len(n), p))
    names, src = np.concatenate(names), np.concatenate(src)
    assert len(set(names.tolist())) == len(names), "twin names not unique"
    dst.mkdir(), warp.mkdir()
    J = link_front(warp, [CR / f"on1_{on}@warp" / "front.npy" for on in ons])
    np.savez(dst / "tab.npz", names=names, src=src)
    rec = dict(name=a.name, rows=int(len(names)), per_pool={p: int((src == p).sum()) for p in POOLS}, on_pools=ons, front_shape=J["shape"],
               harmonizer="ON (HARMONIZER=1, ALPASIM_FAST=1, pai_rows H recipe)", gates=str(Path(a.out) / "gates.json"))
    (dst / "build.json").write_text(json.dumps(rec, indent=1) + "\n")
    f = open_front(warp / "front.npy")
    assert len(f) == len(names)
    print(json.dumps(rec))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--out", required=True)
    p = sp.add_parser("gates")
    p.add_argument("--out", required=True)
    p.add_argument("--on", default="PAIHON", help="ON pool prefix: PAIH -> <on>, PAIH05 -> <on>05, PAIH15 -> <on>15")
    p.add_argument("--offc", default="PAIHOFFC", help="the OFF re-render pool (gate C)")
    p = sp.add_parser("twin")
    p.add_argument("--out", required=True)
    p.add_argument("--on", default="PAIHON")
    p.add_argument("--name", default="PAIHTW")
    a = ap.parse_args()
    dict(select=cmd_select, gates=cmd_gates, twin=cmd_twin)[a.cmd](a)
