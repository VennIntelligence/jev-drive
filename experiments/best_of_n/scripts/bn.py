#!/usr/bin/env python3
"""BEST-OF-N (plans/2026-10-19-bn-prereg.md): four own checkpoints roll the same practice-start windows natively; per window the best-scoring
rollout's own states with its own served plan as the imitation target (+ S hinges) are mixed into a short JTN-L18-s1 continuation. Box, repo root.

  select   AlpaSim venv. Stratified 1 200 windows of alpasim_nuplan/practice-starts (QUOTA, seed 0) -> split alpasim_nuplan/best-of-n;
           $W/select/{windows.csv, meta.json, lists/c<j>.txt (<= CHUNK windows inside one shard cache), lists/smoke.txt (10 windows, one shard)}
  stage2   the staged-launch checklist on finished roll chunks (= practice_nuplan pn.py stage2, amendment 2's reading) -> exit 0 / 2
  pick     --label L --chunks c.. [--gap G]: per window the best of the four rollouts; windows with best - worst >= G get rows: per shard a stage dir
           $W/stage/<L><k>/manifest.json over view run dirs that hold only the winners' rollouts (symlinks + a filtered results-summary.json),
           $W/pick/<L>.json and results/pick_<L>.{json,md} (win shares by stratum, per-model means)
  target   --pools P..: tab `fut` := `served` (the winner's own served plan, row frame); the human target kept in <pool>/fut_human.npz (idempotent)
  weights  --pool on1_BNX --base JT00 --parts P.. --out BNX: JT00's weights x (K_ON - K_BN) / K_ON on the on1_JT prefix, K_BN / K_ON uniform over
           the BN rows that pass their pool gates and are not on JT00's held logs -> $DATA_DIR/runs/defect_fix/w/<out>.npz, results/w_<out>.json
  gate     --arm BNX2K-L18-s1 --control BNC2K-L18-s1 --base JTN-L18-s1: arena held native paired + held 460 -> results/gate.{json,md}
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import zlib
from collections import Counter
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/best_of_n"
PN = DATA / "runs/practice_nuplan/select"
CR = DATA / "runs/op_parity/cache"
ROWS = DATA / "runs/onpol_nuplan/rows"
RES = REPO / "experiments/best_of_n/results"
TAGS = ("JTN-L18-s1", "NTV0-L18-s0", "RSG-L18-s0", "NR1FB10-L18-s0")          # tie order: the student first (prereg 4)
SHORT = dict(zip(TAGS, ("jtn1", "ntv0", "rsg", "fb10")))
QUOTA = dict(static=400, launch=300, braking=200, turn=120, other=120, cruise=60)
SMOKE = dict(static=3, launch=3, braking=2, turn=1, other=1)
STRATA = ("static", "launch", "braking", "turn", "cruise", "other")
CHUNK, GAP, K_ON, K_BN = 110, 0.2, 70, 8
ZERO_TYPES = ("collision_at_fault", "left_corridor_laterally", "offroad")


def log_of(s):
    return "_".join(s.split("_")[:2])


def windows():
    return {r["id"]: r for r in csv.DictReader(open(W / "select/windows.csv"))}


def cmd_select(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("best_of_n", "select", config=vars(a)) as run:
        ps, held = splits.load("alpasim_nuplan/practice-starts"), splits.load("alpasim_nuplan/onpol-held")
        run.use_split(ps), run.use_split(held)
        mem = set(ps.members)
        R = [r for r in csv.DictReader(open(PN / "windows.csv")) if r["keep"] == "True" and r["id"] in mem]
        assert len(R) == len(mem), (len(R), len(mem))
        rng = np.random.default_rng(0)
        K = []
        for st in STRATA:
            c = sorted((r for r in R if r["stratum"] == st), key=lambda r: r["id"])
            K += [c[i] for i in sorted(rng.choice(len(c), min(QUOTA[st], len(c)), replace=False))]
        K.sort(key=lambda r: (int(r["shard"]), r["log"], int(r["start_idx"])))
        hl = {log_of(s) for s in held.members}
        assert not any(log_of(r["id"]) in hl for r in K)
        bn = splits.define("alpasim_nuplan", "best-of-n", [r["id"] for r in K], unit="scene",
                           origin="experiments/best_of_n/scripts/bn.py select: stratified sample of alpasim_nuplan/practice-starts "
                                  f"(quota {QUOTA}, seed 0, uniform without replacement inside a stratum)",
                           used_by=["experiments/best_of_n"], status="frozen",
                           notes="best-of-N self-imitation windows (plans/2026-10-19-bn-prereg.md); a subset of practice-starts, never a held log")
        splits.check_disjoint(bn, held)
        run.use_split(bn)
        S = W / "select"
        (S / "lists").mkdir(parents=True, exist_ok=True)
        chunks, j = [], 0
        for k in sorted({int(r["shard"]) for r in K}):
            g = [r for r in K if int(r["shard"]) == k]
            m = -(-len(g) // CHUNK)
            for q in range(m):
                part = g[q * len(g) // m:(q + 1) * len(g) // m]
                (S / "lists" / f"c{j}.txt").write_text("".join(r["id"] + "\n" for r in part))
                for r in part:
                    r["chunk"] = j
                chunks.append(dict(chunk=j, shard=k, windows=len(part)))
                j += 1
        ksm = max({int(r["shard"]) for r in K}, key=lambda k: sum(r["stratum"] == "launch" for r in K if int(r["shard"]) == k))
        sm = []
        for st, n in SMOKE.items():
            c = [r["id"] for r in K if int(r["shard"]) == ksm and r["stratum"] == st]
            sm += [c[i] for i in sorted(rng.choice(len(c), min(n, len(c)), replace=False))]
        (S / "lists/smoke.txt").write_text("".join(x + "\n" for x in sm))
        with open(S / "windows.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, ["id", "log", "shard", "chunk", "start_idx", "stratum"])
            w.writeheader()
            w.writerows([{q: r[q] for q in w.fieldnames} for r in K])
        meta = dict(candidates=len(R), kept=len(K), strata=dict(Counter(r["stratum"] for r in K)), strata_candidates=dict(Counter(r["stratum"] for r in R)),
                    logs=len({log_of(r["id"]) for r in K}), chunks=chunks, smoke=sm, smoke_shard=ksm, split=bn.id)
        (S / "meta.json").write_text(json.dumps(meta, indent=1))
        run.summary.update({k: v for k, v in meta.items() if k not in ("chunks", "smoke")})
        print(json.dumps({k: v for k, v in meta.items() if k not in ("chunks", "smoke")}, indent=1))


def summary(d):
    f = Path(d) / "aggregate/results-summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def cmd_stage2(a):
    sys.path[:0] = [str(REPO / "experiments/practice_nuplan/scripts")]
    import pn
    rows = pn.roll_stats(a.runs)
    ok = [r for r in rows if r["done"]]
    n, fail = sum(r["rollouts"] for r in ok), sum(r["failed"] for r in ok)
    per_win = sum(r["dumps"] for r in ok) / max(n - fail, 1)
    mean = float(np.mean([r["mean"] for r in ok])) if ok else float("nan")
    need_gb = sum(r["dump_mb"] for r in ok) / max(n, 1) * a.total_rollouts / 1024
    st = os.statvfs(DATA)
    free_gb = st.f_bavail * st.f_frsize / 2 ** 30
    chk = dict(all_done=len(ok) == len(rows), fail_rate=fail / max(n, 1), dumps_per_window=per_win, mean_score=mean, dump_gb_projected_all=need_gb,
               disk_free_gb=free_gb)
    chk["pass"] = bool(chk["all_done"] and chk["fail_rate"] <= 0.05 and 8.0 <= per_win <= 11.0 and 0.4 <= mean <= 1.0 and need_gb <= free_gb - 300)
    print(json.dumps(dict(check=chk, chunks=rows), indent=1))
    sys.exit(0 if chk["pass"] else 2)


def cmd_pick(a):
    Wn = windows()
    sc, bad = {}, Counter()                                                  # window -> {tag: score}
    runs = {}
    for c in a.chunks:
        ids = (W / "select/lists" / f"{c}.txt").read_text().split()
        for t in TAGS:
            D = W / "roll" / t / c / "run"
            runs[t, c] = D
            s = summary(D)
            ro = {r["clipgt_id"]: r for r in (s["rollouts"] if s else [])}
            for i in ids:
                r = ro.get(i)
                if r is None or (r.get("failure_reason") and r["failure_reason"] not in ZERO_TYPES):
                    bad[t] += 1
                    sc.setdefault(i, {})[t] = None
                else:
                    sc.setdefault(i, {})[t] = 0.0 if r.get("failure_reason") else float(r.get("score") or 0.0)
        for i in ids:
            sc[i]["_chunk"] = c
    rec, win = [], {}
    for i, v in sc.items():
        s = [v[t] for t in TAGS]
        if any(x is None for x in s):
            continue
        b, w = max(s), min(s)
        wt = TAGS[s.index(b)]
        ties = [t for t, x in zip(TAGS, s) if x == b]
        info = b - w >= a.gap
        rec.append(dict(id=i, stratum=Wn[i]["stratum"], shard=int(Wn[i]["shard"]), chunk=v["_chunk"], scores=dict(zip(TAGS, s)), best=b, worst=w,
                        winner=wt, ties=len(ties), info=bool(info)))
        if info:
            win[i] = (wt, v["_chunk"])
    # views: only the winners' rollouts, everything else of the run dir linked
    stages = {}
    for (t, c), D in runs.items():
        keep = {i for i, (wt, cc) in win.items() if wt == t and cc == c}
        if not keep:
            continue
        V = W / "view" / a.label / t / c / "run"
        (V / "aggregate").mkdir(parents=True, exist_ok=True)
        for e in D.iterdir():
            if e.name != "aggregate" and not (V / e.name).exists():
                (V / e.name).symlink_to(e)
        s = summary(D)
        s["rollouts"] = [r for r in s["rollouts"] if r["clipgt_id"] in keep]
        assert len(s["rollouts"]) == len(keep)
        (V / "aggregate/results-summary.json").write_text(json.dumps(s))
        k = int(Wn[next(iter(keep))]["shard"])
        stages.setdefault(k, {})[f"{t}@{c}"] = [str(V)]
    for k, m in stages.items():
        (W / "stage" / f"{a.label}{k}").mkdir(parents=True, exist_ok=True)
        (W / "stage" / f"{a.label}{k}" / "manifest.json").write_text(json.dumps(m, indent=1))
    # report
    I = [r for r in rec if r["info"]]
    out = dict(label=a.label, gap=a.gap, chunks=a.chunks, windows=len(sc), complete=len(rec), incomplete_by_tag=dict(bad), informative=len(I),
               shards=sorted(stages), winners={t: sum(r["winner"] == t for r in I) for t in TAGS},
               ties_informative=sum(r["ties"] > 1 for r in I), by_stratum={}, mean={}, zeros={})
    for st in STRATA + ("all",):
        A_ = [r for r in rec if st == "all" or r["stratum"] == st]
        Is = [r for r in A_ if r["info"]]
        out["by_stratum"][st] = dict(windows=len(A_), informative=len(Is), wins={t: sum(r["winner"] == t for r in Is) for t in TAGS},
                                     best_share_all={t: float(sum((r["scores"][t] == r["best"]) / r["ties"] for r in A_) / max(len(A_), 1)) for t in TAGS})
        out["mean"][st] = {t: float(np.mean([r["scores"][t] for r in A_])) if A_ else None for t in TAGS} | dict(best=float(np.mean([r["best"] for r in A_])) if A_ else None)
        out["zeros"][st] = {t: int(sum(r["scores"][t] == 0 for r in A_)) for t in TAGS}
    (W / "pick").mkdir(parents=True, exist_ok=True)
    (W / "pick" / f"{a.label}.json").write_text(json.dumps(dict(summary=out, windows=rec), indent=1))
    (W / "pick" / f"{a.label}_shards.txt").write_text(" ".join(map(str, sorted(stages))) + "\n")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"pick_{a.label}.json").write_text(json.dumps(out, indent=1) + "\n")
    hd = " | ".join(SHORT[t] for t in TAGS)
    L = [f"# Best-of-N pick `{a.label}`", "",
         f"{len(rec)} complete windows of {len(sc)} (a window is dropped when any model's rollout failed outside the scorer's zero types: "
         f"{dict(bad)}); informative = best - worst >= {a.gap}: **{len(I)}**; ties among the best on informative windows {out['ties_informative']} "
         "(broken toward JTN-L18-s1, then NTV0, RSG, NR1FB10).", "",
         f"Wins on informative windows (models: {', '.join(f'{SHORT[t]} = {t}' for t in TAGS)}):", "",
         f"| stratum | windows | informative | {hd} | mean {hd} | best-of-4 mean | zeros {hd} |", "|:--|--:|--:|" + "--:|" * (3 * len(TAGS) + 1)]
    for st in STRATA + ("all",):
        b, m, z = out["by_stratum"][st], out["mean"][st], out["zeros"][st]
        if not b["windows"]:
            continue
        L.append(f"| {st} | {b['windows']} | {b['informative']} | " + " | ".join(str(b["wins"][t]) for t in TAGS) + " | "
                 + " | ".join(f"{m[t]:.3f}" for t in TAGS) + f" | {m['best']:.3f} | " + " | ".join(str(z[t]) for t in TAGS) + " |")
    (RES / f"pick_{a.label}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_target(a):
    for p in a.pools:
        d = CR / p
        tab = dict(np.load(d / "tab.npz"))
        if not (d / "fut_human.npz").exists():
            np.savez(d / "fut_human.npz", names=tab["names"], fut=tab["fut"])
        h = np.load(d / "fut_human.npz")
        assert np.array_equal(h["names"], tab["names"])
        sv = tab["served"].astype(np.float32)
        assert sv.shape == h["fut"].shape and np.isfinite(sv).all(), p
        tab["fut"] = sv
        np.savez(d / "tab.npz", **tab)
        ade = np.linalg.norm(sv[..., :2] - h["fut"][..., :2], axis=-1).mean(1)
        print(json.dumps(dict(pool=p, rows=len(sv), served_vs_human_ade_m=dict(median=float(np.median(ade)), mean=float(ade.mean()),
                                                                                 p90=float(np.quantile(ade, 0.9))))))


def gates(pool_name):
    """Per row of rows/<pool_name>: passes its pool gates (R2 amended per shard; R1 (a) per tag; R1 (b) as on_rows) -> (ok (n,) bool, record)."""
    d = ROWS / pool_name
    r1, r2 = json.loads((d / "r1.json").read_text()), json.loads((d / "r2.json").read_text())
    M = np.load(d / "meta.npz")
    ok2 = r2["ego_max_m"] <= 0.3 and r2["frame_err_max_m"] <= 0.1 and r2["act_share_within"] >= 0.995 and r2["act_median_m"] <= 0.05
    tag_ok = {t: bool(q["mu_share"] >= 0.99) for t, q in r1["tags"].items()}
    ok = np.array([ok2 and r1["b"] and tag_ok[t] for t in M["tag"].astype(str)])
    return ok, dict(r2=ok2, r1b=r1["b"], r1a={t: [q["n"], q["mu_share"]] for t, q in r1["tags"].items()},
                    r1c={t: [q["control"], q["control_mu_share"]] for t, q in r1["tags"].items()}, rows=len(ok), rows_ok=int(ok.sum()))


def cmd_weights(a):
    tb = np.load(CR / a.pool / "tab.npz")
    names, mode, logs, pool = (tb[k].astype(str) for k in ("names", "mode", "log", "pool"))
    z = np.load(DATA / "runs/defect_fix/w" / f"{a.base}.npz")
    n0 = len(z["names"])
    assert np.array_equal(z["names"].astype(str), names[:n0]), "the merged pool's prefix != the base weights' rows"
    S0 = mode[:n0] == "self"
    hl = {g for g in set(logs[:n0][S0]) if zlib.crc32(g.encode()) % 100 < 15}               # jt.py held_logs, --hold-pct 15 (= rm_train's)
    bn = np.arange(len(names)) >= n0
    ok = np.zeros(len(names), bool)
    grec = {}
    for p in a.parts:
        pre = p.split(":")[1]
        g, grec[p] = gates(p.split(":")[0].removeprefix("on1_"))
        m = pool == pre
        assert m.sum() == len(g), p
        ok[m] = g
    held = bn & np.isin(logs, sorted(hl))
    use = bn & ok & ~held
    w = np.zeros(len(names))
    w[:n0] = z["w"] * (K_ON - K_BN) / K_ON
    w[use] = K_BN / K_ON / use.sum()
    grp = np.concatenate([z["grp"].astype("U12"), np.where(held[n0:], "BN_held", np.where(use[n0:], "BN", "BN_gated"))]).astype("U12")
    w = w / w.sum()
    out = DATA / "runs/defect_fix/w" / f"{a.out}.npz"
    assert not out.exists() or a.force, f"{out} exists"
    np.savez(out, names=tb["names"], w=w, grp=grp)
    M = {p: np.load(ROWS / p.split(":")[0].removeprefix("on1_") / "meta.npz")["tag"].astype(str) for p in a.parts}
    tags = np.concatenate([M[p] for p in a.parts])
    rec = dict(pool=a.pool, base=a.base, rows=len(names), base_rows=n0, bn_rows=int(bn.sum()), bn_used=int(use.sum()), bn_held_log=int(held.sum()),
               bn_gated=int((bn & ~ok).sum()), held_logs=sorted(hl), k_on=K_ON, k_bn=K_BN,
               bn_rows_by_tag={t: int((tags == t).sum()) for t in TAGS}, bn_used_by_tag={t: int((tags[use[n0:]] == t).sum()) for t in TAGS},
               share={g: float(w[grp == g].sum()) for g in np.unique(grp)}, per_batch={g: float(w[grp == g].sum() * K_ON) for g in np.unique(grp)},
               draws_per_bn_row_2k=float(K_BN * 2000 / max(use.sum(), 1)), gates=grec)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"w_{a.out}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(rec, indent=1, default=str))


def cmd_gate(a):
    import pandas as pd
    from jevdrive import stats
    sys.path[:0] = [str(REPO / "experiments/defect_fix/scripts")]
    import df
    J = DATA / "runs/defect_fix/jt/arena"
    tmp = W / "gate_reads"
    tmp.mkdir(parents=True, exist_ok=True)
    tags = (a.arm, a.control, a.base)
    for t in tags:
        (tmp / t).unlink(missing_ok=True)
        (tmp / t).symlink_to(J / f"{t}@off")
    os.system(f"{sys.executable} {REPO}/experiments/alpasim_proxy/scripts/arena.py collect --reads {tmp} --out {W}/gate_rows.csv.gz > /dev/null")
    Sx = pd.read_csv(DATA / "runs/alpasim_proxy/arena/set/set.csv.gz").set_index("id")
    Hs = Sx[Sx.split == "held"]
    R = pd.read_csv(W / "gate_rows.csv.gz")
    X = {t: R[(R.tag == t) & R.id.isin(Hs.index)].drop_duplicates("id").set_index("id") for t in tags}
    assert all(len(X[t]) >= 0.99 * len(Hs) for t in tags), {t: len(X[t]) for t in tags}

    def pair(t, ref, strat=None):
        A_, B = X[t], X[ref]
        bad = set(A_.index[A_.type == "route-fail"]) | set(B.index[B.type == "route-fail"])
        c = A_.index.intersection(B.index).difference(bad)
        if strat:
            c = c[Hs.loc[c, "stratum"].values == strat]
        x, y, g = A_.loc[c, "score"].values, B.loc[c, "score"].values, Hs.loc[c, "log"].values
        d = stats.paired(x, y, groups=g) if len(c) >= 20 else None
        return dict(n=len(c), mean=float(x.mean()), ref=float(y.mean()), d=d, zeros=int((x == 0).sum()), zeros_ref=int((y == 0).sum()),
                    new_zeros=int(((y > 0) & (x == 0)).sum()), fixed_zeros=int(((y == 0) & (x > 0)).sum()))
    ar = {f"{t} vs {r}": {s or "all": pair(t, r, s) for s in (None,) + STRATA} for t, r in ((a.arm, a.control), (a.arm, a.base), (a.control, a.base))}
    h = {t: df.cl_read(t, [a.control, a.base]) for t in (a.arm, a.control)}
    H = {t: {k: v for k, v in (h[t]["held"] or {}).items() if k != "pair"} for t in h}
    assert all(H[t].get("scenes") == 460 for t in H), "held 460 reads incomplete"
    p0 = ar[f"{a.arm} vs {a.control}"]["all"]
    g1 = bool(p0["d"]["mean"] >= 0.01 and p0["d"]["lo"] > 0)
    g2 = bool(H[a.arm]["zeros"] <= H[a.control]["zeros"] + 2)
    v = dict(arena_paired=p0["d"], arena_rule="paired >= +0.01 and CI lo > 0", arena_pass=g1, held_zeros=H[a.arm]["zeros"],
             held_zeros_control=H[a.control]["zeros"], held_rule="zeros <= control + 2", held_pass=g2, pass_=g1 and g2,
             held_pair_vs_control=(h[a.arm]["held"] or {}).get("pair", {}).get(a.control))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate.json").write_text(json.dumps(dict(verdict=v, arena=ar, held=H, held_pairs={t: (h[t]["held"] or {}).get("pair") for t in h}),
                                              indent=1, default=str) + "\n")
    f = lambda d: "-" if d is None else stats.fmt(d, "+.4f")  # noqa: E731
    L = [f"# BEST-OF-N gate: {'PASS' if v['pass_'] else 'FAIL'}", "",
         f"Arena held split native (no hold, {len(Hs)} windows, `jt.sh aread <tag> off`), paired on the common windows without a route-fail, "
         "95 % CI clustered by log (jevdrive.stats). Held 460 @18: `rm_chain.sh read`, paired session-clustered (defect_fix df.cl_read).", "",
         f"- arena `{a.arm}` - `{a.control}`: **{f(p0['d'])}** (rule {v['arena_rule']}: {'pass' if g1 else 'fail'})",
         f"- held 460 zeros {H[a.arm]['zeros']} vs control {H[a.control]['zeros']} (rule {v['held_rule']}: {'pass' if g2 else 'fail'})", "",
         "| pair | stratum | windows | tag mean | ref mean | paired [CI] | zeros tag / ref | new / fixed zeros |", "|:--|:--|--:|--:|--:|:--|--:|--:|"]
    for k, by in ar.items():
        for s, p in by.items():
            L.append(f"| {k} | {s} | {p['n']} | {p['mean']:.4f} | {p['ref']:.4f} | {f(p['d'])} | {p['zeros']} / {p['zeros_ref']} | {p['new_zeros']} / {p['fixed_zeros']} |")
    L += ["", "Held 460 @18:", "", "| tag | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | "
          f"paired vs {a.control} | paired vs {a.base} |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|"]
    for t in h:
        x, pr = H[t], (h[t]["held"] or {}).get("pair", {})
        pp = lambda r: (f"{pr[r]['mean']:+.4f} [{pr[r]['lo']:+.4f}, {pr[r]['hi']:+.4f}]" if pr.get(r) and r != t else "")  # noqa: E731
        L.append(f"| {t} | {x['zeros']} | {x['mean']:.4f} | {x['at_fault']} | {x['offroad']} | {x['lat_zeros']} | {x['coll']['front']} / {x['coll']['lateral']} / "
                 f"{x['coll']['rear']} | {x['progress_rel']:.3f} | {x['creep']} / {x['no_launch']} | {pp(a.control)} | {pp(a.base)} |")
    (RES / "gate.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    p = sub.add_parser("stage2")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--total-rollouts", type=int, required=True)
    p = sub.add_parser("pick")
    p.add_argument("--label", required=True), p.add_argument("--chunks", nargs="+", required=True), p.add_argument("--gap", type=float, default=GAP)
    p = sub.add_parser("target")
    p.add_argument("--pools", nargs="+", required=True)
    p = sub.add_parser("weights")
    p.add_argument("--pool", required=True), p.add_argument("--base", default="JT00"), p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True), p.add_argument("--force", action="store_true")
    p = sub.add_parser("gate")
    p.add_argument("--arm", required=True), p.add_argument("--control", required=True), p.add_argument("--base", default="JTN-L18-s1")
    a = ap.parse_args()
    dict(select=cmd_select, stage2=cmd_stage2, pick=cmd_pick, target=cmd_target, weights=cmd_weights, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    main()
