"""P1 read-out (any probed model): score every arm's pose file with jevdrive.bench navtest, wait, paired tables vs `off`,
then the pre-registered k95 / k99 (point and CI-backed) and the explained variance per k.

  .venv/bin/python experiments/wm_surplus/scripts/p1_readout.py --tag wajepa-ctx --arms off k8 k16 ... k512 \
      --ev $DATA_DIR/runs/wm_surplus/p1_usage/ev.json --out experiments/wm_surplus/results/p1_usage
Arms are bench models PROBE-<tag>-<arm> (pred files written by the probe's merge step). Runs inside a CPU pool job (blocks on
bench.wait inside the pool, so no agent waits by hand). --v1 also scores every arm with the NAVSIM v1.1 devkit (PDMS, the metric
third-party subjects publish; <out>/v1/<arm>/pdms.csv, NAVSIM_THREADS from the env) and adds PDMS with paired log-cluster CIs.
"""
import argparse
import json
from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive import bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ref", default="off")
    ap.add_argument("--ref-spec", default="", help="bench model of the reference arm (default PROBE-<tag>-<ref>)")
    ap.add_argument("--ev", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--priority", type=float, default=0.0)
    ap.add_argument("--v1", action="store_true", help="also v1.1 PDMS per arm, paired vs the reference arm")
    ap.add_argument("--v1-only", action="store_true", help="only score v1.1 PDMS of --arms into <out>/v1/<arm>/ (a shard of a fanned-out v1 pass) and exit")
    ap.add_argument("--subset", default="", help="score and pair on a jevdrive.data.splits subset of navtest only (subset bench runs)")
    a = ap.parse_args()
    spec = {x: f"PROBE-{a.tag}-{x}" for x in a.arms}
    if a.ref_spec:
        spec[a.ref] = a.ref_spec
    if a.v1_only:
        return v1_scores(a, spec, Path(a.out))
    dirs = [bench.run(s, "navtest", priority=a.priority, subset=a.subset) for s in spec.values()]
    if not bench.wait(dirs):
        raise SystemExit("a scoring run failed: " + ", ".join(str(d) for d in dirs if (d / "ERROR").exists()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [f"{x}={s}" for x, s in spec.items() if x != a.ref]
    res = bench.report("navtest", arms, vs=[f"{a.ref}={spec[a.ref]}"], out=str(out), subset=a.subset)
    A = res["arms"].set_index("arm") if "arm" in res["arms"] else pd.read_csv(out / "navtest_arms.csv").set_index("arm")
    P = pd.read_csv(out / "navtest_paired.csv")
    P["arm"] = P.pair.str.split(" - ").str[0]
    P = P.set_index("arm")
    base = float(A.loc[a.ref, "EPDMS"])
    ev = json.loads(Path(a.ev).read_text()) if a.ev else {}
    rows = []
    for x in a.arms:
        if x == a.ref:
            continue
        r = dict(arm=x, EPDMS=float(A.loc[x, "EPDMS"]), retained=float(A.loc[x, "EPDMS"]) / base,
                 d=float(P.loc[x, "mean"]), lo=float(P.loc[x, "lo"]), hi=float(P.loc[x, "hi"]))
        for k in ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"):
            r[k] = float(A.loc[x, k])
        r["ev_navtrain"] = ev.get("explained_navtrain", {}).get(x)
        r["ev_navtest"] = ev.get("explained_navtest", {}).get(x)
        r["plan_shift_m"] = ev.get("plan_shift_vs_off_m", {}).get(x)
        r["ev_readin"] = ev.get("explained_readin", {}).get(x)
        r["norm_ratio"] = ev.get("norm_ratio", {}).get(x)
        rows.append(r)
    T = pd.DataFrame(rows)
    pd1 = v1_table(a, spec, out) if a.v1 else None
    if pd1 is not None:
        T = T.merge(pd1, on="arm", how="left")
    # arms k<k> (the protocol) or <variant>-k<k> (projection variants, one k95 / k99 set per variant)
    m = T.arm.str.extract(r"^(?:([A-Za-z]\w*)-)?k(\d+)$")
    T["group"], T["k"] = m[0].fillna(""), pd.to_numeric(m[1], errors="coerce")
    T = T.sort_values(["group", "k"], na_position="last")
    b1 = float(pd1.attrs["ref_PDMS"]) if pd1 is not None else None
    kq = {}
    for g in dict.fromkeys(T.group):
        G = T[(T.group == g) & T.k.notna()]
        pre = f"{g}_" if g else ""

        def first(mask):
            s = G.k[mask]
            return int(s.iloc[0]) if len(s) else None
        # k95 / k99 per the pre-registration: point (retained >= q) and CI-backed (CI lower bound of the difference >= -(1-q) x base)
        kq |= {f"{pre}k{int(q * 100)}_point": first(G.retained >= q) for q in (0.95, 0.99)}
        kq |= {f"{pre}k{int(q * 100)}_ci": first(G.lo >= -(1 - q) * base) for q in (0.95, 0.99)}
        if pd1 is not None:
            kq |= {f"{pre}PDMS_k{int(q * 100)}_point": first(G.PDMS / b1 >= q) for q in (0.95, 0.99)}
            kq |= {f"{pre}PDMS_k{int(q * 100)}_ci": first(G.dPDMS_lo >= -(1 - q) * b1) for q in (0.95, 0.99)}
    extra = T.ev_readin.notna().any()
    summ = dict(tag=a.tag, ref=a.ref, ref_EPDMS=base, ref_PDMS=None if pd1 is None else float(pd1.attrs["ref_PDMS"]), **kq, ref_subscores={k: float(A.loc[a.ref, k]) for k in
                ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC")}, ev=ev)
    T.to_csv(out / "p1_readout.csv", index=False)
    (out / "p1_readout.json").write_text(json.dumps(summ, indent=1))
    lines = [f"# P1 usage read-out: {a.tag} (ref arm `{a.ref}` EPDMS {base:.2f})" + (f"; subset {a.subset}" if a.subset else ""), "",
             "| arm | EPDMS | retained | d vs off [95 % CI] | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EV navtrain | EV navtest | plan shift m |"
             + (" EV read-in | norm ratio |" if extra else ""),
             "|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|" + ("--:|--:|" if extra else "")]
    f = lambda v, p=3: "" if v is None or v != v else f"{v:.{p}f}"  # noqa: E731
    for r in T.itertuples():
        lines.append(f"| {r.arm} | {r.EPDMS:.2f} | {r.retained:.4f} | {r.d:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | "
                     + " | ".join(f"{getattr(r, k):.2f}" for k in ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"))
                     + f" | {f(r.ev_navtrain)} | {f(r.ev_navtest)} | {f(r.plan_shift_m)} |"
                     + (f" {f(r.ev_readin)} | {f(r.norm_ratio)} |" if extra else ""))
    if pd1 is not None:
        lines += ["", f"NAVSIM v1.1 PDMS (ref arm `{a.ref}` {pd1.attrs['ref_PDMS']:.2f}; same poses):", "",
                  "| arm | PDMS | d vs off [95 % CI] |", "|---|--:|---|"]
        lines += [f"| {r.arm} | {r.PDMS:.2f} | {r.dPDMS:+.2f} [{r.dPDMS_lo:+.2f}, {r.dPDMS_hi:+.2f}] |" for r in T.itertuples()]
    groups = list(dict.fromkeys(T.group))
    if len(groups) > 1:
        lines += ["", "Smallest k per variant (`` = the protocol's PCA arms; point = retained >= q, CI = CI lower bound >= -(1-q) x ref):", "",
                  "| variant | EPDMS k95 point / CI | EPDMS k99 point / CI |" + (" PDMS k95 point / CI | PDMS k99 point / CI |" if pd1 is not None else ""),
                  "|---|---|---|" + ("---|---|" if pd1 is not None else "")]
        for g in groups:
            pre = f"{g}_" if g else ""
            cell = lambda m, q: f"{kq.get(f'{pre}{m}k{q}_point')} / {kq.get(f'{pre}{m}k{q}_ci')}"  # noqa: E731
            lines.append(f"| {g or 'pca (protocol)'} | {cell('', 95)} | {cell('', 99)} |" + (f" {cell('PDMS_', 95)} | {cell('PDMS_', 99)} |" if pd1 is not None else ""))
    lines += ["", "k95 / k99 (pre-registered): " + ", ".join(f"{k} = {v}" for k, v in kq.items()),
              "", "Paired CIs: per-token EPDMS x 100, cluster bootstrap over navtest logs (B 10000, seed 0); bench tables "
              "`navtest_{arms,paired,shapley,strata}.md` in this directory."]
    (out / "p1_readout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def v1_scores(a, spec: dict, out: Path) -> dict:
    """arm -> <out>/v1/<arm>/pdms.csv, scored with the v1.1 devkit unless already there (so fanned-out shards are reused)."""
    from jevdrive.bench.models import resolve
    from jevdrive.bench.navsim import pred_file, score_v1
    fs = {}
    for arm, s in spec.items():
        f = fs[arm] = out / "v1" / arm / "pdms.csv"
        if not f.exists():
            score_v1(pred_file(resolve(s), "navtest"), f"probe_{a.tag}_{arm}" + (f"_{a.subset.replace('/', '-')}" if a.subset else ""),
                     f.parent, "navtest", a.subset)
    return fs


def v1_table(a, spec: dict, out: Path) -> pd.DataFrame:
    """v1.1 PDMS per arm (devkit per-token scores, failed tokens 0 as the devkit's average row) and paired differences vs
    the reference arm with the log-cluster bootstrap of jevdrive.stats."""
    from jevdrive import stats
    from jevdrive.bench.sets import navsim_tokens
    toks, logs = navsim_tokens("navtest", a.subset)
    sc = {}
    for arm, f in v1_scores(a, spec, out).items():
        t = pd.read_csv(f)
        t = t[~t.token.astype(str).str.startswith("average")].set_index("token")
        sc[arm] = t.score.fillna(0).reindex(toks.tolist()).to_numpy(float)
    ref = sc[a.ref]
    rows = []
    for arm, x in sc.items():
        if arm == a.ref:
            continue
        r = stats.paired(100 * x, 100 * ref, groups=logs)
        rows.append(dict(arm=arm, PDMS=100 * float(x.mean()), dPDMS=r["mean"], dPDMS_lo=r["lo"], dPDMS_hi=r["hi"]))
    D = pd.DataFrame(rows)
    D.attrs["ref_PDMS"] = 100 * float(ref.mean())
    return D


if __name__ == "__main__":
    main()
