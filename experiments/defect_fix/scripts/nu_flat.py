#!/usr/bin/env python3
"""NU-FLAT (plans/2026-10-20-nu-flat-prereg.md): checks, the step rule and the read-out of the three B2-s1 continuations (CTRL / F / FR).

  $TR nu_flat.py ident                        (box) the two bitcmp results + the code each identity run used -> results/nu_flat/identity.json
  $TR nu_flat.py smoke --tag T --cmd-file F   (box) the 60-step smoke's registered checks -> results/nu_flat/smoke.json; the step rule
                                              N = round(T_total / (0.75 nN), 100), W = round(0.05 N) -> results/nu_flat/steps.json
  $TR nu_flat.py held --tags A B ..           (box) held 460 @18 of each tag, paired against CTRL and B2-s1 -> results/nu_flat/held.json
  $TR nu_flat.py costs --tags A B ..          (box) training wall time -> results/nu_flat/costs.{json,md}
  tmp/pcs_local/.venv/bin/python nu_flat.py gate   (Mac; tmp/nu_flat/ext_full = the box's $S/ext_full) the registered lines per arm against
                                              CTRL: guards from the stand.py tables (716-window subset, wait set, held 460), frozen-pool PCS and easy-miss
                                              classes (stand_r.py's _ctx / classes) on the 18 chunks and on the subset -> results/nu_flat/gate.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/onpol_nuplan/scripts"),
                str(REPO / "experiments/alpasim_proxy/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RES = REPO / "experiments/defect_fix/results/nu_flat"
B2 = "STANDFMXH-L18-s1"
ARMS = {"C": "B2C-L18-s1", "F": "B2F-L18-s1", "FR": "B2FR-L18-s1"}
CTL = ARMS["C"]
BATCH, D_FRAC = 128, 0.25                                       # rm_train --batch default; pp_train d_frac (anchor rows never flattened)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def run_dir(tag):
    d = sorted(glob.glob(str(DATA / f"runs/route_mem/train-{tag}/*")))
    assert d, f"no run dir for {tag}"
    return Path(d[-1])


def cmd_ident(a):
    d = REPO / "experiments/defect_fix/results"
    m = lambda t: json.loads((run_dir(t) / "meta.json").read_text())  # noqa: E731
    code = {t: dict(sha=m(t)["git"]["sha"][:8], dirty=m(t)["git"]["dirty"], cwd=m(t)["cwd"]) for t in ("NFGG-old", "NFGG-new", "NFGGP0-new")}
    out = dict(off=json.loads((d / "nu_flat/identity_off.json").read_text()), p0=json.loads((d / "nu_flat/identity_p0.json").read_text()), code=code)
    out["two_codes"] = code["NFGG-old"]["sha"] != code["NFGG-new"]["sha"] and code["NFGG-old"]["cwd"] != code["NFGG-new"]["cwd"]
    out["identical"] = bool(out["off"]["identical"] and out["p0"]["identical"] and out["two_codes"])
    dump(RES / "identity.json", out)
    print(json.dumps(dict(identical=out["identical"], code=code)))
    raise SystemExit(0 if out["identical"] else 2)


def nn_of(cmd):
    """navtrain normal rows per batch = batch - yaw-rate rows (first --on-k, self) - pool rows (second --on-k)."""
    t = cmd.split()
    ks = [int(t[i + 1]) for i, x in enumerate(t) if x == "--on-k"]
    assert len(ks) == 2 and "--batch" not in t, f"unexpected --on-k / --batch in {cmd}"
    return BATCH - ks[0] - ks[1], ks


def cmd_smoke(a):
    d = run_dir(a.tag)
    S = {}
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            S.setdefault(e["tag"], []).append(e["value"])
    dn = json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else {}
    meta = json.loads((d / "meta.json").read_text())
    fl = dn.get("flat_brk", {})
    nn, ks = nn_of(Path(a.cmd_file).read_text())
    share = fl.get("train_rows", 0) / max(fl.get("train_rows_total", 1), 1)
    exp_flat = a.flat_p * nn * (1 - D_FRAC) * share
    mean = lambda k: float(np.mean(S[k])) if k in S else float("nan")  # noqa: E731
    argv = " ".join(map(str, meta.get("argv", [])))
    ck = {
        "finite": all(np.isfinite(v).all() for k, v in S.items() if k.startswith("loss/")),
        "flat_rows": (mean("loss/flat/rows"), exp_flat, 0.6 * exp_flat <= mean("loss/flat/rows") <= 1.4 * exp_flat),
        "standr_pos_some_step": (max(S.get("loss/on/lon_standr_pos", [0])), 0, max(S.get("loss/on/lon_standr_pos", [0])) > 0),
        "flat_p_recorded": (fl.get("p"), a.flat_p, fl.get("p") == a.flat_p),
        "init": (B2 in json.dumps(dn.get("init")) and f"--init {B2}" in argv, B2, B2 in json.dumps(dn.get("init")) and f"--init {B2}" in argv),
        "dev_ade": (S.get("dev/ade", [None])[-1], None, True),
    }
    ok = bool(ck["finite"] and all(v[2] for k, v in ck.items() if k != "finite"))
    dump(RES / "smoke.json", dict(tag=a.tag, run=str(d), pass_=ok, checks=ck, steps=len(S.get("loss/total", [])), flat_brk=fl))
    T = int(fl["train_rows_total"])
    N = int(round(T / ((1 - D_FRAC) * nn) / 100) * 100)
    st = dict(rule="N = round(T / ((1 - d_frac) nN), 100), W = round(0.05 N)", T=T, nN=nn, on_k=ks, d_frac=D_FRAC, steps=N, warmup=int(round(0.05 * N)),
              listed_rows=fl["train_rows"], eligible_draws_per_listed_row=N * (1 - D_FRAC) * nn / T,
              flattened_per_listed_row=N * (1 - D_FRAC) * nn / T * a.flat_p, expected_flat_rows_per_batch=exp_flat)
    dump(RES / "steps.json", st)
    print(json.dumps(dict(pass_=ok, checks=ck, steps=st), indent=1, default=float))
    raise SystemExit(0 if ok else 2)


def cmd_held(a):
    import df
    out = {t: df.cl_read(t, [r for r in (CTL, B2, "RSG-L18-s0") if r != t])["held"] for t in a.tags}
    dump(RES / "held.json", out)
    print(json.dumps({t: dict(zeros=r["zeros"], mean=r["mean"], front=r["coll"]["front"]) for t, r in out.items()}, indent=1, default=float))


def cmd_costs(a):
    L = ["# NU-FLAT training cost", "", "| tag | steps | wall (min) | steps / s | gpu peak (GB) |", "|:--|--:|--:|--:|--:|"]
    out = {}
    for t in a.tags:
        d = run_dir(t)
        dn, m = json.loads((d / "DONE").read_text()), json.loads((d / "meta.json").read_text())
        st = m["config"]["steps"]
        out[t] = dict(steps=st, wall_s=dn["wall_s"], run=str(d), card_h=dn["wall_s"] / 3600, gpu_peak_gb=dn.get("gpu_peak_gb"))
        L.append(f"| {t} | {st} | {dn['wall_s'] / 60:.1f} | {st / dn['wall_s']:.2f} | {dn.get('gpu_peak_gb', float('nan')):.1f} |")
    dump(RES / "costs.json", out)
    (RES / "costs.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_gate(a):
    import pandas as pd
    import stand_r as SR
    cx = SR._ctx()
    R = pd.read_csv(RES / "arena_windows.csv.gz")
    tags = [*ARMS.values(), B2]
    keep = set.intersection(*(set(R[(R.tag == t) & R.score.notna()].id) for t in tags))
    R0 = pd.read_csv(REPO / "experiments/defect_fix/results/stand_r2/arena_windows.csv.gz")
    sub = keep & set(R0[(R0.tag == "STANDFMXH1K-L18-s0") & R0.score.notna()].id)   # d399's 716-window subset (chunks 0 / 3 / .. / 15)
    E = {t: SR.classes(cx, R, t, a.dec, keep) for t in tags}
    Es = {t: SR.classes(cx, R, t, a.dec, sub) for t in tags}
    H = json.loads((RES / "held.json").read_text())
    AF = {A: json.loads((RES / f"arena_full_{A}.json").read_text()) for A in ARMS}
    J = dict(windows_full=len(keep & set(cx["items"])), windows_subset=len(sub & set(cx["items"])), full=E, subset=Es, arms={})
    r3 = lambda x: round(float(x), 3)  # noqa: E731   (delays: 20 Hz log frames; rounding keeps 0.567 <= 0.417 + 0.15 from failing on float error)
    L = ["# NU-FLAT read-out (plans/2026-10-20-nu-flat-prereg.md; native reads; continuations of B2 `STANDFMXH-L18-s1`, control CTRL `B2C-L18-s1`)", "",
         f"Arena held native: {J['windows_full']} frozen board-mix windows every read has (18 chunks); d399's subset: {J['windows_subset']} of them.", "",
         "## Overview (full 18 chunks unless noted)", "",
         f"| read | {' | '.join(tags)} |", "|:--|" + "--:|" * len(tags),
         "| arena mean / zeros | " + " | ".join(f"{E[t]['mean']:.4f} / {E[t]['zeros']}" for t in tags) + " |",
         "| frozen-pool PCS | " + " | ".join(f"{E[t]['pcs']:.0f}" for t in tags) + " |",
         "| easy misses (red-line zeros) | " + " | ".join(f"{E[t]['easy_miss']} ({E[t]['red']})" for t in tags) + " |",
         "| subset: PCS / easy misses (red-line) | " + " | ".join(f"{Es[t]['pcs']:.0f} / {Es[t]['easy_miss']} ({Es[t]['red']})" for t in tags) + " |",
         "| held 460 mean / zeros / front | " + " | ".join(f"{H[t]['mean']:.4f} / {H[t]['zeros']} / {H[t]['coll']['front']}" for t in tags) + " |"]
    pair = []
    for A, T in ARMS.items():
        for ref in ([B2] if A == "C" else [CTL, B2]):
            r = AF[A][ref]
            hp = H[T]["pair"].get(ref, {})
            pair.append(f"| {T} - {ref} | {r['windows']} | {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['zeros_b']} -> {r['zeros_a']} | "
                        f"{r['af_front_b']} -> {r['af_front_a']} | {E[T]['pcs'] - E[ref]['pcs']:+.0f} | "
                        f"{hp.get('mean', float('nan')):+.4f} [{hp.get('lo', float('nan')):+.4f}, {hp.get('hi', float('nan')):+.4f}] |")
    L += ["", "| pair | windows | arena mean diff [log-clustered 95 % CI] | arena zeros | AF-front | PCS diff | held 460 mean diff |",
          "|:--|--:|--:|--:|--:|--:|--:|", *pair, ""]
    for A in ("C", "F", "FR"):
        T = ARMS[A]
        ref = B2 if A == "C" else CTL
        t = json.loads((RES / f"s17table_{A}.json").read_text())
        Wt, Ar, Dr = t["wait"], t["arena"], next(iter(t["drift"].values()))
        c, e, cs, es = E[ref], E[T], Es[ref], Es[T]
        cls = sorted((set(cs["by_class"]) | set(es["by_class"])) - {SR.RED})
        rows = [("guard: wait b launched " + k, r3(Wt[ref][k]["launched"]), r3(Wt[T][k]["launched"]), Wt[T][k]["launched"] >= 0.95 * Wt[ref][k]["launched"] - 1e-9,
                 ">= 0.95 x control (binding)") for k in ("other_lead", "red_queue")]
        rows += [("guard: wait b median launch delay " + k + " (s)", r3(Wt[ref][k]["delay_med"]), r3(Wt[T][k]["delay_med"]),
                  r3(Wt[T][k]["delay_med"]) <= r3(Wt[ref][k]["delay_med"] + 0.15), "<= control + 0.15 (binding)") for k in ("other_lead", "red_queue")]
        rows += [("guard: wait a " + k, r3(Wt[ref][k]["wait"]), r3(Wt[T][k]["wait"]), Wt[T][k]["wait"] >= Wt[ref][k]["wait"] - 0.03 - 1e-9, ">= control - 0.03")
                 for k in ("other_lead", "red_queue")]
        rows += [("guard: held 460 zeros", H[ref]["zeros"], H[T]["zeros"], H[T]["zeros"] <= H[ref]["zeros"] + 2, "<= control + 2"),
                 ("guard: held 460 front collisions", H[ref]["coll"]["front"], H[T]["coll"]["front"], H[T]["coll"]["front"] <= H[ref]["coll"]["front"] + 2, "<= control + 2"),
                 ("guard: held 460 right drift, non-turn k 11-17 (arm - control, m)", "", r3(Dr["k11_17"]["mean"]), Dr["k11_17"]["mean"] > -0.05, "> -0.05"),
                 ("guard: subset AF-front windows", Ar["af_front_b"], Ar["af_front_a"], Ar["af_front_a"] <= Ar["af_front_b"] + 2, "<= control + 2"),
                 *[(f"guard: subset easy {k}", cs["by_class"].get(k, 0), es["by_class"].get(k, 0), es["by_class"].get(k, 0) <= cs["by_class"].get(k, 0) + 2,
                    "<= control + 2") for k in cls],
                 ("guard: subset slow partial windows", t["partials"][ref]["slow"], t["partials"][T]["slow"], t["partials"][T]["slow"] <= t["partials"][ref]["slow"] + 3,
                  "<= control + 3"),
                 ("guard: subset k = 9 speed ratio", round(Ar["k9"]["b"], 4), round(Ar["k9"]["a"], 4), Ar["k9"]["a"] >= Ar["k9"]["b"] - 0.03, ">= control - 0.03"),
                 ("main: frozen-pool PCS (18 chunks)", round(c["pcs"]), round(e["pcs"]), e["pcs"] >= c["pcs"] + 15, ">= control + 15")]
        if A == "FR":
            rows.append(("main: easy red-line zeros (18 chunks)", c["red"], e["red"], e["red"] <= c["red"] - 3, "<= control - 3"))
        rows.append(("description: subset easy red-line zeros (d399 line <= control - 6)", cs["red"], es["red"], True, "description only"))
        ok = all(bool(r[3]) for r in rows)
        launch_ok = all(bool(r[3]) for r in rows if "(binding)" in r[4])
        J["arms"][A] = dict(tag=T, control=ref, pass_=ok, launch_guard=launch_ok, rows=[dict(read=r[0], control=r[1], arm=r[2], ok=bool(r[3]), line=r[4]) for r in rows],
                            red_fixed=sorted(set(c["red_ids"]) - set(e["red_ids"])), red_new=sorted(set(e["red_ids"]) - set(c["red_ids"])))
        head = ("continuation effect (CTRL against stored B2-s1; lines shown for reference, not a verdict)" if A == "C"
                else f"{'PASS' if ok else 'FAIL'} (launch guard {'ok' if launch_ok else 'FAIL'})")
        L += [f"## {A} `{T}` against `{ref}`: {head}", "", f"| read | {ref} | {T} | line |", "|:--|--:|--:|:--|",
              *[f"| {r[0]} | {r[1]} | {r[2]} | {r[4]}: {'ok' if r[3] else 'FAIL'} |" for r in rows], "",
              f"- red-line easy zeros (18 chunks) fixed {len(J['arms'][A]['red_fixed'])}, new {len(J['arms'][A]['red_new'])}; easy misses {c['easy_miss']} -> {e['easy_miss']}",
              "- full easy classes (control -> arm): " + "; ".join(f"{k} {c['by_class'].get(k, 0)} -> {e['by_class'].get(k, 0)}"
                                                                  for k in sorted(set(c["by_class"]) | set(e["by_class"]))), ""]
    dump(RES / "gate.json", J)
    (RES / "gate.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("ident")
    p = sp.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--cmd-file", required=True), p.add_argument("--flat-p", type=float, default=0.5)
    p = sp.add_parser("held")
    p.add_argument("--tags", nargs="+", required=True)
    p = sp.add_parser("costs")
    p.add_argument("--tags", nargs="+", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--dec", default=str(REPO / "tmp/nu_flat/ext_full"))
    a = ap.parse_args()
    dict(ident=cmd_ident, smoke=cmd_smoke, held=cmd_held, costs=cmd_costs, gate=cmd_gate)[a.cmd](a)
