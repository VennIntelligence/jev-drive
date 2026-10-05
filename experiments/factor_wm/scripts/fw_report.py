"""factor_wm G0 readouts exactly as registered (plans/2026-10-05-stage1-prereg.md section 3.1), from $DATA_DIR/runs/factor_wm/roll/*.npz.

  G0a  per anchor: mean over steps 1..10 of |phi1(engine) - phi1(real frames)| (deg); per group median; lines launch / sharp:
       estar <= 0.7 x plane, cruise: estar <= plane + 0.1 deg; bootstrap 95% CI over anchors of the ratio of medians (descriptive)
  G0b  failure events per arm group (estar closed): perturbation arms (kick, swerve) any-failure rate >= 15% and at least two of stall / heading /
       lane >= 5% each; descriptive: free arm, plane engine, closedlat (logged speed); replay identity
Writes <out>.md and <out>.json; --sheet adds a contact-sheet PNG of real vs plane vs estar frames for representative g0a anchors.

  $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_report.py --out experiments/factor_wm/results/g0 [--sheet]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402


def load(prefix):
    fs = sorted(f for f in glob.glob(str(C.root("roll") / f"{prefix}-*of*.npz")) if "smoke" not in f)
    if not fs:
        return None
    zs = [dict(np.load(f, allow_pickle=True)) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}


def boot_ratio(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = np.median(a[idx], 1) / np.maximum(np.median(b[idx], 1), 1e-9)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def g0a(Z):
    ref = {c: Z["phi1"][i] for i, c in enumerate(Z["c"]) if Z["arm"][i] == "ref"}
    err = {}
    for i, arm in enumerate(Z["arm"]):
        if arm == "ref":
            continue
        e = float(np.mean(np.abs(Z["phi1"][i][1:] - ref[Z["c"][i]][1:])))
        err.setdefault(arm, {}).setdefault(str(Z["cat"][i]), {})[int(Z["c"][i])] = e
    out = {}
    for cat in ("launch", "sharp", "cruise"):
        row = {}
        for arm in sorted(err):
            d = err[arm].get(cat, {})
            row[arm] = dict(n=len(d), median=float(np.median(list(d.values()))) if d else None)
        cs = sorted(err["plane-anchor"].get(cat, {}))
        a = np.array([err["estar-anchor"][cat][c] for c in cs])
        b = np.array([err["plane-anchor"][cat][c] for c in cs])
        ratio = float(np.median(a) / max(np.median(b), 1e-9))
        line = ratio <= 0.7 if cat in ("launch", "sharp") else float(np.median(a)) <= float(np.median(b)) + 0.1
        out[cat] = dict(arms=row, ratio_estar_plane=ratio, ratio_ci=boot_ratio(a, b), line_pass=bool(line))
    out["pass"] = bool(all(out[c]["line_pass"] for c in ("launch", "sharp", "cruise")))
    return out


def reclassify(Z):
    """prereg 3.1 correction: a heading / lane event while the ego is slower than STALL_V and > 1 m behind the log is a stall."""
    ev = Z["event"].astype(object).copy()
    for i, e in enumerate(ev):
        if e in ("heading", "lane"):
            j = int(round(Z["t_event"][i] / C.DT))
            if Z["vs"][i][j] < C.STALL_V and Z["off"][i][j, 0] < -1.0:
                ev[i] = "stall"
    Z["event_raw"], Z["event"] = Z["event"], ev.astype(str)
    return int(np.sum(Z["event_raw"] != Z["event"]))


def launch_accel(ZA):
    """descriptive: the model's acceleration on the real logged frames (g0a ref arm) at launch clips vs the logged acceleration, first 2 s."""
    S = C.Clips("g0a")
    m = (ZA["arm"] == "ref") & (ZA["cat"] == "launch")
    acc = ZA["acc"][m][:, :10]
    c = ZA["c"][m]
    v = S.t["v"][c][:, C.T0: C.T0 + 11]
    a_log = np.diff(v, axis=1) / C.DT
    return dict(n=int(m.sum()), model_acc_t0_median=float(np.median(acc[:, 0])), model_acc_2s_median=float(np.median(acc.mean(1))),
                log_acc_2s_median=float(np.median(a_log.mean(1))), share_model_acc_t0_below_0p3=float(np.mean(acc[:, 0] < 0.3)))


def g0b(Z):
    out = {"reclassified_to_stall": reclassify(Z)}
    kinds = [("estar closed", "closed", "estar"), ("plane closed", "closed", "plane"), ("estar closedlat", "closedlat", "estar")]
    pert = np.array([a.startswith(("kick", "swerve")) for a in Z["arm"]])
    for label, kind, eng in kinds:
        m = (Z["kind"] == kind) & (Z["engine"] == eng)
        res = {}
        for grp, gm in (("perturbed", pert), ("free", Z["arm"] == "free")):
            mm = m & gm
            n = int(mm.sum())
            ev = Z["event"][mm]
            r = dict(n=n, **{k: float(np.mean(ev == k)) if n else None for k in ("stall", "heading", "lane", "behind", "ahead")})
            r["any_failure"] = float(np.mean(np.isin(ev, ["stall", "heading", "lane"]))) if n else None
            r["by_cat"] = {}
            for cat in ("launch", "turn", "cruise"):
                cm = mm & (Z["cat"] == cat)
                e2 = Z["event"][cm]
                r["by_cat"][cat] = dict(n=int(cm.sum()), **{k: int(np.sum(e2 == k)) for k in ("stall", "heading", "lane", "behind", "ahead")})
            res[grp] = r
        out[label] = res
    p = out["estar closed"]["perturbed"]
    two = sum(p[k] >= 0.05 for k in ("stall", "heading", "lane"))
    out["pass"] = bool(p["any_failure"] >= 0.15 and two >= 2)
    rp = Z["kind"] == "replay"
    if rp.any():
        out["replay_identity_max_abs_offset"] = float(np.abs(Z["off"][rp]).max())
    return out


def fmt(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def write_md(path, A, B):
    L = ["# factor_wm G0 readouts (generated by scripts/fw_report.py)", ""]
    if A:
        L += ["## G0a engine fidelity (WOD val, mean |dphi1| over steps 1-10, deg, median per group)", "",
              "| group | n | plane anchor | depth anchor | estar anchor | estar / plane [95% CI] | line | plane step1 | estar step1 |", "|:--|--:|--:|--:|--:|--:|:--|--:|--:|"]
        for cat in ("launch", "sharp", "cruise"):
            r = A[cat]
            ar = r["arms"]
            L.append(f"| {cat} | {ar['plane-anchor']['n']} | {fmt(ar['plane-anchor']['median'])} | {fmt(ar['depth-anchor']['median'])} | "
                     f"{fmt(ar['estar-anchor']['median'])} | {r['ratio_estar_plane']:.2f} [{r['ratio_ci'][0]:.2f}, {r['ratio_ci'][1]:.2f}] | "
                     f"{'pass' if r['line_pass'] else 'fail'} | {fmt(ar['plane-step1']['median'])} | {fmt(ar['estar-step1']['median'])} |")
        la = A.get("launch_accel_real_frames", {})
        L += ["", f"G0a overall: **{'pass' if A['pass'] else 'fail'}**", "",
              f"Launch, real logged frames (descriptive, n {la.get('n')}): model acceleration at t0 median {fmt(la.get('model_acc_t0_median'))} m/s2, "
              f"over 2 s {fmt(la.get('model_acc_2s_median'))}; logged {fmt(la.get('log_acc_2s_median'))}; share with t0 acceleration < 0.3: "
              f"{fmt(la.get('share_model_acc_t0_below_0p3'))}", ""]
    if B:
        L += ["## G0b shipped Cinque closed loop 8 s (WOD val g0b)", "",
              "| engine / kind | arms | n | any failure | stall | heading | lane | behind (not failure) | ahead (validity end) |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
        for label in ("estar closed", "plane closed", "estar closedlat"):
            for grp in ("perturbed", "free"):
                r = B[label][grp]
                L.append(f"| {label} | {grp} | {r['n']} | {fmt(r['any_failure'])} | {fmt(r['stall'])} | {fmt(r['heading'])} | {fmt(r['lane'])} | "
                         f"{fmt(r['behind'])} | {fmt(r['ahead'])} |")
        L += ["", "Per category (estar closed, perturbed arms; counts):", "", "| category | n | stall | heading | lane | behind | ahead |", "|:--|--:|--:|--:|--:|--:|--:|"]
        for cat, r in B["estar closed"]["perturbed"]["by_cat"].items():
            L.append(f"| {cat} | {r['n']} | {r['stall']} | {r['heading']} | {r['lane']} | {r['behind']} | {r['ahead']} |")
        L += ["", f"heading / lane events reclassified to stall (slow and behind): {B['reclassified_to_stall']}", ""]
        L += ["", f"G0b: **{'pass' if B['pass'] else 'fail'}**; replay identity max |offset| {B.get('replay_identity_max_abs_offset', float('nan')):.2g}", ""]
    Path(path).with_suffix(".md").write_text("\n".join(L))


def sheet(path, Z):
    """Representative g0a anchors (closest to the group median of plane-anchor error): real frame vs plane vs estar at steps 5 and 10, road view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from jevdrive import op_interp as I
    import fw_g0 as G
    S = C.Clips("g0a")
    ref = {c: Z["phi1"][i] for i, c in enumerate(Z["c"]) if Z["arm"][i] == "ref"}
    pick = []
    for cat in ("launch", "sharp", "cruise"):
        m = (Z["arm"] == "plane-anchor") & (Z["cat"] == cat)
        e = np.array([np.mean(np.abs(Z["phi1"][i][1:] - ref[Z["c"][i]][1:])) for i in np.flatnonzero(m)])
        pick.append(int(Z["c"][np.flatnonzero(m)[np.argmin(np.abs(e - np.median(e)))]]))
    rgb = lambda f: I.to_rgb(torch.from_numpy(np.asarray(f[0]))).permute(1, 2, 0).numpy()  # noqa: E731
    fig, ax = plt.subplots(len(pick) * 2, 3, figsize=(6.875, 1.25 * 2 * len(pick)))
    for r, c in enumerate(pick):
        for k, j in enumerate((5, 10)):
            pose = S.t["pose"][c][j]
            ims = [S.imgs[c, C.T0 + j], G.frame_job(("g0a", c, j, "plane", "anchor", pose)), G.frame_job(("g0a", c, j, "estar", "anchor", pose))]
            for q, (im, t) in enumerate(zip(ims, ("real", "plane (t0 frame)", "estar (t0 frame)"))):
                a = ax[2 * r + k, q]
                a.imshow(np.clip(rgb(im), 0, 1))
                a.set_xticks([]), a.set_yticks([])
                a.set_title(f"{S.t['cat'][c]} {S.t['id'][c][:8]} +{j * C.DT:.1f} s {t}", fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    ZA, ZB = load("g0a"), load("g0b")
    A = g0a(ZA) if ZA is not None else None
    B = g0b(ZB) if ZB is not None else None
    if A is not None:
        A["launch_accel_real_frames"] = launch_accel(ZA)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(g0a=A, g0b=B), open(out.with_suffix(".json"), "w"), indent=1)
    write_md(out, A, B)
    print(out.with_suffix(".md").read_text())
    if a.sheet and ZA is not None:
        sheet(str(out) + "-sheet.png", ZA)


if __name__ == "__main__":
    main()
