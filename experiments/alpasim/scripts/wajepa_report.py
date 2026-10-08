"""Read-outs of AlpaSim runs with the WA-JEPA driver (experiments/alpasim/lib/wajepa_driver.py). Run with the repo .venv on the box.

  table   sh30_report's per-scene table of several runs side by side + the driver's counters / latency / VRAM, then the NAVSIM cross-checks of
          the WA-JEPA run: a scene `<log>-<token>` starts 1.5 s before that navtest token, so decision 3 is the token's t0 (route command vs NAVSIM
          `driving_command`, fed ego state vs the index, online plan vs the stored plan of the 91.71 run and vs the log)
            wajepa_report.py table --runs wajepa=<run dir> sh30=<run dir> ltf=<run dir> [--out results.md]
  figs    from a run with WAJ_DUMP > 0: the images the model was fed (4 cameras at t0 and the oldest cold-start frame per decision) and a bird's-eye
          view of every plan against the driven path
            wajepa_report.py figs --run <run dir> --out <png prefix> [--session 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import sh30_report as R  # noqa: E402


def navsim_check(dr) -> list:
    from jevdrive.common import data_dir
    import pickle
    D = data_dir()
    tab = np.load(D / "runs/op_parity/cache/lb_navtest/tab.npz")
    row = {t: k for k, t in enumerate(tab["names"].tolist())}
    stored = pickle.load(open(D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl", "rb"))["trajectories"]
    rows = []
    for x in dr:
        tok = x["scene"].rsplit("-", 1)[-1]
        if x["k"] != 3 or tok not in row:
            continue
        r, p, hist = row[tok], np.array(x["poses"]), np.array(x["hist"])
        d = lambda q: float(np.linalg.norm(p[:, :2] - np.asarray(q)[:, :2], axis=1).mean())  # noqa: E731
        ego, v, a = np.array(x["ego"]), tab["vel"][r][-1], tab["acc"][r][-1]
        rows.append(dict(cmd=x["cmd"], nav=int(np.argmax(tab["cmd"][r][-1])), dv=ego[0] - v[0], dvy=ego[1] - v[1], dax=ego[2] - a[0], day=ego[3] - a[1],
                         dh=float(np.linalg.norm(hist[0, :2] - tab["pose"][r][0, :2])), dhy=float(abs(hist[0, 2] - tab["pose"][r][0, 2])),
                         off=d(stored[tok]) if tok in stored else np.nan, log=d(tab["fut"][r]),
                         off_log=float(np.linalg.norm(np.asarray(stored[tok])[:, :2] - tab["fut"][r][:, :2], axis=1).mean()) if tok in stored else np.nan))
    if not rows:
        return []
    c = np.zeros((4, 4), int)
    for q in rows:
        c[q["nav"], q["cmd"]] += 1
    m = lambda k: float(np.nanmean([abs(q[k]) for q in rows]))  # noqa: E731
    names = ("left", "straight", "right", "unknown")
    L = ["", f"NAVSIM cross-check at decision 3 (= the token's t0), {len(rows)} scenes:", "",
         f"- route command vs NAVSIM `driving_command` (rows NAVSIM, columns route rule): agreement {sum(q['cmd'] == q['nav'] for q in rows)} / {len(rows)}",
         "", "| NAVSIM \\ route | " + " | ".join(names) + " |", "|:--|" + "--:|" * 4]
    L += [f"| {names[i]} | " + " | ".join(str(v) for v in c[i]) + " |" for i in range(4) if c[i].sum()]
    L += ["", f"- fed ego state minus the NAVSIM index (mean abs; the rollout is closed loop from 0.5 s on, so these are not errors of the driver alone): "
          f"vx {m('dv'):.2f} m/s, vy {m('dvy'):.2f} m/s, ax {m('dax'):.2f}, ay {m('day'):.2f} m/s^2, oldest history pose {m('dh'):.2f} m / {m('dhy'):.3f} rad",
          f"- 8-pose plan, mean distance: online vs stored plan of the token {m('off'):.2f} m; online vs logged future {m('log'):.2f} m; "
          f"stored plan vs logged future {m('off_log'):.2f} m"]
    return L


def cmd_table(a):
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    ns = argparse.Namespace(runs=a.runs, out="", navsim=False, tag="")
    with redirect_stdout(buf):
        R.cmd_table(ns)
    L = buf.getvalue().rstrip("\n").split("\n")
    for k, v in (x.split("=", 1) for x in a.runs):
        if k.startswith("wajepa"):
            dr, _ = R.drive_log(Path(v))
            L += ["", f"### NAVSIM cross-check of `{k}`"] + navsim_check(dr)
    out = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(out)
    print(out)


def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from jevdrive import op_interp as I
    run = Path(a.run)
    dr, _ = R.drive_log(run)
    ks = [0, 1, 2, 3, 6, 9]
    cams = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")
    fig, ax = plt.subplots(len(ks), 5, figsize=(3.4 * 5, 1.9 * len(ks)))
    for i, k in enumerate(ks):
        z = np.load(run / f"driver-logs/dump/s{a.session:02d}_k{k}.npz")
        m = int(sum(f"CAM_F0_{j}" in z.files for j in range(4)))
        for j, c in enumerate(cams):
            ax[i, j].imshow(z[f"{c}_{m - 1}"])
            ax[i, j].set_title(f"decision {k}: {c} at t0", fontsize=7)
        ax[i, 4].imshow(z["CAM_F0_0"])
        ax[i, 4].set_title(f"decision {k}: CAM_F0, oldest of the 4 history frames" + (" (= t0, repeated)" if k < 3 else ""), fontsize=6)
    for x in ax.ravel():
        x.set_xticks([]), x.set_yticks([])
    fig.suptitle(f"{str(z['scene'])}: images WA-JEPA was fed (512 x 256 as received from the renderer; B0 is the rear camera)", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{a.out}_frames.jpg", dpi=100)
    sess = {}
    for x in dr:
        sess.setdefault(x["session"], []).append(x)
    S = list(sess.values())[: a.bev]
    fig, ax = plt.subplots(2, (len(S) + 1) // 2, figsize=(3.4 * ((len(S) + 1) // 2), 8), squeeze=False)
    for x, s in zip(ax.ravel(), S):
        an = np.array([r["anchor"] for r in s])
        Rm = I.rot2(-an[0, 2] + np.pi / 2)                      # start heading up
        for r in s:
            p = (np.array(r["poses"])[:, :2] @ I.rot2(r["anchor"][2]).T + r["anchor"][:2] - an[0, :2]) @ Rm.T
            x.plot(p[:, 0], p[:, 1], "-", color=plt.cm.viridis(r["k"] / 9), lw=0.9, alpha=0.9)
        d = (an[:, :2] - an[0, :2]) @ Rm.T
        x.plot(d[:, 0], d[:, 1], "k.-", lw=1.6, ms=5)
        x.set_xlim(d[:, 0].mean() - 12, d[:, 0].mean() + 12), x.set_aspect("equal")
        x.set_title(s[0]["scene"][-16:] + f"  cmd {[r['cmd'] for r in s]}", fontsize=6), x.tick_params(labelsize=6)
    for x in ax.ravel()[len(S):]:
        x.axis("off")
    fig.suptitle("driven rear-axle path (black, one dot per decision) and each decision's 4 s plan (dark = first, yellow = last); metres, start heading up, x window 24 m", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{a.out}_bev.jpg", dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="")
    p = sub.add_parser("figs")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--session", type=int, default=0)
    p.add_argument("--bev", type=int, default=8)
    a = ap.parse_args()
    {"table": cmd_table, "figs": cmd_figs}[a.cmd](a)
