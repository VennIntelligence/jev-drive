"""Read-outs of AlpaSim runs with the SH30 driver (experiments/alpasim/lib/sh30_driver.py). Run with the repo .venv on the box.

  table   per-scene scores of several runs side by side + driver counters, stage latency, VRAM, and (SH30 runs) the NAVSIM cross-checks:
          a scene `<log>-<token>` starts 1.5 s before that navtest token, so decision 3 is the token's t0: route command vs NAVSIM
          `driving_command`, fed ego state vs the index, online plan vs the stored offline bench plan
            sh30_report.py table --runs sh30=<run dir> ltf=<run dir> [--out results.md]
  frames  from a run with SH30_DUMP > 0: what the model saw (road / wide model frames of chosen slots, the plan drawn on the wide frame
          at the road 1.87 m below the camera) and a bird's-eye view of every plan against the driven path
            sh30_report.py frames --run <run dir> --out <png prefix> [--session 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")


def rollouts(run: Path) -> dict:
    d = json.loads((run / "aggregate/results-summary.json").read_text())
    return {r["clipgt_id"]: r for r in d["rollouts"]}


def drive_log(run: Path):
    f = run / "driver-logs/drive.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    return [r for r in rows if r["kind"] == "drive"], [r for r in rows if r["kind"] == "close"]


def why(r) -> str:
    m = r["score_metrics"]
    bad = ", ".join(k for k in FAIL if m.get(k))
    if r.get("failure_reason") and not bad:
        return "no score: " + str(r["failure_reason"])[:60]
    return bad or ("" if r["score"] >= 1 else f"progress {m['progress_clipped_rel']:.2f}")


def cmd_table(a):
    runs = {k: Path(v) for k, v in (x.split("=", 1) for x in a.runs)}
    R = {k: rollouts(p) for k, p in runs.items()}
    scenes = sorted(set.intersection(*(set(r) for r in R.values())))
    L = [f"Scenes: {len(scenes)} common to all runs. Score = AlpaSim scene score (0 on collision_at_fault / offroad / left_corridor_laterally, "
         "else min(progress / 0.8, 1)).", "", "| run | scenes | mean scene score | score 1 | score 0 | zeros by reason | mean progress_clipped_rel | mean lateral dist to GT (m) |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for k, r in R.items():
        s = np.array([r[x]["score"] for x in scenes])
        z = ", ".join("%d %s" % (n, f) for f in FAIL if (n := sum(bool(r[x]["score_metrics"].get(f)) for x in scenes))) or "-"
        L.append(f"| {k} | {len(scenes)} | {s.mean():.4f} | {(s >= 1).sum()} | {(s == 0).sum()} | {z} | "
                 f"{np.mean([r[x]['score_metrics']['progress_clipped_rel'] for x in scenes]):.3f} | "
                 f"{np.mean([r[x]['score_metrics']['lateral_dist_to_gt_trajectory'] for x in scenes]):.3f} |")
    L += ["", "| scene | " + " | ".join(f"{k} score | {k} progress | {k} why not 1" for k in R) + " |", "|:--|" + "--:|--:|:--|" * len(R)]
    for x in scenes:
        L.append(f"| {x} | " + " | ".join(f"{r[x]['score']:.3f} | {r[x]['score_metrics']['progress_clipped_rel']:.3f} | {why(r[x])}" for r in R.values()) + " |")
    for k, p in runs.items():
        dr, cl = drive_log(p)
        if not dr:
            continue
        ns = json.loads((p / "native_summary.json").read_text())
        vf = p / "driver-logs/vram.json"
        cnt = {c: sum(x[c] for x in cl) for c in ("drive", "inference", "inference_error", "input_error", "state_rotated", "cold")}
        L += ["", f"### Driver `{k}` ({p.name})", "", f"Sessions closed {len(cl)}; counters {json.dumps(cnt)}; trajectory poses per response "
              f"{sorted(set(x['n_out'] for x in dr))}; keyframes per decision k: " +
              ", ".join(f"k{j}: {sorted(set(x['n_keys'] for x in dr if x['k'] == j))}" for j in range(4)) + ".",
              f"VRAM: driver process peak {ns['peak'].get('driver', {}).get('gpu_mib', float('nan')) / 1024:.2f} GiB (nvidia-smi, 1 Hz)" +
              (f", torch max allocated {json.loads(vf.read_text())['max_allocated_gib']:.2f} GiB" if vf.exists() else "") +
              f"; runtime {ns['times'].get('runtime_s', float('nan')):.0f} s for {ns['times'].get('rollouts')} rollouts.", "",
              "| stage (ms per `drive`) | median | p95 | max |", "|:--|--:|--:|--:|"]
        sh = "frames" in dr[0]["ms"]                              # SH30 stage names; the WA-JEPA driver logs prep_in / infer
        for st in ("frames", "encode", "policy", "export", "prep", "wait", "total") if sh else ("prep_in", "infer", "prep", "wait", "total"):
            v = np.array([x["ms"][st] for x in dr])
            L.append(f"| {st} | {np.median(v):.1f} | {np.quantile(v, .95):.1f} | {v.max():.1f} |")
        im = p / "driver-logs/images.jsonl"
        if im.exists():
            v = np.array([json.loads(x)["pack_ms"] for x in im.read_text().splitlines()])
            L.append(f"| {'CAM_F0 JPEG decode + pack' if sh else 'one-camera JPEG decode + resize'} (in `submit_image_observation`) | {np.median(v):.1f} | {np.quantile(v, .95):.1f} | {v.max():.1f} |")
        L += ["", "frames = CPU ego-motion warp of the slot frames; encode = vision encoder on the image pairs; policy = adapter + policy; "
              "export = lever arm + resampling; prep = session bookkeeping; wait = queueing for the single inference lock; total = inside `drive`."
              if sh else "prep_in = AgentInput assembly; infer = the shipped agent's compute_trajectory (feature builder + 4-step flow sampling; fp32, or bf16 autocast when WAJ_AMP=1) "
              "incl. CUDA sync; prep = session bookkeeping; wait = queueing for the single inference lock; total = inside `drive`."]
        if a.navsim:
            L += navsim_check(dr, a.tag)
    out = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(out)
    print(out)


def navsim_check(dr, tag) -> list:
    from jevdrive import navsim_zs as Z
    from jevdrive.common import data_dir
    idx = {e["token"]: e for e in Z.load_index("navtest", slim=True)}
    pf = data_dir() / f"runs/bench/ol/lb_navtest/preds/{tag}-warp__base.npz"
    pz = np.load(pf) if pf.exists() else None
    prow = {t: k for k, t in enumerate(pz["tokens"].tolist())} if pz is not None else {}
    fz = np.load(Z.root("index") / "navtest_future.npz")
    frow = {t: k for k, t in enumerate(fz["tokens"].tolist())}
    rows = []
    for x in dr:
        tok = x["scene"].rsplit("-", 1)[-1]
        if x["k"] != 3 or tok not in idx:
            continue
        e, ego, p = idx[tok], np.array(x["ego"]), np.array(x["poses"])
        hist = np.array(x["hist"])
        d = lambda q: float(np.linalg.norm(p[:, :2] - q[:, :2], axis=1).mean())  # noqa: E731
        rows.append(dict(cmd=x["cmd"], nav=int(np.argmax(e["cmd"][-1])), dv=10 * ego[4] - e["vel"][-1][0], dvy=10 * ego[5] - e["vel"][-1][1],
                         dax=3 * ego[6] - e["acc"][-1][0], day=3 * ego[7] - e["acc"][-1][1],
                         dh=float(np.linalg.norm(hist[0, :2] - np.asarray(e["pose"])[0, :2])),
                         off=d(pz["poses"][prow[tok]]) if tok in prow else np.nan, log=d(fz["poses"][frow[tok]]) if tok in frow else np.nan,
                         off_log=float(np.linalg.norm(pz["poses"][prow[tok]][:, :2] - fz["poses"][frow[tok]][:, :2], axis=1).mean())
                         if tok in prow and tok in frow else np.nan))
    if not rows:
        return []
    c = np.zeros((4, 4), int)
    for r in rows:
        c[r["nav"], r["cmd"]] += 1
    m = lambda k: float(np.nanmean([abs(r[k]) for r in rows]))  # noqa: E731
    names = ("left", "straight", "right", "unknown")
    L = ["", f"NAVSIM cross-check at decision 3 (= the token's t0), {len(rows)} scenes:", "",
         "- route command vs NAVSIM `driving_command` (rows NAVSIM, columns route rule): agreement "
         f"{sum(r['cmd'] == r['nav'] for r in rows)} / {len(rows)}", "", "| NAVSIM \\ route | " + " | ".join(names) + " |", "|:--|" + "--:|" * 4]
    L += [f"| {names[i]} | " + " | ".join(str(v) for v in c[i]) + " |" for i in range(4) if c[i].sum()]
    L += ["", f"- fed ego state minus the NAVSIM index (mean abs; the rollout is closed loop from 0.5 s on, so these are not errors of the driver "
          f"alone): vx {m('dv'):.2f} m/s, vy {m('dvy'):.2f} m/s, ax {m('dax'):.2f}, ay {m('day'):.2f} m/s^2, oldest history pose {m('dh'):.2f} m",
          f"- 8-pose plan, mean distance: online vs offline bench plan of the token {m('off'):.2f} m; online vs logged future {m('log'):.2f} m; "
          f"offline bench plan vs logged future {m('off_log'):.2f} m"]
    return L


def rgb(p):
    """(6, 128, 256) packed YUV420 (BT.601 full range) -> (256, 512, 3) uint8."""
    Y = np.empty((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = p[0], p[1], p[2], p[3]
    U, V = (np.repeat(np.repeat(p[k].astype(np.float32), 2, 0), 2, 1) - 128 for k in (4, 5))
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def cmd_frames(a):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from jevdrive import op_interp as I
    run = Path(a.run)
    dr, _ = drive_log(run)
    ks, slots = [0, 1, 2, 3, 6, 9], [0, 2, 5, 6, 7]
    fig, ax = plt.subplots(len(ks), len(slots) + 2, figsize=(3.2 * (len(slots) + 2), 1.75 * len(ks)))
    for i, k in enumerate(ks):
        z = np.load(run / f"driver-logs/dump/s{a.session:02d}_k{k}.npz")
        h = float(z["cam_t"][2]) + 0.35
        ax[i, 0].imshow(Image.open(io.BytesIO(z["jpeg"].tobytes())).resize((480, 270)))
        ax[i, 0].set_title(f"decision {k}: CAM_F0 as received", fontsize=7)
        for j, s in enumerate(slots):
            ax[i, j + 1].imshow(rgb(z["cur"][s, 0]))
            ax[i, j + 1].set_title(f"road, slot {round(-1.4 + 0.2 * s, 1)} s" + ("" if z["valid"][s] else " (not fed)"), fontsize=7)
        ax[i, -1].imshow(rgb(z["cur"][7, 1]))
        mu, K = z["mu"], I.OP_K["wide"]
        ok = mu[:, 0] > 2.0
        ax[i, -1].plot(K[0, 0] * mu[ok, 1] / mu[ok, 0] + K[0, 2], K[1, 1] * (mu[ok, 2] + h) / mu[ok, 0] + K[1, 2], "-", color="lime", lw=1.5)
        ax[i, -1].set_xlim(0, 511), ax[i, -1].set_ylim(255, 0)
        ax[i, -1].set_title(f"wide, slot 0 s + plan ({mu[-1, 0]:.0f} m in 10 s)", fontsize=7)
    for x in ax.ravel():
        x.set_xticks([]), x.set_yticks([])
    fig.suptitle(f"{str(z['scene'])}: model frames per decision", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{a.out}_frames.jpg", dpi=110)
    sess = {}
    for x in dr:
        sess.setdefault(x["session"], []).append(x)
    S = list(sess.values())[: a.bev]
    fig, ax = plt.subplots(2, (len(S) + 1) // 2, figsize=(3.4 * ((len(S) + 1) // 2), 7), squeeze=False)
    for x, s in zip(ax.ravel(), S):
        an = np.array([r["anchor"] for r in s])
        R = I.rot2(-an[0, 2] + np.pi / 2)                       # start heading up
        for r in s:
            p = (np.array(r["poses"])[:, :2] @ I.rot2(r["anchor"][2]).T + r["anchor"][:2] - an[0, :2]) @ R.T
            x.plot(p[:, 0], p[:, 1], "-", color=plt.cm.viridis(r["k"] / 9), lw=0.9, alpha=0.9)
        d = (an[:, :2] - an[0, :2]) @ R.T
        x.plot(d[:, 0], d[:, 1], "k.-", lw=1.6, ms=5)
        x.set_aspect("equal"), x.set_title(s[0]["scene"][-16:] + f"  cmd {[r['cmd'] for r in s]}", fontsize=6), x.tick_params(labelsize=6)
    for x in ax.ravel()[len(S):]:
        x.axis("off")
    fig.suptitle("driven rear-axle path (black, one dot per decision) and each decision's 4 s plan (dark = first, yellow = last); metres, start heading up", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{a.out}_bev.jpg", dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--navsim", action="store_true")
    p.add_argument("--tag", default="SH30-F-s0")
    p = sub.add_parser("frames")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--session", type=int, default=0)
    p.add_argument("--bev", type=int, default=8)
    a = ap.parse_args()
    {"table": cmd_table, "frames": cmd_frames}[a.cmd](a)
