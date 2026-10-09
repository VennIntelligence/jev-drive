"""Read-outs of PAI-track runs made by pai_run.sh (driver experiments/alpasim/lib/pai_driver.py). Plain python3 + numpy on the Tokyo box
(frames also needs matplotlib and Pillow).

  table   scene scores of one or more runs next to the organisers' reference subjects on the same scenes (mean of their 3 rollouts),
          zeros by kind, slow scenes, driver counters and latency, per-container VRAM / RAM peaks, wall time
            pai_report.py table --runs p2h10=<out dir> [...] --ref <dir with <subject>.json> [--out results.md]
  frames  from a run with PAI_DUMP > 0: per dumped session one row per chosen second: the front-wide JPEG as received, the road model
          frames of four slots, the wide model frame of slot 0 with the plan drawn on the road, and the bird's-eye plans of the session
            pai_report.py frames --run <out dir> --out <jpg prefix> [--sessions 0 1 2] [--at 20 60 100 140 180]
"""
import argparse
import json
from pathlib import Path

import numpy as np

FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")


def rollouts(run: Path) -> dict:
    d = json.loads((run / "sim/aggregate/results-summary.json").read_text())
    return {r["clipgt_id"]: r for r in d["rollouts"]}


def drive_log(run: Path):
    f = run / "driver/drive.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    return [r for r in rows if r["kind"] == "drive"], [r for r in rows if r["kind"] == "close"]


def refs(ref: Path) -> dict:
    out = {}
    for p in sorted(ref.glob("*.json")):
        s = {}
        for r in json.loads(p.read_text())["rollouts"]:
            s.setdefault(r["clipgt_id"], []).append(r)
        out[p.stem] = s
    return out


def why(r) -> str:
    m = r["score_metrics"]
    z = [f for f in FAIL if m.get(f)]
    if r.get("failure_reason") and not z:                   # a scored zero carries its kind there too; this is a rollout that broke
        return "FAILED: " + str(r["failure_reason"])[:60]
    return ", ".join(z) if z else "" if r["score"] >= 1 else f"progress {m.get('progress_clipped_rel', float('nan')):.2f}"


def gib(s: str) -> float:
    n = float("".join(c for c in s if c.isdigit() or c == "."))
    return n * {"KiB": 2**-20, "MiB": 2**-10, "GiB": 1.0}.get(s.lstrip("0123456789. "), 2**-30)


def cmd_table(a):
    runs = {k: Path(v) for k, v in (x.split("=", 1) for x in a.runs)}
    R, ref = {k: rollouts(v) for k, v in runs.items()}, refs(Path(a.ref))
    scenes = sorted(set.intersection(*(set(v) for v in R.values())))
    L = [f"Scenes scored in every run: {len(scenes)}. Reference subjects: mean of 3 rollouts on the same scenes "
         "(e2e_challenge/local_evaluation/data/pai).", "", "| Row | Mean scene score | Score 1 | Zeros | at-fault collision | offroad | left corridor | Slow (0 < s < 1) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for k, v in R.items():
        s = np.array([v[x]["score"] for x in scenes])
        zk = [[f for f in FAIL if v[x]["score_metrics"].get(f)] for x in scenes]
        L.append(f"| **{k}** | {s.mean():.4f} | {(s >= 1).sum()} | {(s == 0).sum()} | " + " | ".join(str(sum(f in z for z in zk)) for f in FAIL)
                 + f" | {((s > 0) & (s < 1)).sum()} |")
    for k, v in ref.items():
        if not all(x in v for x in scenes):
            continue
        s = np.array([np.mean([r["score"] for r in v[x]]) for x in scenes])
        n = sum(len(v[x]) for x in scenes)
        fr = [sum(bool(r["score_metrics"].get(f)) for x in scenes for r in v[x]) / n for f in FAIL]
        L.append(f"| {k} (reference) | {s.mean():.4f} | {(s >= 1).sum()} | {(s == 0).sum()} | " + " | ".join(f"{x:.0%}" for x in fr) + " | |")
    L += ["", "| Scene | Reference mean | alpamayo1 | " + " | ".join(f"{k} | why" for k in R) + " | GT distance m | " + " | ".join(f"{k} driven m" for k in R) + " |",
          "|---|--:|--:|" + "--:|---|" * len(R) + "--:|" + "--:|" * len(R)]
    rm = lambda x: np.mean([np.mean([r["score"] for r in v[x]]) for v in ref.values()])  # noqa: E731
    for x in sorted(scenes, key=rm):
        L.append(f"| {x[7:15]} | {rm(x):.2f} | {np.mean([r['score'] for r in ref['alpamayo1'][x]]):.2f} | "
                 + " | ".join(f"{v[x]['score']:.2f} | {why(v[x])}" for v in R.values())
                 + f" | {next(iter(R.values()))[x]['score_metrics'].get('gt_dist_traveled_m', float('nan')):.0f} | "
                 + " | ".join(f"{(v[x].get('metrics') or {}).get('dist_traveled_m', float('nan')):.0f}" for v in R.values()) + " |")
    for k, run in runs.items():
        dr, cl = drive_log(run)
        inf = [r for r in dr if r.get("infer")]
        L += ["", f"**{k}** ({run.name}): " + (json.dumps(json.loads((run / 'run.json').read_text())) if (run / "run.json").exists() else "")]
        if inf:
            tot = np.array([r["total_ms"] for r in inf])
            ms = {s: np.median([r["ms"][s] for r in inf]) for s in inf[0]["ms"]}
            cmd = np.bincount([r["cmd"] for r in inf], minlength=4)
            L.append(f"- {len(dr)} `drive` calls, {len(inf)} inferences; per call median {np.median(tot):.1f} ms, p95 {np.percentile(tot, 95):.1f}, max {tot.max():.0f}; "
                     f"stage medians " + ", ".join(f"{s} {v:.1f}" for s, v in ms.items()) + " ms")
            L.append(f"- commands fed (left / straight / right / unknown): {cmd.tolist()}; real slots per inference: "
                     f"{dict(zip(*map(lambda z: z.tolist(), np.unique([r['n_real'] for r in inf], return_counts=True))))}; "
                     f"history keys: {dict(zip(*map(lambda z: z.tolist(), np.unique([r['n_keys'] for r in inf], return_counts=True))))}")
        if cl:
            L.append("- session counters: " + ", ".join(f"{c} {sum(x[c] for x in cl)}" for c in ("drive", "inference", "inference_error", "input_error", "state_rotated", "cold", "images"))
                     + f"; delivered sizes {sorted({tuple(x['wh']) for x in cl if x.get('wh')})}; model-frame coverage (road, wide) "
                     f"{np.round(np.min([x['coverage'] for x in cl if x.get('coverage')], 0), 3).tolist()} (minimum over sessions)")
        u = [json.loads(x) for x in (run / "usage.jsonl").read_text().splitlines()] if (run / "usage.jsonl").exists() else []
        if u:
            names = sorted({n for x in u for n in list(x["vram_mib"]) + list(x["ram"])})
            L.append("- peaks: " + "; ".join(f"{n.split('-', 2)[-1]} VRAM {max(x['vram_mib'].get(n, 0) for x in u) / 1024:.1f} GiB, RAM "
                                             f"{max(gib(x['ram'].get(n, '0GiB')) for x in u):.1f} GiB" for n in names))
    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        Path(a.out).write_text(out)


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
    run = Path(a.run)
    dr, _ = drive_log(run)
    R = rollouts(run)
    Kw, rot = np.array([[455.0, 0, 256.0], [0, 455.0, 151.8], [0, 0, 1]]), lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    slots = [0, 3, 6, 7]
    for n in a.sessions:
        files = {int(f.stem.split("_k")[1]): f for f in sorted((run / "driver/dump").glob(f"s{n:02d}_k*.npz"))}
        ks = [k for k in a.at if k in files]
        if not ks:
            continue
        fig = plt.figure(figsize=(3.2 * (len(slots) + 3.2), 1.75 * len(ks)))
        gs = fig.add_gridspec(len(ks), len(slots) + 3, width_ratios=[1] * (len(slots) + 2) + [1.2])
        for i, k in enumerate(ks):
            z = np.load(files[k])
            h = float(z["cam_t"][2])
            x = fig.add_subplot(gs[i, 0])
            x.imshow(Image.open(io.BytesIO(z["jpeg"].tobytes())).resize((480, 270)))
            x.set_title(f"t = {k / 10:.0f} s: front wide as received", fontsize=7)
            for j, s in enumerate(slots):
                x = fig.add_subplot(gs[i, j + 1])
                x.imshow(rgb(z["cur"][s, 0]))
                x.set_title(f"road, slot {round(-1.4 + 0.2 * s, 1)} s" + ("" if z["real"][s] else " (warped)" if z["valid"][s] else " (not fed)"), fontsize=7)
            x = fig.add_subplot(gs[i, len(slots) + 1])
            x.imshow(rgb(z["cur"][7, 1]))
            mu = z["mu"]
            ok = mu[:, 0] > 2.0
            x.plot(Kw[0, 0] * mu[ok, 1] / mu[ok, 0] + Kw[0, 2], Kw[1, 1] * (mu[ok, 2] + h) / mu[ok, 0] + Kw[1, 2], "-", color="lime", lw=1.5)
            x.set_xlim(0, 511), x.set_ylim(255, 0)
            x.set_title(f"wide, slot 0 s + plan ({mu[-1, 0]:.0f} m in 10 s)", fontsize=7)
        scene = str(z["scene"])
        s = [r for r in dr if r["scene"] == scene and r.get("infer")]
        x = fig.add_subplot(gs[:, -1])
        an = np.array([r["anchor"] for r in s])
        Rm = rot(-an[0, 2] + np.pi / 2)                          # start heading up
        for r in s[::5]:
            p = (np.array(r["poses"])[:, :2] @ rot(r["anchor"][2]).T + r["anchor"][:2] - an[0, :2]) @ Rm.T
            x.plot(p[:, 0], p[:, 1], "-", color=plt.cm.viridis(r["k"] / max(1, s[-1]["k"])), lw=0.8, alpha=0.9)
        d = (an[:, :2] - an[0, :2]) @ Rm.T
        x.plot(d[:, 0], d[:, 1], "k-", lw=1.8)
        x.plot(d[ks, 0] if max(ks) < len(d) else [], d[ks, 1] if max(ks) < len(d) else [], "r.", ms=6)
        x.set_aspect("equal", "datalim"), x.tick_params(labelsize=6)
        x.set_title("driven path (black), 4 s plans every 0.5 s\n(dark = first), red = the rows; m, start heading up", fontsize=7)
        for x in fig.axes[:-1]:
            x.set_xticks([]), x.set_yticks([])
        r = R.get(scene)
        fig.suptitle(f"{scene}: score {r['score']:.2f} {why(r)}" if r else scene, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{a.out}_{scene[7:15]}.jpg", dpi=110)
        plt.close(fig)
        print(f"{a.out}_{scene[7:15]}.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--ref", required=True), p.add_argument("--out", default="")
    p = sub.add_parser("frames")
    p.add_argument("--run", required=True), p.add_argument("--out", required=True)
    p.add_argument("--sessions", type=int, nargs="+", default=[0, 1, 2]), p.add_argument("--at", type=int, nargs="+", default=[20, 60, 100, 140, 180])
    a = ap.parse_args()
    {"table": cmd_table, "frames": cmd_frames}[a.cmd](a)
