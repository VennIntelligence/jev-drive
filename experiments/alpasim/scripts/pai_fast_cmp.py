#!/usr/bin/env python3
"""PAI_FAST check (results/pai_fast.md): cost and scores of pai_box.sh reads, plain python3 + numpy.
  pai_fast_cmp.py --ref <read root> <read root>...   per chunk run: scenes, wall (native_summary total_s), wall s / scene, core s / scene
                                                     (sum of the services' cpu_s), peak VRAM and RSS of the stack (max over usage.jsonl samples of
                                                     the sum over services); per read vs --ref on the scenes both scored: mean difference
                                                     (paired bootstrap over scenes, 10 000 draws, seed 0), identical scores, within 0.05, zero flips,
                                                     identical controller traces (the ego's whole driven path, byte for byte).
  Phases (from native_summary.json and the runtime log): start-up = wizard + services ready (s per job); per rollout, mean s: load = "Requesting
  available cameras" -> "Session STARTING" (usdz map, traffic, the renderer's scene open / ego masks), roll = the event loop's wall (render +
  driver + controller + physics), eval = loop end -> "Session COMPLETED" (scoring, eval video when on); idle = rollout slots (CONC x runtime)
  holding no rollout, per scene; tail = last COMPLETED -> runtime exit (aggregation). Wall / scene ~ (start-up + tail) / n + (load + roll +
  eval + idle) / CONC.
"""
import argparse, csv, glob, hashlib, json, os, re

import numpy as np


def runs(root):
    return sorted(glob.glob(os.path.join(root, "runs", "*", "sim")))


def cost(sim):
    s = json.load(open(os.path.join(sim, "native_summary.json")))
    n = s["times"]["rollouts"]
    vram = rss = 0.0
    for line in open(os.path.join(sim, "usage.jsonl")):
        u = json.loads(line)
        procs = [v for k, v in u.items() if isinstance(v, dict)]
        vram, rss = max(vram, sum(p["gpu_mib"] for p in procs) / 1024), max(rss, sum(p["rss_gib"] for p in procs))
    cpu = sum(p["cpu_s"] for p in s["peak"].values())
    return dict(n=n, wall=s["times"]["total_s"], cpu=cpu, vram=vram, rss=rss)


def phases(sim):
    s = json.load(open(os.path.join(sim, "native_summary.json")))["times"]
    conc = int(re.search(r"renderer.n_concurrent_rollouts=(\d+)", open(os.path.join(sim, "native_summary.json")).read()).group(1))
    t0, req, start, done, last = None, {}, {}, [], 0.0
    for line in open(os.path.join(sim, "native-logs", "runtime-0.log"), errors="replace"):
        m = re.match(r"(\d\d):(\d\d):(\d\d\.\d+) ", line)
        if not m:
            continue
        t = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        t0 = t if t0 is None else t0
        t += 86400 * (t < t0 - 3600)
        last = t
        if (g := re.search(r"Requesting available cameras for scene_id='([^']+)'", line)):
            req[g[1]] = t
        elif (g := re.search(r"Session STARTING: uuid=\S+ scene=(\S+)", line)):
            start[g[1]] = t
        elif (g := re.search(r"Session COMPLETED: uuid=\S+ scene=(\S+) simulated \S+ sim seconds in ([\d.]+) wall", line)):
            done.append((g[1], t, float(g[2])))
    load = [start[k] - req[k] for k in start if k in req]
    roll = [w for _, _, w in done]
    ev = [t - start[k] - w for k, t, w in done if k in start]
    n = len(done)
    busy = sum(load) + sum(roll) + sum(ev)
    tail = s["runtime_s"] - (max(t for _, t, _ in done) - t0) if done else float("nan")
    return dict(conc=conc, n=n, startup=s["services_ready_s"] + s["wizard_s"], load=np.mean(load), roll=np.mean(roll), eval=np.mean(ev),
                idle=(conc * s["runtime_s"] - busy) / max(n, 1), tail=tail, wall=s["total_s"] / max(n, 1))


def scores(root):
    out = {}
    for sim in runs(root):
        f = os.path.join(sim, "aggregate", "results-summary.json")
        if not os.path.exists(f):
            continue
        for r in json.load(open(f))["rollouts"]:
            c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
            h = hashlib.md5(open(c, "rb").read()).hexdigest() if os.path.exists(c) else None
            out[r["clipgt_id"]] = (float(r["score"] or 0), h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("roots", nargs="+")
    a = ap.parse_args()
    print("| read | chunk | scenes | wall s | wall s / scene | core s / scene | peak VRAM GB | peak RSS GiB |\n|:--|:--|--:|--:|--:|--:|--:|--:|")
    for root in [a.ref] + a.roots:
        for sim in runs(root):
            if os.path.exists(os.path.join(sim, "native_summary.json")):
                c = cost(sim)
                print(f"| {os.path.basename(root)} | {os.path.basename(os.path.dirname(sim))} | {c['n']} | {c['wall']:.0f} | {c['wall'] / c['n']:.1f} "
                      f"| {c['cpu'] / c['n']:.0f} | {c['vram']:.1f} | {c['rss']:.1f} |")
    print("\n| read | chunk | CONC | rollouts | start-up s | load s | roll s | eval s | idle slot-s / scene | tail s | wall s / scene |\n|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for root in [a.ref] + a.roots:
        for sim in runs(root):
            if os.path.exists(os.path.join(sim, "native_summary.json")):
                p = phases(sim)
                print(f"| {os.path.basename(root)} | {os.path.basename(os.path.dirname(sim))} | {p['conc']} | {p['n']} | {p['startup']:.0f} | {p['load']:.1f} "
                      f"| {p['roll']:.1f} | {p['eval']:.1f} | {p['idle']:.1f} | {p['tail']:.0f} | {p['wall']:.1f} |")
    ref = scores(a.ref)
    print("\n| read | scenes | ref mean | mean | diff (95 % CI) | identical score | within 0.05 | zeros ref / read | zero flips | identical trace |"
          "\n|:--|--:|--:|--:|:--|--:|--:|--:|--:|--:|")
    for root in a.roots:
        new = scores(root)
        k = sorted(set(ref) & set(new))
        if not k:
            continue
        r, x = np.array([ref[s][0] for s in k]), np.array([new[s][0] for s in k])
        d = x - r
        bs = np.random.default_rng(0).integers(0, len(k), (10000, len(k)))
        lo, hi = np.percentile(d[bs].mean(1), [2.5, 97.5])
        flips = int(((r == 0) != (x == 0)).sum())
        same = sum(ref[s][1] is not None and ref[s][1] == new[s][1] for s in k)
        print(f"| {os.path.basename(root)} | {len(k)} | {r.mean():.4f} | {x.mean():.4f} | {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {int((d == 0).sum())} "
              f"| {int((abs(d) <= 0.05).sum())} | {int((r == 0).sum())} / {int((x == 0).sum())} | {flips} | {same} |")
        for s in k:
            if d[k.index(s)] != 0:
                print(f"  {s}: {ref[s][0]:.4f} -> {new[s][0]:.4f}")


if __name__ == "__main__":
    main()
