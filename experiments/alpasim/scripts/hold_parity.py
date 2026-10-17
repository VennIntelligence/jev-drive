"""JEV_HOLD image parity: the served plans and latch events of a submission image (the containerised smoke on a scene list) against a box
read of the same windows with the same hold (experiments/alpasim/results/image_rsg_leash3.md). Per window: the latch at the handoff call
(state, call time, measured speed; for a stop<m> latch also its deceleration and leash), the per-decision hold state and cut flag, the served poses (hold.jsonl's `poses` when cut, else
drive.jsonl's model plan) as the max xy distance over the 8 poses, the distance from the latch point, and the scene score.
  python hold_parity.py --box <dir with hold.jsonl drive.jsonl results-summary*.json (any depth)> --image <smoke out dir>
                        --scenes <list> [--strata <s1,s2,...> in list order] [--out <md>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(paths, kind=None):
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if kind is None or r.get("kind") == kind:
                yield r


def side(root: Path, scenes: set) -> dict:
    hold = [r for r in rows(sorted(root.rglob("hold.jsonl"))) if r["scene"] in scenes]
    drive = [r for r in rows(sorted(root.rglob("drive.jsonl")), "drive") if r["scene"] in scenes]
    score = {}
    for f in sorted(root.rglob("results-summary.json")):
        for r in json.load(open(f))["rollouts"]:
            if r["clipgt_id"] in scenes:
                score[r["clipgt_id"]] = r.get("score")
    out = {}
    for s in scenes:
        H = sorted((r for r in hold if r["scene"] == s), key=lambda r: r["t0"])
        D = {r["t0"]: r for r in drive if r["scene"] == s}
        assert len({r["session"] for r in H}) <= 1, f"{root}: {s} has several sessions"
        served = {r["t0"]: np.asarray(r["poses"] if r.get("cut") else D[r["t0"]]["poses"], float) for r in H if r["t0"] in D}
        out[s] = dict(hold=H, served=served, score=score.get(s))
    return out


def latch(H):
    r = next((r for r in H if r["state"] != "pre"), None)          # stop<m>: a stop latch also shows its deceleration and leash
    return (r["state"] + (f" stop a {r['stop_a']} cap {r['cap']}" if "cap" in r else ""), r["t_s"], r["v"]) if r else ("-", None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--strata", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    ids = Path(a.scenes).read_text().split()
    strata = dict(zip(ids, a.strata.split(","))) if a.strata else {}
    B, I = side(Path(a.box), set(ids)), side(Path(a.image), set(ids))
    L = ["| window | stratum | latch box / image (state, t s, v m/s) | decisions box / image | state equal | cut equal | served plan diff k0 / median / max (m) | max dist from latch box / image (m) | score box / image |",
         "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    tot = dict(win=0, latch=0, dec=0, state=0, cut=0, d0=[], dall=[])
    for s in ids:
        b, i = B[s], I[s]
        lb, li = latch(b["hold"]), latch(i["hold"])
        hb, hi = {r["t0"]: r for r in b["hold"]}, {r["t0"]: r for r in i["hold"]}
        ts = sorted(set(hb) & set(hi))
        st = sum(hb[t]["state"] == hi[t]["state"] for t in ts)
        cu = sum(bool(hb[t].get("cut")) == bool(hi[t].get("cut")) for t in ts)
        dif = [float(np.hypot(*(b["served"][t][:, :2] - i["served"][t][:, :2]).T).max()) for t in ts if t in b["served"] and t in i["served"]]
        mx = lambda H: max((r.get("d0", 0.0) for r in H), default=0.0)  # noqa: E731
        f = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
        L.append(f"| {s} | {strata.get(s, '')} | {lb[0]} {lb[1]} {lb[2]} / {li[0]} {li[1]} {li[2]} | {len(hb)} / {len(hi)} | {st} / {len(ts)} | "
                 f"{cu} / {len(ts)} | {dif[0]:.3f} / {np.median(dif):.3f} / {max(dif):.3f} | {mx(b['hold']):.2f} / {mx(i['hold']):.2f} | "
                 f"{f(b['score'])} / {f(i['score'])} |")
        tot["win"] += 1; tot["latch"] += lb[0] == li[0]; tot["dec"] += len(ts); tot["state"] += st; tot["cut"] += cu   # noqa: E702
        tot["d0"].append(dif[0]); tot["dall"] += dif
    sb = [B[s]["score"] for s in ids if B[s]["score"] is not None and I[s]["score"] is not None]
    si = [I[s]["score"] for s in ids if B[s]["score"] is not None and I[s]["score"] is not None]
    held = sum(latch(B[s]["hold"])[0].startswith("held") for s in ids)
    L += ["", f"Windows {tot['win']}; latch state equal {tot['latch']} / {tot['win']} (box held {held}); decisions compared {tot['dec']}: hold state equal "
          f"{tot['state']}, cut flag equal {tot['cut']}; served plan diff at decision 0 median {np.median(tot['d0']):.4f} m / max {max(tot['d0']):.4f} m, "
          f"all decisions median {np.median(tot['dall']):.4f} m / p90 {np.percentile(tot['dall'], 90):.4f} m / max {max(tot['dall']):.4f} m; "
          f"mean score box {np.mean(sb):.4f} / image {np.mean(si):.4f} (n {len(sb)}), zeros {sum(x == 0 for x in sb)} / {sum(x == 0 for x in si)}, "
          f"scores equal {sum(x == y for x, y in zip(sb, si))}, within 0.01 {sum(abs(x - y) <= 0.01 for x, y in zip(sb, si))}."]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
