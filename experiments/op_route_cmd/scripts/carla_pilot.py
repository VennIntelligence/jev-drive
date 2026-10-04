"""CARLA counterfactual route pairs, stage 3 pilot (CPU only, no server): for a few recorded op_img_cmd junction approaches
(train split, tag d10/d20) build the clean route polyline of EVERY exit of the approach road from the map, in the
sample's t0 rear-axle frame, and draw a check figure. The image is exit-independent: one approach frame, k polylines =
one counterfactual group. Frames are the existing op_img_cmd renders (P4 3-camera rig, camera 1.806 m), not yet the
openpilot rig at 1.22 m (no card was free for a re-render).

  build (envs/carla):    carla_pilot.py build --topo <carla_topo run dir>  -> $DATA_DIR/runs/op_route_cmd/carla_pilot/pilot.pkl
  fig   (envs/openpilot): carla_pilot.py fig   -> same dir: carla_pilot.png (-> experiments/op_route_cmd/figs/), pilot_samples.json

Per polyline (the sample schema proposed for stage 3): id, group (counterfactual group = one approach pose), poly (16, 2)
ego frame x fwd / y left at 0, 10, ..., 150 m along the path from the ego's nearest path point, mask (16,), cls, angle
(deg, left +), index_from_left, n_exits, dist (m, rear axle to the connector start along the approach), legal (the
ego's lane has the connector) / lane_change, exit (road, dir), conn (road, lane).
"""
import argparse, json, math, os, pickle, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
DATA = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data")))
IMG = DATA / "runs" / "op_img_cmd" / "carla"
OUT = DATA / "runs" / "op_route_cmd" / "carla_pilot"
REAR_AXLE_X = -1.388633220          # p4_carla_agent.REAR_AXLE_X
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


def build(a):
    import carla_topo as CT
    import pair_inv_carla as PI
    G = pickle.load(open(IMG / "geom_rec.pkl", "rb"))
    split = json.loads((IMG / "split.json").read_text())
    T = {t["route"]: t for t in json.load(open(REPO / "experiments" / "op_common_cause" / "results" / "carla_traversals.json"))}
    topo = {}

    def exits(town):
        if town not in topo:
            topo[town] = json.load(open(Path(a.topo) / f"topo_{town}.json"))
        return topo[town]

    # candidates: train-split approach frames, one per route, with the approach road's exit count
    cand = []
    for s in G:
        if split.get(s["token"]) != "train" or s["tag"] not in ("d10", "d20"):
            continue
        t = T[s["log"]]
        k = "%d:%d:%d" % (t["junction"], t["entry"][0], int(t["entry"][1] > 0))
        ex = exits(t["town"])["exits"].get(k, [])
        cand.append((s, t, ex))
    pick, towns = [], set()
    for want in (4, 3, 3, 3, 2):
        c = [x for x in cand if len(x[2]) == want and x[1]["town"] not in towns and x[0]["log"] not in {p[0]["log"] for p in pick}]
        c.sort(key=lambda x: (x[0]["tag"] != "d20", x[0]["token"]))
        if c:
            pick.append(c[0])
            towns.add(c[0][1]["town"])
    pick = pick[: a.n]
    out = []
    for g, (s, t, ex) in enumerate(pick):
        town = t["town"]
        m = PI.get_map(town)
        adir = next(p for p in Path(s["files"][-1][0]).parents if (p / "pose.jsonl").exists())
        bf = {p["frame"]: p for p in map(json.loads, open(adir / "pose.jsonl"))}
        p0 = bf[s["frame"]]
        yaw = -math.radians(p0["yaw"])
        ego = np.array([p0["x"], -p0["y"]]) + REAR_AXLE_X * np.array([math.cos(yaw), math.sin(yaw)])
        lane = t["entry"][1]
        polys = []
        for e in ex:
            c = min(e["conns"], key=lambda c: (abs(c["inc_lane"] - lane), c["out_lane"]))
            P, _ = CT.walk_path(m, town, c["conn"], s["dist"] + 25.0, s["dist"] + 175.0)
            q, mask = CT.poly_ego(P, ego, yaw)
            cs, sn = math.cos(yaw), math.sin(yaw)
            d = P - ego
            dense = np.stack([cs * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + cs * d[:, 1]], -1).astype(np.float32)
            polys.append(dict(id=f"{s['token']}-x{e['index_from_left']}", group=s["token"], poly=q, mask=mask, dense=dense,
                              cls=e["cls"], angle=e["angle"], index_from_left=e["index_from_left"], n_exits=len(ex),
                              dist=float(s["dist"]), legal=c["inc_lane"] == lane, lane_change=c["inc_lane"] != lane,
                              exit=e["exit"], conn=c["conn"]))
        out.append(dict(token=s["token"], town=town, junction=t["junction"], taken=t["taken"], v=s["v"], dist=s["dist"],
                        frames=str(IMG / "frames" / f"{s['token']}.npz"), cam=s["cam"], polys=polys))
        print(s["token"], town, t["junction"], "exits", [(p["cls"], p["index_from_left"], round(p["angle"]), p["legal"]) for p in polys])
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / "pilot.pkl", "wb"))


def fig(a):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "experiments" / "op_img_cmd" / "scripts"))
    import img_carla_check as C
    import img_overlay as O
    P = pickle.load(open(OUT / "pilot.pkl", "rb"))
    fig, axs = plt.subplots(len(P), 3, figsize=(15, 3.3 * len(P)), gridspec_kw=dict(width_ratios=[0.8, 2, 2]))
    axs = np.atleast_2d(axs)
    rows = []
    for r, s in enumerate(P):
        fr = np.load(s["frames"])["frames"][-1]                 # t0 packed (2, 6, 128, 256)
        ax = axs[r, 0]
        for k, p in enumerate(s["polys"]):
            col = COLORS[k % len(COLORS)]
            ax.plot(-p["dense"][:, 1], p["dense"][:, 0], "-", color=col, lw=1, alpha=0.5)
            q = p["poly"][p["mask"]]
            ax.plot(-q[:, 1], q[:, 0], "o-", color=col, ms=3, lw=1.5,
                    label=f"#{p['index_from_left']} {p['cls']} {p['angle']:+.0f}°" + ("" if p["legal"] else " (lane chg)"))
        ax.plot(0, 0, "k^", ms=8)
        ax.set_xlim(-80, 80), ax.set_ylim(-10, 155), ax.set_aspect("equal"), ax.grid(alpha=0.3)
        ax.set_title(f"{s['town']} J{s['junction']} d={s['dist']:.0f} m v={s['v']:.1f} m/s", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=6)
        lay = [(O.ribbon(p["dense"][p["dense"][:, 0] > 0.5], 0.12), tuple(int(COLORS[k % len(COLORS)][i:i + 2], 16) for i in (1, 3, 5)), 0.9, 0)
               for k, p in enumerate(s["polys"])]
        drawn = O.draw(fr, lay, np.zeros(3), np.asarray(s["cam"], float))
        axs[r, 1].imshow(C.rgb(fr)[1]), axs[r, 1].set_title("t0 wide model frame (the input; exit-independent)", fontsize=8)
        axs[r, 2].imshow(C.rgb(drawn)[1]), axs[r, 2].set_title("check only: exit polylines projected (never in the input)", fontsize=8)
        for ax in axs[r, 1:]:
            ax.axis("off")
        rows.append(dict(token=s["token"], town=s["town"], junction=s["junction"], taken=s["taken"], dist=round(s["dist"], 1),
                         polys=[dict(id=p["id"], cls=p["cls"], angle=p["angle"], index_from_left=p["index_from_left"],
                                     n_exits=p["n_exits"], legal=bool(p["legal"]), exit=p["exit"],
                                     poly=np.round(p["poly"], 2).tolist(), mask=p["mask"].astype(int).tolist()) for p in s["polys"]]))
    fig.tight_layout()
    dst = OUT / "carla_pilot.png"         # copied to experiments/op_route_cmd/figs/ on the Mac
    fig.savefig(dst, dpi=90)
    json.dump(rows, open(OUT / "pilot_samples.json", "w"), indent=0)
    print(dst, os.path.getsize(dst) // 1024, "KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "fig"))
    ap.add_argument("--topo", default="")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    build(a) if a.mode == "build" else fig(a)
