"""HUGSIM 64 section of the op_parity gap page (results/gap/index.html): where P2 (seed mean of P2-F-s0 / s1) loses to WA-JEPA,
per scenario, with P0 (shipped weights, parity path) as the third arm. Existing outputs only: no model, no simulator, CPU only.

Inputs: results/hugsim_full/extract.csv (pp_hugsim_report.py extract full: runner rows + spin analysis of every arm-preset run),
experiments/hugsim/results/hugsim-exam/scored_wajepa.csv + wajepa_ref/wajepa_extract.csv (WA-JEPA, one run, used for both presets).
End class as pp_hugsim_report.py report_full: spin (heading error >= 60 deg vs the recorded route) takes precedence, else the runner's
end (max_steps -> stuck, fg_collision -> fg_coll, bg_collision -> bg_coll, off_route, complete).

  tables  (Mac or box)          -> <out>/fragment.html, <out>/selection.json
  gifs    (box, envs/hugsim)    -> <out>/clips/<scenario>.gif for the cases in selection.json
  all     both
    DATA_DIR=... $DATA_DIR/envs/hugsim/bin/python experiments/op_parity/scripts/pp_gap_hugsim.py all <out>
env REPO overrides the repository root (to run a copy of this file from outside the checkout).

GIF panels per frame: P2 = the openpilot road and wide model frames rebuilt from the logged front cameras (video.mp4, 2x3 grid) with
the agent's own jevdrive.hugsim_zs.OpenpilotFrames (the model input of the current step); WA-JEPA = the input mosaic its client saved
(viz/input: L0 F0 / R0 B0 as rendered); BEV (scene frame, from infos.pkl / plan logs, ground and obstacle points of the scene's
ground.ply / scene.ply) with both egos, their paths, the plan each sent to the simulator, the scripted actors and the recorded route.
"""
import html
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("REPO") or Path(__file__).resolve().parents[3])
HR = REPO / "experiments/hugsim/results"
FULL = REPO / "experiments/op_parity/results/hugsim_full"
PRESETS = ("spec", "exam")
SEEDS = ("P2-F-s0", "P2-F-s1")
CLASSES = ["complete", "fg_coll", "bg_coll", "off_route", "stuck", "spin"]
CLEAR = 0.2              # GIF candidates: WA-JEPA HD - P2 seed-mean HD >= CLEAR
N_GIF = 4                # at most one GIF per P2 loss class, the classes with the largest loss sums first
MIN_SHARE = 0.1          # ... that carry at least this share of P2's losses
CAMYAML = "third_party/HUGSIM/configs/sim/{}_camera.yaml"


# ---------------------------------------------------------------------------------------------------------------- tables
def cls_of(end, spin):
    c = {"max_steps": "stuck", "fg_collision": "fg_coll", "bg_collision": "bg_coll"}.get(str(end), str(end))
    return "spin" if bool(spin) else c


def load():
    import pandas as pd
    W = pd.read_csv(HR / "hugsim-exam/scored_wajepa.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    WX = pd.read_csv(HR / "wajepa_ref/wajepa_extract.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    W = W.join(WX[["spin"]])
    W["cls"] = [cls_of(e, s) for e, s in zip(W.end, W.spin)]
    ex = pd.read_csv(FULL / "extract.csv")
    ex["cls"] = [cls_of(e, s) for e, s in zip(ex.end, ex.spin)]
    arms = {pr: {t: ex[(ex.preset == pr) & (ex.arm == t)].set_index("scenario").reindex(W.index) for t in ("P0",) + SEEDS} for pr in PRESETS}
    return W, arms


def per_scenario(W, A):
    import pandas as pd
    s0, s1, p0 = A[SEEDS[0]], A[SEEDS[1]], A["P0"]
    df = pd.DataFrame({"dataset": W.dataset, "difficulty": W.difficulty, "wa_hd": W.hdscore, "wa_rc": W.rc, "wa_cls": W.cls,
                       "p2_hd": (s0.hdscore + s1.hdscore) / 2, "p2_rc": (s0.rc + s1.rc) / 2, "s0_hd": s0.hdscore, "s1_hd": s1.hdscore,
                       "s0_cls": s0.cls, "s1_cls": s1.cls, "s0_steps": s0.steps, "s1_steps": s1.steps, "s0_dir": s0.run_dir, "s1_dir": s1.run_dir,
                       "p0_hd": p0.hdscore, "p0_rc": p0.rc, "p0_cls": p0.cls, "wa_dir": W.run_dir, "wa_steps": W.steps, "scene": W.scene})
    df["d"] = df.p2_hd - df.wa_hd
    return df.sort_values("d", kind="stable")


def attribution(df):
    """Per seed s and scenario: loss_s = max(0, WA - P2_s) goes to P2_s's end class, gain_s = max(0, P2_s - WA) to WA-JEPA's end
    class; each seed weighs 1/2. Sum over classes of (gains - losses) / 64 = P2 seed-mean HD - WA-JEPA HD exactly."""
    n = len(df)
    loss, gain = {c: [0.0, 0] for c in CLASSES}, {c: [0.0, 0] for c in CLASSES}
    for s in ("s0", "s1"):
        d = df[f"{s}_hd"] - df.wa_hd
        for sc in df.index:
            if d[sc] < 0:
                loss[df.at[sc, f"{s}_cls"]][0] += -d[sc] / 2
                loss[df.at[sc, f"{s}_cls"]][1] += 1
            elif d[sc] > 0:
                gain[df.at[sc, "wa_cls"]][0] += d[sc] / 2
                gain[df.at[sc, "wa_cls"]][1] += 1
    return {c: (loss[c][0] / n, loss[c][1], gain[c][0] / n, gain[c][1]) for c in CLASSES}


def counts(W, A):
    rows = [("WA-JEPA (one run)", W.cls)] + [(t, A[t].cls) for t in ("P0",) + SEEDS]
    out = [(n, [int((c == k).sum()) for k in CLASSES]) for n, c in rows]
    out.append(("P2 seed mean", [(a + b) / 2 for a, b in zip(out[2][1], out[3][1])]))
    return out


def select(df, att):
    """Per P2 loss class with >= MIN_SHARE of the loss sum (largest first): (scenario, seed) pairs where that seed ended in the class
    and WA-JEPA beats both that seed and the P2 seed mean by >= CLEAR; pick the scenario at the (lower) median of the seed-mean loss
    (s0 when both seeds qualify)."""
    picks, used = [], set()
    tot = sum(v[0] for v in att.values())
    for c in sorted(CLASSES, key=lambda k: -att[k][0]):
        if att[c][0] < MIN_SHARE * tot or len(picks) >= N_GIF:
            continue
        ok = {s: (df[f"{s}_cls"] == c) & (df.wa_hd - df[f"{s}_hd"] >= CLEAR) for s in ("s0", "s1")}
        cand = df[(-df.d >= CLEAR) & (ok["s0"] | ok["s1"]) & ~df.index.isin(used)].sort_values("d", ascending=False)
        if not len(cand):
            continue
        sc = cand.index[(len(cand) - 1) // 2]
        r = cand.loc[sc]
        seed = "s0" if ok["s0"][sc] else "s1"
        used.add(sc)
        picks.append(dict(cls=c, scenario=sc, n_candidates=len(cand), seed=seed, tag=SEEDS[0 if seed == "s0" else 1], dataset=r.dataset, scene=r.scene,
                          p2_dir=r[f"{seed}_dir"], wa_dir=r.wa_dir, p2_hd=float(r[f"{seed}_hd"]), p2_mean=float(r.p2_hd), wa_hd=float(r.wa_hd),
                          p2_cls=c, wa_cls=r.wa_cls, p2_steps=int(r[f"{seed}_steps"]), wa_steps=int(r.wa_steps)))
    return picks


def fragment(out):
    W, arms = load()
    E = html.escape
    f3 = lambda x: "-" if x != x else f"{x:.3f}"  # noqa: E731
    short = lambda s: s.replace("scene-", "")  # noqa: E731
    H, picks = [], []
    H.append('<section id="hugsim64">\n<h2>HUGSIM 64: P2 vs WA-JEPA per scenario</h2>')
    tabs, atts = {}, {}
    for pr in PRESETS:
        tabs[pr] = per_scenario(W, arms[pr])
        atts[pr] = attribution(tabs[pr])
    spec, att = tabs["spec"], atts["spec"]
    tl = sum(v[0] for v in att.values())
    tg = sum(v[2] for v in att.values())
    top = sorted(CLASSES, key=lambda k: -att[k][0])
    picks = select(spec, att)
    H.append(
        f"<p>64 scenarios (derot_all64), one closed-loop run per arm and scenario, 400-step cap (0.25 s steps). P2 = seed mean of P2-F-s0 / P2-F-s1, P0 = "
        "shipped weights through the parity path, both under the <code>spec</code> preset (tree opctrl, openpilot lateral path) unless marked "
        "<code>exam</code> (tree fixed, PR #57 controller); WA-JEPA = its single run (GT route command and ego state), the same row for both presets. "
        "End class as <code>pp_hugsim_report.py</code>: <b>spin</b> (heading error &ge; 60&deg; vs the recorded route) takes precedence over the "
        "runner's end: <b>complete</b>, <b>fg_coll</b> (foreground actor), <b>bg_coll</b> (static background), <b>off_route</b> (far from the preset "
        "trajectory), <b>stuck</b> (400 steps without ending).</p>")
    H.append(
        "<p><b>What to look at.</b> "
        f"Under <code>spec</code> P2 is {-spec.d.mean():.3f} HD below WA-JEPA on the mean, but that is {tg:.3f} of gains minus {tl:.3f} of losses "
        f"(per-scenario HD, /64): the gap is the net of large offsetting per-scenario differences. "
        f"Of P2's losses, {att[top[0]][0] / tl:.0%} come from runs that end in <b>{top[0]}</b> and {att[top[1]][0] / tl:.0%} from <b>{top[1]}</b>"
        f" ({att[top[2]][0] / tl:.0%} {top[2]}); P2 never stays stuck. fg_coll losses are partly offset by scenarios where WA-JEPA ends in fg_coll "
        f"and P2 scores higher (net {att['fg_coll'][2] - att['fg_coll'][0]:+.3f}); bg_coll (P2 {(spec.s0_cls == 'bg_coll').sum()} / {(spec.s1_cls == 'bg_coll').sum()} per seed, WA-JEPA {int((W.cls == 'bg_coll').sum())}) "
        f"has no offset (net {att['bg_coll'][2] - att['bg_coll'][0]:+.3f}) and is the largest single net deficit. "
        "The table is sorted by P2 - WA-JEPA HD, largest loss first: read down the P2 class column against the WA-JEPA class column. "
        "Single runs: the rerun spread is about 0.2 HD per scenario, so single rows are noisy; the sums are the evidence.</p>")
    # gap attribution
    H.append("<h3>Which P2 failure types make the HD gap</h3>")
    H.append("<p>Attribution rule: for each P2 seed and scenario, a loss <i>max(0, HD<sub>WA-JEPA</sub> - HD<sub>P2 seed</sub>)</i> is booked to that "
             "seed's end class, and a gain <i>max(0, HD<sub>P2 seed</sub> - HD<sub>WA-JEPA</sub>)</i> to WA-JEPA's end class there; each seed weighs "
             "1/2 and sums are divided by 64, so (gains - losses) summed over classes equals the seed-mean HD difference exactly. n = scenario-seed "
             "pairs (out of 128).</p>")
    for pr in PRESETS:
        a, t = atts[pr], tabs[pr]
        H.append(f"<table>\n<caption>preset <code>{pr}</code>: P2 seed mean {t.p2_hd.mean():.3f}, WA-JEPA {t.wa_hd.mean():.3f}, "
                 f"difference {t.d.mean():+.3f}</caption>")
        H.append("<tr><th>class</th><th>P2 loss (by P2 class)</th><th>n</th><th>share of losses</th><th>P2 gain (by WA-JEPA class)</th><th>n</th>"
                 "<th>net</th></tr>")
        lt = sum(v[0] for v in a.values())
        for c in CLASSES:
            lo, nl, ga, ng = a[c]
            if nl or ng:
                H.append(f"<tr><td>{c}</td><td>{-lo:+.3f}</td><td>{nl}</td><td>{lo / lt:.0%}</td><td>{ga:+.3f}</td><td>{ng}</td><td>{ga - lo:+.3f}</td></tr>")
        gt = sum(v[2] for v in a.values())
        H.append(f"<tr><th>total</th><th>{-lt:+.3f}</th><th>{sum(v[1] for v in a.values())}</th><th>100%</th><th>{gt:+.3f}</th>"
                 f"<th>{sum(v[3] for v in a.values())}</th><th>{gt - lt:+.3f}</th></tr>\n</table>")
    # class counts
    H.append("<h3>End classes per arm</h3>")
    for pr in PRESETS:
        H.append(f"<table>\n<caption>preset <code>{pr}</code> (WA-JEPA: its single run)</caption>\n<tr><th>arm</th>"
                 + "".join(f"<th>{c}</th>" for c in CLASSES) + "</tr>")
        for n, v in counts(W, arms[pr]):
            H.append(f"<tr><td>{n}</td>" + "".join(f"<td>{x:g}</td>" for x in v) + "</tr>")
        H.append("</table>")
    # per-scenario tables
    def cls2(r):
        return r.s0_cls if r.s0_cls == r.s1_cls else f"{r.s0_cls} / {r.s1_cls}"

    def table(t, pr):
        L = [f"<table>\n<caption>preset <code>{pr}</code>, sorted by P2 - WA-JEPA HD (largest loss first). HD = HD-Score, RC = route completion; "
             "P2 class shows s0 / s1 when the seeds differ.</caption>",
             "<tr><th>#</th><th>scenario</th><th>dataset</th><th>difficulty</th><th>P2 - WA-JEPA HD</th><th>P2 HD (s0, s1)</th><th>P2 RC</th>"
             "<th>P2 end</th><th>WA-JEPA HD</th><th>WA-JEPA RC</th><th>WA-JEPA end</th><th>P0 HD</th><th>P0 RC</th><th>P0 end</th></tr>"]
        for k, (sc, r) in enumerate(t.iterrows(), 1):
            col = "#c0392b" if r.d < -0.02 else "#1e8449" if r.d > 0.02 else "inherit"
            L.append(f"<tr><td>{k}</td><td>{E(short(sc))}</td><td>{r.dataset}</td><td>{r.difficulty}</td>"
                     f"<td style=\"color:{col}\">{r.d:+.3f}</td><td>{f3(r.p2_hd)} ({f3(r.s0_hd)}, {f3(r.s1_hd)})</td><td>{f3(r.p2_rc)}</td><td>{cls2(r)}</td>"
                     f"<td>{f3(r.wa_hd)}</td><td>{f3(r.wa_rc)}</td><td>{r.wa_cls}</td><td>{f3(r.p0_hd)}</td><td>{f3(r.p0_rc)}</td><td>{r.p0_cls}</td></tr>")
        L.append("</table>")
        return L
    H.append("<h3>Per scenario</h3>")
    H += table(spec, "spec")
    H.append("<details>\n<summary>The same under preset <code>exam</code></summary>")
    H += table(tabs["exam"], "exam")
    H.append("</details>")
    # clips
    H.append("<h3>Clips: typical losses</h3>")
    H.append(f"<p>Selection rule (preset <code>spec</code>): for each P2 loss class, in order of its loss sum, the scenarios where WA-JEPA beats the P2 "
             f"seed mean by &ge; {CLEAR} HD and at least one P2 seed ended in that class; the clip is the scenario at the (lower) median of that "
             "loss, i.e. typical, not worst, and shows the seed that ended in the class (s0 if both). "
             "Panels: <b>P2 model view</b> = the openpilot road (top) and wide (bottom) model frames of the current step, rebuilt offline from "
             "the logged front cameras with the agent's own frame builder (<code>jevdrive.hugsim_zs.OpenpilotFrames</code>); <b>WA-JEPA model view</b> = "
             "the camera mosaic its client saved (front-left, front / front-right, back); <b>BEV</b> = scene frame, both egos (P2 orange, "
             "WA-JEPA blue) with their driven paths and the 3 s plan each sent to the simulator, scripted actors (red), the recorded route "
             "(dashed green), ground (light) and obstacle points (dark) from the scene's point clouds. One GIF frame per 0.25 s step, "
             "played at 8 fps (2x real time); a run that has ended is frozen and marked. The two runs are separate episodes from the same "
             "start, aligned by step.</p>")
    for p in picks:
        cap = (f"{E(short(p['scenario']))} ({p['dataset']}): P2 end class <b>{p['cls']}</b>, the median of {p['n_candidates']} candidates. "
               f"{p['tag']} HD {p['p2_hd']:.3f} (seed mean {p['p2_mean']:.3f}, {p['p2_steps']} steps) vs WA-JEPA {p['wa_hd']:.3f} "
               f"({p['wa_cls']}, {p['wa_steps']} steps).")
        H.append(f"<figure>\n<img src=\"clips/{short(p['scenario'])}.gif\" alt=\"{E(short(p['scenario']))}\" loading=\"lazy\">\n"
                 f"<figcaption>{cap}</figcaption>\n</figure>")
    H.append("<p>Sources: <code>experiments/op_parity/results/hugsim_full/extract.csv</code>, "
             "<code>experiments/hugsim/results/hugsim-exam/scored_wajepa.csv</code>; generator "
             "<code>experiments/op_parity/scripts/pp_gap_hugsim.py</code>.</p>\n</section>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fragment.html").write_text("\n".join(H) + "\n")
    json.dump(picks, open(out / "selection.json", "w"), indent=1)
    for pr in PRESETS:
        a = atts[pr]
        print(pr, "P2-WA", f"{tabs[pr].d.mean():+.3f}", {c: (round(-v[0], 3), v[1], round(v[2], 3), v[3]) for c, v in a.items() if v[1] or v[3]})
    for p in picks:
        print("pick", p["cls"], p["scenario"], p["tag"], round(p["p2_hd"], 3), round(p["wa_hd"], 3), p["n_candidates"])


# ---------------------------------------------------------------------------------------------------------------- gifs
FPS, MAX_FRAMES, TAIL = 8, 64, 24       # frames: steps 0 .. P2's end + TAIL (capped at the longer run), every k-th step to fit MAX_FRAMES
CW, BEV_W, ROW_H = 208, 264, 208        # model-view column width, BEV size, P2 panel height (road over wide)
P2C, WAC, ACT = (255, 140, 0), (40, 120, 255), (220, 40, 40)


def yuv_rgb(packed):
    """(6, 128, 256) BT.601 limited-range packed YUV420 (OpenpilotFrames.pack) -> (256, 512, 3) uint8 RGB."""
    Y = np.empty((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = packed[:4]
    U, V = (np.repeat(np.repeat(packed[i].astype(np.float32), 2, 0), 2, 1) - 128 for i in (4, 5))
    Y = 1.164 * (Y - 16)
    return np.clip(np.stack([Y + 1.596 * V, Y - 0.392 * U - 0.813 * V, Y + 2.017 * U], -1), 0, 255).astype(np.uint8)


def ego_track(infos):
    from scipy.spatial.transform import Rotation
    pos = np.array([[i["ego_pos"][0], i["ego_pos"][2]] for i in infos], np.float64)
    th = np.array([np.arctan2(*Rotation.from_euler("XYZ", i["ego_rot"]).as_matrix()[[0, 2], 2]) for i in infos])
    v = np.array([float(np.ravel(i["ego_velo"])[0]) for i in infos])
    return pos, th, v                                                    # world x right / z forward, heading right-positive


def plan_world(plan, p, th):
    """plan (N, 2) x right, y forward at the ego -> world (x, z)."""
    plan = np.asarray(plan, np.float64).reshape(-1, 2)
    f, r = np.array([np.sin(th), np.cos(th)]), np.array([np.cos(th), -np.sin(th)])
    return p + plan[:, 1:2] * f + plan[:, 0:1] * r


def box_world(b):
    """scene box [x fwd, y left, z, w, l, h, yaw left+] -> 4 world (x, z) corners."""
    x, y, w, l, yaw = float(b[0]), float(b[1]), float(b[3]), float(b[4]), float(b[6])
    f, lf = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    c = np.array([x, y]) + np.array([s * l / 2 * f + t * w / 2 * lf for s, t in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return np.stack([-c[:, 1], c[:, 0]], -1)


def bev_background(d, lo, hi, px, route_y):
    """Ground (light) / obstacle points within the ego height band (dark) rasterised at px metres, as export_review_cases.topdown."""
    import open3d as o3d
    shape = (int(np.ceil((hi[1] - lo[1]) / px)), int(np.ceil((hi[0] - lo[0]) / px)))      # rows = z (flipped), cols = x
    img = np.full(shape + (3,), 250, np.uint8)
    for name, colr, need in (("ground", (215, 215, 215), 1), ("scene", (110, 110, 110), 3)):
        p = np.asarray(o3d.io.read_point_cloud(str(d / f"{name}.ply")).points)
        p = p[(p[:, 0] >= lo[0]) & (p[:, 0] < hi[0]) & (p[:, 2] >= lo[1]) & (p[:, 2] < hi[1])]
        if name == "scene":                                               # camera height band: y is down, ego box 1.5 m below the camera
            p = p[(p[:, 1] > route_y) & (p[:, 1] < route_y + 1.5)]
        h = np.zeros(shape, np.int32)
        np.add.at(h, (((hi[1] - p[:, 2]) / px).astype(int).clip(0, shape[0] - 1), ((p[:, 0] - lo[0]) / px).astype(int).clip(0, shape[1] - 1)), 1)
        img[h >= need] = colr
    return img


def render_case(args):
    import cv2
    from PIL import Image
    sys.path.insert(0, str(REPO))
    from jevdrive import hugsim_zs as Z
    p, out = args
    dp, dw = Path(p["p2_dir"]), Path(p["wa_dir"])
    ip, iw = pickle.load(open(dp / "infos.pkl", "rb")), pickle.load(open(dw / "infos.pkl", "rb"))
    rp = [json.loads(x) for x in open(dp / "zs_steps.jsonl")][1:]
    rw = pickle.load(open(dw / "plan_log.pkl", "rb"))
    (pp, thp, vp), (pw, thw, vw) = ego_track(ip), ego_track(iw)
    np_, nw = len(ip), len(iw)
    last = min(max(np_, nw), np_ + TAIL) - 1
    stride = int(np.ceil((last + 1) / MAX_FRAMES))
    steps = list(range(0, last + 1, stride))
    if steps[-1] != last:
        steps.append(last)
    # P2 model frames
    cal = Z.calibs(ip[0]["cam_params"], Z.rect_matrix(str(Path(os.environ["DATA_DIR"]) / CAMYAML.format(p["dataset"]))))
    op = Z.OpenpilotFrames(cal)
    cap, vid = cv2.VideoCapture(str(dp / "video.mp4")), []
    while len(vid) <= min(last, np_ - 1):
        ok, f = cap.read()
        if not ok:
            break
        vid.append(f)
    def p2_view(k):
        f = cv2.cvtColor(vid[min(k, len(vid) - 1)], cv2.COLOR_BGR2RGB)
        h, w = f.shape[0] // 2, f.shape[1] // 3
        pk = op.pack({"CAM_FRONT_LEFT": f[:h, :w], "CAM_FRONT": f[:h, w:2 * w], "CAM_FRONT_RIGHT": f[:h, 2 * w:]})
        return np.concatenate([cv2.resize(yuv_rgb(pk[m]), (CW, CW // 2), interpolation=cv2.INTER_AREA) for m in (0, 1)])
    def wa_view(k):
        f = cv2.imread(str(dw / "viz/input" / f"{min(k, nw - 1):04d}.jpg"))
        return cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (CW, CW * f.shape[0] // f.shape[1]), interpolation=cv2.INTER_AREA)
    # BEV extent: both paths over the shown steps, the actors there, 12 m margin, square
    routes = json.load(open(Path(os.environ["DATA_DIR"]) / "runs/op_parity/hugsim/routes.json"))
    route = np.asarray(routes[p["scene"]]["xz"]) if p["scene"] in routes else np.zeros((0, 2))
    pts = [pp[:min(last, np_ - 1) + 1], pw[:min(last, nw - 1) + 1]]
    objs = [np.concatenate([box_world(b) for b in ip[k]["obj_boxes"]]) for k in range(min(last, np_ - 1) + 1) if len(ip[k]["obj_boxes"])]
    near = lambda q: q[np.linalg.norm(q - pp[0], axis=1) < 60] if len(q) else q  # noqa: E731
    allp = np.concatenate(pts + [near(o) for o in objs[:1]])
    c, half = (allp.min(0) + allp.max(0)) / 2, max(np.ptp(allp, 0).max() / 2 + 12, 25)
    lo, hi = c - half, c + half
    px = 2 * half / BEV_W
    bg = bev_background(dp, lo, hi, px, 0.0)
    bg = cv2.resize(bg, (BEV_W, BEV_W), interpolation=cv2.INTER_NEAREST)
    uv = lambda q: np.stack([(q[:, 0] - lo[0]) / px, (hi[1] - q[:, 1]) / px], -1).round().astype(np.int32)  # noqa: E731
    rt = route[(np.abs(route - c) < half).all(1)] if len(route) else route
    for a, b in zip(uv(rt)[:-1:2], uv(rt)[1::2]):
        cv2.line(bg, tuple(map(int, a)), tuple(map(int, b)), (40, 160, 60), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    TOP, BOT = 18, 6
    END_C = (210, 0, 0)
    frames = []
    for k in steps:
        canvas = np.full((TOP + max(ROW_H, BEV_W) + BOT, 2 * CW + BEV_W + 8, 3), 255, np.uint8)
        kp, kw = min(k, np_ - 1), min(k, nw - 1)
        a = p2_view(kp)
        canvas[TOP:TOP + a.shape[0], 0:CW] = a
        b = wa_view(kw)
        canvas[TOP:TOP + b.shape[0], CW + 4:2 * CW + 4] = b
        ev = bg.copy()
        for o in ip[kp]["obj_boxes"]:
            cv2.fillPoly(ev, [uv(box_world(o))], ACT, cv2.LINE_AA)
        for pos, th, n, kk, col, plan in ((pw, thw, nw, kw, WAC, rw[kw]["traj_lidar"] if kw < len(rw) else None),
                                          (pp, thp, np_, kp, P2C, rp[kp]["plan"] if kp < len(rp) else None)):
            cv2.polylines(ev, [uv(pos[:kk + 1])], False, col, 1, cv2.LINE_AA)
            if plan is not None and k < n:
                cv2.polylines(ev, [uv(np.concatenate([pos[kk:kk + 1], plan_world(plan, pos[kk], th[kk])]))], False, col, 2, cv2.LINE_AA)
            eb = np.array([pos[kk, 1], -pos[kk, 0], 0, 1.8, 4.2, 0, -th[kk]])
            cv2.fillPoly(ev, [uv(box_world(eb))], col, cv2.LINE_AA)
        x0 = 2 * CW + 8
        for j, (txt, col) in enumerate((("BEV:", (0, 0, 0)), ("orange = P2, blue = WA-JEPA", (0, 0, 0)), ("thin = driven, thick = plan sent", (0, 0, 0)),
                                        ("red = actors, dashed green = route", (0, 0, 0)), ("grey = ground / obstacle points", (0, 0, 0)))):
            cv2.putText(canvas, txt, (CW + 6, TOP + b.shape[0] + 44 + 13 * j), font, 0.36, (70, 70, 70), 1, cv2.LINE_AA)
        canvas[TOP:TOP + BEV_W, x0:x0 + BEV_W] = ev
        cv2.putText(canvas, f"{p['scenario'].replace('scene-', '')}   step {k}  t {k * 0.25:.1f} s", (4, 13), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{(hi[0] - lo[0]):.0f} m", (x0 + BEV_W - 46, 13), font, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
        for x, name, n, kk, v, cl, col in ((0, f"P2 {p['seed']} model view", np_, kp, vp, p["p2_cls"], P2C),
                                           (CW + 4, "WA-JEPA model view", nw, kw, vw, p["wa_cls"], WAC)):
            yb = TOP + (ROW_H if x == 0 else b.shape[0]) + 12
            cv2.putText(canvas, name, (x + 2, yb), font, 0.38, col, 1, cv2.LINE_AA)
            msg, mc = (f"ENDED {cl} @ {n - 1}", END_C) if k >= n - 1 and k > 0 else (f"v {v[kk]:.1f} m/s", (0, 0, 0))
            cv2.putText(canvas, msg, (x + 2, yb + 14), font, 0.38, mc, 1, cv2.LINE_AA)
            if k >= n - 1 and k > 0:
                cv2.rectangle(canvas, (x, TOP), (x + CW - 1, TOP + (ROW_H if x == 0 else b.shape[0]) - 1), END_C, 2)
        frames.append(Image.fromarray(canvas))
    W_ = frames[0].width                            # one palette for the clip: 4 frames + swatches of the drawing colours
    sw = np.concatenate([np.full((24, W_, 3), c, np.uint8) for c in (P2C, WAC, ACT, END_C, (40, 160, 60), (0, 0, 0), (255, 255, 255))])
    src = np.concatenate([np.asarray(frames[i]) for i in np.linspace(0, len(frames) - 1, 4).astype(int)] + [sw])
    pal = Image.fromarray(src).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    dur = [1000 // FPS] * len(q)
    dur[-1] = 1500
    gif = out / "clips" / f"{p['scenario'].replace('scene-', '')}.gif"
    q[0].save(gif, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True)
    return f"{gif.name}: {len(q)} frames (stride {stride}), {gif.stat().st_size / 1e6:.2f} MB"


def gifs(out):
    from multiprocessing import Pool
    sys.path.insert(0, str(REPO))
    from jevdrive.common import n_cpus
    picks = json.load(open(out / "selection.json"))
    (out / "clips").mkdir(parents=True, exist_ok=True)
    with Pool(min(len(picks), n_cpus())) as pool:
        for r in pool.imap_unordered(render_case, [(p, out) for p in picks]):
            print(r, flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[2] if len(sys.argv) > 2 else REPO / "tmp/gap_hugsim")
    if sys.argv[1] in ("tables", "all"):
        fragment(out)
    if sys.argv[1] in ("gifs", "all"):
        gifs(out)
