"""System 1 (openpilot plan) + System 2 (a discrete go / hold decision) late fusion on WOD-E2E val: what the decision is worth (privileged
oracle ceiling) and how much zero-shot Qwen3-VL-4B delivers (plans/2026-10-08-s2-gohold-prereg.md, results/s2_gohold.md).

  fit     (jevdrive env, CPU)      decision codebook from WOD train logs (wod/r2-train): class edges and prototype arc-length profiles
                                   -> results/s2_gohold/codebook.json; k-means trajectory anchors (K = 8 .. 64, global and per speed bin)
                                   -> results/s2_gohold/anchors.npz
  qwen    (jevdrive env, one GPU)  zero-shot Qwen3-VL-4B on the 479 rater frames: decomposed questions, the go / hold decision, the path choice
                                   among five candidate paths drawn on the front image and the speed choice (option scoring on the first token
                                   after a forced `ANSWER:`), per-forward latency -> results/s2_gohold/qwen.csv
  overlay (jevdrive env, CPU)      a few set-of-mark overlays written to a directory, to check the projection by eye
  report  (jevdrive env, CPU)      A: longitudinal go / hold: oracle ceilings by vocabulary size, Qwen3 agreement, RFS of the fused plans;
                                   B: System 2 as a selector among K candidate trajectories (System 1 outputs, k-means anchors, path family x
                                   speed profiles): oracle ceilings with the longitudinal / lateral / joint split, Qwen3 as the selector
                                   (paired bootstrap over sequences everywhere)
  figs    (jevdrive env, CPU)      per selected standstill frame: Qwen3's image, the frames openpilot is fed, the answers, a BEV

The oracle decision is derived from the top-rated rater trajectory of the val frame: privileged, a ceiling, not a method.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

OUT, FIG = _R / "experiments/op_parity/results/s2_gohold", _R / "experiments/op_parity/figs/s2_gohold"
T = np.arange(1, 21) * 0.25
V_STOP, HOLD_D5, STOP_DS = 0.5, 2.0, 0.5            # standstill speed; hold = under 2 m in 5 s (standstill) / under 0.5 m over the last second (moving)
VEDGES = (0.5, 2.0, 5.0, 8.0, 12.0)                 # regime bins: 0 = standstill, 1..5 = moving speed bins
KS = (2, 3, 5)
FOLDS, REPEATS, TAUS = 5, 20, np.round(np.arange(0.1, 0.91, 0.1), 1)
CTX_ORDER = ("red", "green", "stop_sign", "lead", "cross", "open")
KM, KM_ROWS = (8, 16, 32, 64), 300_000
# path family of a plan: (letter, name, lateral shift m (+ = left), ramp arc length floor m, ramp seconds x v0), smoothstep ramp along the arc
PATHS = (("A", "lane_left", 3.5, 20.0, 4.0), ("B", "nudge_left", 1.2, 10.0, 2.0), ("C", "keep", 0.0, 1.0, 0.0), ("D", "nudge_right", -1.2, 10.0, 2.0),
         ("E", "lane_right", -3.5, 20.0, 4.0))
KEEP = 2
COLS = {"A": (255, 0, 255), "B": (0, 255, 255), "C": (0, 220, 0), "D": (255, 140, 0), "E": (40, 90, 255)}
COL_NAMES = {"A": "magenta", "B": "cyan", "C": "green", "D": "orange", "E": "blue"}
SPEEDS = ("follow", "hold", "creep", "go")              # follow = the plan's own profile; the rest = V3 prototypes (classes 0, 1, 2)


# ---------------------------------------------------------------- codebook
def arc(p):
    return np.cumsum(np.linalg.norm(np.diff(np.concatenate([np.zeros_like(p[:, :1]), p], 1), axis=1), axis=-1), 1)


def vbin(v0):
    return np.digitize(v0, VEDGES)


def key(s, v0):
    """Sub-bucket key of a go trajectory: distance at 5 s at standstill, equivalent mean acceleration when moving."""
    return np.where(v0 < V_STOP, s[:, -1], 2 * (s[:, -1] - 5 * v0) / 25)


def is_go(s, v0):
    return np.where(v0 < V_STOP, s[:, -1] >= HOLD_D5, s[:, -1] - s[:, 15] >= STOP_DS)


def cls(s, v0, K, cb):
    """Class 0 .. K - 1 of arc-length profiles s (n, 20) at speeds v0: 0 = hold, 1 .. = go sub-buckets (train quantiles of the key)."""
    b, k, c = vbin(v0), key(s, v0), is_go(s, v0).astype(int)
    for j in range(len(VEDGES) + 1):
        m = (b == j) & (c > 0)
        c[m] = 1 + np.digitize(k[m], cb["edges"][str(K)][j])
    return c


def decode(c, v0, K, cb):
    """Prototype arc-length profile (n, 20) of class c at speed v0."""
    P = np.asarray(cb["proto"][str(K)])[vbin(v0), c]                       # (n, 20)
    return np.where((v0 < V_STOP)[:, None], P, np.maximum.accumulate(np.clip(v0[:, None] * T + P, 0, None), 1))


def along(path, sq):
    """path (n, 20, 2) plans, sq (n, m) arc lengths -> (n, m, 2) positions on each plan's path (origin prepended; beyond its end the path continues
    along its last moving direction, straight ahead for a plan at rest). Same construction as pp_wod_diag.retime, any number of query points."""
    out = np.empty(sq.shape + (2,))
    for i in range(len(path)):
        p = np.vstack([[0, 0], path[i]])
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        mv = np.flatnonzero(seg > 1e-3)
        d = (p[mv[-1] + 1] - p[mv[-1]]) / seg[mv[-1]] if len(mv) else np.array([1.0, 0.0])
        keep = np.concatenate([[True], seg > 1e-6])
        sx, px, ex = s[keep], p[keep], np.maximum(sq[i] - s[-1], 0)
        out[i, :, 0] = np.interp(sq[i], sx, px[:, 0]) + ex * d[0]
        out[i, :, 1] = np.interp(sq[i], sx, px[:, 1]) + ex * d[1]
    return out


at_arc = along


def offset_path(plan, v0, sq, dy, l0, lt):
    """The plan's path shifted sideways by dy (+ = left), ramped in by a smoothstep over the arc length max(l0, lt * v0), sampled at arc lengths sq (n, m)."""
    c = along(plan, sq)
    if dy == 0:
        return c
    t = along(plan, sq + 0.5) - along(plan, np.maximum(sq - 0.5, 0))
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-9)
    u = np.clip(sq / np.maximum(l0, lt * v0)[:, None], 0, 1)
    return c + np.stack([-t[..., 1], t[..., 0]], -1) * (dy * u * u * (3 - 2 * u))[..., None]


def factored(plan, v0, K, cb):
    """Path family x speed profiles of a plan -> (n, 5 * (K + 1), 20, 2), index = path * (K + 1) + speed; speed 0 = the plan's own profile, 1 .. K = the
    codebook prototypes of vocabulary K."""
    sp = [arc(plan)] + [decode(np.full(len(plan), c), v0, K, cb) for c in range(K)]
    return np.stack([offset_path(plan, v0, s, *pp[2:]) for pp in PATHS for s in sp], 1)


def first_best(J):
    """(20, n) scores of the F20 candidates -> index of the best one per frame; ties go to the plan itself, then to the smaller change
    (path order keep, nudges, lane shifts; speed order follow, hold, creep, go)."""
    pr = np.array([p_ * 4 + s_ for p_ in (KEEP, 1, 3, 0, 4) for s_ in range(4)])
    return pr[(J[pr] >= J.max(0) - 1e-9).argmax(0)]


def project(xy, cal):
    """Ground points (m, 2) in the vehicle frame -> pixels (m, 2) of a WOD camera (calibration dict of op_calib.json) and a validity mask."""
    E = np.asarray(cal["extrinsic"], np.float64).reshape(4, 4)
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = cal["intrinsic"]
    pc = (np.linalg.inv(E) @ np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))].T).T
    x = np.maximum(pc[:, 0], 1e-6)
    un, vn = -pc[:, 1] / x, -pc[:, 2] / x
    r2 = un * un + vn * vn
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    ud, vd = un * rad + 2 * p1 * un * vn + p2 * (r2 + 2 * un * un), vn * rad + p1 * (r2 + 2 * vn * vn) + 2 * p2 * un * vn
    uv = np.c_[cu + fu * ud, cv + fv * vd]
    return uv, (pc[:, 0] > 1.0) & (uv[:, 0] >= 0) & (uv[:, 0] < cal["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < cal["height"])


def overlay(img, cal, plan, v0, cb):
    """Set-of-mark image: the five candidate paths of one plan (20, 2) drawn as thin coloured lines on the front image, a letter at each far end.
    Uses the plan and fixed offsets only (no map, no labels); nothing is filled, so the road ahead stays visible."""
    from PIL import ImageDraw, ImageFont
    im = img.copy()
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=34)
    except TypeError:
        font = ImageFont.load_default()
    v = np.array([v0])
    far = float(np.clip(max(arc(plan[None])[0, -1], decode(np.array([2]), v, 3, cb)[0, -1]), 15, 50))
    s = np.linspace(6.0, far, 40)[None]
    for letter in "AEBDC":                                                      # the kept path on top
        pp = next(q for q in PATHS if q[0] == letter)
        uv, ok = project(offset_path(plan[None], v, s, *pp[2:])[0], cal)
        if ok.sum() < 2:
            continue
        pts = [tuple(x) for x in uv[ok]]
        dr.line(pts, fill=(0, 0, 0), width=7)
        dr.line(pts, fill=COLS[letter], width=4)
        u, w = pts[-1]
        dx = {"A": -34, "B": -30, "C": -10, "D": 10, "E": 14}[letter]
        dr.text((u + dx, w - (78 if letter == "C" else 44)), letter, fill=COLS[letter], font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    return im


def fuse(plan, v0, dec, K, cb, mode, scope):
    """Late fusion: the plan's own path re-timed to the prototype of decision `dec` on the rows of `scope`; mode gate keeps the plan where its own
    class already equals the decision, mode replace re-times every row."""
    use = np.asarray(scope, bool) & ((cls(arc(plan), v0, K, cb) != dec) if mode == "gate" else True)
    return np.where(use[:, None, None], at_arc(plan, decode(dec, v0, K, cb)), plan)


def cmd_fit(a):
    from jevdrive import waymo as W
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-gohold-fit", seed=0, config=vars(a)) as run:
        tr = splits.load("wod/r2-train")
        run.use_split(tr)
        df = W.load_index()
        past, fut = W.load_ego()
        m = tr.mask(df.sequence.astype(str).to_numpy()) & df.has_future.to_numpy().astype(bool) & np.isfinite(fut).all((1, 2)) & np.isfinite(past[:, -1]).all(1)
        s, v0 = arc(fut[m][..., :2].astype(np.float64)), W.init_speed(past[m]).astype(np.float64)
        b, go, k = vbin(v0), is_go(s, v0), key(s, v0)
        cb = {"T": T.tolist(), "v_stop": V_STOP, "hold_d5": HOLD_D5, "stop_ds": STOP_DS, "vedges": list(VEDGES), "split": tr.id, "rows": int(m.sum()),
              "edges": {}, "proto": {}, "n": {}, "go_share": [float(go[b == j].mean()) for j in range(len(VEDGES) + 1)]}
        for K in KS:
            cb["edges"][str(K)] = [np.quantile(k[(b == j) & go], np.arange(1, K - 1) / (K - 1)).tolist() for j in range(len(VEDGES) + 1)]
            c = cls(s, v0, K, cb)
            base = np.where((v0 < V_STOP)[:, None], s, s - v0[:, None] * T)       # standstill: s(t); moving: s(t) - v0 t
            cb["proto"][str(K)] = [[base[(b == j) & (c == q)].mean(0).tolist() for q in range(K)] for j in range(len(VEDGES) + 1)]
            cb["n"][str(K)] = [[int(((b == j) & (c == q)).sum()) for q in range(K)] for j in range(len(VEDGES) + 1)]
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "codebook.json").write_text(json.dumps(cb, indent=1))
        run.info("rows %d; go share per bin %s", m.sum(), np.round(cb["go_share"], 3).tolist())
        from sklearn.cluster import KMeans                                         # trajectory vocabulary: k-means anchors in the ego frame
        xy = fut[m][..., :2].astype(np.float32)
        sub = np.random.default_rng(0).choice(len(xy), min(KM_ROWS, len(xy)), replace=False)
        X, bs = xy[sub].reshape(len(sub), -1), b[sub]
        anc = {}
        for K in KM:
            anc[f"g{K}"] = KMeans(K, n_init=3, random_state=0).fit(X).cluster_centers_.reshape(K, 20, 2)
            anc[f"v{K}"] = np.stack([KMeans(K, n_init=3, random_state=0).fit(X[bs == j]).cluster_centers_.reshape(K, 20, 2) for j in range(len(VEDGES) + 1)])
            run.info("k-means K=%d: global d5 range %.1f .. %.1f m; rows per speed bin %s", K, np.linalg.norm(anc[f"g{K}"][:, -1], axis=-1).min(),
                     np.linalg.norm(anc[f"g{K}"][:, -1], axis=-1).max(), np.bincount(bs, minlength=len(VEDGES) + 1).tolist())
        np.savez_compressed(OUT / "anchors.npz", **{k: v.astype(np.float32) for k, v in anc.items()})
        for K in KS:
            run.info("K=%d edges %s\n n %s\n standstill d5 of prototypes %s", K, [np.round(e, 2).tolist() for e in cb["edges"][str(K)]], cb["n"][str(K)],
                     [round(p[-1], 2) for p in cb["proto"][str(K)][0]])


# ---------------------------------------------------------------- Qwen3 prompts (fixed with the pre-registration)
ONE = "The image was taken by the front camera of a car (the ego vehicle)."
TWO = ("The two images were taken by the front camera of a car (the ego vehicle): image 1 one second before image 2. "
       "Answer for the moment of image 2.")
FR = "The two images were taken at the same instant by a car (the ego vehicle): image 1 is its front camera, image 2 its front-right camera."
THREE = ("The three images were taken at the same instant by a car (the ego vehicle): image 1 is its front-left camera, image 2 its front camera, "
         "image 3 its front-right camera.")
FOUR = ("The four images were taken by the cameras of a car (the ego vehicle): image 1 is its front camera one second ago; images 2, 3 and 4 are "
        "its front-left, front and front-right cameras now.")
TAIL = "End your reply with one line of the form `ANSWER: <option>`, <option> being one of: %s. Reply with that line only."
# question -> (head, body, options, images); images: P = FRONT one second ago, L / F / R = front-left / front / front-right now.
# light, lead and cross are the wod-launch questions verbatim (scripts/wod_launch.py QS), stop_ctrl its label2 question.
QS = {
    "light": (ONE, "Traffic light status controlling the ego vehicle lane ahead. Options:\n"
              "  red_or_yellow_for_ego: a red or yellow traffic light controls the ego lane ahead\n"
              "  green_for_ego: a green traffic light controls the ego lane ahead\n"
              "  no_light_for_ego: no traffic light controls the ego lane ahead (none visible, or only lights for other lanes / directions)",
              ["red_or_yellow_for_ego", "green_for_ego", "no_light_for_ego"], "F"),
    "stop_ctrl": (FR, "Is the ego vehicle at, or just before, a junction where a stop sign controls the ego's direction? Evidence: a stop sign facing the ego "
                  "(usually on the right kerb, image 2), a painted STOP on the ego lane, or the backs of stop signs for the other directions at an all-way "
                  "stop. A junction with traffic lights is not stop-controlled. Options:\n"
                  "  stop_controlled: a stop sign controls the ego's direction at the junction just ahead\n"
                  "  not_stop_controlled: no stop sign controls the ego here",
                  ["stop_controlled", "not_stop_controlled"], "FR"),
    "row": (THREE, "Right of way at the junction or crossing just ahead of the ego vehicle. Options:\n"
            "  others_first: another road user (a vehicle in or entering the junction, a pedestrian or cyclist on or stepping onto the crossing) must be "
            "let through before the ego vehicle can proceed\n"
            "  ego_may_go: nobody has to be let through first: the junction and crossing ahead are clear for the ego vehicle, or there is no junction or "
            "crossing just ahead",
            ["others_first", "ego_may_go"], "LFR"),
    "lead": (TWO, "Vehicle directly ahead of the ego vehicle in the ego lane, within about 20 metres. Options:\n"
             "  none_ahead: no vehicle is directly ahead in the ego lane within about 20 metres\n"
             "  stopped_lead: a vehicle is directly ahead and it is standing still (same position in both images)\n"
             "  moving_lead: a vehicle is directly ahead and it is moving away or driving (its position changes between the images)",
             ["none_ahead", "stopped_lead", "moving_lead"], "PF"),
    "cross": (ONE, "Pedestrians, cyclists or crossing vehicles in the ego vehicle's path. Options:\n"
              "  crossing: a pedestrian, a cyclist or a vehicle is crossing or standing in the road space directly in front of the ego vehicle\n"
              "  clear: nothing is crossing or blocking the road space directly in front of the ego vehicle (a vehicle queued ahead in the "
              "ego lane does not count)",
              ["crossing", "clear"], "F"),
}
OBS = {"light": ("traffic light for the ego lane", {"red_or_yellow_for_ego": "red or yellow", "green_for_ego": "green", "no_light_for_ego": "none"}),
       "stop_ctrl": ("stop sign controlling the ego at the junction just ahead", {"stop_controlled": "yes", "not_stop_controlled": "no"}),
       "row": ("right of way", {"others_first": "another road user must be let through first", "ego_may_go": "nobody has to be let through first"}),
       "lead": ("vehicle directly ahead in the ego lane within 20 m", {"none_ahead": "none", "stopped_lead": "yes, standing still",
                                                                         "moving_lead": "yes, moving away"}),
       "cross": ("pedestrian, cyclist or vehicle crossing directly in front", {"crossing": "yes", "clear": "no"})}
D2 = ("Decide what the ego vehicle should do over the next 5 seconds, driving as a careful and efficient human driver would. Options:\n"
      "  hold: remain stopped, or brake to a stop (a red or yellow light for the ego, a stopped vehicle directly ahead, pedestrians or cross traffic "
      "that must be let through first, not yet the ego's turn at a stop sign)\n"
      "  go: start moving, or keep driving (a green light, a clear way ahead, the ego's turn at a stop sign, the vehicle ahead moving off)")
D3 = ("Decide what the ego vehicle should do over the next 5 seconds, driving as a careful and efficient human driver would. Options:\n"
      "  hold: remain stopped for the whole 5 seconds (a red light, a stopped vehicle directly ahead, others must be let through first)\n"
      "  creep: move forward slowly by a few metres only (inch up to the line or into the junction, close a gap, a cautious start)\n"
      "  go: move off and drive away normally (a green light, a clear way ahead, the ego's turn at a stop sign, the vehicle ahead moving off)")
# addendum B: the selector. One more scene question, then the path choice on the set-of-mark image and the speed choice in text.
BLOCK = (THREE, "Obstruction of the ego vehicle's own lane ahead, within about 40 metres. Options:\n"
         "  left_pass: the ego lane ahead is obstructed (an object or debris, a parked, stopped or very slow vehicle, a cyclist, a construction zone or "
         "cones) and the free space to go around it is on the left\n"
         "  right_pass: the ego lane ahead is obstructed and the free space to go around it is on the right\n"
         "  clear_lane: the ego lane ahead is not obstructed (ordinary traffic queued or moving in the lane does not count)",
         ["left_pass", "right_pass", "clear_lane"], "LFR")
OBS_BLOCK = ("obstruction of the ego lane ahead", {"left_pass": "yes, free space to pass on the left", "right_pass": "yes, free space to pass on the right",
                                                    "clear_lane": "no"})
PATH_HEAD = ("The image was taken by the front camera of a car (the ego vehicle). Five candidate paths for the next 5 seconds are drawn on the road as "
             "coloured lines, each with its letter at its far end: A (magenta) shifts one lane to the left, B (cyan) moves about one metre to the left, "
             "C (green) is the path the driving system currently plans, D (orange) moves about one metre to the right, E (blue) shifts one lane to the right.")
PATH_Q = ("Choose the path a careful and efficient human driver would take. Keep C unless the scene gives a reason to leave it: an obstacle or debris, a "
          "parked or stopped vehicle, a cyclist or pedestrian that needs clearance, a construction zone or lane closure, a slow vehicle that should be "
          "overtaken, or a lane the ego must move into. Options:\n"
          "  A: shift one lane to the left\n  B: move about one metre to the left\n  C: keep the planned path\n  D: move about one metre to the right\n"
          "  E: shift one lane to the right")
SPEED_Q = ("Choose the speed behaviour of the ego vehicle over the next 5 seconds, driving as a careful and efficient human driver would. Options:\n"
           "  hold: stay stopped, or brake to a stop (a red or yellow light for the ego, a stopped vehicle directly ahead, others that must be let through)\n"
           "  creep: from standstill move forward a few metres only; when moving, slow down clearly without stopping (approaching a hazard, a queue or a turn)\n"
           "  follow: keep the present behaviour: hold the current speed, follow the traffic ahead, or stay as the driving system plans\n"
           "  go: from standstill move off and drive away normally; when moving, accelerate (a green light, a clear way ahead, the vehicle ahead moving off)")
DEC = {"direct": (D2, ["hold", "go"]), "decomp": (D2, ["hold", "go"]), "decomp3": (D3, ["hold", "creep", "go"]), "block": (BLOCK[1], BLOCK[2]),
       "path": (PATH_Q, ["A", "B", "C", "D", "E"]), "speed4": (SPEED_Q, ["hold", "creep", "follow", "go"])}


def state_line(v0):
    return "The ego vehicle is standing still." if v0 < V_STOP else f"The ego vehicle is moving at about {max(5, int(round(v0 * 3.6 / 5) * 5))} km/h."


def obs_text(ans):
    return "Observations already made from these images:\n" + "\n".join(f"- {OBS[q][0]}: {OBS[q][1][ans[q]]}" for q in QS)


def obs_text2(ans):
    return obs_text(ans) + f"\n- {OBS_BLOCK[0]}: {OBS_BLOCK[1][ans['block']]}"


def rule(d):
    """D-rule: hold when any decomposed answer says so (pandas frame or dict of answers) -> 1 = go, 0 = hold."""
    return 1 - ((d["light"] == "red_or_yellow_for_ego") | (d["lead"] == "stopped_lead") | (d["cross"] == "crossing") | (d["row"] == "others_first"))


def cmd_qwen(a):
    import glob
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-gohold-qwen", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        p = glob.glob(str(data_dir() / "cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*"))[0]
        proc = AutoProcessor.from_pretrained(p)
        m = AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16).to("cuda").eval()
        tok = proc.tokenizer
        pre = tok("ANSWER:", add_special_tokens=False).input_ids
        opts = {q: v[2] for q, v in QS.items()} | {q: v[1] for q, v in DEC.items()}
        first = {q: [tok("ANSWER: " + o, add_special_tokens=False).input_ids[len(pre)] for o in o_] for q, o_ in opts.items()}
        for q, f in first.items():
            assert len(set(f)) == len(opts[q]), (q, f)
        r = Z.load_sets()["rater"]
        names, v0 = r["name"].astype(str)[: a.limit or None], W.init_speed(r["past"]).astype(np.float64)
        t2 = Z.root("t2_jpg")
        from pp_wod import load_preds
        cb, cal = json.loads((OUT / "codebook.json").read_text()), json.loads((Z.root() / "op_calib.json").read_text())
        wp2 = load_preds("WP2-full-s0", names)                                    # the plan whose path family is drawn

        def ask(q, text, ims):
            t0 = time.perf_counter()
            msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": text + "\n" + TAIL % ", ".join(opts[q])}]}]
            x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + "ANSWER:"
            x = proc(text=[x], images=ims, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            pr = torch.softmax(m(**x, use_cache=False).logits[0, -1].float()[first[q]], -1).cpu().numpy()
            torch.cuda.synchronize()
            t2_ = time.perf_counter()
            return ({q: opts[q][int(pr.argmax())], f"{q}_ms_pre": round(1e3 * (t1 - t0), 1), f"{q}_ms_fwd": round(1e3 * (t2_ - t1), 1),
                     f"{q}_tokens": int(x["input_ids"].shape[1])} | {f"{q}_p_{o}": round(float(v), 4) for o, v in zip(opts[q], pr)})
        rows, t_start = [], time.time()
        with torch.no_grad():
            for i, n in enumerate(names):
                s, f = n.rsplit("-", 1)
                prev = t2 / f"{s}-{int(f) - 10:03d}" / "1.jpg"
                im = {c: Image.open(t2 / n / f"{k}.jpg").convert("RGB") for c, k in (("L", 2), ("F", 1), ("R", 3))}
                im["P"] = Image.open(prev).convert("RGB") if prev.exists() else im["F"]
                row = {"name": n, "v0": round(float(v0[i]), 3), "has_prev": prev.exists()}
                for q, (head, body, _, cams) in QS.items():
                    row |= ask(q, f"{head}\n{body}", [im[c] for c in cams])
                st, four = state_line(v0[i]), [im[c] for c in "PLFR"]
                row |= ask("direct", f"{FOUR} {st}\n{D2}", four)
                row |= ask("decomp", f"{FOUR} {st}\n{obs_text(row)}\n{D2}", four)
                if v0[i] < V_STOP:
                    row |= ask("decomp3", f"{FOUR} {st}\n{obs_text(row)}\n{D3}", four)
                row |= ask("block", f"{BLOCK[0]}\n{BLOCK[1]}", [im[c] for c in BLOCK[3]])
                row |= ask("path", f"{PATH_HEAD} {st}\n{obs_text2(row)}\n{PATH_Q}", [overlay(im["F"], cal[s]["1"], wp2[i], v0[i], cb)])
                row |= ask("speed4", f"{FOUR} {st}\n{obs_text2(row)}\n{SPEED_Q}", four)
                rows.append(row)
                if (i + 1) % 40 == 0 or i + 1 == len(names):
                    run.info(f"[{i + 1}/{len(names)}] {(time.time() - t_start) / (i + 1):.2f} s/frame")
        df = pd.DataFrame(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / ("qwen.csv" if not a.limit else f"qwen_first{a.limit}.csv"), index=False)
        lat = {q: float(df[f"{q}_ms_pre"].iloc[10:].median() + df[f"{q}_ms_fwd"].iloc[10:].median()) for q in opts}
        run.info("median ms per forward (pre + model): %s", {q: round(v) for q, v in lat.items()})
        run.info("counts: %s", {q: df[q].value_counts().to_dict() for q in opts})
        run.summary.update(n=len(df), ms=lat)


# ---------------------------------------------------------------- report
def cmd_report(a):
    import pandas as pd
    import wod_launch_report as R
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-gohold-report", seed=0, config=vars(a) | dict(B=R.B, folds=FOLDS, repeats=REPEATS)) as run:
        val = splits.load("wod/val")
        run.use_split(val)
        cb = json.loads((OUT / "codebook.json").read_text())
        C = R.Ctx()
        n, v0, ii = C.n, C.v0, np.arange(C.n)
        assert set(C.seq[:n]) <= set(val.members), "rater frames outside wod/val"
        names = C.names[:n]
        P = {"WP2": [C.preds("WP2-full-s0"), C.preds("WP2-full-s1")], "shipped": [C.preds("shipped")]}
        base = {k: np.mean([C.rfs(p) for p in v], 0) for k, v in P.items()}
        top, log = C.top, C.fut[:n]
        from jevdrive import wod_zeroshot as Z
        from pp_wod_diag import retime
        cluster = Z.load_sets()["rater"]["cluster"].astype(str)
        st = {"stopped": C.st["stopped"], "SL": C.st["SL (stopped, log moves)"], "SS": C.st["SS (stopped, log stays)"], "moving": C.st["moving (v>=0.5)"],
              "all": C.st["all"], "turn": C.intent >= 2}
        BST = {"all": st["all"], "stopped": st["stopped"], "moving": st["moving"], "turn": st["turn"]} | {f"cluster {c}": cluster == c for c in sorted(set(cluster))}
        assert np.abs(along(P["WP2"][0], arc(top)) - retime(P["WP2"][0], top)).max() < 1e-9      # along == pp_wod_diag.retime
        stop = st["stopped"]
        fr = pd.read_csv(_R / "experiments/op_parity/results/wod_launch/frames.csv").set_index("name").reindex(names)
        ctx = fr.context.to_numpy().astype(str)
        scopes = {"standstill-only": stop, "all": np.ones(n, bool)}
        run.info("reproduce: WP2 %.3f shipped %.3f log %.3f top %.3f; stopped WP2 %.3f shipped %.3f", C.cm(base["WP2"]), C.cm(base["shipped"]),
                 C.cm(C.rfs(log)), C.cm(C.rfs(top)), C.cm(base["WP2"], stop), C.cm(base["shipped"], stop))
        for k in P:                                                               # at_arc of a plan onto its own arc length returns it
            assert np.abs(at_arc(P[k][0], arc(P[k][0])) - P[k][0]).max() < 1e-9

        def score(S1, dec, K, mode, scope):
            return np.mean([C.rfs(fuse(p, v0, dec, K, cb, mode, scope)) for p in P[S1]], 0)

        def cells(d, extra=()):
            out = {}
            for nm in ("stopped", "moving", "all") + tuple(extra):
                c = C.ci(d, st[nm])
                out |= {f"d {nm}": c[0], f"d {nm} lo": c[1], f"d {nm} hi": c[2]}
            return out
        f3 = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"  # noqa: E731

        # ---- classes: oracle (top-rated), log, System 1's own
        CL = {K: {"oracle": cls(arc(top), v0, K, cb), "log": cls(arc(log), v0, K, cb)} | {k: [cls(arc(p), v0, K, cb) for p in v] for k, v in P.items()} for K in KS}
        crow = []
        for K in KS:
            for nm in ("stopped", "moving"):
                m = st[nm]
                row = {"K": K, "stratum": nm, "n": int(m.sum()), "oracle class counts": " / ".join(str(int((CL[K]["oracle"][m] == q).sum())) for q in range(K))}
                for k in ("log", "WP2", "shipped"):
                    c = [CL[K][k]] if k == "log" else CL[K][k]
                    row[f"{k} = oracle"] = float(np.mean([(x[m] == CL[K]["oracle"][m]).mean() for x in c]))
                    row[f"{k} class counts"] = " / ".join(f"{np.mean([(x[m] == q).sum() for x in c]):.0f}" for q in range(K))
                crow.append(row)
        stats.write_table(crow, OUT / "classes")
        run.info("classes:\n%s", pd.DataFrame(crow).to_string())
        proto_self = {K: float(np.mean([(cls(decode(np.full(n, q), v0, K, cb), v0, K, cb) == q).mean() for q in range(K)])) for K in KS}

        # ---- 1. oracle ceilings
        orow, keep = [], {}
        for S1 in P:
            o1 = np.mean([C.rfs(along(p, arc(top))) for p in P[S1]], 0)
            for sc_nm, sc in scopes.items():
                d = np.where(sc, o1 - base[S1], 0.0)
                orow.append({"S1": S1, "decision": "continuous (top-rated speed profile, O1 swap)", "K": "inf", "fusion": "replace", "scope": sc_nm} | cells(d, ("SL", "SS")))
            for K in KS:
                for mode in ("gate", "replace"):
                    per = np.stack([score(S1, np.full(n, q), K, mode, np.ones(n, bool)) for q in range(K)])     # (K, n) every class applied everywhere
                    for sc_nm, sc in scopes.items():
                        f_top = np.where(sc, per[CL[K]["oracle"], ii], base[S1])
                        f_best = np.where(sc, np.maximum(per.max(0), base[S1]) if mode == "gate" else per.max(0), base[S1])
                        keep[(S1, K, mode, sc_nm)] = f_top
                        orow.append({"S1": S1, "decision": "oracle: class of the top-rated trajectory (privileged)", "K": K, "fusion": mode, "scope": sc_nm}
                                    | cells(f_top - base[S1], ("SL", "SS")))
                        orow.append({"S1": S1, "decision": "oracle: best class per frame (privileged, codebook bound)", "K": K, "fusion": mode, "scope": sc_nm}
                                    | cells(f_best - base[S1], ("SL", "SS")))
        stats.write_table(orow, OUT / "oracle")
        od = pd.DataFrame(orow)
        run.info("oracle:\n%s", od[["S1", "decision", "K", "fusion", "scope", "d stopped", "d stopped lo", "d stopped hi", "d moving", "d moving lo", "d moving hi", "d all",
                                      "d all lo", "d all hi"]].to_string(float_format=lambda v: f"{v:+.3f}"))
        inc = []                                                                  # increments between vocabularies (gate, top-rated class), paired
        for S1 in P:
            for sc_nm in scopes:
                for Ka, Kb in ((2, 3), (3, 5), (2, 5)):
                    inc.append({"S1": S1, "scope": sc_nm, "contrast": f"V{Kb} - V{Ka} (gate, top-rated class)"} | cells(keep[(S1, Kb, "gate", sc_nm)] - keep[(S1, Ka, "gate", sc_nm)]))
        stats.write_table(inc, OUT / "oracle_increments")
        octx = []                                                                 # oracle V2 gate per context of the standstill frames
        for S1 in P:
            for c in CTX_ORDER:
                m = stop & (ctx == c)
                if m.sum() < 5:
                    continue
                row = {"S1": S1, "context": c, "n": int(m.sum()), "oracle go share": float((CL[2]["oracle"][m] == 1).mean()), "RFS S1": C.cm(base[S1], m)}
                for K in KS:
                    cc = C.ci(keep[(S1, K, "gate", "standstill-only")] - base[S1], m)
                    row |= {f"V{K} gate d": cc[0], f"V{K} lo": cc[1], f"V{K} hi": cc[2]}
                octx.append(row)
        stats.write_table(octx, OUT / "oracle_context")

        # ---- B1. System 2 as a selector among K candidates: privileged best-of-set ceilings, longitudinal / lateral / joint
        anc = np.load(OUT / "anchors.npz")
        s_top = C.rfs(top)

        def cand_sets(seed):
            p = P["WP2"][seed]
            S = {"S1x2": np.stack([p, P["shipped"][0]], 1),
                 "S1x5": np.stack([p, P["WP2"][1 - seed], P["shipped"][0], C.preds("WP1-full-s0"), C.preds("WP1-full-s1")], 1)}
            for K in KM:
                S[f"KM{K}"] = np.broadcast_to(anc[f"g{K}"].astype(np.float64), (n, K, 20, 2))
                S[f"KMv{K}"] = anc[f"v{K}"].astype(np.float64)[vbin(v0)]
            return S | {"F20": factored(p, v0, 3, cb), "F30": factored(p, v0, 5, cb)}

        def three(p, cands):
            """Per candidate (M, n): RFS of the candidate, of p's path at the candidate's arc-length profile, of the candidate's path at p's profile."""
            sp, M_ = arc(p), cands.shape[1]
            return (np.stack([C.rfs(np.ascontiguousarray(cands[:, m])) for m in range(M_)]), np.stack([C.rfs(along(p, arc(cands[:, m]))) for m in range(M_)]),
                    np.stack([C.rfs(along(np.ascontiguousarray(cands[:, m]), sp)) for m in range(M_)]))
        SETS, bw = {}, [C.rfs(p) for p in P["WP2"]]
        for seed in (0, 1):
            for nm, cands in cand_sets(seed).items():
                SETS.setdefault(nm, []).append(three(P["WP2"][seed], cands))
        assert np.abs(SETS["F20"][0][0][KEEP * 4] - bw[0]).max() < 1e-9          # keep x follow is the plan itself
        agg = lambda nm, k: np.mean([np.maximum(SETS[nm][s_][k].max(0), bw[s_]) for s_ in (0, 1)], 0)  # noqa: E731  best of set U {WP2}
        SEL, brow = {}, []
        for nm in SETS:
            alone = np.mean([SETS[nm][s_][0].max(0) for s_ in (0, 1)], 0)
            joint, lon, lat = agg(nm, 0), agg(nm, 1), agg(nm, 2)
            either = np.mean([np.maximum(np.maximum(SETS[nm][s_][1].max(0), SETS[nm][s_][2].max(0)), bw[s_]) for s_ in (0, 1)], 0)
            SEL[nm] = dict(alone=alone, joint=joint, lon=lon, lat=lat, either=either)
            for sn, m in BST.items():
                row = {"set": nm, "M": SETS[nm][0][0].shape[0], "stratum": sn, "n": int(m.sum()), "RFS WP2": C.cm(base["WP2"], m), "RFS top-rated": C.cm(s_top, m),
                       "RFS best of set alone": C.cm(alone, m), "RFS best of set U WP2": C.cm(joint, m)}
                for k, d in (("joint", joint - base["WP2"]), ("lon only", lon - base["WP2"]), ("lat only", lat - base["WP2"]), ("either", either - base["WP2"]),
                             ("needs joint", joint - either)):
                    c = C.ci(d, m)
                    row |= {f"d {k}": c[0], f"d {k} lo": c[1], f"d {k} hi": c[2]}
                row |= {"lon share": row["d lon only"] / row["d joint"] if row["d joint"] > 1e-9 else np.nan,
                        "lat share": row["d lat only"] / row["d joint"] if row["d joint"] > 1e-9 else np.nan,
                        "joint share": row["d needs joint"] / row["d joint"] if row["d joint"] > 1e-9 else np.nan,
                        "share of gap to top-rated closed": row["d joint"] / (row["RFS top-rated"] - row["RFS WP2"])}
                brow.append(row)
        o1s, o2s = [C.rfs(along(p, arc(top))) for p in P["WP2"]], [C.rfs(along(top, arc(p))) for p in P["WP2"]]
        o1w, o2w = np.mean(o1s, 0), np.mean(o2s, 0)
        eith = np.mean([np.maximum(np.maximum(o1s[s_], o2s[s_]), bw[s_]) for s_ in (0, 1)], 0)       # the better of the two swaps or the plan, per frame
        for sn, m in BST.items():                                                 # continuous references (decision 164): top-rated speed / path / trajectory
            row = {"set": "reference: top-rated trajectory (continuous, privileged)", "M": np.nan, "stratum": sn, "n": int(m.sum()), "RFS WP2": C.cm(base["WP2"], m),
                   "RFS top-rated": C.cm(s_top, m), "RFS best of set alone": C.cm(s_top, m), "RFS best of set U WP2": C.cm(s_top, m)}
            for k, d in (("joint", s_top - base["WP2"]), ("lon only", o1w - base["WP2"]), ("lat only", o2w - base["WP2"]), ("either", eith - base["WP2"]),
                         ("needs joint", s_top - eith)):
                c = C.ci(d, m)
                row |= {f"d {k}": c[0], f"d {k} lo": c[1], f"d {k} hi": c[2]}
            row |= {"lon share": row["d lon only"] / row["d joint"], "lat share": row["d lat only"] / row["d joint"], "joint share": row["d needs joint"] / row["d joint"]}
            brow.append(row)
        stats.write_table(brow, OUT / "select_oracle")
        bd = pd.DataFrame(brow)
        run.info("select oracle (all / stopped / turn):\n%s", bd[bd.stratum.isin(["all", "stopped", "turn"])][
            ["set", "M", "stratum", "RFS best of set alone", "d joint", "d joint lo", "d joint hi", "d lon only", "d lat only", "d either", "d needs joint",
             "d needs joint lo", "d needs joint hi"]].to_string(float_format=lambda v: f"{v:+.3f}"))
        run.info("select oracle by cluster (d joint):\n%s", bd.pivot_table(index="set", columns="stratum", values="d joint", sort=False).to_string(
            float_format=lambda v: f"{v:+.2f}"))

        # ---- per-frame table (oracle decisions)
        pf = pd.DataFrame({"name": names, "v0": v0, "vbin": vbin(v0), "stopped": stop, "context": ctx, "top_d5": arc(top)[:, -1], "log_d5": arc(log)[:, -1],
                           "WP2_s0_d5": arc(P["WP2"][0])[:, -1], "shipped_d5": arc(P["shipped"][0])[:, -1], "rfs_WP2": base["WP2"], "rfs_shipped": base["shipped"]}
                          | {f"oracle_V{K}": CL[K]["oracle"] for K in KS} | {f"log_V{K}": CL[K]["log"] for K in KS}
                          | {f"WP2_s0_V{K}": CL[K]["WP2"][0] for K in KS} | {f"WP2_s1_V{K}": CL[K]["WP2"][1] for K in KS} | {f"shipped_V{K}": CL[K]["shipped"][0] for K in KS}
                          | {f"rfs_WP2_oracle_V{K}_gate": keep[("WP2", K, "gate", "all")] for K in KS}
                          | {f"rfs_shipped_oracle_V{K}_gate": keep[("shipped", K, "gate", "all")] for K in KS}
                          | {"cluster": cluster, "turn": st["turn"]} | {f"rfs_best_{nm}": SEL[nm]["joint"] for nm in SEL}
                          | {"rfs_best_F20_lon": SEL["F20"]["lon"], "rfs_best_F20_lat": SEL["F20"]["lat"],
                             "oracle_F20_s0": [f"{PATHS[j // 4][0]}/{SPEEDS[j % 4]}" for j in first_best(SETS["F20"][0][0])]})
        if not (OUT / "qwen.csv").exists():
            pf.to_csv(OUT / "frames.csv", index=False, float_format="%.4f")
            run.info("no qwen.csv yet: oracle part only")
            run.summary.update(oracle_only=True)
            return

        # ---- 2. Qwen3 decisions
        q = pd.read_csv(OUT / "qwen.csv").set_index("name").reindex(names)
        assert q.decomp.notna().all()
        D = {"decomp": (q.decomp == "go").to_numpy().astype(int), "direct": (q.direct == "go").to_numpy().astype(int), "rule": rule(q).to_numpy().astype(int)}
        d3 = q.decomp3.map({"hold": 0, "creep": 1, "go": 2}).fillna(0).to_numpy().astype(int)
        same = {k: float((q[k].to_numpy() == fr[k].to_numpy()).mean()) for k in ("light", "stop_ctrl", "lead", "cross")}
        run.info("decomposed answers identical to wod-launch's labels: %s", same)
        o2 = CL[2]["oracle"]
        arow = []

        def agree(nm, pred, m, ref=o2, K=2):
            pr = [pred] if not isinstance(pred, list) else pred
            go, hold = m & (ref > 0), m & (ref == 0)
            f = lambda mm: float(np.mean([((x[mm] > 0) == (ref[mm] > 0)).mean() for x in pr])) if mm.sum() else np.nan  # noqa: E731
            return {f"{nm} acc": f(m), f"{nm} go recall": f(go), f"{nm} hold recall": f(hold), f"{nm} says go": float(np.mean([(x[m] > 0).mean() for x in pr]))}
        groups = [("stopped", stop), ("SL", st["SL"]), ("SS", st["SS"])] + [(f"stopped, {c}", stop & (ctx == c)) for c in CTX_ORDER] + [("moving", st["moving"])] + \
                 [(f"moving, {c}", st["moving"] & (ctx == c)) for c in CTX_ORDER]
        for nm, m in groups:
            if m.sum() < 5:
                continue
            row = {"group": nm, "n": int(m.sum()), "oracle go": int((o2[m] > 0).sum()), "oracle hold": int((o2[m] == 0).sum())}
            for k in ("decomp", "direct", "rule"):
                row |= agree(f"Qwen3 {k}", D[k], m)
            row |= agree("WP2 own", CL[2]["WP2"], m) | agree("shipped own", CL[2]["shipped"], m) | agree("log", CL[2]["log"], m)
            row["always go acc"] = float((o2[m] > 0).mean())
            if nm.startswith("stopped") or nm in ("SL", "SS"):
                row["Qwen3 decomp3 = oracle V3"] = float((d3[m] == CL[3]["oracle"][m]).mean())
                row["WP2 own V3 = oracle V3"] = float(np.mean([(x[m] == CL[3]["oracle"][m]).mean() for x in CL[3]["WP2"]]))
            arow.append(row)
        stats.write_table(arow, OUT / "agreement")
        run.info("agreement:\n%s", pd.DataFrame(arow).T.to_string(float_format=lambda v: f"{v:.3f}"))
        conf = pd.crosstab(pd.Series(np.where(o2[stop] > 0, "oracle go", "oracle hold"), name="oracle"), pd.Series(q.decomp.to_numpy()[stop], name="Qwen3 decomp"))
        conf.to_csv(OUT / "confusion_stopped.csv")
        run.info("confusion on stopped:\n%s", conf.to_string())

        # ---- fused RFS
        frow, F = [], {}
        for S1 in P:
            for dn, dec, K in (("decomp", D["decomp"], 2), ("direct", D["direct"], 2), ("rule", D["rule"], 2), ("decomp3", d3, 3)):
                for mode in ("gate", "replace"):
                    for sc_nm, sc in scopes.items():
                        if dn == "decomp3" and sc_nm == "all":
                            continue
                        f = score(S1, dec, K, mode, sc)
                        F[(S1, dn, mode, sc_nm)] = f
                        orc = keep[(S1, K, mode, sc_nm)]
                        g = C.ci(f - orc, stop)
                        frow.append({"S1": S1, "decision": f"Qwen3 {dn}", "K": K, "fusion": mode, "scope": sc_nm, "primary": bool(S1 == "WP2" and dn == "decomp" and mode == "gate"),
                                     "RFS stopped": C.cm(f, stop), "RFS all": C.cm(f)} | cells(f - base[S1], ("SL", "SS"))
                                    | {"vs oracle stopped": g[0], "vs oracle lo": g[1], "vs oracle hi": g[2],
                                       "frames changed (stopped)": float(np.mean([(np.abs(fuse(p, v0, dec, K, cb, mode, sc) - p).max((1, 2)) > 1e-6)[stop].mean() for p in P[S1]]))})
            for dn, dec in (("always go", np.ones(n, int)), ("always hold", np.zeros(n, int))):             # trivial decisions, standstill only
                for mode in ("gate", "replace"):
                    f = score(S1, dec, 2, mode, stop)
                    F[(S1, dn, mode, "standstill-only")] = f
                    g = C.ci(f - keep[(S1, 2, mode, "standstill-only")], stop)
                    frow.append({"S1": S1, "decision": dn, "K": 2, "fusion": mode, "scope": "standstill-only", "primary": False, "RFS stopped": C.cm(f, stop), "RFS all": C.cm(f)}
                                | cells(f - base[S1], ("SL", "SS")) | {"vs oracle stopped": g[0], "vs oracle lo": g[1], "vs oracle hi": g[2]})
        stats.write_table(frow, OUT / "fused")
        fd = pd.DataFrame(frow)
        run.info("fused:\n%s", fd[["S1", "decision", "fusion", "scope", "d stopped", "d stopped lo", "d stopped hi", "d moving", "d moving lo", "d moving hi", "d all", "d all lo",
                                     "d all hi", "vs oracle stopped"]].to_string(float_format=lambda v: f"{v:+.3f}"))
        xrow = []                                                                 # per context, primary arm and its oracle
        for S1 in P:
            for c in CTX_ORDER:
                m = stop & (ctx == c)
                if m.sum() < 5:
                    continue
                f, orc = F[(S1, "decomp", "gate", "standstill-only")], keep[(S1, 2, "gate", "standstill-only")]
                a_, b_, g_ = C.ci(f - base[S1], m), C.ci(orc - base[S1], m), C.ci(F[(S1, "always go", "gate", "standstill-only")] - base[S1], m)
                xrow.append({"S1": S1, "context": c, "n": int(m.sum()), "oracle go share": float((o2[m] > 0).mean()), "Qwen3 says go": float(D["decomp"][m].mean()),
                             "Qwen3 acc": float((D["decomp"][m] == o2[m]).mean()), "S1 own acc": float(np.mean([(x[m] == o2[m]).mean() for x in CL[2][S1]])),
                             "RFS S1": C.cm(base[S1], m), "fused - S1": a_[0], "fused lo": a_[1], "fused hi": a_[2], "oracle - S1": b_[0], "oracle lo": b_[1], "oracle hi": b_[2],
                             "always go - S1": g_[0], "always go lo": g_[1], "always go hi": g_[2]})
        stats.write_table(xrow, OUT / "fused_context")
        run.info("fused by context:\n%s", pd.DataFrame(xrow).to_string(float_format=lambda v: f"{v:+.3f}"))

        # ---- threshold on p(go) of D-decomp, out of fold by sequence (target: fused RFS on standstill frames; gate, standstill-only)
        scode = pd.factorize(pd.Series(C.seq[:n]))[0]
        ns, pgo = scode.max() + 1, q.decomp_p_go.to_numpy()
        trow = []
        for S1 in P:
            SK = np.stack([score(S1, (pgo >= t).astype(int), 2, "gate", stop) for t in TAUS])                 # (n_tau, n)
            i5 = int(np.flatnonzero(TAUS == 0.5)[0])
            pick = lambda rows: (lambda tot: np.flatnonzero(tot >= tot.max() - 1e-12)[np.abs(np.flatnonzero(tot >= tot.max() - 1e-12) - i5).argmin()])(  # noqa: E731
                (SK[:, rows & stop] * C.w[rows & stop]).sum(1))
            k_in = pick(np.ones(n, bool))
            acc, picks = np.zeros(n), []
            for rep in range(REPEATS):
                fold = (np.random.default_rng(rep).permutation(ns) % FOLDS)[scode]
                for f_ in range(FOLDS):
                    k_ = pick(fold != f_)
                    te = fold == f_
                    acc[te] += SK[k_, np.flatnonzero(te)]
                    picks.append(TAUS[k_])
            oo = acc / REPEATS
            F[(S1, "decomp-tau-oof", "gate", "standstill-only")] = oo
            a_, b_, c_ = C.ci(SK[k_in] - base[S1], stop), C.ci(oo - base[S1], stop), C.ci(SK[i5] - base[S1], stop)
            trow.append({"S1": S1, "tau in-sample": float(TAUS[k_in]), "d stopped in-sample": a_[0], "in lo": a_[1], "in hi": a_[2], "d stopped out-of-fold": b_[0], "oof lo": b_[1],
                         "oof hi": b_[2], "d stopped at tau 0.5": c_[0], "tau over folds (median [min, max])": f"{np.median(picks):.1f} [{min(picks):.1f}, {max(picks):.1f}]",
                         "curve (tau: d stopped)": ", ".join(f"{t:.1f}: {C.cm(SK[j] - base[S1], stop):+.3f}" for j, t in enumerate(TAUS))})
        stats.write_table(trow, OUT / "threshold")
        run.info("threshold:\n%s", pd.DataFrame(trow).T.to_string())

        # ---- B2. Qwen3 as the selector on F20 (path drawn on the image, speed in text)
        pi = q.path.map({pp[0]: j for j, pp in enumerate(PATHS)}).to_numpy().astype(int)
        si = q.speed4.map({k: j for j, k in enumerate(SPEEDS)}).to_numpy().astype(int)
        J = [SETS["F20"][s_][0] for s_ in (0, 1)]                                  # (20, n) per seed
        pick = lambda a_, b_: np.mean([j_[a_ * 4 + b_, ii] for j_ in J], 0)        # noqa: E731
        QS_ = {"joint": pick(pi, si), "path only": pick(pi, 0 * si), "speed only": pick(0 * pi + KEEP, si)}
        best = SEL["F20"]["joint"]
        srow = []
        for arm, f in QS_.items():
            for sn, m in BST.items():
                c, g = C.ci(f - base["WP2"], m), C.ci(f - best, m)
                srow.append({"arm": f"Qwen3 {arm}", "stratum": sn, "n": int(m.sum()), "RFS WP2": C.cm(base["WP2"], m), "RFS selected": C.cm(f, m), "d vs WP2": c[0], "lo": c[1],
                             "hi": c[2], "oracle F20 - WP2": C.cm(best - base["WP2"], m), "vs oracle F20": g[0], "vs oracle lo": g[1], "vs oracle hi": g[2]})
        stats.write_table(srow, OUT / "selector")
        sd = pd.DataFrame(srow)
        run.info("selector:\n%s", sd.to_string(float_format=lambda v: f"{v:+.3f}"))
        tol, grow = 1e-9, []
        mx = [j_.max(0) for j_ in J]
        pbest = [j_.reshape(5, 4, n).max(1) for j_ in J]                           # (5, n) best over speeds per path
        sbest = [j_.reshape(5, 4, n).max(0) for j_ in J]                           # (4, n) best over paths per speed
        for sn, m in BST.items():
            mean2 = lambda fn: float(np.mean([fn(s_)[m].mean() for s_ in (0, 1)]))  # noqa: E731
            grow.append({"stratum": sn, "n": int(m.sum()),
                         "top-1: Qwen3 joint pick reaches the oracle best": mean2(lambda s_: J[s_][pi * 4 + si, ii] >= mx[s_] - tol),
                         "top-1: keep / follow (WP2 itself) reaches it": mean2(lambda s_: J[s_][KEEP * 4, ii] >= mx[s_] - tol),
                         "path: Qwen3's path can reach it": mean2(lambda s_: pbest[s_][pi, ii] >= mx[s_] - tol),
                         "path: keep can reach it": mean2(lambda s_: pbest[s_][KEEP] >= mx[s_] - tol),
                         "speed: Qwen3's speed can reach it": mean2(lambda s_: sbest[s_][si, ii] >= mx[s_] - tol),
                         "speed: follow can reach it": mean2(lambda s_: sbest[s_][0] >= mx[s_] - tol),
                         "Qwen3 path counts A/B/C/D/E": " / ".join(str(int((pi[m] == j).sum())) for j in range(5)),
                         "Qwen3 speed counts follow/hold/creep/go": " / ".join(str(int((si[m] == j).sum())) for j in range(4)),
                         "oracle path counts A/B/C/D/E (seed 0; ties to keep / follow)": " / ".join(str(int(((first_best(J[0]) // 4)[m] == j).sum())) for j in range(5)),
                         "oracle speed counts follow/hold/creep/go (seed 0; ties to keep / follow)": " / ".join(str(int(((first_best(J[0]) % 4)[m] == j).sum()))
                                                                                                                 for j in range(4))})
        stats.write_table(grow, OUT / "selector_agreement")
        run.info("selector agreement:\n%s", pd.DataFrame(grow).T.to_string())
        F[("WP2", "sel_joint", "", "all")], F[("WP2", "sel_path", "", "all")], F[("WP2", "sel_speed", "", "all")] = QS_["joint"], QS_["path only"], QS_["speed only"]

        # ---- latency
        qs = list(QS) + ["direct", "decomp", "decomp3", "block", "path", "speed4"]
        lrow = [{"forward": k, "images": len(QS[k][3]) if k in QS else {"block": 3, "path": 1}.get(k, 4), "n": int(q[f"{k}_ms_fwd"].iloc[10:].notna().sum()), "prompt tokens (median)": float(q[f"{k}_tokens"].median()),
                 "preprocess ms (median)": float(q[f"{k}_ms_pre"].iloc[10:].median()), "model ms (median)": float(q[f"{k}_ms_fwd"].iloc[10:].median()),
                 "model ms (p95)": float(q[f"{k}_ms_fwd"].iloc[10:].quantile(0.95))} for k in qs]
        tot = lambda ks, c: float(sum(q[f"{k}_ms_{c}"].iloc[10:] for k in ks).median())  # noqa: E731
        for nm, ks in (("decision D-direct (1 forward)", ["direct"]), ("decision D-decomp (5 + 1 forwards)", list(QS) + ["decomp"]), ("decision D-rule (5 forwards)", list(QS)),
                       ("selector: 6 scene questions + path + speed (8 forwards)", list(QS) + ["block", "path", "speed4"])):
            lrow.append({"forward": nm, "images": np.nan, "n": n - 10, "preprocess ms (median)": tot(ks, "pre"), "model ms (median)": tot(ks, "fwd")})
        stats.write_table(lrow, OUT / "latency", floatfmt=".1f")
        run.info("latency:\n%s", pd.DataFrame(lrow).to_string())

        # ---- per-frame tables and the verdict
        pf = pf.assign(qwen_decomp=q.decomp.to_numpy(), qwen_decomp_p_go=pgo, qwen_direct=q.direct.to_numpy(), qwen_rule=np.where(D["rule"] > 0, "go", "hold"),
                       qwen_decomp3=q.decomp3.to_numpy(), qwen_block=q.block.to_numpy(), qwen_path=q.path.to_numpy(), qwen_speed4=q.speed4.to_numpy(),
                       rfs_WP2_qwen_sel_joint=QS_["joint"], rfs_WP2_qwen_sel_path=QS_["path only"], rfs_WP2_qwen_sel_speed=QS_["speed only"],
                       **{f"rfs_{S1}_qwen_{dn}_{mode}_{'stop' if sc == 'standstill-only' else 'all'}": v for (S1, dn, mode, sc), v in F.items()
                          if dn in ("decomp", "direct", "rule", "decomp3", "always go")})
        pf.to_csv(OUT / "frames.csv", index=False, float_format="%.4f")
        pri, pri_all = fd[(fd.S1 == "WP2") & (fd.decision == "Qwen3 decomp") & (fd.fusion == "gate") & (fd.scope == "standstill-only")].iloc[0], \
            fd[(fd.S1 == "WP2") & (fd.decision == "Qwen3 decomp") & (fd.fusion == "gate") & (fd.scope == "all")].iloc[0]
        orc = od[(od.S1 == "WP2") & od.decision.str.startswith("oracle: class") & (od.K == 2) & (od.fusion == "gate") & (od.scope == "standstill-only")].iloc[0]
        sj = sd[(sd.arm == "Qwen3 joint") & (sd.stratum == "all")].iloc[0]
        f20 = bd[(bd.set == "F20") & (bd.stratum == "all")].iloc[0]
        stand_ok = bool(pri["d stopped lo"] > 0)
        mov_ok = bool(pri_all["d moving hi"] >= 0 and pri_all["d moving"] >= -0.05)
        verdict = {"oracle V2 gate has value (stopped CI lower bound > 0)": bool(orc["d stopped lo"] > 0),
                   "oracle V2 gate d stopped": [orc["d stopped"], orc["d stopped lo"], orc["d stopped hi"]],
                   "primary d stopped (standstill-only)": [pri["d stopped"], pri["d stopped lo"], pri["d stopped hi"]],
                   "primary d moving (scope all)": [pri_all["d moving"], pri_all["d moving lo"], pri_all["d moving hi"]],
                   "standstill criterion": stand_ok, "moving criterion (scope all)": mov_ok,
                   "System 2 delivers": ("yes" if stand_ok and mov_ok else "only when gated to standstill by ego speed" if stand_ok else
                                         "harmful" if pri["d stopped hi"] < 0 else "weak / undecided" if pri["d stopped"] > 0 else "no"),
                   "B: oracle F20 U WP2 d all": [f20["d joint"], f20["d joint lo"], f20["d joint hi"]],
                   "B: Qwen3 joint selection d all": [sj["d vs WP2"], sj["lo"], sj["hi"]],
                   "B: selector delivers": ("yes" if sj["lo"] > 0 else "harmful" if sj["hi"] < 0 else "weak / undecided" if sj["d vs WP2"] > 0 else "no"),
                   "prototype self-consistency (share of decoded prototypes that fall in their own class)": proto_self,
                   "decomposed answers identical to wod-launch labels": same, "n": n, "B": R.B, "split": val.id, "codebook split": cb["split"]}
        (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float))
        run.info("verdict: %s", json.dumps(verdict, default=float))
        run.summary.update(delivers=verdict["System 2 delivers"], d_stopped=float(pri["d stopped"]), oracle_d_stopped=float(orc["d stopped"]))


# ---------------------------------------------------------------- overlay check
def cmd_overlay(a):
    from PIL import Image
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from pp_wod import load_preds
    r = Z.load_sets()["rater"]
    names, v0 = r["name"].astype(str), W.init_speed(r["past"]).astype(np.float64)
    cb, cal = json.loads((OUT / "codebook.json").read_text()), json.loads((Z.root() / "op_calib.json").read_text())
    idx = np.random.default_rng(0).choice(len(names), a.n, replace=False)
    wp2 = load_preds("WP2-full-s0", names[idx])
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(idx):
        im = overlay(Image.open(Z.root("t2_jpg") / names[i] / "1.jpg").convert("RGB"), cal[names[i].rsplit("-", 1)[0]]["1"], wp2[j], v0[i], cb)
        im.resize((im.width // 2, im.height // 2)).save(out / f"{j:02d}_{names[i][:8]}_v{v0[i]:.0f}.jpg", quality=85)
        print(names[i], round(float(v0[i]), 2), flush=True)


# ---------------------------------------------------------------- figures
def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import wod_launch_report as R
    import wod_zeroshot_openpilot as H
    from PIL import Image
    from jevdrive import wod_zeroshot as Z
    from jevdrive.common import data_dir
    from wod_launch_figs import rgb
    FIG.mkdir(parents=True, exist_ok=True)
    cb, cal = json.loads((OUT / "codebook.json").read_text()), json.loads((Z.root() / "op_calib.json").read_text())
    C = R.Ctx()
    n, v0, ii = C.n, C.v0, np.arange(C.n)
    names = C.names[:n]
    pf = pd.read_csv(OUT / "frames.csv").set_index("name").reindex(names)
    q = pd.read_csv(OUT / "qwen.csv").set_index("name").reindex(names)
    stop = pf.stopped.to_numpy().astype(bool)
    wp2 = C.preds("WP2-full-s0")
    dq, do = (q.decomp == "go").to_numpy().astype(int), pf.oracle_V2.to_numpy()
    fq, fo = fuse(wp2, v0, dq, 2, cb, "gate", stop), fuse(wp2, v0, do, 2, cb, "gate", stop)
    F20 = factored(wp2, v0, 3, cb)                                              # (n, 20, 20, 2)
    J = np.stack([C.rfs(np.ascontiguousarray(F20[:, m])) for m in range(20)])
    pi = q.path.map({pp[0]: j for j, pp in enumerate(PATHS)}).to_numpy().astype(int)
    si = q.speed4.map({k: j for j, k in enumerate(SPEEDS)}).to_numpy().astype(int)
    jb = first_best(J)
    fs, fb = F20[ii, pi * 4 + si], F20[ii, jb]
    s_w, s_q, s_o, s_l, s_s, s_b = C.rfs(wp2), C.rfs(fq), C.rfs(fo), C.rfs(C.fut[:n]), C.rfs(fs), C.rfs(fb)
    d = (pf.rfs_WP2_qwen_decomp_gate_stop - pf.rfs_WP2).to_numpy()            # seed-mean change of primary arm A
    sel = []
    for c, picks in (("stop_sign", ("gain", "loss")), ("green", ("gain", "loss")), ("red", ("loss",)), ("lead", ("loss",)), ("open", ("gain", "loss"))):
        i = np.flatnonzero(stop & (pf.context.to_numpy() == c))
        for w in picks:
            j = int(i[d[i].argmax()] if w == "gain" else i[d[i].argmin()])
            if w == "loss" and d[j] >= 0:
                j, w = int(i[q.decomp_p_go.to_numpy()[i].argmax()]), "no loss in the class: highest p(go)"
            if j not in [s_[2] for s_ in sel]:
                sel.append((f"standstill, {c}", f"go / hold {w}", j))
    ds = (pf.rfs_WP2_qwen_sel_joint - pf.rfs_WP2).to_numpy()                 # seed-mean change of the joint selection (arm B)
    mv = np.flatnonzero(~stop & (pi != KEEP))
    if len(mv) < 6:
        mv = np.flatnonzero(~stop & ((pi != KEEP) | (si != 0)))
    o = mv[np.argsort(-ds[mv], kind="stable")]
    for w, js in (("selector gain", o[:3]), ("selector loss", o[::-1][:3])):
        for r_, j in enumerate(js):
            if int(j) not in [s_[2] for s_ in sel]:
                sel.append((f"moving, {pf.cluster.iloc[j]}", f"{w} {r_ + 1}", int(j)))
    spans, _ = Z.load_spans()
    H._init(spans, json.loads((Z.root() / "op_calib.json").read_text()), str(data_dir() / "datasets" / "waymo_e2e" / "front3"))
    t2, rows = Z.root("t2_jpg"), []
    for k, (c, why, i) in enumerate(sel):
        name = names[i]
        _, _, frames = H.model_frames(name)
        fig = plt.figure(figsize=(15.5, 7.4))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.15], height_ratios=[1.45, 1], hspace=0.1, wspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(overlay(Image.open(t2 / name / "1.jpg").convert("RGB"), cal[name.rsplit("-", 1)[0]]["1"], wp2[i], v0[i], cb))
        ax.set_title("FRONT image given to Qwen3 for the path choice (t0; A-E = candidate paths of WP2)", fontsize=8.5, loc="left")
        ax.axis("off")
        ax = fig.add_subplot(gs[0, 1])
        ax.axis("off")
        r = q.iloc[i]
        d3 = r.decomp3 if isinstance(r.decomp3, str) else "-"
        txt = ["Qwen3-VL-4B, zero-shot (p of the answer)", f"light: {r.light} ({r['light_p_' + r.light]:.2f})", f"stop_ctrl: {r.stop_ctrl} ({r['stop_ctrl_p_' + r.stop_ctrl]:.2f})",
               f"right of way: {r.row} ({r['row_p_' + r.row]:.2f})", f"lead: {r['lead']} ({r['lead_p_' + r['lead']]:.2f})", f"crossing: {r.cross} ({r['cross_p_' + r.cross]:.2f})",
               f"lane obstruction: {r.block} ({r['block_p_' + r.block]:.2f})", "",
               f"A  go / hold (D-decomp): {r.decomp.upper()} (p go {r.decomp_p_go:.2f})", f"   D-direct: {r.direct} (p go {r.direct_p_go:.2f}); 3-way: {d3}",
               f"   oracle V2 (privileged): {'GO' if do[i] else 'HOLD'}", f"   WP2 own: {'go' if pf.WP2_s0_V2.iloc[i] else 'hold'}; log: {'go' if pf.log_V2.iloc[i] else 'hold'}", "",
               f"B  path: {r.path} = {PATHS[pi[i]][1]} ({r['path_p_' + r.path]:.2f})", f"   speed: {r.speed4} ({r['speed4_p_' + r.speed4]:.2f})",
               "   oracle best of F20 (privileged):", f"   {PATHS[jb[i] // 4][0]} = {PATHS[jb[i] // 4][1]} / {SPEEDS[jb[i] % 4]}", "", f"{c}", f"cluster {pf.cluster.iloc[i]}"]
        ax.text(0.0, 0.98, "\n".join(txt), va="top", ha="left", fontsize=8.2, family="monospace", transform=ax.transAxes)
        for m_, lab in ((0, "road frame fed to openpilot (t0)"), (1, "wide frame fed to openpilot (t0)")):
            ax = fig.add_subplot(gs[1, m_])
            ax.imshow(rgb(frames[-1, m_]))
            ax.set_title(lab, fontsize=9, loc="left")
            ax.axis("off")
        ax = fig.add_subplot(gs[:, 2])
        order = np.argsort(-C.sc[i])
        for j, o_ in enumerate(order):
            t = C.traj[i, o_]
            ax.plot(-np.r_[0, t[:, 1]], np.r_[0, t[:, 0]], "-" if j == 0 else "--", color="k" if j == 0 else "0.55", lw=2.2 if j == 0 else 1.3,
                    label=f"rater {j + 1}: score {C.sc[i, o_]:.0f}, {np.linalg.norm(t[19]):.1f} m at 5 s")
            ax.plot(-t[[11, 19], 1], t[[11, 19], 0], "o", color="k" if j == 0 else "0.55", ms=4)
        arms = [("log", C.fut[i], s_l[i], "#ff7f0e", "-", 1.4), ("WP2 (seed 0)", wp2[i], s_w[i], "#d62728", "-", 2.2)]
        if stop[i]:
            arms += [("A: WP2 re-timed by Qwen3 go / hold", fq[i], s_q[i], "#2a78d6", "-", 1.8), ("A: WP2 re-timed by the oracle V2", fo[i], s_o[i], "#2a78d6", (0, (3, 2)), 1.8)]
        arms += [(f"B: Qwen3's pick {r.path} / {r.speed4}", fs[i], s_s[i], "#1baf7a", "-", 1.8),
                 (f"B: oracle best of F20 {PATHS[jb[i] // 4][0]} / {SPEEDS[jb[i] % 4]}", fb[i], s_b[i], "#1baf7a", (0, (3, 2)), 1.8)]
        for lab, p, s_, col, ls, lw in arms:
            ax.plot(-np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], color=col, ls=ls, lw=lw, label=f"{lab}: RFS {s_:.1f}, {np.linalg.norm(p[19]):.1f} m")
            ax.plot(-p[[11, 19], 1], p[[11, 19], 0], "^", color=col, ms=5)
        ax.plot(0, 0, "k*", ms=10)
        allp = np.concatenate([C.traj[i].reshape(-1, 2), C.fut[i]] + [x[1] for x in arms])
        ax.set_xlim(-max(3.0, 1.2 * np.abs(allp[:, 1]).max()), max(3.0, 1.2 * np.abs(allp[:, 1]).max()))
        ax.set_ylim(-0.05 * max(3.0, allp[:, 0].max()), 1.1 * max(3.0, allp[:, 0].max()))
        ax.set_xlabel("lateral (m, left is left; axis stretched)")
        ax.set_ylabel("forward (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.3, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=2)
        ax.set_title(f"{why}; v0 {v0[i]:.2f} m/s; markers at 3 s and 5 s", fontsize=9)
        fig.suptitle(f"{k:02d}  {name}", fontsize=9, x=0.02, ha="left")
        fn = FIG / f"{k:02d}_{c.split(', ')[0]}_{c.split(', ')[1].replace(' ', '_').replace('-', '_')}_{why.split(' ')[-1 if stop[i] else 1]}_{name[:8]}.png"
        fig.savefig(fn, dpi=105, bbox_inches="tight")
        plt.close(fig)
        rows.append({"figure": fn.name, "frame": name, "group": c, "pick": why, "cluster": pf.cluster.iloc[i], "v0": v0[i], "qwen go/hold": r.decomp, "p_go": r.decomp_p_go,
                     "oracle_V2": "go" if do[i] else "hold", "WP2 own": "go" if pf.WP2_s0_V2.iloc[i] else "hold", "light": r.light, "stop_ctrl": r.stop_ctrl, "row": r.row,
                     "lead": r["lead"], "cross": r.cross, "block": r.block, "qwen path": r.path, "qwen speed": r.speed4,
                     "oracle F20": f"{PATHS[jb[i] // 4][0]}/{SPEEDS[jb[i] % 4]}", "scores": " ".join(f"{x:.0f}" for x in C.sc[i][order]),
                     "top d5": np.linalg.norm(C.traj[i, order[0], 19]), "log d5": np.linalg.norm(C.fut[i, 19]), "WP2 d5": np.linalg.norm(wp2[i, 19]),
                     "go/hold fused d5": np.linalg.norm(fq[i, 19]), "RFS WP2 s0": s_w[i], "RFS go/hold fused s0": s_q[i], "RFS oracle V2 s0": s_o[i],
                     "RFS selector s0": s_s[i], "RFS oracle F20 s0": s_b[i], "RFS log": s_l[i]})
    pd.DataFrame(rows).to_csv(OUT / "figures.csv", index=False, float_format="%.2f")
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    sp_.add_parser("fit")
    p = sp_.add_parser("qwen")
    p.add_argument("--limit", type=int, default=0)
    sp_.add_parser("report")
    p = sp_.add_parser("overlay")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", required=True)
    sp_.add_parser("figs")
    a = ap.parse_args()
    {"fit": cmd_fit, "qwen": cmd_qwen, "report": cmd_report, "figs": cmd_figs, "overlay": cmd_overlay}[a.cmd](a)
