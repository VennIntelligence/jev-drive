"""Q-light sweep: can a model tell the ego traffic light's state at a junction, and does the wording change that?

  python vlm_arb_lightsweep.py select            frames.csv, items.jsonl (frame x prompt), crops
  python vlm_arb_lightsweep.py run --model NAME  one model over items.jsonl (replies cached, a rerun skips them)
  python vlm_arb_lightsweep.py report            results/lightsweep.{md,json}

An exploratory sweep (plan deviation D16): every prompt is read on the same frames, every variant is reported, and a
prompt found here unlocks no closed-loop arm. It replaces the Q-light diagnosis (select on debug frames, read once on
dev frames) and the all-frame model comparison.

Frames, fixed before any answer on the subset was read: the answered requests with both frames of the shadow units
UNITS, nothing new is collected.
  signalized  a segment = consecutive requests (gap <= 2 s) of one route with the same ego light and the same class
              (red or yellow / green), ego light within 40 m of its stop line
              per segment at most 5 requests, evenly spaced in time (first and last included)
  no light    per unit and route, the requests with no ego light and no light actor within 60 m; those with a junction
              entry within 40 m ahead where the route has any (in the collected data only route 37969 does: the
              first form of this rule, junction approaches only, gave 5 frames on one route and was widened before
              any answer on the subset was read), else all of them; 10 requests evenly spaced in time

Prompts (PROMPTS): the four-choice Q_light question as run; a plain colour question; describe every light head, then
pick the facing one; the far-side mounting convention spelled out; lamp position instead of hue; and the plain
question on the road camera alone, the wide camera alone, and with an upper-centre crop of the wide frame added.
Every reply ends in `ANSWER: <option>`; yellow counts as red, "light for another lane" as none, an unparseable reply
is no answer (it stays in the denominator). Models: vlm_arb_models.MODELS as chat models, and `openjev` (the lane's
System One endpoint, constrained answers) on the variants it has (lib/vlm_protocol.VARIANTS).

Readout per model x prompt, route-cluster bootstrap where a cell has >= 3 routes, else raw counts: ego-red recall, red
answered green, ego-green recall, no-light answered red; the two red readouts again by "another approach green" /
"no other light green" and for route 27043 / the rest. Guard against a lucky prompt: routes are split in two halves
(sorted ids, alternating); per model the prompt with the best score (red recall - red as green + green recall -
no-light false alarm) on one half is read on the other, both ways.
Also: on route 27043 the lit lamps found in the frames by the logged answer, and the junctions' light structure.
"""
import argparse
import colorsys
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import REPO, RUN, boot_ratio, write_json  # noqa: E402
from vlm_arb_models import MODELS, WORKER, box_quiet  # noqa: E402

sys.path.insert(0, str(REPO / "lib"))
from vlm_client import VLMClient, crop_wide  # noqa: E402

OUT = RUN / "results/lightsweep"
UNITS = ["v2-dbg-shadow-s0-light", "v2-drive-s1-dev", "v2-drive-s1-tgt", "v2-drive-s0-tgt"]
PER_SEGMENT, PER_ROUTE_NONE, NEAR_M, GAP_S = 5, 10, 40.0, 2.0

CAMS = {("wide", "road"): "The two images were taken at the same instant by the front cameras of a car (the ego vehicle): "
                          "image 1 is the wide-angle camera, image 2 is the narrow road camera.",
        ("road",): "The image was taken by the narrow front camera of a car (the ego vehicle).",
        ("wide",): "The image was taken by the wide-angle front camera of a car (the ego vehicle).",
        ("wide", "road", "crop"): "The three images were taken at the same instant by the front cameras of a car (the ego "
                                  "vehicle): image 1 is the wide-angle camera, image 2 is the narrow road camera, image 3 "
                                  "is an enlarged crop of the upper centre of image 1."}
COLOURS = "red, yellow, green, none"
Q_FOUR = ("Traffic light status controlling the ego vehicle lane ahead. Options:\n"
          "  no_light: no traffic light controlling ego lane ahead\n"
          "  red_or_yellow_for_ego: red or yellow traffic light controlling ego lane ahead\n"
          "  green_for_ego: green traffic light controlling ego lane ahead\n"
          "  light_for_other_lane: traffic light visible but controlling another lane, not ego lane")
Q_PLAIN = ("What colour is the traffic light that applies to you (the ego vehicle) right now: red, yellow, green, or none "
           "if no traffic light applies to you?")
Q_DESCRIBE = ("First list every traffic light head visible, one per line, with its position (left / centre / right of the "
              "image, near or far side of the junction) and the colour of its lit lamp. Then pick the one that faces the "
              "ego vehicle across the junction it is approaching and give its colour: red, yellow, green, or none if no "
              "traffic light applies to the ego vehicle.")
Q_CONVENTION = ("Convention of this road network: the traffic lights of a junction are mounted on its far side, one light "
                "per approach. The light that applies to the ego vehicle is the one facing it from across the junction, "
                "straight ahead beyond the stop line. The lights of the other approaches are visible too, from the side or "
                "at an angle, and may show a different colour: ignore them. What colour is the light that applies to the "
                "ego vehicle: red, yellow, green, or none if the ego vehicle is not approaching a traffic light?")
Q_POSITION = ("Find the traffic light that faces the ego vehicle at the junction ahead; ignore lights seen from the side "
              "that serve cross traffic. Decide by which lamp of its vertical housing is lit, not by the hue: top lamp lit "
              "= red, middle lamp lit = yellow, bottom lamp lit = green. Answer red, yellow, green, or none if no traffic "
              "light faces the ego vehicle.")
FOUR_OPTS = "no_light, red_or_yellow_for_ego, green_for_ego, light_for_other_lane"
# name -> (cameras, question, options, max new tokens, the openjev System One variant with the same input, if any)
PROMPTS = {
    "four": (("wide", "road"), Q_FOUR, FOUR_OPTS, 160, "base"),
    "plain": (("wide", "road"), Q_PLAIN, COLOURS, 160, None),
    "describe": (("wide", "road"), Q_DESCRIBE, COLOURS, 384, None),
    "convention": (("wide", "road"), Q_CONVENTION, COLOURS, 160, None),
    "position": (("wide", "road"), Q_POSITION, COLOURS, 160, "pos"),
    "plain_road": (("road",), Q_PLAIN, COLOURS, 160, None),
    "plain_wide": (("wide",), Q_PLAIN, COLOURS, 160, None),
    "plain_crop": (("wide", "road", "crop"), Q_PLAIN, COLOURS, 160, None),
}
OPENJEV = {"four": "base", "four_road": "road", "four_crop": "crop", "position": "pos"}    # System One variants
CLASS = {"red": "red", "yellow": "red", "amber": "red", "orange": "red", "red_or_yellow_for_ego": "red", "green": "green",
         "green_for_ego": "green", "none": "none", "no": "none", "no_light": "none", "light_for_other_lane": "none"}


def prompt_text(name):
    cams, q, opts, _, _ = PROMPTS[name]
    tail = "End your reply with one line of the form `ANSWER: <option>`, <option> being one of: %s." % opts
    return "%s\n%s\n%s%s" % (CAMS[cams], q, tail, "" if name == "describe" else " Reply with that line only.")


def parse(text):
    """Reply -> red | green | none | "" (no parseable answer). The last `ANSWER:` wins; a bare option also counts."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.findall(r"ANSWER\s*[:=]\s*[\"'`<*\s]*([A-Za-z_]+)", text, flags=re.I)
    word = (m[-1] if m else text.strip("\"'`.*<> \n")).lower()
    return CLASS.get(word, "")


# ---------------------------------------------------------------------------------------------- frames
def load_requests():
    rows = []
    for u in UNITS:
        udir = RUN / "arms" / u
        for done in sorted(udir.glob("done/*.json")):
            a = udir / "attempts" / done.stem / str(json.loads(done.read_text()).get("attempt", 1))
            if not (a / "vlm_decisions.jsonl").exists():
                continue
            for line in open(a / "vlm_decisions.jsonl"):
                d = json.loads(line)
                if d.get("k") != "a" or not d["ans"].get("ok"):
                    continue
                f = {c: a / "vlm_frames" / ("%08.2f_%s.jpg" % (d["t_q"], c)) for c in ("wide", "road")}
                if not all(p.exists() for p in f.values()):
                    continue
                g = d["gt"]
                lights = g.get("lights", [])
                rows.append(dict(id="%s/%s/%.2f" % (u, done.stem, d["t_q"]), unit=u, route=done.stem, t=d["t_q"],
                                 tl=-1 if g.get("tl") is None else g["tl"], tl_id=g.get("tl_id"),
                                 tl_dist=np.nan if g.get("tl") is None else g["tl_dist"], junc_dist=g.get("junc_dist", 999.0),
                                 n_lights=len(lights), lights=json.dumps(lights),
                                 other_green=any(x[1] == 0 for x in lights if x[0] != g.get("tl_id")),
                                 logged=CLASS.get(d["ans"].get("Q_light", ""), ""), wide=str(f["wide"]), road=str(f["road"])))
    return pd.DataFrame(rows)


def segments(df):
    """Label every request near an ego light with its segment (see the module docstring), the others with ''."""
    seg, n = [], 0
    for _, g in df.sort_values("t").groupby(["unit", "route"], sort=True):
        prev = None
        for r in g.itertuples():
            if r.tl >= 0 and r.tl_dist <= NEAR_M:
                key = ("light", r.tl_id, "red" if r.tl in (1, 2) else "green")
            else:
                key = None
            if key is not None and (prev is None or prev[0] != key or r.t - prev[1] > GAP_S):
                n += 1
            seg.append((r.Index, "" if key is None else "%s-%04d" % (key[0], n)))
            prev = None if key is None else (key, r.t)
    return pd.Series(dict(seg)).reindex(df.index)


def spaced(g, n):
    return list(g.index[np.unique(np.round(np.linspace(0, len(g) - 1, min(n, len(g)))).astype(int))])


def select(df):
    df = df.assign(seg=segments(df))
    keep = []
    for _, g in df[df.seg != ""].sort_values("t").groupby("seg", sort=True):
        keep += spaced(g, PER_SEGMENT)
    for (u, r), g in df[(df.tl < 0) & (df.n_lights == 0)].sort_values("t").groupby(["unit", "route"], sort=True):
        near = g[g.junc_dist <= NEAR_M]
        pick = spaced(near if len(near) else g, PER_ROUTE_NONE)
        df.loc[pick, "seg"] = "none-%s-%s" % (u, r)
        keep += pick
    s = df.loc[keep].sort_values("id").reset_index(drop=True)
    s["truth"] = np.where(s.tl.isin([1, 2]), "red", np.where(s.tl == 0, "green", "none"))
    routes = sorted(s.route.unique(), key=lambda x: int(x))
    s["half"] = s.route.map({r: "AB"[i % 2] for i, r in enumerate(routes)})
    return s


def cmd_select():
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    s = select(load_requests())
    items = []
    for r in s.itertuples():
        crop = OUT / "crops" / (r.id.replace("/", "_") + ".jpg")
        if not crop.exists():
            crop.write_bytes(crop_wide(Path(r.wide).read_bytes()))
        path = dict(wide=r.wide, road=r.road, crop=str(crop))
        for name, (cams, _, _, max_new, _) in PROMPTS.items():
            items.append(dict(id="%s|%s" % (r.id, name), images=[path[c] for c in cams], prompt=prompt_text(name), max_new=max_new))
    s.to_csv(OUT / "frames.csv", index=False)
    tmp = OUT / "items.jsonl.tmp"
    tmp.write_text("".join(json.dumps(x) + "\n" for x in items))
    tmp.replace(OUT / "items.jsonl")
    comp = dict(frames=len(s), requests=len(items), routes=int(s.route.nunique()), segments=int(s.seg.nunique()),
                truth=s.truth.value_counts().to_dict(), red_other_green=int(((s.truth == "red") & s.other_green).sum()),
                red_27043=int(((s.truth == "red") & (s.route == "27043")).sum()), by_unit=s.unit.value_counts().to_dict())
    write_json(OUT / "selection.json", comp)
    print(json.dumps(comp))
    return 0


# ---------------------------------------------------------------------------------------------- run
def run_openjev(s):
    """The System One endpoint on its own variants (constrained answers, Q_light alone), sequential."""
    out = OUT / "openjev.jsonl"
    have = {json.loads(x)["id"] for x in open(out)} if out.exists() else set()
    cli, load = VLMClient(timeout_s=30.0), int(not box_quiet())
    with open(out, "a") as f:
        for r in s.itertuples():
            jp = {"wide": Path(r.wide).read_bytes(), "road": Path(r.road).read_bytes()}
            for name, variant in OPENJEV.items():
                rid = "%s|%s" % (r.id, name)
                if rid in have:
                    continue
                a = cli.ask(jp, variant, only_light=True)
                row = dict(id=rid, load=load, lat_ms=a["latency_ms"])
                row.update(raw="ANSWER: " + a.get("Q_light", "")) if a["ok"] else row.update(err=a.get("error", ""))
                f.write(json.dumps(row) + "\n")
                f.flush()
    return 0


def cmd_run(name):
    s = pd.read_csv(OUT / "frames.csv")
    if name == "openjev":
        return run_openjev(s)
    m = next(x for x in MODELS if x["name"] == name)
    e = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
    cmd = [m["py"], str(WORKER), "--kind", m["kind"], "--path", m["path"], "--sample", str(OUT / "items.jsonl"),
           "--out", str(OUT / (name + ".jsonl")), "--load", str(int(not box_quiet()))]
    with open(OUT / (name + ".log"), "a") as log:
        log.write("\n==> %s  %s\n" % (time.strftime("%F %T"), " ".join(cmd)))
        log.flush()
        rc = subprocess.run(cmd, env=e, stdout=log, stderr=subprocess.STDOUT).returncode
    print("%s: worker exit %d" % (name, rc), flush=True)
    return 0                                            # a model that does not load is a row of the table, not a lane error


# ---------------------------------------------------------------------------------------------- report
def cell(hit, den, routes):
    """estimate [95% route-cluster CI] k/n, or raw counts when fewer than 3 routes carry the cell."""
    n, k = int(den.sum()), int((hit & den).sum())
    if n == 0:
        return "n/a", dict(n=0, k=0)
    r = boot_ratio(hit & den, den, routes)
    txt = "%.0f%% [%.0f, %.0f] %d/%d" % (100 * r["est"], 100 * r["lo"], 100 * r["hi"], k, n) if r["groups"] >= 3 else "%d/%d" % (k, n)
    return txt, dict(n=n, k=k, est=r["est"], lo=r["lo"], hi=r["hi"], routes=r["groups"])


COLS = [("red_recall", "ego red: answered red"), ("red_green", "ego red: answered green"), ("green_recall", "ego green: answered green"),
        ("none_fp", "no light: answered red")]
SPLITS = [("og", "another approach green"), ("allred", "no other light green"), ("r27043", "route 27043"), ("ex27043", "without 27043")]


def readouts(s, ans):
    red, green, none, r27 = s.truth == "red", s.truth == "green", s.truth == "none", s.route.astype(str) == "27043"
    R, T = {}, {}
    def put(k, hit, den):
        T[k], R[k] = cell(hit, den, s.route)
    put("red_recall", ans == "red", red)
    put("red_green", ans == "green", red)
    put("green_recall", ans == "green", green)
    put("none_fp", ans == "red", none)
    for tag, m in (("og", s.other_green), ("allred", ~s.other_green), ("r27043", r27), ("ex27043", ~r27)):
        put("red_recall_" + tag, ans == "red", red & m)
        put("red_green_" + tag, ans == "green", red & m)
    R["unparsed"] = int((ans == "").sum())
    R["score"] = float(sum(sg * R[k].get("est", 0.0) for k, sg in (("red_recall", 1), ("red_green", -1), ("green_recall", 1), ("none_fp", -1))))
    return R, T


def answers(s, name):
    """{prompt: (answers aligned to s, latency ms, requests answered)} from one model's cached replies."""
    p = OUT / (name + ".jsonl")
    rep = {}
    for line in (open(p) if p.exists() else ()):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("err"):
            rep[r["id"]] = r
    out = {}
    for pr in (OPENJEV if name == "openjev" else PROMPTS):
        got = [rep.get("%s|%s" % (i, pr)) for i in s.id]
        if any(got):
            out[pr] = (pd.Series([parse(g["raw"]) if g else "" for g in got], index=s.index),
                       [g["lat_ms"] for g in got if g and g.get("load") == 0] or [g["lat_ms"] for g in got if g],
                       sum(bool(g) for g in got), all(g.get("load") == 0 for g in got if g))
    return out


def hue_name(h):
    return "red" if h < 20 or h >= 330 else "amber" if h < 70 else "green" if h < 190 else "other"


def lamps(path):
    """Lit lamps in the upper 55% of a frame: {hue name: count} over connected blobs of >= 4 very bright saturated pixels."""
    from scipy import ndimage
    img = Image.open(path).convert("RGB")
    a = np.asarray(img, np.float32)[: int(img.height * .55)] / 255.0
    mx, mn = a.max(-1), a.min(-1)
    lab, n = ndimage.label((mx > 0.9) & ((mx - mn) / np.maximum(mx, 1e-6) > 0.35))
    out = {"red": 0, "amber": 0, "green": 0, "other": 0}
    for i in range(1, n + 1):
        px = a[lab == i]
        if len(px) >= 4:
            out[hue_name(360.0 * colorsys.rgb_to_hsv(*map(float, px.mean(0)))[0])] += 1
    return out


def in_view_27043(df):
    """Route 27043, ego red within 50 m, every answered request: lit lamps found in the two frames, by the logged answer."""
    rows = []
    for r in df[(df.route == "27043") & df.tl.isin([1, 2]) & (df.tl_dist < 50)].itertuples():
        w, n = lamps(r.wide), lamps(r.road)
        rows.append(dict(answer=r.logged or "unparsed", og=r.other_green, wr=w["red"], wa=w["amber"], wg=w["green"],
                         nr=n["red"], na=n["amber"], ng=n["green"]))
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    return pd.DataFrame([{
        "logged answer": k, "frames": len(g), "truth: another approach green": "%d/%d" % (g.og.sum(), len(g)),
        "wide: frames with a green lamp": "%d/%d" % ((g.wg > 0).sum(), len(g)),
        "wide: frames with a red or amber lamp": "%d/%d" % (((g.wr + g.wa) > 0).sum(), len(g)),
        "wide: mean lamps red / amber / green": "%.1f / %.1f / %.1f" % (g.wr.mean(), g.wa.mean(), g.wg.mean()),
        "road: frames with a green lamp": "%d/%d" % ((g.ng > 0).sum(), len(g)),
        "road: frames with a red or amber lamp": "%d/%d" % (((g.nr + g.na) > 0).sum(), len(g)),
        "road: mean lamps red / amber / green": "%.1f / %.1f / %.1f" % (g.nr.mean(), g.na.mean(), g.ng.mean())}
        for k, g in t.groupby("answer")])


def lamp_check(df, n=60):
    """Does the lamp heuristic see green lamps at all? Ego green within 50 m on the other routes, n frames evenly spaced."""
    g = df[(df.route != "27043") & (df.tl == 0) & (df.tl_dist < 50)].sort_values("id")
    g = g.loc[spaced(g, n)] if len(g) else g
    w, r = [lamps(x)["green"] > 0 for x in g.wide], [lamps(x)["green"] > 0 for x in g.road]
    return "Detector check on %d ego-green frames within 50 m of the other routes: a green lamp is found in the wide frame of %d, in the road frame of %d." % (len(g), sum(w), sum(r))


def junctions(df):
    """Per route and ego light: light actors within 60 m, how many are green at once, the ego light's distance rank."""
    rows = []
    for r in df[df.tl >= 0].itertuples():
        ls = sorted(json.loads(r.lights), key=lambda x: x[2])
        ids = [x[0] for x in ls]
        rows.append(dict(route=r.route, ego_light=int(r.tl_id), n=len(ls), green=sum(x[1] == 0 for x in ls),
                         rank=ids.index(r.tl_id) + 1 if r.tl_id in ids else np.nan))
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    g = t.groupby(["route", "ego_light"])
    return pd.DataFrame({"frames": g.size(), "lights within 60 m (median)": g.n.median(), "green at once (median)": g.green.median(),
                         "green at once (max)": g.green.max(), "ego light distance rank (median)": g["rank"].median(),
                         "ego light is the nearest (share)": g["rank"].apply(lambda x: (x == 1).mean()).round(2)}).reset_index()


def cmd_report():
    s = pd.read_csv(OUT / "frames.csv", dtype={"route": str})
    comp = json.loads((OUT / "selection.json").read_text())
    res, L = dict(selection=comp, models={}), []
    L += ["# Q-light sweep: ego traffic light state at junction approaches, model x prompt", "",
          "**Exploratory**: every prompt is read on the same %d frames (%d routes, %d segments; ego red or yellow %d, ego "
          "green %d, no light %d), all variants are listed, and no result here unlocks a closed-loop arm (plan D16). "
          "Cells: estimate [95%% route-cluster CI] hits/frames; raw counts alone where fewer than 3 routes carry the cell. "
          "Yellow counts as red; a reply without a parseable answer is no answer and stays in the denominator." % (
              comp["frames"], comp["routes"], comp["segments"], comp["truth"].get("red", 0), comp["truth"].get("green", 0),
              comp["truth"].get("none", 0)), "",
          "| model | prompt | " + " | ".join(t for _, t in COLS) + " | unparsed | latency p50 ms |", "|:--|:--|" + ":--|" * len(COLS) + "--:|:--|"]
    split, halves, names = [], [], ["openjev-logged", "openjev"] + [m["name"] for m in MODELS]
    for name in names:
        A = {"as run (4 questions, in the drive)": (s.logged.fillna(""), [], len(s), False)} if name == "openjev-logged" else answers(s, name)
        if not A:
            L.append("| %s | no replies | " % name + " | " * len(COLS) + " | |")
            continue
        res["models"][name], per_half = {}, {}
        for pr, (ans, lat, n, quiet) in A.items():
            R, T = readouts(s, ans)
            R["requests"], R["lat_p50"] = n, float(np.median(lat)) if lat else None
            res["models"][name][pr] = R
            L.append("| %s | %s | " % (name, pr) + " | ".join(T[k] for k, _ in COLS) + " | %d | %s |" % (
                R["unparsed"], "" if not lat else "%.0f (%s)" % (R["lat_p50"], "quiet box" if quiet else "under load")))
            split.append("| %s | %s | " % (name, pr) + " | ".join("%s / %s" % (T["red_recall_" + t], T["red_green_" + t]) for t, _ in SPLITS) + " |")
            per_half[pr] = {h: readouts(s[s.half == h], ans[s.half == h]) for h in "AB"}
        if len(per_half) > 1:
            for a, b in ("AB", "BA"):
                best = max(per_half, key=lambda p: per_half[p][a][0]["score"])
                T = per_half[best][b][1]
                halves.append("| %s | %s | %s | %s | " % (name, a, best, b) + " | ".join(T[k] for k, _ in COLS) + " |")
                res["models"][name].setdefault("_split_half", {})[a] = dict(best=best, read_on=b, readouts=per_half[best][b][0])
    L += ["", "## Ego red by what the other approaches show, and route 27043", "",
          "Cells: answered red / answered green.", "",
          "| model | prompt | " + " | ".join(t for _, t in SPLITS) + " |", "|:--|:--|" + ":--|" * len(SPLITS)] + split
    L += ["", "## Guard against a lucky prompt: chosen on one half of the routes, read on the other", "",
          "Routes sorted by id, alternating halves A / B. Score = red recall - red answered green + green recall - no-light "
          "false alarm.", "", "| model | chosen on | prompt | read on | " + " | ".join(t for _, t in COLS) + " |",
          "|:--|:--|:--|:--|" + ":--|" * len(COLS)] + halves
    df = load_requests()
    v, j = in_view_27043(df), junctions(df)
    L += ["", "## Route 27043, ego red within 50 m: what is lit in the frame, by the logged answer", "",
          "Lamps = blobs of very bright saturated pixels in the upper 55% of the frame, named by hue (red < 20 or >= 330 "
          "deg, amber < 70, green < 190). A heuristic: it also picks up tail lights and signs and does not know which head "
          "serves which approach.", "", v.to_markdown(index=False) if len(v) else "no frames", "", lamp_check(df),
          "", "## Junction structure on these routes (truth labels, every answered request with an ego light)", "",
          "Rank 1 = the ego light is the nearest light actor; a higher rank = it is mounted beyond other approaches' lights, "
          "on the far side.", "", j.to_markdown(index=False) if len(j) else "no frames",
          "", "## Prompts", ""]
    for name in PROMPTS:
        L += ["`%s`:" % name, "", "```", prompt_text(name), "```", ""]
    L += ["`openjev` rows use the System One endpoint with Q_light alone and its own variants: " +
          ", ".join("`%s` = %s" % kv for kv in OPENJEV.items()) + " (lib/vlm_protocol.VARIANTS).", "",
          "Frames: `results/lightsweep/frames.csv`; raw replies: `results/lightsweep/<model>.jsonl`."]
    res.update(in_view_27043=v.to_dict("records") if len(v) else [], junctions=j.to_dict("records") if len(j) else [])
    (RUN / "results/lightsweep.md").write_text("\n".join(L) + "\n")
    write_json(RUN / "results/lightsweep.json", res)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "run", "report"])
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.exit(cmd_select() if a.cmd == "select" else cmd_run(a.model) if a.cmd == "run" else cmd_report())
