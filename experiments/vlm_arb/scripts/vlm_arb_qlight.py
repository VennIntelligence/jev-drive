"""Q-light diagnosis: why the VLM answers green on a red light (route 27043), and one pass over input variants.

  python vlm_arb_qlight.py

1. What the model is shown. For every saved frame with the ego light within 50 m, the lit lamp of the light is
   located (bright saturated pixels in the upper half of the wide frame) and its hue is tabulated against the true
   state, on the old route 27043 frames and on the new frames. A red lamp that renders amber explains a colour
   reader that does not say red.
2. Variants (lib/vlm_protocol.VARIANTS, Q_light only): road (narrow front camera alone), pos (lamp position instead
   of colour in the wording), crop (an upper-centre crop of the wide frame added). They are scored on the debug-route
   frames only (unit v2-dbg-shadow-s0); the selected one is the best red recall (0-50 m) among those that keep both
   false-alarm lines on the debug frames, else the best recall minus false alarms.
3. The selected variant is read once on the dev-route frames (v2-drive-s1-dev) against the registered Q-light lines.
4. Listed, no line (plan deviation D15): ego red within 50 m split by whether a light of another approach is green at
   that moment, with route 27043 on its own (logged answers on every new shadow unit; the selected variant on the dev
   frames); on 27043, the lit lamps found in the frame by the answer given; the junctions' structure (lights per
   junction, how many are green at once, the ego light's distance rank).
An offline read: it runs once the shadow frames exist, whatever gates/phase_a.json says. vred / vall are queued by the
chain only if this read passes and the registered Phase A latency line holds.
Writes results/qlight.{json,md} and gates/qlight.json {"variant", "pass"}.
"""
import colorsys
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import REPO, RUN, boot_ratio, fmt, unit_dir, write_json  # noqa: E402
from vlm_arb_phase_a import STRATA, SUBSETS  # noqa: E402

sys.path.insert(0, str(REPO / "lib"))
from vlm_client import VLMClient  # noqa: E402

RED = "red_or_yellow_for_ego"
CANDIDATES = ("road", "pos", "crop")
THREADS = 6


def frames_of(udir):
    """One row per answered request of a new-format shadow unit, with its frame paths and truth."""
    rows = []
    for done in sorted(Path(udir).glob("done/*.json")):
        a = Path(udir) / "attempts" / done.stem / str(json.loads(done.read_text()).get("attempt", 1))
        if not (a / "vlm_decisions.jsonl").exists():
            continue
        for line in open(a / "vlm_decisions.jsonl"):
            d = json.loads(line)
            if d.get("k") != "a" or not d["ans"].get("ok"):
                continue
            g, base = d["gt"], d["ans"]
            f = {c: a / "vlm_frames" / ("%08.2f_%s.jpg" % (d["t_q"], c)) for c in ("wide", "road")}
            if not all(p.exists() for p in f.values()):
                continue
            others = [x for x in g.get("lights", []) if x[0] != g.get("tl_id")]
            rows.append(dict(route=done.stem, t=d["t_q"], tl=-1 if g.get("tl") is None else g["tl"],
                             tl_dist=np.nan if g.get("tl") is None else g["tl_dist"],
                             other_red=any(x[1] == 2 for x in others), any_light=bool(g.get("lights")),
                             other_green=any(x[1] == 0 for x in others), tl_id=g.get("tl_id"), lights=g.get("lights", []),
                             base=base["Q_light"], wide=str(f["wide"]), road=str(f["road"])))
    return pd.DataFrame(rows)


def lamp_hue(img):
    """Median hue (degrees) of the lit-lamp pixels in the upper 55% of a frame, or None: very bright and saturated."""
    a = np.asarray(img.convert("RGB"), np.float32)[: int(img.height * .55)] / 255.0
    mx, mn = a.max(-1), a.min(-1)
    lit = (mx > 0.9) & ((mx - mn) / np.maximum(mx, 1e-6) > 0.35)
    if lit.sum() < 6:
        return None
    r, g, b = a[lit].mean(0)
    return 360.0 * colorsys.rgb_to_hsv(float(r), float(g), float(b))[0]


def hue_name(h):
    return "none" if h is None else "red" if h < 20 or h >= 330 else "amber" if h < 70 else "green" if h < 190 else "other"


def lamp_table(df, load):
    rows = []
    for r in df[(df.tl >= 0) & (df.tl_dist < 50)].itertuples():
        rows.append(dict(truth={0: "green", 1: "yellow", 2: "red"}[r.tl], lamp=hue_name(lamp_hue(load(r)))))
    t = pd.DataFrame(rows)
    return pd.crosstab(t.truth, t.lamp) if len(t) else pd.DataFrame()


def lamps(img):
    """Lit lamps in the upper 55% of a frame: {hue name: count} over connected blobs of >= 4 lit pixels."""
    from scipy import ndimage
    a = np.asarray(img.convert("RGB"), np.float32)[: int(img.height * .55)] / 255.0
    mx, mn = a.max(-1), a.min(-1)
    lab, n = ndimage.label((mx > 0.9) & ((mx - mn) / np.maximum(mx, 1e-6) > 0.35))
    out = {"red": 0, "amber": 0, "green": 0, "other": 0}
    for i in range(1, n + 1):
        px = a[lab == i]
        if len(px) >= 4:
            r, g, b = px.mean(0)
            out[hue_name(360.0 * colorsys.rgb_to_hsv(float(r), float(g), float(b))[0])] += 1
    return out


def strata(df, col):
    """Ego red / yellow within 50 m by "another approach green", all routes / 27043 / the rest: (recall, as green)."""
    ok, red, r27 = df[col] != "", df.tl.isin([1, 2]) & (df.tl_dist < 50), df.route == "27043"
    R = {}
    for tag, m in (("og", df.other_green), ("allred", ~df.other_green)):
        for sub, mm in (("", True), ("_27043", r27), ("_ex27043", ~r27)):
            d = red & m & mm & ok
            R[tag + sub] = (boot_ratio((df[col] == RED) & d, d, df.route), boot_ratio((df[col] == "green_for_ego") & d, d, df.route))
    return R


def strata_md(name, R):
    c = lambda r: "n/a" if r["n"] == 0 else "%s n=%d" % (fmt(r, True), r["n"])   # noqa: E731
    return "| %s | " % name + " | ".join("%s / %s" % (c(R[t + u][0]), c(R[t + u][1])) for t, _ in STRATA for u, _ in SUBSETS) + " |"


STRATA_HEAD = ["| answers | " + " | ".join("%s, %s" % (a, b) for _, a in STRATA for _, b in SUBSETS) + " |",
               "|:--|" + ":--|" * (len(STRATA) * len(SUBSETS))]


def in_view_27043(df):
    """Route 27043, ego red within 50 m: the lit lamps found in the wide and the road frame, by the answer given."""
    d = df[(df.route == "27043") & df.tl.isin([1, 2]) & (df.tl_dist < 50)]
    rows = []
    for r in d.itertuples():
        w, n = lamps(Image.open(r.wide)), lamps(Image.open(r.road))
        rows.append(dict(answer=r.base, truth_other_green=r.other_green, wide_red=w["red"], wide_amber=w["amber"],
                         wide_green=w["green"], road_red=n["red"], road_amber=n["amber"], road_green=n["green"]))
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    g = t.groupby("answer")
    return pd.DataFrame({
        "frames": g.size(), "truth: another light green": g.truth_other_green.mean().round(2),
        "wide: frames with a green lamp": (g.wide_green.apply(lambda x: (x > 0).mean())).round(2),
        "wide: frames with a red / amber lamp": g.apply(lambda x: ((x.wide_red + x.wide_amber) > 0).mean()).round(2),
        "wide: mean lamps red / amber / green": g.apply(lambda x: "%.1f / %.1f / %.1f" % (x.wide_red.mean(), x.wide_amber.mean(), x.wide_green.mean())),
        "road: frames with a green lamp": (g.road_green.apply(lambda x: (x > 0).mean())).round(2),
        "road: frames with a red / amber lamp": g.apply(lambda x: ((x.road_red + x.road_amber) > 0).mean()).round(2)})


def junctions(df):
    """Per route and ego light (one junction approach): lights within 60 m, green at once, the ego light's distance rank."""
    rows = []
    for r in df[df.tl >= 0].itertuples():
        ls = sorted(r.lights, key=lambda x: x[2])
        ids = [x[0] for x in ls]
        rows.append(dict(route=r.route, ego_light=r.tl_id, n=len(ls), green=sum(x[1] == 0 for x in ls),
                         rank=ids.index(r.tl_id) + 1 if r.tl_id in ids else np.nan))
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    g = t.groupby(["route", "ego_light"])
    return pd.DataFrame({"frames": g.size(), "lights within 60 m (median)": g.n.median(), "green at once (median)": g.green.median(),
                         "green at once (max)": g.green.max(), "ego light distance rank (median)": g["rank"].median(),
                         "ego light nearest (share of frames)": g["rank"].apply(lambda x: (x == 1).mean()).round(2),
                         "ego light not in the 60 m list (share)": g["rank"].apply(lambda x: x.isna().mean()).round(2)}).reset_index()


def replay(df, variant):
    cli = VLMClient()
    def one(r):
        jp = {"wide": Path(r.wide).read_bytes(), "road": Path(r.road).read_bytes()}
        a = cli.ask(jp, variant, only_light=True)
        return a.get("Q_light", "") if a["ok"] else ""
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(one, df.itertuples()))


def lines(df, col):
    ok = df[col] != ""
    red = df.tl.isin([1, 2]) & (df.tl_dist < 50)
    other = df.other_red & ~red
    none = (df.tl == -1) & ~df.any_light
    R = dict(red_recall=boot_ratio(((df[col] == RED) & red & ok), (red & ok), df.route),
             red_as_green=boot_ratio(((df[col] == "green_for_ego") & red & ok), (red & ok), df.route),
             other_fp=boot_ratio(((df[col] == RED) & other & ok), (other & ok), df.route),
             nolight_fp=boot_ratio(((df[col] == RED) & none & ok), (none & ok), df.route))
    e = lambda k: R[k]["est"]   # noqa: E731
    R["pass"] = bool(e("red_recall") >= .80 and e("other_fp") <= .10 and e("nolight_fp") <= .02)
    R["score"] = float(np.nan_to_num(e("red_recall")) - np.nan_to_num(e("other_fp")) - np.nan_to_num(e("nolight_fp")))
    return R


def md_lines(name, R):
    return "| %s | %s | %s | %s | %s | %d | %s |" % (name, fmt(R["red_recall"], True), fmt(R["red_as_green"], True),
                                                   fmt(R["other_fp"], True), fmt(R["nolight_fp"], True),
                                                   R["red_recall"]["n"], "yes" if R["pass"] else "no")


def main():
    out = RUN / "results"
    out.mkdir(parents=True, exist_ok=True)
    doc, res = ["# Q-light diagnosis", ""], {}
    # 1. what the model is shown
    old = RUN / "arms/shadow-drive-s0/attempts/27043/1"
    rows = []
    if (old / "vlm_decisions.jsonl").exists():
        gt = {}
        for line in open(old / "vlm_decisions.jsonl"):
            d = json.loads(line)
            gt[round(d["t"], 2)] = d["gt"]
        for p in sorted((old / "vlm_frames").glob("frame_*.npy")):
            g = gt.get(round(float(p.name.split("_")[1][:-1]), 2))
            if g and g.get("tl_state") is not None and g["tl_dist"] < 50:
                rows.append(dict(truth={0: "green", 1: "yellow", 2: "red"}[g["tl_state"]],
                                 lamp=hue_name(lamp_hue(Image.fromarray(np.load(p))))))
    t = pd.DataFrame(rows)
    doc += ["## Lit-lamp hue in the wide frame against the true state", "",
            "Route 27043 (old frames, the wide camera as sent to the model):", "",
            pd.crosstab(t.truth, t.lamp).to_markdown() if len(t) else "no frames", ""]
    res["lamp_27043"] = pd.crosstab(t.truth, t.lamp).to_dict() if len(t) else {}
    dbg, dev = frames_of(unit_dir("dbg-shadow", 0, "light")), frames_of(unit_dir("drive", 1, "dev"))
    for name, df in (("debug routes", dbg), ("dev routes, seed 1", dev)):
        for cam in ("wide", "road"):
            tab = lamp_table(df, lambda r, cam=cam: Image.open(getattr(r, cam))) if len(df) else pd.DataFrame()
            doc += ["%s, %s camera:" % (name, cam), "", tab.to_markdown() if len(tab) else "no frames", ""]
            res["lamp_%s_%s" % (name.split()[0], cam)] = tab.to_dict() if len(tab) else {}
    # 2. variants on the debug frames
    doc += ["## Variants on the debug-route frames (selection only)", "",
            "| variant | red recall 0-50 m | red answered green | other direction red -> red | no light -> red | red frames | lines kept |",
            "|:--|:--|:--|:--|:--|--:|:--|"]
    scores = {}
    if len(dbg):
        scores["base"] = lines(dbg, "base")
        doc.append(md_lines("base (as run)", scores["base"]))
        for v in CANDIDATES:
            dbg[v] = replay(dbg, v)
            scores[v] = lines(dbg, v)
            doc.append(md_lines(v, scores[v]))
        cand = {v: scores[v] for v in CANDIDATES}
        keep = {v: s for v, s in cand.items() if s["pass"]} or cand
        chosen = max(keep, key=lambda v: (keep[v]["red_recall"]["est"] if keep is not cand else keep[v]["score"]))
    else:
        chosen = None
        doc.append("| no debug frames | | | | | 0 | |")
    res.update(debug=scores, chosen=chosen)
    # 3. one read on the dev frames
    passed = False
    doc += ["", "## The selected variant on the dev-route frames (read once)", ""]
    if chosen and len(dev):
        dev[chosen] = replay(dev, chosen)
        final, ref = lines(dev, chosen), lines(dev, "base")
        passed = final["pass"]
        doc += ["| variant | red recall 0-50 m | red answered green | other direction red -> red | no light -> red | red frames | lines kept |",
                "|:--|:--|:--|:--|:--|--:|:--|", md_lines("base (as run)", ref), md_lines(chosen + " (selected)", final), "",
                "Registered Q-light lines (recall >= 80%, other-direction false alarm <= 10%, no-light false alarm <= 2%): "
                + ("**pass**: `vred` and the R2 row of `vall` are queued with this variant if the registered Phase A "
                   "latency line holds (gates/phase_a.json)." if passed else "**fail**: `vred` and `vall` are not run.")]
        res.update(dev=final, dev_base=ref)
    else:
        doc.append("Not read: no variant selected or no dev frames.")
    # 4. listed reads (deviation D15)
    units = [unit_dir("dbg-shadow", 0, "light"), unit_dir("drive", 1, "dev"), unit_dir("drive", 1, "tgt"), unit_dir("drive", 0, "tgt")]
    parts = [frames_of(u).assign(unit=u.name) for u in units]
    allf = pd.concat([x for x in parts if len(x)], ignore_index=True) if any(len(x) for x in parts) else pd.DataFrame()
    doc += ["", "## Ego red within 50 m: another approach green or not (listed, no line)", "",
            "Cells: answered red / answered green, estimate [95% route-cluster CI] n. `no other light green` includes "
            "frames where the ego light is the only one within 60 m.", ""] + STRATA_HEAD
    if len(allf):
        res["strata_all_base"] = strata(allf, "base")
        doc.append(strata_md("as run, all new shadow units (%d routes)" % allf.route.nunique(), res["strata_all_base"]))
    if len(dev):
        res["strata_dev_base"] = strata(dev, "base")
        doc.append(strata_md("as run, dev routes seed 1", res["strata_dev_base"]))
        if chosen:
            res["strata_dev_chosen"] = strata(dev, chosen)
            doc.append(strata_md("variant `%s`, dev routes seed 1" % chosen, res["strata_dev_chosen"]))
    doc += ["", "## Route 27043, ego red within 50 m: lit lamps found in the frame, by the answer given", "",
            "Lamps are blobs of very bright saturated pixels in the upper 55% of the frame, named by hue (red < 20 or "
            ">= 330 deg, amber < 70, green < 190). A heuristic: it also picks up tail lights and signs, and it does not "
            "know which head belongs to which approach. Frames: v2-drive-s1-dev (seed 1).", ""]
    v = in_view_27043(dev) if len(dev) else pd.DataFrame()
    doc.append(v.to_markdown() if len(v) else "no frames")
    res["in_view_27043"] = v.reset_index().to_dict("records") if len(v) else []
    doc += ["", "## Junction structure on these routes (truth labels)", "",
            "One row per route and ego light, over the frames where an ego light is set. Rank 1 = the ego light is the "
            "nearest light actor; a high rank means it is mounted on the far side of the junction.", ""]
    j = junctions(allf) if len(allf) else pd.DataFrame()
    doc.append(j.to_markdown(index=False) if len(j) else "no frames")
    res["junctions"] = j.to_dict("records") if len(j) else []
    (out / "qlight.md").write_text("\n".join(doc) + "\n")
    write_json(out / "qlight.json", res)
    write_json(RUN / "gates/qlight.json", {"variant": chosen, "pass": bool(passed)})
    print("\n".join(doc))


if __name__ == "__main__":
    main()
