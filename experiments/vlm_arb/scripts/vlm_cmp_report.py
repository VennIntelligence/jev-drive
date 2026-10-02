"""Tables of the vlm_cmp study (Qwen3-VL-4B vs 8B) from the chain's run dir -> <run>/results/vlm_4b_vs_8b.md (+ csv, png).

  python vlm_cmp_report.py --run RUN [--out DIR]

Plan: experiments/vlm_arb/plans/2026-10-03-vlm-4b-vs-8b.md. Everything is computed from the cached option log-probabilities
(features/, twoframe/), the head predictions (fit/), the latency bench (bench/) and frames.csv; the hand-written summary of the
result file is added by hand on top of this output.
Cells: estimate [95% route-cluster bootstrap CI, 2000 resamples, seed 0] hits/n (routes); paired differences use
jevdrive.stats.paired (same resample for both models, route clusters, 10000 resamples, seed 0).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import REPO, boot_ratio  # noqa: E402

sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402
from vlm_cmp_frames import DIRECTIVES  # noqa: E402
from vlm_cmp_model import OPTIONS  # noqa: E402

MODELS = ["4b", "8b"]
RESN = ["r559", "r1153", "r2335", "r4573"]
SINGLE = ("light", "sign", "block", "side", "dir")
RED, GREEN = "red_or_yellow_for_ego", "green_for_ego"
KEEP = list(range(2, 37, 2))


# ---------------------------------------------------------------------------------------------------- loading
def load_df(run):
    df = pd.read_csv(Path(run) / "frames.csv", dtype={"route": str, "attempt": str}, keep_default_na=False, na_values=[""])
    for c in ("ego_red", "ego_green", "yellow", "ego_moving", "ev_yellow_onset", "ev_red_onset", "ev_red_to_green", "ev_hold",
              "sign_near", "sign_far", "tf"):
        df[c] = df[c].astype(bool)
    df["dir_ok"] = df.dir_ok.map(lambda s: set(eval(s)) if isinstance(s, str) else s)
    df["dir_note"] = df.dir_note.fillna("")
    df["pos"] = df.pos.fillna("")
    return df


def load_chunks(path, df, keys):
    parts = [np.load(p, allow_pickle=False) for p in sorted(Path(path).glob("shard*-chunk*.npz"))]
    if not parts:
        return None
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys + ["ids"]}
    order = pd.Series(range(len(out["ids"])), index=out["ids"]).reindex(df.id).to_numpy()
    assert not np.isnan(order).any(), "frames missing in " + str(path)
    return {k: v[order.astype(int)] for k, v in out.items()}


def answers(lp, qs):
    return {q: np.array(OPTIONS[q])[lp["lp_" + q].argmax(1)] for q in qs}


def compose(df, A):
    """Directive from the separate answers: the arbitration of the questions without its timing rules (plan section 3)."""
    v = df.v.to_numpy()
    out = np.full(len(df), "proceed", dtype=object)
    block = A["block"] == "static_block"
    side = np.where(A["side"] == "left_free", "pass_left", np.where(A["side"] == "right_free", "pass_right", "wait"))
    out[block] = side[block]
    out[A["sign"] == "yes"] = "stop_at_line"
    g = A["light"] == GREEN
    out[g] = np.where(v[g] < 1.0, "go_now", "proceed")
    out[A["light"] == RED] = "stop_at_line"
    return out


def accepted(df, d):
    return np.array([x in s for x, s in zip(d, df.dir_ok)])


# ---------------------------------------------------------------------------------------------------- cells
def cell(hit, m, groups):
    m = np.asarray(m, bool)
    if not m.any():
        return "n/a"
    r = boot_ratio((np.asarray(hit) & m).astype(float), m.astype(float), groups)
    h = int((np.asarray(hit) & m).sum())
    return "%.0f%% [%.0f, %.0f] %d/%d (%d)" % (100 * r["est"], 100 * r["lo"], 100 * r["hi"], h, int(m.sum()), r["groups"])


def diff(ha, hb, m, groups, scale=100):
    """mean(a - b) over the rows of m with a route-cluster CI, text and verdict."""
    m = np.asarray(m, bool)
    if m.sum() < 2:
        return "n/a", "n/a"
    r = stats.paired(np.asarray(ha, float)[m], np.asarray(hb, float)[m], groups=np.asarray(groups)[m])
    txt = "%+.1f [%+.1f, %+.1f]" % (scale * r["mean"], scale * r["lo"], scale * r["hi"])
    if r["units"] < 3:
        return txt, "too few routes"
    return txt, ("higher" if r["lo"] > 0 else "lower" if r["hi"] < 0 else "within noise")


def pct(hit, m):
    m = np.asarray(m, bool)
    return "n/a" if not m.any() else "%.0f%%" % (100 * (np.asarray(hit) & m).sum() / m.sum())


# ---------------------------------------------------------------------------------------------------- zero-shot readouts
def readouts(df, A, D, C):
    """[(group, name, mask, hit)] for one model at one resolution: A = single-question answers, D = directive answers,
    C = composed directive."""
    d = df
    other = (d.other_red == 1) & ~(d.tl.isin([1, 2]) & (d.tl_dist < 50))
    nolight = (d.tl == -1) & (d.any_light == 0)
    red, green = d.ego_red.to_numpy(), d.ego_green.to_numpy()
    L = A["light"]
    R = []
    add = lambda g, n, m, h: R.append((g, n, np.asarray(m, bool), np.asarray(h, bool)))   # noqa: E731
    add("light", "ego red answered red, all", red, L == RED)
    add("light", "... 0-20 m", red & (d.tl_dist < 20), L == RED)
    add("light", "... 20-50 m", red & (d.tl_dist >= 20), L == RED)
    for p, nm in (("A", "A before the stop line"), ("B", "B between line and junction entrance"), ("C", "C inside the junction")):
        add("light", "... position " + nm, red & (d.pos == p), L == RED)
    add("light", "ego red answered green, all", red, L == GREEN)
    for p, nm in (("A", "A"), ("B", "B"), ("C", "C")):
        add("light", "... position " + nm, red & (d.pos == p), L == GREEN)
    add("light", "ego green answered green, all", green, L == GREEN)
    for p, nm in (("A", "A"), ("B", "B"), ("C", "C")):
        add("light", "... position " + nm, green & (d.pos == p), L == GREEN)
    add("light", "ego green answered red", green, L == RED)
    add("light", "other-direction red (ego not red) answered red", other, L == RED)
    add("light", "no light at all answered red", nolight, L == RED)
    S = A["sign"]
    add("sign", "stop sign within 25 m answered yes", d.sign_near, S == "yes")
    add("sign", "no sign within 80 m answered yes", d.stop_dist.isna(), S == "yes")
    add("sign", "sign at 25-80 m answered yes (listed)", d.sign_far, S == "yes")
    B = A["block"]
    sb = d.block == "static_block"
    add("block", "static block answered static_block, all", sb, B == "static_block")
    for k, nm in (("cones", "cones routes"), ("vehicle", "stopped-vehicle routes"), ("other", "other routes (queues)")):
        add("block", "... " + nm, sb & (d.kind == k), B == "static_block")
    add("block", "stopped lead (< 0.5 m/s) answered static_block", d.lead_state == "stopped", B == "static_block")
    add("block", "clear answered static_block", d.block == "clear", B == "static_block")
    add("block", "moving_lead truth answered moving_lead", d.block == "moving_lead", B == "moving_lead")
    add("block", "lead moving (>= 0.5 m/s) answered moving_lead", d.lead_state == "moving", B == "moving_lead")
    add("side", "bypass side correct, static-block rows with a side truth", sb & d.side.isin(["left_free", "right_free", "none_free"]),
        A["side"] == d.side.to_numpy())
    strict = (d.dir_note == "").to_numpy()
    prim = d.dir.to_numpy()
    add("directive", "directive: strict accuracy (rows without an ambiguity note)", strict, D == prim)
    add("directive", "directive: accepted accuracy, all rows", np.ones(len(d), bool), accepted(d, D))
    add("directive", "directive: accepted accuracy, ambiguous rows only", ~strict, accepted(d, D))
    for c in DIRECTIVES:
        add("directive", "directive: strict recall of %s" % c, strict & (prim == c), D == c)
    add("composed", "composed from the questions: strict accuracy", strict, C == prim)
    add("composed", "composed: accepted accuracy, all rows", np.ones(len(d), bool), accepted(d, C))
    add("composed", "composed: accepted accuracy, ambiguous rows only", ~strict, accepted(d, C))
    for c in DIRECTIVES:
        add("composed", "composed: strict recall of %s" % c, strict & (prim == c), C == c)
    add("composed", "composed: strict accuracy, rows whose truth is not slow", strict & (prim != "slow"), C == prim)
    add("directive", "directive: strict accuracy, rows whose truth is not slow", strict & (prim != "slow"), D == prim)
    return R


def zs_table(df, RA, RB, groups, title):
    """RA / RB readouts of 4B / 8B at one resolution -> markdown table with paired 8B - 4B."""
    L = ["| readout | 4B | 8B | 8B - 4B (points) | read |", "|:--|:--|:--|:--|:--|"]
    rows = []
    for (g, n, m, ha), (_, _, _, hb) in zip(RA, RB):
        dtxt, verdict = diff(hb, ha, m, groups)
        ca, cb = cell(ha, m, groups), cell(hb, m, groups)
        L.append("| %s | %s | %s | %s | %s |" % (n, ca, cb, dtxt, verdict))
        rows.append(dict(res=title, group=g, readout=n, n=int(m.sum()), est4=float((ha & m).sum() / max(m.sum(), 1)),
                         est8=float((hb & m).sum() / max(m.sum(), 1)), diff=dtxt, read=verdict))
    return L, rows


def confusion(df, ans, title):
    rows = DIRECTIVES
    t = pd.crosstab(pd.Categorical(df.dir, rows), pd.Categorical(ans, rows), dropna=False)
    t.index.name = "truth (primary)"
    return ["**%s**" % title, "", t.to_markdown(), ""]


# ---------------------------------------------------------------------------------------------------- heads
def bacc_counts(pred, y, groups, ncls):
    ids, inv = np.unique(groups, return_inverse=True)
    n = np.zeros((len(ids), ncls))
    c = np.zeros((len(ids), ncls))
    for k in range(ncls):
        mk = y == k
        n[:, k] = np.bincount(inv[mk], minlength=len(ids))
        c[:, k] = np.bincount(inv[mk & (pred == k)], minlength=len(ids))
    return n, c


def bacc_from(n, c, idx=None):
    if idx is not None:
        n, c = n[idx].sum(0), c[idx].sum(0)
    else:
        n, c = n.sum(0), c.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = c / n
    return float(np.nanmean(r)) if np.isfinite(r).any() else np.nan


def bacc_ci(pred, y, groups, ncls, other=None, B=2000):
    """balanced accuracy of pred on rows y >= 0 with a route-cluster CI; `other` (second prediction) -> paired difference
    (other - pred) with CI."""
    ok = y >= 0
    n, c = bacc_counts(pred[ok], y[ok], groups[ok], ncls)
    idx = np.random.default_rng(0).integers(0, n.shape[0], (B, n.shape[0]))
    est = bacc_from(n, c)
    bs = np.array([bacc_from(n, c, i) for i in idx])
    out = dict(est=est, lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)), routes=int((n.sum(1) > 0).sum()))
    if other is not None:
        n2, c2 = bacc_counts(other[ok], y[ok], groups[ok], ncls)
        d = np.array([bacc_from(n2, c2, i) - bacc_from(n, c, i) for i in idx])
        out.update(d_est=bacc_from(n2, c2) - est, d_lo=float(np.nanpercentile(d, 2.5)), d_hi=float(np.nanpercentile(d, 97.5)))
    return out


def readable(curve):
    """curve: {layer: cv_bacc}; smallest layer within 0.05 of the best layer's CV balanced accuracy if the best is >= 0.70."""
    ok = {k: v for k, v in curve.items() if v is not None and np.isfinite(v)}
    if not ok:
        return None, None
    best = max(ok.values())
    if best < 0.70:
        return None, best
    return min(k for k, v in ok.items() if v >= best - 0.05), best


LAB_NC = dict(light=3, sign=2, block=3, lead=2)


def zero_shot_label_pred(A, lab):
    """Zero-shot answers mapped to the head's classes (-1 = an answer outside the classes: counted wrong)."""
    if lab == "light":
        m = {RED: 0, GREEN: 1, "no_light": 2, "light_for_other_lane": 2}
        return np.array([m[x] for x in A["light"]])
    if lab == "sign":
        return (A["sign"] == "yes").astype(int)
    if lab == "block":
        return np.array([{"clear": 0, "moving_lead": 1, "static_block": 2}[x] for x in A["block"]])
    return np.array([{"static_block": 1, "moving_lead": 0}.get(x, -1) for x in A["block"]])      # lead: stopped vs moving


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = load_df(run)
    g = df.route.to_numpy()
    md, csv = [], []
    P = md.append

    # ---- loads
    F, ANS, DIRC, COMP = {}, {}, {}, {}
    for m in MODELS:
        for r in RESN:
            f = load_chunks(run / "features" / m / r, df, ["lp_" + q for q in SINGLE])
            if f is None:
                continue
            F[m, r] = f
            ANS[m, r] = answers(f, SINGLE)
            DIRC[m, r] = ANS[m, r]["dir"]
            COMP[m, r] = compose(df, ANS[m, r])
    have = [r for r in RESN if all((m, r) in F for m in MODELS)]
    bench = {}
    for m in MODELS:
        for p in sorted((run / "bench" / m).glob("*.json")) if (run / "bench" / m).exists() else []:
            d = json.loads(p.read_text())
            bench[m, d["cfg"]] = d

    # ---- frame set
    P("## 1. Frames\n")
    P("Core set of %d requests on %d routes (train %d, val %d, test %d), %d with an earlier frame, %d two-frame instants. Requests per route and"
      " category are in `frames.csv` of the run dir. The all-request count was 33 806 (plan section 2).\n" % (
          len(df), df.route.nunique(), (df.part == "train").sum(), (df.part == "val").sum(), (df.part == "test").sum(),
          int(df.prev_t.notna().sum()), int(df.tf.sum())))
    cnt = {k: int(v) for k, v in df.dir.value_counts().items()}
    P("Directive truth of the core set: " + ", ".join("%s %d" % (k, cnt.get(k, 0)) for k in DIRECTIVES)
      + "; rows without an ambiguity note (strict): %d of %d. Ambiguity notes: %s.\n" % (
          int((df.dir_note == "").sum()), len(df), ", ".join("%s %d" % (k, v) for k, v in df[df.dir_note != ""].dir_note.value_counts().items())))
    P("Requests by position of the car (ego light within range): red A %d / B %d / C %d, green A %d / B %d / C %d.\n" % tuple(
        [int((df.ego_red & (df.pos == p)).sum()) for p in "ABC"] + [int((df.ego_green & (df.pos == p)).sum()) for p in "ABC"]))

    # ---- zero-shot tables
    P("## 2. Zero-shot readings, one forward pass with option scoring (same frames, same code path)\n")
    P("How to read: each cell is the rate of the named answer among the rows of the named truth, with the route-cluster CI, hits / rows and, in "
      "brackets, the number of routes the rows come from. `8B - 4B` is the paired difference in percentage points on the same rows; `read` is "
      "mechanical (CI above 0 = 8B higher, below 0 = 4B higher, else within noise; fewer than 3 routes = too few routes). Position A / B / C = "
      "before the stop line / between line and junction entrance / inside the junction. Directive rows are scored against the truth mapping of "
      "plan section 3; `composed` is the directive built from the separate questions (never answers `slow`).\n")
    zs_rows = []
    ZSR = {}
    for r in have:
        RA = readouts(df, ANS["4b", r], DIRC["4b", r], COMP["4b", r])
        RB = readouts(df, ANS["8b", r], DIRC["8b", r], COMP["8b", r])
        ZSR[r] = (RA, RB)
        P("### 2.%d Resolution %s (%d visual tokens for the pair at 4B counts: %s)\n" % (
            have.index(r) + 1, r, int(np.median(F["4b", r].get("ntok", [0]))) if "ntok" in F["4b", r] else 0, r))
        L, rows = zs_table(df, RA, RB, g, r)
        zs_rows += rows
        md += L + [""]
    pd.DataFrame(zs_rows).to_csv(out / "vlm_cmp_zeroshot.csv", index=False)

    # ---- directive vs separate questions: confusion tables
    if "r1153" in have:
        P("## 3. Directive prompt against the separate questions\n")
        P("Accuracy rows are in section 2 (`directive:` and `composed:` rows). Paired difference directive - composed on the same rows, per model, "
          "and the confusion tables at r1153 (all rows of the core set, counts; truth = primary directive).\n")
        L = ["| row | 4B directive - composed | 8B directive - composed |", "|:--|:--|:--|"]
        RA, RB = ZSR["r1153"]
        names = {n: i for i, (_, n, _, _) in enumerate(RA)}
        for nm_d, nm_c in (("directive: strict accuracy (rows without an ambiguity note)", "composed from the questions: strict accuracy"),
                           ("directive: accepted accuracy, all rows", "composed: accepted accuracy, all rows"),
                           ("directive: accepted accuracy, ambiguous rows only", "composed: accepted accuracy, ambiguous rows only"),
                           ("directive: strict accuracy, rows whose truth is not slow", "composed: strict accuracy, rows whose truth is not slow")) + tuple(
                ("directive: strict recall of %s" % c, "composed: strict recall of %s" % c) for c in DIRECTIVES):
            cells = []
            for RR in (RA, RB):
                i, j = names[nm_d], names[nm_c]
                m = RR[i][2]
                dt, v = diff(RR[i][3], RR[j][3], m, g)
                cells.append("%s (%s vs %s) %s" % (dt, pct(RR[i][3], m), pct(RR[j][3], m), v) if m.any() else "n/a")
            L.append("| %s | %s | %s |" % (nm_d.replace("directive: ", ""), cells[0], cells[1]))
        md += L + [""]
        for mdl in MODELS:
            md += confusion(df, DIRC[mdl, "r1153"], "%s directive prompt, r1153: truth x answer" % mdl.upper())
            md += confusion(df, COMP[mdl, "r1153"], "%s composed from the separate questions, r1153: truth x composed" % mdl.upper())
        L = ["Directive accuracy over resolutions (strict / accepted; composed strict / accepted):", "",
             "| model | " + " | ".join(have) + " |", "|:--|" + ":--|" * len(have)]
        strict = (df.dir_note == "").to_numpy()
        for mdl in MODELS:
            cs = []
            for r in have:
                d, c = DIRC[mdl, r], COMP[mdl, r]
                cs.append("%s / %s; %s / %s" % (pct(d == df.dir.to_numpy(), strict), pct(accepted(df, d), np.ones(len(df), bool)),
                                                 pct(c == df.dir.to_numpy(), strict), pct(accepted(df, c), np.ones(len(df), bool))))
            L.append("| %s | %s |" % (mdl.upper(), " | ".join(cs)))
        md += L + [""]

    # ---- two frames
    tf_ok = all((run / "twoframe" / m / "r1153").exists() for m in MODELS) and "r1153" in have
    TF = {}
    if tf_ok:
        sub = df[df.tf].reset_index(drop=True)
        gs = sub.route.to_numpy()
        for m in MODELS:
            t = load_chunks(run / "twoframe" / m / "r1153", sub, ["lp_dir", "lp_light", "lp_block", "lp_dir_prev", "prev_dir"])
            if t is None:
                tf_ok = False
                break
            TF[m] = dict(dir=np.array(DIRECTIVES)[t["lp_dir"].argmax(1)], light=np.array(OPTIONS["light"])[t["lp_light"].argmax(1)],
                         block=np.array(OPTIONS["block"])[t["lp_block"].argmax(1)], prev=t["prev_dir"])
    if tf_ok:
        idx = df.index[df.tf].to_numpy()
        P("## 4. Two frames (current + about 1 s earlier) with the previous directive, r1153, four images\n")
        P("%d instants (every event instant, a systematic subsample of the rest). The single-frame numbers are the same model at the same "
          "instants (section 2 answers restricted to them). Cells: estimate [CI] hits/rows (routes); `2f - 1f` = paired difference in points. "
          "The previous directive is the model's own single-frame directive at the earlier frame.\n" % len(sub))
        s1 = {m: dict(dir=DIRC[m, "r1153"][idx], light=ANS[m, "r1153"]["light"][idx], block=ANS[m, "r1153"]["block"][idx]) for m in MODELS}
        prim, strict_s = sub.dir.to_numpy(), (sub.dir_note == "").to_numpy()
        held = sub.ev_red_to_green & (sub.prev_v < 1.0)
        rows2 = [
            ("directive strict accuracy, all instants", strict_s, lambda s, m_: s["dir"] == prim),
            ("directive accepted accuracy, all instants", np.ones(len(sub), bool), lambda s, m_: accepted(sub, s["dir"])),
            ("yellow onset (green 1 s ago, yellow now): directive stop_at_line", sub.ev_yellow_onset.to_numpy(), lambda s, m_: s["dir"] == "stop_at_line"),
            ("... light answered red_or_yellow", sub.ev_yellow_onset.to_numpy(), lambda s, m_: s["light"] == RED),
            ("red onset (green 1 s ago, red now): directive stop_at_line", sub.ev_red_onset.to_numpy(), lambda s, m_: s["dir"] == "stop_at_line"),
            ("... light answered red_or_yellow", sub.ev_red_onset.to_numpy(), lambda s, m_: s["light"] == RED),
            ("red to green, car was held (speed < 1 m/s 1 s ago): directive go_now", held.to_numpy(), lambda s, m_: s["dir"] == "go_now"),
            ("... light answered green", held.to_numpy(), lambda s, m_: s["light"] == GREEN),
            ("... directive stop_at_line (still held)", held.to_numpy(), lambda s, m_: s["dir"] == "stop_at_line"),
            ("red to green, car moving: directive proceed / go_now", (sub.ev_red_to_green & ~held).to_numpy(), lambda s, m_: np.isin(s["dir"], ["proceed", "go_now"])),
            ("... light answered green", (sub.ev_red_to_green & ~held).to_numpy(), lambda s, m_: s["light"] == GREEN),
            ("ego red, all instants: light answered red", sub.ego_red.to_numpy(), lambda s, m_: s["light"] == RED),
            ("ego red, ego stopped (hold): directive stop_at_line", sub.ev_hold.to_numpy(), lambda s, m_: s["dir"] == "stop_at_line"),
            ("ego green: light answered green", sub.ego_green.to_numpy(), lambda s, m_: s["light"] == GREEN),
            ("stopped lead (< 0.5 m/s), ego moving: answered static_block", ((sub.lead_state == "stopped") & sub.ego_moving).to_numpy(), lambda s, m_: s["block"] == "static_block"),
            ("stopped lead, ego stopped: answered static_block", ((sub.lead_state == "stopped") & ~sub.ego_moving).to_numpy(), lambda s, m_: s["block"] == "static_block"),
            ("moving lead (>= 0.5 m/s), ego moving: answered moving_lead", ((sub.lead_state == "moving") & sub.ego_moving).to_numpy(), lambda s, m_: s["block"] == "moving_lead"),
            ("moving lead, ego stopped: answered moving_lead", ((sub.lead_state == "moving") & ~sub.ego_moving).to_numpy(), lambda s, m_: s["block"] == "moving_lead"),
            ("static block, stopped-vehicle routes: answered static_block", ((sub.block == "static_block") & (sub.kind == "vehicle")).to_numpy(), lambda s, m_: s["block"] == "static_block"),
            ("... directive pass_left / pass_right / wait (strict rows)", ((sub.block == "static_block") & (sub.kind == "vehicle") & strict_s).to_numpy(), lambda s, m_: s["dir"] == prim),
            ("clear: answered static_block", (sub.block == "clear").to_numpy(), lambda s, m_: s["block"] == "static_block"),
        ]
        L = ["| instants | 4B 1f | 4B 2f | 2f - 1f | read | 8B 1f | 8B 2f | 2f - 1f | read |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
        for nm, m, fn in rows2:
            cs = []
            for mdl in MODELS:
                h1, h2 = fn(s1[mdl], m), fn(TF[mdl], m)
                dt, v = diff(h2, h1, m, gs)
                cs += [cell(h1, m, gs), cell(h2, m, gs), dt, v]
            L.append("| %s | %s |" % (nm, " | ".join(cs)))
        md += L + [""]
        L = ["| instants | 4B 2f | 8B 2f | 8B - 4B (2f) | read |", "|:--|:--|:--|:--|:--|", ]
        for nm, m, fn in rows2:
            ha, hb = fn(TF["4b"], m), fn(TF["8b"], m)
            dt, v = diff(hb, ha, m, gs)
            L.append("| %s | %s | %s | %s | %s |" % (nm, cell(ha, m, gs), cell(hb, m, gs), dt, v))
        P("Two-frame answers, 8B against 4B on the same instants:\n")
        md += L + [""]
        # previous directive chain: how often does the model keep its previous answer
        for mdl in MODELS:
            keep = (TF[mdl]["dir"] == TF[mdl]["prev"])
            P("%s: the two-frame directive equals the previous (single-frame) directive in %.0f%% of the %d instants; the single-frame directive at the current frame differs from "
              "the previous one in %.0f%%.\n" % (mdl.upper(), 100 * keep.mean(), len(sub), 100 * (s1[mdl]["dir"] != TF[mdl]["prev"]).mean()))
        rows_l = []
        for mdl in MODELS:
            if (mdl, "dir_r1153") in bench and (mdl, "two_r1153") in bench:
                b1, b2 = bench[mdl, "dir_r1153"], bench[mdl, "two_r1153"]
                rows_l.append("%s: directive p50 / p95 %.0f / %.0f ms (peak %.1f GiB), two-moment directive %.0f / %.0f ms (peak %.1f GiB), ratio of p50 %.2f." % (
                    mdl.upper(), b1["p50"], b1["p95"], b1["peak_gb"], b2["p50"], b2["p95"], b2["peak_gb"], b2["p50"] / b1["p50"]))
        if rows_l:
            P("Latency cost of the second moment (bench, section 6): " + " ".join(rows_l) + "\n")

    # ---- heads
    fits = {}
    for m in MODELS:
        for r in RESN:
            p = run / "fit" / ("%s_%s.json" % (m, r))
            if p.exists():
                fits[m, r] = json.loads(p.read_text())
    preds = {}
    for k in fits:
        pz = np.load(run / "fit" / ("%s_%s_preds.npz" % k), allow_pickle=False)
        preds[k] = {n: pz[n] for n in pz.files if n != "ids"}
    ycache = {}
    from vlm_cmp_fit import labels as fit_labels
    Y = fit_labels(df)
    part = df.part.to_numpy()
    head_rows = []
    if fits:
        P("## 5. Frozen features and a linear head per layer (in-domain supervised: fitted on CARLA frames, not zero-shot)\n")
        P("How to read: `src` = where the feature is taken (`ans_<q>`: hidden state at the answer position of question q after layer N; `pool`: mean of the image "
          "tokens of both cameras at layer N, the same for every question). `readable N` = the smallest layer whose 7-fold route-grouped CV balanced accuracy is "
          "within 0.05 of the best layer's, if that best is >= 0.70 (plan section 5). `CV bacc` = out-of-fold balanced accuracy over all 21 routes at that layer "
          "with a route-cluster CI; `test bacc` = head fitted on the 9 train routes, scored on the 8 test routes (n/a: a class is missing from the train "
          "routes, so no head can be fitted on the registered split); `zero-shot` = balanced accuracy of the zero-shot answers of the same model and resolution "
          "on the same rows; `cut latency` = bench p50 / p95 of running only the first N layers (light prompt, linear head cost excluded), `full` = the whole "
          "model, one forward pass. The sign label rests on two routes (both test routes) and the lead label on the routes with a logged lead.\n")
        cutfull = lambda mdl, r: bench.get((mdl, "q1_" + r))   # noqa: E731
        for r in [x for x in ("r1153", "r559", "r2335", "r4573") if all((m, x) in fits for m in MODELS)]:
            P("### 5.%d Resolution %s\n" % (["r1153", "r559", "r2335", "r4573"].index(r) + 1, r))
            L = ["| model | label | src | readable N | CV bacc at N | test bacc at N | best N (CV bacc) | zero-shot bacc | cut latency p50 / p95 ms at N | full p50 ms |",
                 "|:--|:--|:--|--:|:--|:--|:--|:--|:--|--:|"]
            for mdl in MODELS:
                rows = fits[mdl, r]
                zs = {lab: zero_shot_label_pred(ANS[mdl, r], lab) for lab in LAB_NC}
                for lab in ("light", "sign", "block", "lead"):
                    for src in (["ans_light"] if lab == "light" else ["ans_sign"] if lab == "sign" else ["ans_block"]) + ["pool"] + (["ans_dir"] if r == "r1153" else []):
                        rs = [x for x in rows if x["src"] == src and x["label"] == lab]
                        if not rs:
                            continue
                        curve = {x["layer"]: x["cv_bacc"] for x in rs}
                        N, best = readable(curve)
                        bestN = max((k for k in curve if curve[k] is not None and np.isfinite(curve[k])), key=lambda k: curve[k], default=None)
                        y = Y[lab]
                        zsp = zs[lab]
                        zb = bacc_ci(zsp, y, g, LAB_NC[lab])
                        if N is None:
                            L.append("| %s | %s | %s | not readable | best %s | | N=%s | %.2f [%.2f, %.2f] | | %s |" % (
                                mdl.upper(), lab, src, "n/a" if best is None else "%.2f" % best, bestN, zb["est"], zb["lo"], zb["hi"],
                                "" if not cutfull(mdl, r) else "%.0f" % cutfull(mdl, r)["p50"]))
                            continue
                        key = "%s|%d|%s" % (src, N, lab)
                        cvp = preds[mdl, r]["cv|" + key].astype(int)
                        ci = bacc_ci(cvp, np.where(cvp >= 0, y, -1), g, LAB_NC[lab])
                        x = [x for x in rs if x["layer"] == N][0]
                        tb = x.get("test_bacc")
                        cut = bench.get((mdl, "q1_" + r)) if N == 36 else bench.get((mdl, "cut_%s_%d" % (r, N)))
                        L.append("| %s | %s | %s | %d | %.2f [%.2f, %.2f] | %s | %d (%.2f) | %.2f [%.2f, %.2f] | %s | %s |" % (
                            mdl.upper(), lab, src, N, ci["est"], ci["lo"], ci["hi"], "n/a" if tb is None or not np.isfinite(tb) else "%.2f" % tb,
                            bestN, curve[bestN], zb["est"], zb["lo"], zb["hi"],
                            "" if not cut else "%.0f / %.0f" % (cut["p50"], cut["p95"]), "" if not cutfull(mdl, r) else "%.0f" % cutfull(mdl, r)["p50"]))
                        head_rows.append(dict(model=mdl, res=r, label=lab, src=src, readable_N=N, cv_bacc=ci["est"], cv_lo=ci["lo"], cv_hi=ci["hi"],
                                              test_bacc=tb, zeroshot_bacc=zb["est"]))
            md += L + [""]
        pd.DataFrame(head_rows).to_csv(out / "vlm_cmp_heads.csv", index=False)
        # layer curves and paired 8B - 4B at fixed layers, r1153
        if all((m, "r1153") in fits for m in MODELS):
            P("### 5.5 CV balanced accuracy per layer at r1153, 4B and 8B, with the paired difference 8B - 4B at the same layer\n")
            P("Rows = layers; per label and source `4B / 8B (8B - 4B [CI])`. Same frames and folds for both; layer N of the 4B and of the 8B are both of 36 layers.\n")
            combos = [("light", "ans_light"), ("light", "pool"), ("block", "ans_block"), ("block", "pool"), ("lead", "ans_block"), ("lead", "pool"), ("sign", "ans_sign")]
            L = ["| N | " + " | ".join("%s (%s)" % c for c in combos) + " |", "|--:|" + ":--|" * len(combos)]
            curves = {}
            for N in [0] + KEEP:
                cells = []
                for lab, src in combos:
                    if N == 0 and src != "pool":
                        cells.append("")
                        continue
                    key = "%s|%d|%s" % (src, N, lab)
                    try:
                        c4 = preds["4b", "r1153"]["cv|" + key].astype(int)
                        c8 = preds["8b", "r1153"]["cv|" + key].astype(int)
                    except KeyError:
                        cells.append("")
                        continue
                    y = Y[lab]
                    v4 = np.where(c4 >= 0, y, -1)
                    if (v4 < 0).all():
                        cells.append("n/a")
                        continue
                    both = (c4 >= 0) & (c8 >= 0)
                    yy = np.where(both, y, -1)
                    r4 = bacc_ci(c4, yy, g, LAB_NC[lab], other=c8)
                    curves[lab, src, N] = (r4["est"], r4["est"] + r4["d_est"])
                    cells.append("%.2f / %.2f (%+.2f [%+.2f, %+.2f])" % (r4["est"], r4["est"] + r4["d_est"], r4["d_est"], r4["d_lo"], r4["d_hi"]))
                L.append("| %d | %s |" % (N, " | ".join(cells)))
            md += L + [""]
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                labs = ["light", "block", "lead", "sign"]
                fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
                for ax, lab in zip(axs, labs):
                    for mdl, col in (("4b", "#1f77b4"), ("8b", "#d62728")):
                        for src, ls in ((("ans_light" if lab == "light" else "ans_sign" if lab == "sign" else "ans_block"), "-"), ("pool", "--")):
                            xs = [N for N in ([0] + KEEP) if (lab, src, N) in curves]
                            ys = [curves[lab, src, N][0 if mdl == "4b" else 1] for N in xs]
                            if xs:
                                ax.plot(xs, ys, ls, color=col, label="%s %s" % (mdl.upper(), "answer position" if ls == "-" else "pooled image tokens"))
                    ax.axhline(0.70, color="gray", lw=0.6)
                    ax.set_title(lab)
                    ax.set_xlabel("layer N")
                ax_ = axs[0]
                ax_.set_ylabel("CV balanced accuracy")
                ax_.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(out / "vlm_cmp_layers.png", dpi=110)
                P("![CV balanced accuracy per layer](vlm_cmp_layers.png)\n")
                P("Figure: out-of-fold balanced accuracy of the linear head against the layer N it reads (r1153), solid = answer-position state of the question's own prompt, dashed = pooled image tokens, blue 4B, red 8B, grey line = 0.70. Look at where each curve first leaves the floor and whether the 8B curve sits left of the 4B one.\n")
            except Exception as e:  # noqa: BLE001
                P("(figure not drawn: %r)\n" % e)

    # ---- latency
    if bench:
        P("## 6. Latency and memory (batch 1, quiet card, JPEG bytes in, answer out)\n")
        wgb = {m: next((b["weights_gb"] for (mm, c), b in bench.items() if mm == m), float("nan")) for m in MODELS}
        P("n = %d core instants evenly spaced over the sorted ids, 5 warm-up requests discarded, one process alone on the card. `q1` = one question in one forward "
          "pass (the closed-loop path), `dir` = the directive prompt, `four` = the four separate questions on one prefill, `two` = the two-moment directive (4 images). "
          "`peak` = peak allocated GPU memory during the variant, weights included (resident weights: 4B %.1f GiB, 8B %.1f GiB; both include a float32 copy of "
          "the output head used for exact option scoring, 4B %.1f GiB and 8B %.1f GiB, that a deployment would not need).\n" % (
              next((b["frames"] for b in bench.values()), 0), wgb["4b"], wgb["8b"],
              next((b.get("wf_gb", float("nan")) for (mm, c), b in bench.items() if mm == "4b"), float("nan")),
              next((b.get("wf_gb", float("nan")) for (mm, c), b in bench.items() if mm == "8b"), float("nan"))))
        L = ["| variant | 4B p50 / p95 / p99 ms | 8B p50 / p95 / p99 ms | 8B / 4B (p50) | 4B peak GiB | 8B peak GiB |", "|:--|:--|:--|--:|--:|--:|"]
        cfgs = [c for c in ["q1_" + r for r in RESN] + ["dir_" + r for r in RESN] + ["four_r559", "four_r1153", "two_r559", "two_r1153"]
                if ("4b", c) in bench or ("8b", c) in bench]
        lat_rows = []
        for c in cfgs:
            b4, b8 = bench.get(("4b", c)), bench.get(("8b", c))
            f = lambda b: "failed" if b is None or "p50" not in b else "%.0f / %.0f / %.0f" % (b["p50"], b["p95"], b["p99"])   # noqa: E731
            ratio = "" if not (b4 and b8 and "p50" in b4 and "p50" in b8) else "%.2f" % (b8["p50"] / b4["p50"])
            L.append("| %s | %s | %s | %s | %s | %s |" % (c, f(b4), f(b8), ratio, "" if not b4 or "peak_gb" not in b4 else "%.1f" % b4["peak_gb"],
                                                       "" if not b8 or "peak_gb" not in b8 else "%.1f" % b8["peak_gb"]))
            lat_rows.append(dict(cfg=c, p50_4b=b4 and b4.get("p50"), p95_4b=b4 and b4.get("p95"), p50_8b=b8 and b8.get("p50"), p95_8b=b8 and b8.get("p95")))
        md += L + [""]
        L = ["Cut latency (first N layers, light prompt, p50 / p95 ms):", "", "| N | 4B r1153 | 8B r1153 | 8B / 4B | 4B r559 | 8B r559 | 8B / 4B |", "|--:|:--|:--|--:|:--|:--|--:|"]
        for N in KEEP:
            cs, rt = [], []
            for r in ("r1153", "r559"):
                b4, b8 = bench.get(("4b", "cut_%s_%d" % (r, N))), bench.get(("8b", "cut_%s_%d" % (r, N)))
                cs += ["" if not b4 or "p50" not in b4 else "%.0f / %.0f" % (b4["p50"], b4["p95"]), "" if not b8 or "p50" not in b8 else "%.0f / %.0f" % (b8["p50"], b8["p95"]),
                       "" if not (b4 and b8 and "p50" in b4 and "p50" in b8) else "%.2f" % (b8["p50"] / b4["p50"])]
            L.append("| %d | %s |" % (N, " | ".join(cs)))
        md += L + [""]
        pd.DataFrame(lat_rows).to_csv(out / "vlm_cmp_latency.csv", index=False)

    # ---- summary (mechanical)
    if "r1153" in have:
        RA, RB = ZSR["r1153"]
        nm = {n: i for i, (_, n, _, _) in enumerate(RA)}
        P("## 7. Summary table 4B vs 8B at r1153 (the resolution of the closed-loop server)\n")
        P("Every capability row with both estimates, the paired difference (same frames, route-cluster CI) and the mechanical read. Zero-shot rows only; head and "
          "latency rows below. Rows with fewer than 3 routes behind them cannot carry a conclusion.\n")
        keys = [("light", "ego red answered red, all"), ("light", "... position A before the stop line"), ("light", "... position B between line and junction entrance"),
                ("light", "... position C inside the junction"), ("light", "ego red answered green, all"), ("light", "ego green answered green, all"),
                ("light", "other-direction red (ego not red) answered red"), ("light", "no light at all answered red"),
                ("sign", "stop sign within 25 m answered yes"), ("sign", "no sign within 80 m answered yes"),
                ("block", "static block answered static_block, all"), ("block", "... cones routes"), ("block", "... stopped-vehicle routes"),
                ("block", "... other routes (queues)"), ("block", "stopped lead (< 0.5 m/s) answered static_block"), ("block", "clear answered static_block"),
                ("block", "moving_lead truth answered moving_lead"), ("side", "bypass side correct, static-block rows with a side truth"),
                ("directive", "directive: strict accuracy (rows without an ambiguity note)"), ("directive", "directive: accepted accuracy, all rows"),
                ("composed", "composed from the questions: strict accuracy"), ("composed", "composed: accepted accuracy, all rows")]
        L = ["| capability | 4B | 8B | 8B - 4B (points) | read |", "|:--|:--|:--|:--|:--|"]
        # the light position rows are listed under both red-as-red and red-as-green; take the first occurrence of each name after the group's header row
        def find(group, name):
            for i, (gg, n, m, h) in enumerate(RA):
                if gg == group and n == name:
                    return i
        for grp, name in keys:
            i = find(grp, name)
            if i is None:
                continue
            m, ha, hb = RA[i][2], RA[i][3], RB[i][3]
            dt, v = diff(hb, ha, m, g)
            L.append("| %s | %s | %s | %s | %s |" % (name if not name.startswith("...") else name, cell(ha, m, g), cell(hb, m, g), dt, v))
        md += L + [""]
        if tf_ok:
            P("Two-frame rows (4B / 8B, r1153): see section 4. Head rows: section 5. Latency ratios: section 6.\n")

    # ---- equivalence checks
    P("## 8. Checks\n")
    for m in MODELS:
        p = run / ("selftest-%s.json" % m)
        if p.exists():
            d = json.loads(p.read_text())
            mx = max(c[q]["max_abs_lp_diff_unique_first"] for c in d["checks"] for q in ("light", "dir"))
            hx = max(c[q]["hid_rel_diff"] for c in d["checks"] for q in ("light", "dir"))
            ok = all(c[q]["same_argmax_cache_vs_full"] and c[q]["repeat_equal"] for c in d["checks"] for q in ("light", "dir"))
            P("- %s selftest (%d frame x resolution checks): shared-prefix cache path vs one full forward without cache: max |log-prob difference| of the first-token options %.2f nats "
              "(bf16 rounding of the forward kernels; the option log-probs here are mostly -20 to 0), max relative difference of the answer-position hidden states %.4f, "
              "same answer in all: %s." % (m.upper(), len(d["checks"]), mx, hx, ok))
    old = list((Path(REPO) / "x").glob("*")) if False else []
    try:
        from vlm_thin_common import CACHE
        dirs = sorted((CACHE / "features").glob("*/r1153"))
        if dirs and ("4b", "r1153") in F:
            parts = [np.load(p) for p in sorted(dirs[-1].glob("shard-*.npz"))]
            oid = np.concatenate([p["ids"] for p in parts])
            ozs = np.concatenate([p["zs"] for p in parts])
            key = df.unit + "/" + df.route + "/" + df.t.map(lambda t: "%.2f" % t)
            om = pd.Series(range(len(oid)), index=oid)
            hit = key.isin(om.index) & (df.attempt == "1")
            if hit.sum():
                j = om.loc[key[hit]].to_numpy()
                oldp = np.exp(ozs[j] - ozs[j].max(1, keepdims=True))
                oldp /= oldp.sum(1, keepdims=True)
                lp = F["4b", "r1153"]["lp_light"][hit.to_numpy()]
                newp = np.exp(lp - lp.max(1, keepdims=True))
                newp /= newp.sum(1, keepdims=True)
                P("- 4B light option scores against the earlier vlm_thin cache (r1153) on the %d core frames that are in both (same unit, route, time, attempt 1): same answer in %.1f%%, "
                  "max |probability difference| %.3f (different preprocessing path details: prefix cache, float32 head over the whole vocabulary)." % (
                      int(hit.sum()), 100 * (oldp.argmax(1) == newp.argmax(1)).mean(), np.abs(oldp - newp).max()))
    except Exception as e:  # noqa: BLE001
        P("- earlier-cache comparison not run: %r" % e)
    (out / "vlm_4b_vs_8b_generated.md").write_text("\n".join(md) + "\n")
    print("wrote", out / "vlm_4b_vs_8b_generated.md")


if __name__ == "__main__":
    main()
