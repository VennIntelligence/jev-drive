"""Phase A readout: VLM answers against ground truth on every shadow frame collected so far, with the fixed labels.

  python vlm_arb_phase_a.py [--stage old|final]

Sources (all `drive`, the VLM never controls):
  old   the first executor's runs: shadow-drive-s0 (10 dev routes) and the three drive runs on the 58 routes. Their
        log has the ego light and its distance, and the obstacle label; no stop-sign actor and no other lights.
  new   v2 drive units with VLM_SHADOW=1 (fixed labels: every light within 60 m, on-route stop-sign actors, bypass side).
One row per request. Everything is a ratio with a route-cluster bootstrap (2000 resamples, seed 0).

Labels (fixed before the new frames were read, plan deviation D3):
  light     ego light red / yellow / green by distance of the bumper to its stop line: 0-20, 20-50, 50-80 m. The
            registered recall line is read on 0-50 m; 50-80 m is listed on its own and is neither a recall miss nor
            a false alarm.
  other     "a light of another direction is red while the ego has no red": new frames = some non-ego light within
            60 m is red and the ego light is green or absent; old frames = ego light green within 50 m (in CARLA the
            other directions of a junction are red then; assumption, checked on the new frames). The registered
            line: P(answer red_for_ego) <= 10%.
  no light  no ego light within 80 m (new frames: and no light at all within 60 m): P(answer red_for_ego) <= 2%.
  sign      new frames only: positive = an on-route stop-sign actor within 25 m ahead (R3's window), negative = none
            within 80 m; 25-80 m is listed, not scored.
  block     static_block = a stationary actor in the ego lane 0-50 m ahead outside junctions (the bypass trigger).
Writes results/phase_a.{json,md}, results/phase_a.png and (stage final) gates/phase_a.json.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import OLD_DRIVE, REPO, RUN, boot_ratio, fmt, write_json  # noqa: E402

sys.path.insert(0, str(REPO))
from jevdrive.common import n_cpus  # noqa: E402

LINES = dict(red_recall=0.80, other_fp=0.10, nolight_fp=0.02, sign_recall=0.70, sign_fp=0.05, block_recall=0.80,
             latency_p95_ms=600.0)
RED = "red_or_yellow_for_ego"


def read_old(path):
    """Old log: one line per tick carrying the current answer; one row per distinct request."""
    route, rows, gt_at, seen = path.parts[-3], [], {}, set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gt_at[round(d["t"], 2)] = d["gt"]
            a = d.get("vlm_answer")
            if a and a["sim_t_queried"] not in seen:
                seen.add(a["sim_t_queried"])
                rows.append(a)
    out = []
    for a in rows:
        g = gt_at.get(round(a["sim_t_queried"], 2))
        if g is None or not a.get("ok"):
            continue
        tl = g.get("tl_state")
        out.append(dict(src="old", unit=path.parts[-5], route=route, t=a["sim_t_queried"], tl=-1 if tl is None else tl,
                        tl_dist=g["tl_dist"] if tl is not None else np.nan, other_red=np.nan, any_light=np.nan,
                        stop_dist=np.nan, has_sign_label=False, block=g["gt_block"], side="", lead=np.nan,
                        a_light=a["Q_light"], a_sign=a["Q_sign"], a_block=a["Q_block"], a_side=a["Q_side"],
                        lat=a["latency_ms"], ok=True))
    return out


def read_new(path):
    route, out = path.parts[-3], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("k") != "a":
                continue
            g, a = d["gt"], d["ans"]
            tl = g.get("tl")
            others = [x for x in g.get("lights", []) if x[0] != g.get("tl_id")]
            row = dict(src="new", unit=path.parts[-5], route=route, t=d["t_q"], tl=-1 if tl is None else tl,
                       tl_dist=g["tl_dist"] if tl is not None else np.nan,
                       other_red=float(any(x[1] == 2 for x in others)), any_light=float(bool(g.get("lights"))),
                       other_differs=float(any(x[1] != tl for x in others)) if tl is not None and others else np.nan,
                       stop_dist=np.nan if g.get("stop_dist") is None else g["stop_dist"], has_sign_label=True,
                       block=g["block"], side=g["side"], lat=a.get("latency_ms", np.nan), ok=bool(a.get("ok")))
            row.update(a_light=a.get("Q_light", ""), a_sign=a.get("Q_sign", ""), a_block=a.get("Q_block", ""),
                       a_side=a.get("Q_side", ""))
            out.append(row)
    return out


def load(extra_dirs=()):
    jobs = []
    for u in OLD_DRIVE:
        jobs += [(read_old, p) for p in sorted((RUN / "arms" / u).glob("attempts/*/*/vlm_decisions.jsonl"))]
    for d in sorted((RUN / "arms").glob("v2-drive-*")) + [Path(x) for x in extra_dirs]:
        for done in sorted(d.glob("done/*.json")):
            p = d / "attempts" / done.stem / str(json.loads(done.read_text()).get("attempt", 1)) / "vlm_decisions.jsonl"
            if p.exists():
                jobs.append((read_new, p))
    with Pool(min(n_cpus(), 32)) as pool:
        parts = pool.starmap(_call, jobs, chunksize=2)
    return pd.DataFrame([r for p in parts for r in p])


def _call(fn, p):
    return fn(p)


def rate(df, mask, hit):
    m = mask & df.ok
    return boot_ratio((hit & m).astype(float), m.astype(float), df.route)


def green_delay(df, K=2):
    """Seconds from the ego light turning green (after red) to the K-th consecutive green answer, per transition."""
    out = []
    for _, g in df[df.ok].sort_values("t").groupby(["unit", "route"]):
        tl, ans, t = g.tl.to_numpy(), g.a_light.to_numpy(), g.t.to_numpy()
        for i in range(1, len(g)):
            if tl[i] == 0 and tl[i - 1] in (1, 2):
                run = 0
                for j in range(i, len(g)):
                    if tl[j] != 0:
                        break
                    run = run + 1 if ans[j] == "green_for_ego" else 0
                    if run >= K:
                        out.append(t[j] - t[i])
                        break
                else:
                    out.append(np.nan)
    return out


def evaluate(df):
    red, green, none = df.tl.isin([1, 2]), df.tl == 0, df.tl == -1
    R = {}
    for name, lo, hi in (("0-20", -5, 20), ("20-50", 20, 50), ("0-50", -5, 50), ("50-80", 50, 81)):
        b = (df.tl_dist >= lo) & (df.tl_dist < hi)
        R["red_recall_" + name] = rate(df, red & b, df.a_light == RED)
        R["red_as_green_" + name] = rate(df, red & b, df.a_light == "green_for_ego")
        R["green_recall_" + name] = rate(df, green & b, df.a_light == "green_for_ego")
    new = df.src == "new"
    other_old = (~new) & green & (df.tl_dist < 50)
    other_new = new & (df.other_red == 1) & ~(red & (df.tl_dist < 50))
    R["other_fp"] = rate(df, other_old | other_new, df.a_light == RED)
    R["other_fp_new"] = rate(df, other_new, df.a_light == RED)
    R["other_lane_answer_used"] = rate(df, pd.Series(True, df.index), df.a_light == "light_for_other_lane")
    nolight = none & ((~new) | (df.any_light == 0))
    R["nolight_fp"] = rate(df, nolight, df.a_light == RED)
    R["nolight_green"] = rate(df, nolight, df.a_light == "green_for_ego")
    R["group_assumption_new"] = rate(df, new & green & (df.tl_dist < 50), df.other_red == 1)   # others red when ego green
    R["red27043_as_green_old"] = rate(df, red & (df.route == "27043") & (~new), df.a_light == "green_for_ego")
    lab = df.has_sign_label
    R["sign_recall"] = rate(df, lab & (df.stop_dist <= 25), df.a_sign == "yes")
    R["sign_fp"] = rate(df, lab & df.stop_dist.isna(), df.a_sign == "yes")
    R["sign_far_yes"] = rate(df, lab & (df.stop_dist > 25), df.a_sign == "yes")
    R["block_recall"] = rate(df, df.block == "static_block", df.a_block == "static_block")
    R["block_fp_clear"] = rate(df, df.block == "clear", df.a_block == "static_block")
    R["side_acc"] = rate(df, new & (df.block == "static_block") & (df.side != "none_free") & (df.a_block == "static_block"),
                         df.a_side == df.side)
    lat = df.lat[df.ok].to_numpy()
    R["latency_ms"] = dict(n=int(len(lat)), p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
                           p99=float(np.percentile(lat, 99)), ok_rate=float(df.ok.mean()))
    gd = green_delay(df)
    R["green_delay_s"] = dict(n=len(gd), censored=int(np.isnan(gd).sum()) if gd else 0,
                              median=float(np.nanmedian(gd)) if gd and not np.all(np.isnan(gd)) else None)
    e = lambda k: R[k]["est"]   # noqa: E731
    gate = dict(
        q_light=bool(e("red_recall_0-50") >= LINES["red_recall"] and e("other_fp") <= LINES["other_fp"]
                     and e("nolight_fp") <= LINES["nolight_fp"]),
        q_sign=bool(R["sign_recall"]["n"] > 0 and e("sign_recall") >= LINES["sign_recall"] and e("sign_fp") <= LINES["sign_fp"]),
        q_block=bool(e("block_recall") >= LINES["block_recall"]),
        latency=bool(R["latency_ms"]["p95"] <= LINES["latency_p95_ms"]))
    gate["L_s"] = float(np.ceil(R["latency_ms"]["p95"] / 50.0) * 0.05)       # registered: L = measured p95, on the tick grid
    return R, gate


def table(R, gate, df):
    L = ["| readout | value [95% CI] | requests | routes | registered line | pass |", "|:--|:--|--:|--:|:--|:--|"]
    def row(k, label, line="", ok=None):
        r = R[k]
        L.append("| %s | %s | %d | %d | %s | %s |" % (label, fmt(r, True), r["n"], r["groups"], line,
                                                     "" if ok is None else ("yes" if ok else "**no**")))
    e = lambda k: R[k]["est"]   # noqa: E731
    row("red_recall_0-50", "ego red / yellow answered red, 0-50 m", ">= 80%", e("red_recall_0-50") >= .8)
    row("red_recall_0-20", "... 0-20 m")
    row("red_recall_20-50", "... 20-50 m")
    row("red_recall_50-80", "... 50-80 m (listed, not scored)")
    row("red_as_green_0-50", "ego red / yellow answered green, 0-50 m")
    row("red27043_as_green_old", "... route 27043 alone (old frames)")
    row("green_recall_0-50", "ego green answered green, 0-50 m")
    row("other_fp", "another direction red, ego not red: answered red", "<= 10%", e("other_fp") <= .1)
    row("other_fp_new", "... new frames only (lights logged)")
    row("other_lane_answer_used", "answer light_for_other_lane, any frame")
    row("nolight_fp", "no light: answered red", "<= 2%", e("nolight_fp") <= .02)
    row("nolight_green", "no light: answered green")
    row("group_assumption_new", "check: ego green -> another light red (new frames)")
    row("sign_recall", "stop sign within 25 m answered yes", ">= 70%", gate["q_sign"] or (R["sign_recall"]["n"] > 0 and e("sign_recall") >= .7))
    row("sign_fp", "no stop sign within 80 m answered yes", "<= 5%", R["sign_fp"]["n"] > 0 and e("sign_fp") <= .05)
    row("sign_far_yes", "stop sign 25-80 m answered yes (listed)")
    row("block_recall", "static block answered static_block", ">= 80%", gate["q_block"])
    row("block_fp_clear", "clear answered static_block")
    row("side_acc", "bypass side correct given block seen (new frames)")
    la, gd = R["latency_ms"], R["green_delay_s"]
    L += ["", "Latency over %d requests: p50 %.0f ms, p95 %.0f ms, p99 %.0f ms (line: p95 <= 600 ms: %s); answered %.1f%%. "
          "L for Phase B = %.2f s." % (la["n"], la["p50"], la["p95"], la["p99"], "yes" if gate["latency"] else "**no**",
                                      100 * la["ok_rate"], gate["L_s"]),
          "Green after red: %d transitions, %d never reached K = 2 green answers, median delay %s s." % (
              gd["n"], gd["censored"], "n/a" if gd["median"] is None else "%.1f" % gd["median"]),
          "", "Gates: Q-light %s, Q-sign %s, Q-block %s, latency %s." % tuple(
              "pass" if gate[k] else "FAIL" for k in ("q_light", "q_sign", "q_block", "latency")),
          "", "Requests by source: " + ", ".join("%s %d (%d routes)" % (s, len(g), g.route.nunique()) for s, g in df.groupby("src")),
          "", "Q-light confusion (rows: truth within 50 m, else no_light / far; columns: answer):", "",
          pd.crosstab(np.where(df.tl.isin([1, 2]) & (df.tl_dist < 50), "red_or_yellow", np.where((df.tl == 0) & (df.tl_dist < 50), "green",
                      np.where(df.tl == -1, "no_light", "light 50-80 m"))), df.a_light).to_markdown()]
    return "\n".join(L) + "\n"


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("red_recall_0-20", "ego red, 0-20 m", .8), ("red_recall_20-50", "ego red, 20-50 m", .8),
            ("red_recall_50-80", "ego red, 50-80 m", None), ("green_recall_0-50", "ego green, 0-50 m", None),
            ("sign_recall", "stop sign <= 25 m", .7), ("block_recall", "static block", .8),
            ("other_fp", "other direction red -> red", .1), ("nolight_fp", "no light -> red", .02),
            ("sign_fp", "no sign -> yes", .05), ("block_fp_clear", "clear -> block", None)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    for i, (k, label, line) in enumerate(keys):
        r, y = R[k], len(keys) - 1 - i
        if r["n"]:
            ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color="#3b6ea5", lw=2, solid_capstyle="round")
            ax.plot(100 * r["est"], y, "o", color="#3b6ea5", ms=6)
            ax.text(103, y, "%.0f%%  (n=%d, %d routes)" % (100 * r["est"], r["n"], r["groups"]), va="center", fontsize=7, color="#444")
        else:
            ax.text(103, y, "no frames", va="center", fontsize=7, color="#888")
        if line is not None:
            ax.plot([100 * line] * 2, [y - .35, y + .35], color="#222", lw=1.2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k[1] for k in keys][::-1], fontsize=8)
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of requests, % (dot: estimate, bar: 95% route-cluster CI, tick: registered line)", fontsize=8)
    ax.axhline(3.5, color="#ccc", lw=.8)
    ax.set_title("Phase A: correct answers (top six) and false alarms (bottom four)", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee", lw=.6)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="final", choices=["old", "final"])
    a = ap.parse_args()
    df = load()
    if a.stage == "old":
        df = df[df.src == "old"]
    R, gate = evaluate(df.reset_index(drop=True))
    out = RUN / "results"
    out.mkdir(parents=True, exist_ok=True)
    tag = "phase_a" if a.stage == "final" else "phase_a_old"
    head = "# Phase A (%s): VLM answers against ground truth, %d requests, %d routes\n\n" % (a.stage, len(df), df.route.nunique())
    (out / (tag + ".md")).write_text(head + table(R, gate, df))
    write_json(out / (tag + ".json"), dict(readouts=R, gate=gate, lines=LINES, n=len(df)))
    figure(R, out / (tag + ".png"))
    if a.stage == "final":
        write_json(RUN / "gates/phase_a.json", dict(gate, readouts={k: v.get("est") for k, v in R.items() if "est" in v}))
    print(head + table(R, gate, df))


if __name__ == "__main__":
    main()
