"""Phase A of the `vred` variant: the registered Q-light lines and latency of zero-shot Qwen3-VL-4B as served for the closed loop.

  python vlm_qwen_phase_a.py run --cards 0,1,2 [--out DIR]    replay every labelled request with frames through the per-card servers
  python vlm_qwen_phase_a.py report [--out DIR]               lines, tables, gate file

The requests go through the exact serving path of the closed loop (vlm_qwen_server.py: JPEG bytes over HTTP, GPU
preprocessing, one forward pass at 1153 visual tokens, option scoring over the four answers, both cameras): one request at
a time per card, a quiet card (nothing else on it), JPEG bytes already in host memory. The frames are all answered requests of the
shadow units with both frames saved (3281 requests, 21 routes; vlm_thin_common.load_frames). Label definitions and the
lines are the registered ones (vlm_arb_phase_a.LINES; masks in vlm_thin_common.light_masks); nothing is changed.
Gate = the registered rule: point estimate against the line (as vlm_arb_phase_a.evaluate), CIs listed.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vlm_arb_common import DATA, REPO, RUN, boot_ratio, write_json  # noqa: E402
from vlm_arb_phase_a import LINES  # noqa: E402
from vlm_thin_common import LIGHTS, RED, light_masks, load_frames, percentiles, read_jpgs  # noqa: E402

sys.path.insert(0, str(REPO / "lib"))
from vlm_qwen_client import PORT0, QwenClient  # noqa: E402

OUT = RUN / "vred"


def replay(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_frames()
    cards = [int(c) for c in a.cards.split(",")]
    print("%d requests on %d cards" % (len(df), len(cards)), flush=True)
    shards = [df.iloc[k::len(cards)] for k in range(len(cards))]
    errors = []

    def work(k):
        cl, f = QwenClient(PORT0 + cards[k], timeout_s=30.0), out / ("replay-%d.jsonl" % cards[k])
        have = {json.loads(l)["id"] for l in open(f)} if f.exists() else set()
        with open(f, "a", buffering=1) as fh:
            for r in shards[k].itertuples():
                if r.id in have:
                    continue
                jp = read_jpgs(r)
                ans = cl.ask(dict(wide=jp[0], road=jp[1]))
                if not ans["ok"]:
                    errors.append(ans)
                    continue
                fh.write(json.dumps(dict(id=r.id, ans=ans["Q_light"], p=ans["Q_light_p"], lat_ms=ans["latency_ms"],
                                         srv_svc_ms=ans["srv_svc_ms"], srv_queue_ms=ans["srv_queue_ms"])) + "\n")
    th = [threading.Thread(target=work, args=(k,)) for k in range(len(cards))]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    print("replay done in %.0f s, %d failed requests" % (time.time() - t0, len(errors)), flush=True)
    if errors:
        print(errors[:3])


def report(a):
    out = Path(a.out)
    df = load_frames()
    rows = [json.loads(l) for f in sorted(out.glob("replay-*.jsonl")) for l in open(f)]
    ans = pd.DataFrame(rows).drop_duplicates("id").set_index("id")
    df = df[df.id.isin(ans.index)].reset_index(drop=True)
    df["ans"] = ans.loc[df.id, "ans"].to_numpy()
    df["lat"] = ans.loc[df.id, "lat_ms"].to_numpy()
    df["svc"] = ans.loc[df.id, "srv_svc_ms"].to_numpy()
    m = light_masks(df)
    near = lambda lo, hi: (df.tl_dist >= lo) & (df.tl_dist < hi)   # noqa: E731
    R = {}

    def rate(k, mask, hit):
        R[k] = boot_ratio((hit & mask).astype(float), mask.astype(float), df.route)
    red = df.tl.isin([1, 2])
    rate("red_recall_0-50", m["red"], df.ans == RED)
    rate("red_recall_0-20", red & near(-5, 20), df.ans == RED)
    rate("red_recall_20-50", red & near(20, 50), df.ans == RED)
    rate("red_recall_50-80", red & near(50, 81), df.ans == RED)
    rate("red_as_green_0-50", m["red"], df.ans == "green_for_ego")
    rate("green_recall_0-50", m["green"], df.ans == "green_for_ego")
    rate("other_fp", m["other"], df.ans == RED)
    rate("nolight_fp", m["nolight"], df.ans == RED)
    rate("nolight_green", m["nolight"], df.ans == "green_for_ego")
    rate("other_lane_answer_used", pd.Series(True, df.index), df.ans == "light_for_other_lane")
    sw = df[df["sweep"]].reset_index(drop=True)
    sm = light_masks(sw)
    sweep = dict(red_red=int(((sw.ans == RED) & sm["red"]).sum()), red_n=int(sm["red"].sum()),
                 red_green=int(((sw.ans == "green_for_ego") & sm["red"]).sum()),
                 green_green=int(((sw.ans == "green_for_ego") & sm["green"]).sum()), green_n=int(sm["green"].sum()),
                 none_red=int(((sw.ans == RED) & sm["nolight"]).sum()), none_n=int(sm["nolight"].sum()))
    lat, svc = percentiles(df.lat), percentiles(df.svc)
    e = lambda k: R[k]["est"]   # noqa: E731
    gate = dict(q_light=bool(e("red_recall_0-50") >= LINES["red_recall"] and e("other_fp") <= LINES["other_fp"]
                             and e("nolight_fp") <= LINES["nolight_fp"]),
                red_recall=bool(e("red_recall_0-50") >= LINES["red_recall"]), other_fp=bool(e("other_fp") <= LINES["other_fp"]),
                nolight_fp=bool(e("nolight_fp") <= LINES["nolight_fp"]), latency=bool(lat["p95"] <= LINES["latency_p95_ms"]),
                latency_p50_ms=lat["p50"], latency_p95_ms=lat["p95"])
    gate["proceed"] = bool(gate["q_light"] and gate["latency"])
    fmt = lambda r: "n/a" if not r["n"] else "%.1f%% [%.1f, %.1f]" % (100 * r["est"], 100 * r["lo"], 100 * r["hi"])   # noqa: E731
    lines = {"red_recall_0-50": ">= 80%", "other_fp": "<= 10%", "nolight_fp": "<= 2%"}
    names = {"red_recall_0-50": "ego red / yellow answered red, 0-50 m", "red_recall_0-20": "... 0-20 m", "red_recall_20-50": "... 20-50 m",
             "red_recall_50-80": "... 50-80 m (listed, not scored)", "red_as_green_0-50": "ego red / yellow answered green, 0-50 m",
             "green_recall_0-50": "ego green answered green, 0-50 m", "other_fp": "another direction red, ego not red: answered red",
             "nolight_fp": "no light: answered red", "nolight_green": "no light: answered green",
             "other_lane_answer_used": "answer light_for_other_lane, any frame"}
    L = ["# vred Phase A: zero-shot Qwen3-VL-4B, one forward pass, 1153 visual tokens, both cameras, as served", "",
         "%d labelled requests with saved frames, %d routes (%d with an ego red within 50 m). Each request went through the "
         "closed-loop serving path (HTTP, JPEG bytes, GPU preprocessing, option scoring), one at a time on a quiet card. Estimate "
         "[95%% route-cluster bootstrap CI, 2000 resamples, seed 0]. Gate = point estimate against the registered line." % (
             len(df), df.route.nunique(), df[m["red"]].route.nunique()), "",
         "| readout | value [95% CI] | requests | routes | registered line | pass |", "|:--|:--|--:|--:|:--|:--|"]
    for k, name in names.items():
        r = R[k]
        ok = "" if k not in lines else ("yes" if gate[{"red_recall_0-50": "red_recall", "other_fp": "other_fp", "nolight_fp": "nolight_fp"}[k]] else "**no**")
        L.append("| %s | %s | %d | %d | %s | %s |" % (name, fmt(r), r["n"], r["groups"], lines.get(k, ""), ok))
    L += ["", "Latency over %d requests (client round trip, quiet card, batch 1): p50 %.0f ms, p95 %.0f ms, p99 %.0f ms; server "
          "service time p50 %.0f ms. Registered line p95 <= 600 ms: %s." % (lat["n"], lat["p50"], lat["p95"], lat["p99"], svc["p50"],
                                                                        "yes" if gate["latency"] else "**no**"), "",
          "Gate: Q-light %s, latency %s. Proceed to the closed loop: **%s**." % ("pass" if gate["q_light"] else "FAIL",
                                                                                 "pass" if gate["latency"] else "FAIL",
                                                                                 "yes" if gate["proceed"] else "NO"), "",
          "Reproduction on the 233-frame sweep subset of decision 85 (fwd at 1153 tokens: ego red answered red 76/85, answered green 5/85, "
          "green answered green 59/68, no light answered red 0/80): here %d/%d, %d/%d, %d/%d, %d/%d." % (
              sweep["red_red"], sweep["red_n"], sweep["red_green"], sweep["red_n"], sweep["green_green"], sweep["green_n"],
              sweep["none_red"], sweep["none_n"]), "",
          "Q-light confusion (rows: truth, columns: answer):", "",
          pd.crosstab(np.where(m["red"], "ego red/yellow 0-50 m", np.where(m["green"], "ego green 0-50 m", np.where(m["nolight"], "no light", "other"))),
                      df.ans).reindex(columns=LIGHTS).fillna(0).astype(int).to_markdown(), ""]
    per = []
    for rid, g in df.groupby("route"):
        mm = light_masks(g)
        per.append(dict(route=rid, requests=len(g), red_n=int(mm["red"].sum()), red_as_red=int(((g.ans == RED) & mm["red"]).sum()),
                        green_n=int(mm["green"].sum()), green_as_green=int(((g.ans == "green_for_ego") & mm["green"]).sum()),
                        nolight_n=int(mm["nolight"].sum()), nolight_as_red=int(((g.ans == RED) & mm["nolight"]).sum()),
                        other_n=int(mm["other"].sum()), other_as_red=int(((g.ans == RED) & mm["other"]).sum())))
    L += ["Per route (counts):", "", pd.DataFrame(per).to_markdown(index=False), ""]
    text = "\n".join(L)
    (out / "phase_a.md").write_text(text)
    write_json(out / "phase_a.json", dict(readouts=R, gate=gate, latency=lat, service=svc, sweep=sweep, lines=LINES, n=len(df)))
    write_json(RUN / "gates/vred_phase_a.json", gate)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--cards", default="0,1,2")
    ap.add_argument("--out", default=str(OUT / "phase_a"))
    a = ap.parse_args()
    {"run": replay, "report": report}[a.cmd](a)
