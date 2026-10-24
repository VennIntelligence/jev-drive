#!/usr/bin/env python3
"""D3-FAIL: where D3 (NT4D3-F-s0) loses points on PAI 281 and on nuPlan, and a video page with one clip per failure type
(research/d3-failures/). Lane: d3nu.sh (Part 1 reads; Part 2 job bodies `vread` / `fail-*`). Box lane dir $DATA_DIR/runs/round9/d3_fail.

Sources (stored reads only for the statistics):
  PAI 281   D3-s0's stored ACC + GAP Harmonizer-off read (decision 473) vs R8K-L18-s2's (decision 443); per scene: pai_fix facts (old scorer,
            failure class fclass) + the new scorer (scorer_v2 new, the board's): D3 runs/round9/ntv1_step4/sv2, R8K runs/alpasim/scorer_v2/report.
  nuPlan    the facts cache (jevdrive/facts.py) of D3's arena held native read and held 460 @18 read (d3nu.sh) vs the stored B2
            (STANDFMXH-L18-s1) and NTV0-L18-s0 reads; class = the cache's ftype; arena = held windows (route-fail windows dropped, 2 228).
Clips: one per failure type (two for the large classes), D3 next to the reference on the same scene / window, preferring scenes where D3
fails and the reference passes. A side is the stored read's own rollout when its rollout.asl survived (PAI zero scores; rendered on CPU with
AlpaSim's eval video, as round 7's loss_videos.py), else a camera rerun of the same checkpoint and serve on the pool (PAI: pai_box.sh, CONC 2;
nuPlan: d3nu.sh vread = jt.sh aread / rm_chain.sh read + eval.video.render_video=true; nuPlan has no usdz so no offline render), one GPU job
at a time at low priority; the rerun's score is recorded next to the stored one.

  D3F_O = the phase's lane dir (runs/round9/d3_fail/pai | nu)
  d3f.py select [--tracks pai|arena,held]  (box, .venv)  statistics -> $O/stats.json; clips -> $O/sel.json + rerun lists $O/lists/
  d3f.py asl         (box, AlpaSim venv)  stored PAI sides -> $O/videos/
  d3f.py collect     (box, AlpaSim venv)  every side: score, event onsets, 10 Hz speed, video -> $O/facts.json; composed clips -> $O/clips/
  d3f.py page DIR    (Mac)  DIR/media/{clips.json, *.mp4} + $O stats -> DIR/index.html + DIR/manifest.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
REPO = Path(__file__).resolve().parents[3]
RUNS = DATA / "runs"
O = Path(os.environ.get("D3F_O", RUNS / "round9/d3_fail/pai"))   # one lane dir per phase: d3_fail/pai, d3_fail/nu
AP = DATA / "third_party/alpasim/.venv/bin/python"
FFMPEG = DATA / "third_party/alpasim/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
D3, R8K, B2, NTV0 = "NT4D3-F-s0", "R8K-L18-s2", "STANDFMXH-L18-s1", "NTV0-L18-s0"
SHORT = {D3: "D3", R8K: "R8K-s2", B2: "B2", NTV0: "NTV0"}
PAI_READ = {D3: RUNS / "round9/ntv1_step4/pai281/NT4D3-F-s0-ag", R8K: RUNS / "round8/s2/pai281/R8K-L18-s2-ag"}
NU_READ = {"arena": lambda t: RUNS / f"defect_fix/jt/arena/{t}@off", "held": lambda t: RUNS / f"route_mem/read/{t}"}
ZERO_PAI = ["offroad", "AF-front", "side", "overshoot", "route-turn", "zero-other"]
PART_PAI = ["rear-hit", "slow", "ok"]
# clips per (track, class): PAI by fclass, nuPlan by ftype; order = page order
PICK = [("pai", "offroad", 2), ("pai", "AF-front", 1), ("pai", "overshoot", 1), ("pai", "route-turn", 1), ("pai", "zero-other", 1),
        ("pai", "rear-hit", 2), ("pai", "slow", 2),
        ("arena", "corridor", 2), ("arena", "offroad", 2), ("arena", "AF-front", 1), ("arena", "AF-front+corridor", 1), ("arena", "rear-hit", 1),
        ("arena", "slow", 1), ("arena", "no-launch", 1), ("arena", "creep", 1),
        ("held", "offroad", 1), ("held", "AF-front", 1), ("held", "corridor", 1), ("held", "rear-hit", 1), ("held", "slow", 1)]
REF = {"pai": R8K, "arena": B2, "held": B2}
SERVE_PAI = "JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"
WIZ = "controller.gains.rel_front_steering_angle_weight=10"


# ---------------------------------------------------------------- statistics
def pai_scores():
    """tag -> DataFrame(scene: score old, fclass, ptype, t_ev_s, rear_t_s, new)."""
    import pandas as pd
    PS = pd.read_csv(RUNS / "alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    x = pd.read_csv(RUNS / "round9/ntv1_step4/sv2/new/rollouts.csv")
    x = x[x.read == f"{D3}-ag"].drop_duplicates("scene", keep="last").set_index("scene").score
    new = {D3: x, R8K: PS[f"{R8K}-ag"]}
    out = {}
    for t in (D3, R8K):
        F = pd.read_csv(RUNS / f"pai_fix/facts/{t}-281ag.csv").set_index("scene")
        F["new"] = new[t].reindex(F.index)
        out[t] = F
    return out


def nu_scores(bench):
    """tag -> DataFrame(id: score, ftype, rollout_id, chunk) from the facts cache; arena drops route-fail windows (the held split)."""
    import pandas as pd
    out = {}
    for t in (D3, B2, NTV0):
        key = f"{t}@off" if bench == "arena" else t
        fs = sorted(glob.glob(f"{RUNS}/facts/{'arena' if bench == 'arena' else 'held460'}/{key}/c*/scores.parquet"))
        if not fs:
            continue
        S = pd.concat([pd.read_parquet(f).assign(chunk=Path(f).parent.name) for f in fs]).drop_duplicates("id", keep="last").set_index("id")
        if bench == "arena":
            S = S[S.ftype != "route-fail"]
        out[t] = S
    return out


def table(S, col, cls_col, a, b, zero_cls=None):
    """per class: zeros / partials, count and points lost (1 - score) for reads a and b on their common scenes."""
    common = S[a][col].dropna().index.intersection(S[b][col].dropna().index)
    rows, tot = [], {}
    for t in (a, b):
        x = S[t].loc[common]
        tot[t] = dict(n=len(common), mean=float(x[col].mean()), zeros=int((x[col] == 0).sum()), lost=float((1 - x[col]).sum()),
                      partial_n=int(((x[col] > 0) & (x[col] < 1)).sum()), partial_lost=float((1 - x[col][(x[col] > 0)]).sum()))
    classes = sorted(set(S[a].loc[common, cls_col]) | set(S[b].loc[common, cls_col]))
    for c in classes:
        r = dict(cls=c)
        for t in (a, b):
            x = S[t].loc[common]
            m = x[cls_col] == c
            r[t] = dict(zero_n=int((m & (x[col] == 0)).sum()), zero_lost=float((m & (x[col] == 0)).sum()),
                        part_n=int((m & (x[col] > 0) & (x[col] < 1)).sum()), part_lost=float((1 - x[col][m & (x[col] > 0)]).sum()))
        if any(r[t]["zero_n"] + r[t]["part_n"] for t in (a, b)):
            rows.append(r)
    rows.sort(key=lambda r: -(r[a]["zero_lost"] + r[a]["part_lost"]))
    return dict(a=a, b=b, total=tot, rows=rows)


def cmd_select(a):
    import pandas as pd
    P = pai_scores()
    for t in P:
        P[t]["cls"] = P[t].fclass
        P[t].loc[(P[t].new == 0) & (P[t].fclass == "ok"), "cls"] = "zero-other"
    st = dict(pai=table(P, "new", "cls", D3, R8K), pai_old=table(P, "score", "cls", D3, R8K))
    N = {b: nu_scores(b) for b in ("arena", "held")}
    for b in N:
        if D3 in N[b] and B2 in N[b]:
            st[b] = table(N[b], "score", "ftype", D3, B2)
            if NTV0 in N[b]:
                st[f"{b}_ntv0"] = table(N[b], "score", "ftype", D3, NTV0)
    (O / "stats.json").write_text(json.dumps(st, indent=1))
    sel = []
    for track, c, k in PICK:
        if track not in a.tracks.split(","):
            continue
        if track == "pai":
            A, R = P[D3], P[R8K]
            m = (A.cls == c) & (A.new < 1)
            col, rcol = "new", "new"
        else:
            if D3 not in N[track]:
                continue
            A, R = N[track][D3], N[track][B2]
            m = (A.ftype == c) & (A.score < 1)
            col = rcol = "score"
        cand = A[m].join(R[[rcol]].rename(columns={rcol: "ref"}), how="inner")
        if not len(cand):
            continue
        cand["pass"] = cand.ref >= 0.98
        cand = cand.sort_values(["pass", col, "ref"], ascending=[False, True, False])
        for sid, r in cand.head(k).iterrows():
            sel.append(dict(track=track, cls=c, id=sid, d3=float(r[col]), ref=float(r.ref), ref_tag=REF[track],
                            n_cls=int(m.sum()), n_cls_refpass=int(cand["pass"].sum()),
                            chunk=None if track == "pai" else r.chunk))
    # sides: stored rollout.asl (PAI only) else a rerun
    (O / "lists").mkdir(parents=True, exist_ok=True)
    lists = {}
    meta = json.loads((RUNS / "alpasim_proxy/arena/hold/meta.json").read_text())
    shard = {f"c{x['chunk']}": x["shard"] for x in meta["chunks"]}
    for s in sel:
        s["sides"] = []
        for tag in (D3, s["ref_tag"]):
            src = "rerun"
            if s["track"] == "pai" and glob.glob(f"{PAI_READ[tag]}/runs/*/sim/rollouts/{s['id']}/*/rollout.asl"):
                src = "stored"
            s["sides"].append(dict(tag=tag, src=src))
            if src == "rerun":
                ln = f"pai_{tag}" if s["track"] == "pai" else (f"arena_s{shard[s['chunk']]}_{tag}" if s["track"] == "arena" else f"held_{tag}")
                lists.setdefault(ln, []).append(s["id"])
                s["sides"][-1]["list"] = ln
    for ln, ids in lists.items():
        (O / "lists" / f"{ln}.txt").write_text("".join(f"{i}\n" for i in dict.fromkeys(ids)))
    (O / "sel.json").write_text(json.dumps(dict(sel=sel, lists=sorted(lists)), indent=1))
    for s in sel:
        print(s["track"], s["cls"], s["id"][:40], f"D3 {s['d3']:.2f} ref {s['ref']:.2f}", [x["src"] for x in s["sides"]])
    print(len(sel), "clips;", {ln: len(v) for ln, v in lists.items()})


# ---------------------------------------------------------------- box renders
def ensure_ffmpeg():
    (O / "bin").mkdir(parents=True, exist_ok=True)
    if not (O / "bin/ffmpeg").exists():
        os.symlink(FFMPEG, O / "bin/ffmpeg")


def cmd_asl(a):
    import yaml
    S = json.loads((O / "sel.json").read_text())["sel"]
    ensure_ffmpeg()
    (O / "videos").mkdir(exist_ok=True)
    work = {}
    for s in S:
        for x in s["sides"]:
            if x["src"] != "stored":
                continue
            r = Path(glob.glob(f"{PAI_READ[x['tag']]}/runs/*/sim/rollouts/{s['id']}/*/rollout.asl")[0]).parent
            sim = r.parents[2]
            dst = O / "asl" / x["tag"] / sim.parent.name
            work.setdefault(dst, sim)
            rd = dst / "rollouts" / r.parent.name / r.name
            rd.mkdir(parents=True, exist_ok=True)
            for f in ("rollout.asl", "_complete"):
                if not (rd / f).exists():
                    shutil.copyfile(r / f, rd / f)
    for dst, sim in work.items():
        for y in sim.glob("*.yaml"):
            shutil.copyfile(y, dst / y.name)
        c = yaml.safe_load(open(dst / "eval-config.yaml"))
        c["video"]["render_video"], c["num_processes"] = True, 4
        yaml.safe_dump(c, open(dst / "eval-config.yaml", "w"))
        usdz = yaml.safe_load(open(sim / "generated-user-config-0.yaml"))["scene_provider"]["usdz"]["data_dir"]
        env = dict(os.environ, PATH=f"{O / 'bin'}:{os.environ['PATH']}")
        rc = subprocess.run([str(AP), "-m", "eval.main", "--asl_search_glob", f"{dst}/rollouts/**/rollout.asl", "--config_path", str(dst / "eval-config.yaml"),
                             "--usdz_glob", f"{usdz}/*.usdz", "--log-level", "WARNING"], cwd=DATA / "third_party/alpasim/src/eval", env=env).returncode
        if rc:
            sys.exit(f"eval.main failed ({rc}) on {dst}")
        for mp4 in dst.glob("rollouts/*/*/*camera_front_wide*.mp4"):
            shutil.copyfile(mp4, O / "videos" / f"pai_{mp4.parents[1].name}_{dst.parent.name}.mp4")
    print(len(list((O / "videos").glob("*.mp4"))), "videos")


def summaries(roots):
    """clip / window id -> (sim dir, rollout id, score, failure_reason) over results-summary files under the roots."""
    out = {}
    for root in roots:
        for p in glob.glob(f"{root}/**/aggregate/results-summary.json", recursive=True):
            for r in json.load(open(p))["rollouts"]:
                out[r["clipgt_id"]] = (Path(p).parents[1], r["rollout_id"], float(r.get("score") or 0.0), r.get("failure_reason"))
    return out


def side_facts(sim, sid, rid):
    import numpy as np
    import polars as pl
    rows = list(csv.reader(open(Path(sim) / f"controller/alpasim_controller_{rid}.csv")))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    T, v = A[:, h["timestamp_us"]], np.hypot(A[:, h["vx"]], A[:, h["vy"]])
    t0 = T[0] - 100_000
    on = {}
    mp = Path(sim) / "rollouts" / sid / rid / "metrics.parquet"
    if mp.exists():
        m = pl.read_parquet(mp)
        for k in ("offroad", "collision_at_fault", "collision_rear", "collision_any", "left_corridor_laterally", "wrong_lane"):
            x = m.filter((pl.col("name") == k) & (pl.col("values") > 0))["timestamps_us"].to_list()
            if x:
                on[k] = round((min(x) - t0) / 1e6, 2)
    return dict(on=on, t=((T - t0) / 1e6).round(2).tolist(), v=np.round(v, 2).tolist())


EV = [("offroad", "出路"), ("collision_at_fault", "碰撞"), ("left_corridor_laterally", "离开走廊"), ("collision_rear", "被追尾")]


def t_event(x, st):
    for k, w in EV:
        if k in x["on"]:
            return x["on"][k], k
    if st is not None and st.get("t_ev_s") not in (None, "") and st["t_ev_s"] == st["t_ev_s"]:   # NaN: no event
        return float(st["t_ev_s"]), "event"
    return None, ""


def compose(sides, dst, te, pre=8.0, post=3.0, W1=640):
    """Sides next to each other (full eval frames scaled to width W1), header with read, score, t, speed, event countdown."""
    import cv2
    import numpy as np
    fr = []
    for x in sides:
        cap, f = cv2.VideoCapture(str(x["video"])), []
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            f.append(im)
        fr.append((f, fps))
    fps = fr[0][1]
    n = min(len(f) for f, _ in fr)
    i0, i1 = (0, n) if te is None else (max(0, int(round((te - pre) * fps))), min(n, int(round((te + post) * fps)) + 1))
    h0, w0 = fr[0][0][0].shape[:2]
    H1 = int(round(h0 * W1 / w0))
    HD = 64
    W, H = len(sides) * W1 + 8 * (len(sides) - 1), HD + H1
    H += H % 2
    p = subprocess.Popen([str(O / "bin/ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:g}",
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "slow", "-movflags", "+faststart", str(dst)],
                         stdin=subprocess.PIPE)
    for i in range(i0, i1):
        ti = i / fps
        im = np.full((H, W, 3), 255, np.uint8)
        for j, x in enumerate(sides):
            X0 = j * (W1 + 8)
            im[HD:HD + H1, X0:X0 + W1] = cv2.resize(fr[j][0][i], (W1, H1), interpolation=cv2.INTER_AREA)
            tej, why = x["ev"]
            bad = x["score"] < 0.98
            hit = tej is not None and ti >= tej and bad
            cv2.rectangle(im, (X0, 0), (X0 + W1, HD - 4), (0, 0, 200) if hit else (40, 40, 40), -1)
            cv2.putText(im, f"{SHORT[x['tag']]}  score {x['score']:.2f}" + (f" ({why})" if bad and why else ""), (X0 + 8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            v = float(np.interp(ti, x["t"], x["v"]))
            tail = f"  {why.upper()}" if hit else (f"  {why} in {tej - ti:.1f} s" if tej is not None and bad else "")
            cv2.putText(im, f"t {ti:5.1f} s   v {v:4.1f} m/s{tail}", (X0 + 8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        p.stdin.write(im.tobytes())
    p.stdin.close(), p.wait()
    dur = (i1 - i0) / fps
    return dict(t0=round(i0 / fps, 2), t1=round((i1 - 1) / fps, 2), fps=fps, w=W, h=H, rate=round(max(2.0, min(3.0, 30.0 / fps, dur / 6.0)), 2))


def cmd_collect(a):
    sel = json.loads((O / "sel.json").read_text())["sel"]
    ensure_ffmpeg()
    P = pai_scores()
    (O / "clips").mkdir(exist_ok=True)
    stored = {("pai", t): summaries([PAI_READ[t]]) for t in PAI_READ}
    for tr in ("arena", "held"):
        for t in (D3, B2):
            stored[(tr, t)] = summaries([NU_READ[tr](t)])
    rer = summaries([O / "rerun"])
    rer_by = {}
    for ln in json.loads((O / "sel.json").read_text())["lists"]:
        rer_by[ln] = summaries([O / "rerun" / ln])
    out, bad = [], []
    for k, s in enumerate(sel):
        sides = []
        for x in s["sides"]:
            tag, tr = x["tag"], s["track"]
            sim, rid, sc, fr = stored[(tr, tag)][s["id"]]
            st = P[tag].loc[s["id"]].to_dict() if tr == "pai" else None
            y = dict(tag=tag, src=x["src"], stored_old=sc, failure=fr, score=(float(st["new"]) if st else sc))
            if x["src"] == "rerun":
                got = rer_by.get(x["list"], {}).get(s["id"])
                if got is None:
                    y["video"] = None
                    sides.append(y)
                    continue
                sim, rid, rsc, _ = got
                y["rerun_old"] = rsc
                if abs(rsc - sc) > 1e-3:
                    bad.append(f"{s['id']}|{tag}: rerun {rsc:.3f} stored {sc:.3f}")
                m = sorted(glob.glob(f"{sim}/rollouts/{s['id']}/{rid}/**/*.mp4", recursive=True), key=os.path.getsize)
                y["video"] = m[-1] if m else None
            else:
                v = O / "videos" / f"pai_{s['id']}_{tag}.mp4"
                y["video"] = str(v) if v.exists() else None
            try:
                y.update(side_facts(sim, s["id"], rid))
            except FileNotFoundError as e:                     # a rerun rollout that died (no controller trace): no clip for this scene
                bad.append(f"{s['id']}|{tag}: {x['src']} side has no controller trace ({Path(e.filename).name})")
                y["video"] = None
                sides.append(y)
                continue
            y["ev"] = t_event(y, st)
            y["stored_facts"] = {kk: (None if str(vv) == "nan" else vv) for kk, vv in (st or {}).items() if kk in ("fclass", "t_ev_s", "rear_t_s", "speed_ratio", "stratum", "turn_deg")}
            sides.append(y)
        s = dict(s, sides=[{kk: vv for kk, vv in y.items() if kk not in ("t", "v")} for y in sides])
        if all(y.get("video") and "t" in y for y in sides):
            te = next((y["ev"][0] for y in sides if y["ev"][0] is not None and y["score"] < 0.98), None)
            name = f"{k:02d}_{s['track']}_{s['cls'].replace('+', '_')}.mp4"
            s["clip"] = compose(sides, O / "clips" / name, te)
            s["file"], s["t_ev"] = f"media/{O.name}/{name}", te
            s["mb"] = round((O / "clips" / name).stat().st_size / 1e6, 2)
        out.append(s)
        print(k, s["track"], s["cls"], s["id"][:36], [(SHORT[y["tag"]], y["src"], round(y["score"], 2), y.get("rerun_old")) for y in sides],
              "clip" if "clip" in s else "NO CLIP", flush=True)
    (O / "clips" / "clips.json").write_text(json.dumps(dict(clips=out, rerun_mismatch=bad), indent=1, default=str))
    shutil.copyfile(O / "stats.json", O / "clips" / "stats.json")
    (O / "DONE").write_text(f"clips {sum('clip' in s for s in out)} / {len(out)} mismatch {len(bad)}\n")
    print(sum("clip" in s for s in out), "clips;", len(bad), "rerun mismatches", bad)


# ---------------------------------------------------------------- Mac page
TRACK = {"pai": "PAI 281（ACC + GAP，Harmonizer 关）", "arena": "nuPlan 大考场 held（原生）", "held": "nuPlan held 460 @18（原生）"}
CLS = {"offroad": "出路", "AF-front": "前撞（本车责任）", "AF-side": "侧撞（本车责任）", "side": "侧撞", "overshoot": "冲过停车点", "route-turn": "路线该转未转",
       "zero-other": "其他零分（走廊）", "rear-hit": "被追尾（部分分）", "slow": "偏慢（部分分）", "ok": "近满分微损", "corridor": "离开走廊",
       "AF-front+corridor": "前撞 + 走廊", "offroad+corridor": "出路 + 走廊", "no-launch": "不起步（部分分）", "creep": "蠕行（部分分）",
       "route-fail": "路线失败（全族同样）"}


def stat_table(T, lab_a, lab_b):
    a, b = T["a"], T["b"]
    ta, tb = T["total"][a], T["total"][b]
    rows = "".join(
        f"<tr><td>{html.escape(CLS.get(r['cls'], r['cls']))}</td>"
        f"<td>{r[a]['zero_n']}</td><td>{r[b]['zero_n']}</td><td>{r[a]['part_n']}</td><td>{r[b]['part_n']}</td>"
        f"<td>{r[a]['zero_lost'] + r[a]['part_lost']:.1f}</td><td>{r[b]['zero_lost'] + r[b]['part_lost']:.1f}</td>"
        f"<td>{(r[a]['zero_lost'] + r[a]['part_lost']) - (r[b]['zero_lost'] + r[b]['part_lost']):+.1f}</td></tr>" for r in T["rows"])
    return (f"<table><thead><tr><th>类别</th><th>零分 {lab_a}</th><th>零分 {lab_b}</th><th>部分分 {lab_a}</th><th>部分分 {lab_b}</th>"
            f"<th>丢分 {lab_a}</th><th>丢分 {lab_b}</th><th>差</th></tr></thead><tbody>{rows}"
            f"<tr><th>合计（{ta['n']} 个）</th><th>{ta['zeros']}</th><th>{tb['zeros']}</th><th>{ta['partial_n']}</th><th>{tb['partial_n']}</th>"
            f"<th>{ta['lost']:.1f}</th><th>{tb['lost']:.1f}</th><th>{ta['lost'] - tb['lost']:+.1f}</th></tr></tbody></table>"
            f"<p class='small'>均分 {lab_a} {ta['mean']:.3f}、{lab_b} {tb['mean']:.3f}；丢分 = Σ(1 − 得分)，单位是场景（窗）。</p>")


def card(i, c, cap):
    look = cap.get(c["id"][:24], cap.get(c["id"], ""))
    sides = "；".join(f"{'左' if j == 0 else '右'} {SHORT[y['tag']]} 得分 {y['score']:.2f}"
                     + ("（存档 rollout，离线渲染）" if y["src"] == "stored" else f"（重读，重读旧分 {y.get('rerun_old', float('nan')):.2f} / 存档 {y['stored_old']:.2f}）")
                     for j, y in enumerate(c["sides"]))
    w = c["clip"]
    ev = f"事件时刻 {c['t_ev']:.1f} s。" if c.get("t_ev") is not None else "无单一事件，放整段。"
    return (f'<section id="c{i}"><h3>{i}. {TRACK[c["track"]]} · {html.escape(CLS.get(c["cls"], c["cls"]))} <code>{html.escape(c["id"][:48])}</code></h3>'
            f'<p>{sides}。{ev}该类 D3 共 {c["n_cls"]} 个，其中参照满分 {c["n_cls_refpass"]} 个。</p>'
            + (f'<p class="cap"><b>看：</b>{html.escape(look)}</p>' if look else "")
            + f'<figure><video data-src="{c["file"]}" data-rate="{w["rate"]}" muted loop playsinline controls preload="none" '
            f'style="aspect-ratio:{w["w"]}/{w["h"]}"></video>'
            f'<figcaption><span class="rate">{w["rate"]:g}×</span> 片段 {w["t0"]:.1f}–{w["t1"]:.1f} s</figcaption></figure></section>')


def cmd_page(a):
    D = Path(a.dir)
    J = dict(clips=[], rerun_mismatch=[])
    for ph in ("pai", "nu"):
        f = D / "media" / ph / "clips.json"
        if f.exists():
            x = json.loads(f.read_text())
            J["clips"] += x["clips"]
            J["rerun_mismatch"] += x["rerun_mismatch"]
    st = json.loads((D / "media" / ("nu" if (D / "media/nu/stats.json").exists() else "pai") / "stats.json").read_text())
    cap = json.loads((D / "captions.json").read_text()) if (D / "captions.json").exists() else {}
    body = json.loads((D / "body.json").read_text()) if (D / "body.json").exists() else {}
    clips = [c for c in J["clips"] if "clip" in c]
    cards = "".join(card(i, c, cap) for i, c in enumerate(clips, 1))
    tabs = []
    for k, la, lb, ttl in (("pai", "D3", "R8K-s2", "PAI 281（新评分器）"), ("arena", "D3", "B2", "nuPlan 大考场 held 原生"),
                           ("held", "D3", "B2", "nuPlan held 460 @18"), ("arena_ntv0", "D3", "NTV0", "nuPlan 大考场（对 NTV0）"),
                           ("held_ntv0", "D3", "NTV0", "nuPlan held 460（对 NTV0）")):
        if k in st:
            tabs.append(f"<h3>{ttl}</h3>" + stat_table(st[k], la, lb))
    mm = "全部与存档读数相同（|差| &lt; 0.001）" if not J["rerun_mismatch"] else "；".join(html.escape(x) for x in J["rerun_mismatch"]) + " 与存档不同"
    mb = sum(c["mb"] for c in clips)
    (D / "index.html").write_text(PAGE.format(head=body.get("head", ""), concl=body.get("concl", ""), tabs="".join(tabs), cards=cards,
                                              n=len(clips), mb=mb, mm=mm, script=SCRIPT))
    man = [dict(i=i, track=c["track"], cls=c["cls"], id=c["id"], file=c["file"], mb=c["mb"], rate=c["clip"]["rate"], t_ev=c.get("t_ev"),
                sides=[dict(tag=y["tag"], src=y["src"], score=y["score"], stored_old=y["stored_old"], rerun_old=y.get("rerun_old")) for y in c["sides"]])
           for i, c in enumerate(clips, 1)]
    (D / "manifest.json").write_text(json.dumps(dict(clips=man, rerun_mismatch=J["rerun_mismatch"], videos="media/*.mp4 (gitignored; box: runs/round9/d3_fail/clips)"),
                                                indent=1, ensure_ascii=False))
    print(len(clips), "clips", round(mb, 1), "MB")


SCRIPT = """<script>
// Speed bar: 0 = per-clip auto rate (data-rate); any other value overrides every clip. localStorage throws in the reader's sandbox.
let G = 0; try { G = +localStorage.vidRate || 0; } catch (e) {}
const R = v => { const r = G || +v.dataset.rate; v.defaultPlaybackRate = r; v.playbackRate = r; };
const setG = r => { G = r; try { localStorage.vidRate = r; } catch (e) {}
  document.querySelectorAll("#rb button").forEach(b => b.classList.toggle("on", +b.dataset.r === r));
  document.querySelector("#rb input").value = r || 2; document.getElementById("rv").textContent = r ? r + "×" : "自动";
  document.querySelectorAll("video[data-src]").forEach(v => { if (v.src) R(v); const s = v.parentNode.querySelector(".rate"); if (s) s.textContent = (G || +v.dataset.rate) + "×"; }); };
document.querySelectorAll("#rb button").forEach(b => b.onclick = () => setG(+b.dataset.r));
document.querySelector("#rb input").oninput = e => setG(+e.target.value);
setG(G);
// Lazy: fetch a clip within about one screen, play it only while >= 50 % visible, pause when it leaves.
const near = new IntersectionObserver(es => es.forEach(e => { const v = e.target;
  if (e.isIntersecting && !v.src) { v.src = v.dataset.src; v.preload = "auto"; v.addEventListener("loadedmetadata", () => R(v)); near.unobserve(v); } }),
  { rootMargin: "100% 0px" });
const seen = new IntersectionObserver(es => es.forEach(e => { const v = e.target;
  if (e.isIntersecting) { if (!v.src) { v.src = v.dataset.src; v.addEventListener("loadedmetadata", () => R(v)); } R(v); v.play().catch(() => {}); } else v.pause(); }),
  { threshold: 0.5 });
document.querySelectorAll("video[data-src]").forEach(v => { near.observe(v); seen.observe(v); });
</script>"""

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D3 在 PAI 与 nuPlan 上丢分在哪</title><style>
body{{font:15px/1.6 -apple-system,"PingFang SC",sans-serif;max-width:1400px;margin:0 auto;padding:0 .8em 3em;color:#222}}
#rb{{position:sticky;top:0;z-index:9;background:#fffffff2;padding:6px 0;border-bottom:1px solid #ddd;font-size:14px}}
#rb button{{margin:0 2px;padding:1px 8px;border:1px solid #999;border-radius:4px;background:#fff}}#rb button.on{{background:#333;color:#fff}}
#rb input{{vertical-align:middle;width:150px}}section{{border-top:1px solid #ddd;margin-top:1.2em;padding-top:.2em}}h3{{margin:.4em 0}}
figure{{margin:0}}video{{width:100%;max-height:86vh;background:#000;display:block;margin:0 auto}}figcaption{{font-size:13px;color:#555}}.cap{{background:#f4f4fb;padding:3px 8px}}
.rate{{background:#333;color:#fff;border-radius:4px;padding:0 5px;font-size:12px}}code{{font-size:13px}}.small{{font-size:13px;color:#555}}
table{{border-collapse:collapse;font-size:13px;margin:.4em 0}}td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#f3f3f3}}
</style></head><body>
<div id="rb">播放速度 <button data-r="0">自动</button><button data-r="1">1×</button><button data-r="2">2×</button><button data-r="3">3×</button><button data-r="5">5×</button><button data-r="10">10×</button><button data-r="16">16×</button> <input type="range" min="0.5" max="16" step="0.5"> <b id="rv"></b></div>
<h1>D3（NT4D3-F-s0）在 PAI 与 nuPlan 上丢分在哪</h1>
{head}
{concl}
<h2>失分统计</h2>
{tabs}
<h2>视频（每类一段，大类两段）</h2>
<p>画面是 AlpaSim 自带的评测视频（俯视图 + 前视相机），左 D3、右参照（PAI 为 R8K-s2，nuPlan 为 B2）。有事件的剪到事件前 8 s 到事件后 3 s，顶条显示读数、得分、时刻、车速、事件倒计时，事件后变红；没有单一事件（偏慢、不起步）的放整段。
PAI 零分的 rollout 用存档读数留下的 rollout.asl 在 CPU 上离线渲染（就是计分的那一次）；其余（部分分、参照满分、nuPlan 全部）用同一 checkpoint、同一 serve 在 GPU 池低优先级重读一次（PAI CONC 2），重读得分{mm}。
PAI 的得分是新评分器（榜用），重读核对用旧评分器。共 {n} 段、{mb:.0f} MB。顶部速度条改全部视频（自动 = 每段 max(2, min(3, 30/fps, 时长/6))，选择记在浏览器里），滑到附近才加载，过半可见才播放。</p>
{cards}
{script}</body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("select").add_argument("--tracks", default="pai,arena,held")
    for c in ("asl", "collect"):
        sp.add_parser(c)
    p = sp.add_parser("page")
    p.add_argument("dir")
    a = ap.parse_args()
    if a.cmd != "page":
        O.mkdir(parents=True, exist_ok=True)
    dict(select=cmd_select, asl=cmd_asl, collect=cmd_collect, page=cmd_page)[a.cmd](a)
