#!/usr/bin/env python3
"""OFFROAD-VIDEOS: a video page of the offroad zeros the PAI-human-row recipe (R3MV-L18-s0 / s1, R5LC-L18-s0 / s1) adds on PAI 281 under
ACC + GAP, each next to NTV0-L18-s0 on the same scene (decisions 427-429: +8..+12 offroad zeros vs NTV0).

  select   (box, .venv)      stored facts ($DATA_DIR/runs/pai_fix/facts/<tag>-281ag.csv) -> $O/sel.json + $O/lists/<tag>.txt
           candidates = scenes offroad in >= 1 of the four MV-family reads where NTV0 is not offroad; every candidate is re-read, one MV
           read per scene (the first offroad read in MV order) plus NTV0
  launch   (box, bash -lc)   one pai_box.sh submit per checkpoint on its list (served config of the stored reads: SERVE / WIZ / CONC 4 /
           ALPASIM_FAST=1, plus KEEP_VIDEO=1 = AlpaSim's eval video, KEEP_ASL=1), pool PRIO 18, then a CPU collect job --after all chunks
  wait     (box)             blocks until $O/DONE or a failed / cancelled job of this lane (one run_in_background waiter)
  collect  (box, AlpaSim venv: polars)  re-read rollouts -> $O/videos/<scene8>_<tag>.mp4 + $O/facts.json (re-read vs stored score,
           offroad onset, 10 Hz ego speed on the video clock); writes $O/DONE
  page     (Mac, .venv: cv2) DIR = research/round5/offroad_videos: $O/{facts.json, sel.json, videos/} copied to DIR/media/src ->
           DIR/media/<scene8>.mp4 (MV read | NTV0 side by side, cut to the event, burnt-in header) + DIR/index.html + DIR/manifest.json

Video: AlpaSim's DEFAULT eval layout (ego-centred bird's-eye view with the served plan in orange, the metric table, and the driver's own
front camera camera_front_wide_120fov with the plan drawn on it), 10 fps; frame i shows the rollout clock t0 + i / 10 s, t0 = first
controller step - 0.1 s (rf_tax.Roll.t0, the clock of the facts' t_ev_s).
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
import time
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
REPO = Path(__file__).resolve().parents[3]
O = DATA / "runs/round5/offroad_videos"
FACTS = DATA / "runs/pai_fix/facts"
NT = "NTV0-L18-s0"
MV = ["R3MV-L18-s0", "R5LC-L18-s0", "R3MV-L18-s1", "R5LC-L18-s1"]
SHORT = {NT: "NTV0", **{t: t.replace("-L18", "") for t in MV}}
SERVE = "JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"
WIZ = "controller.gains.rel_front_steering_angle_weight=10"
PRIO = 18
GEO = {"off_curve_inside": "弯道内侧切出", "off_curve_outside": "弯道外侧冲出", "off_junc_inside": "路口转弯内侧切角",
       "off_junc_outside": "路口转弯外侧过宽", "off_junc_straight": "路口内直行段出路", "off_straight": "直路车道边出路"}
CTX = {"launch": "起步后", "stop": "停车 / 跟车场景", "cruise": "行驶中"}
FCL = {"ok": "满分", "rear-hit": "被追尾", "slow": "偏慢", "overshoot": "冲过人类停车点", "route-turn": "路线转弯失败", "offroad": "出路",
       "zero-other": "其他零分（漂移 / 变道）", "AF-front": "正面碰撞", "side": "侧碰"}


def facts():
    return {t: {r["scene"]: r for r in csv.DictReader(open(FACTS / f"{t}-281ag.csv"))} for t in [NT] + MV}


def ctx(r):
    return "launch" if r["stratum"] == "launch" or float(r["v_hand"] or 0) < 1.0 else "stop" if r["stratum"].startswith("stop") else "cruise"


# ---------------------------------------------------------------- box
def cmd_select(a):
    F = facts()
    ev, shared = [], []
    for s in sorted(set().union(*({k for k, r in F[t].items() if r["fclass"] == "offroad"} for t in MV))):
        off = [t for t in MV if F[t].get(s, {}).get("fclass") == "offroad"]
        n = F[NT].get(s)
        row = dict(scene=s, s8=s[7:15], off=off, scores={t: float(F[t][s]["score"]) if s in F[t] else None for t in [NT] + MV},
                   fclass={t: F[t][s]["fclass"] if s in F[t] else None for t in [NT] + MV})
        if n is not None and n["fclass"] == "offroad":
            shared.append(row)
            continue
        show = off[0]
        x = F[show][s]
        row.update(show=show, geo=x["primary"], ctx=ctx(x), stratum=x["stratum"], t_ev=float(x["t_ev_s"]), v_hand=float(x["v_hand"]),
                   turn_deg=float(x["turn_deg"]), prims={t: F[t][s]["primary"] for t in off})
        ev.append(row)
    ev.sort(key=lambda r: (-len(r["off"]), r["s8"]))
    O.mkdir(parents=True, exist_ok=True), (O / "lists").mkdir(exist_ok=True)
    lists = {t: sorted(r["scene"] for r in ev if r["show"] == t) for t in MV}
    lists[NT] = sorted(r["scene"] for r in ev)
    for t, ids in lists.items():
        if ids:
            (O / "lists" / f"{t}.txt").write_text("\n".join(ids) + "\n")
    (O / "sel.json").write_text(json.dumps(dict(events=ev, shared=shared), indent=1))
    for r in ev:
        print(len(r["off"]), r["s8"], SHORT[r["show"]], r["geo"], r["ctx"], f"t_ev {r['t_ev']:.1f}", "NTV0", r["fclass"][NT], r["scores"][NT])
    print(len(ev), "events,", len(shared), "shared with NTV0;", {SHORT[t]: len(v) for t, v in lists.items()})


def sub(name, *args):
    cmd = [str(REPO / ".venv/bin/python"), "-m", "jevdrive.cl", "submit", "--owner", "round5-offroadvid", "--name", name, "--tries", "1",
           "--priority", str(PRIO), "--no-check", "--log-dir", str(O / "pool" / name), *args]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    jid = out.stdout.split()[-1] if out.returncode == 0 and out.stdout.split() else None
    if not jid:
        sys.exit(f"submit {name} failed: {out.stdout} {out.stderr}")
    return jid


def cmd_launch(a):
    if (O / "jobs.txt").exists() and (O / "jobs.txt").read_text().strip():
        sys.exit(f"already submitted ({O / 'jobs.txt'})")
    ids = []
    env = dict(os.environ, SERVE=SERVE, WIZ=WIZ, CONC="4", ALPASIM_FAST="1", KEEP_VIDEO="1", KEEP_ASL="1", PRIO=str(PRIO))
    for f in sorted((O / "lists").glob("*.txt")):
        t = f.stem
        out = subprocess.run(["bash", str(REPO / "experiments/alpasim/scripts/pai_box.sh"), "submit", str(O / "reads" / t), t, str(f)],
                             cwd=REPO, env=env, capture_output=True, text=True)
        got = [ln.split() for ln in out.stdout.splitlines() if ln.split()[1:2] and ln.split()[1].startswith("pai-")]
        if out.returncode or not got:
            sys.exit(f"pai_box submit {t} failed: {out.stdout} {out.stderr}")
        ids += [g[0] for g in got]
        with open(O / "jobs.txt", "a") as fh:
            fh.writelines(f"{g[0]} {g[1]}\n" for g in got)
    py = DATA / "third_party/alpasim/.venv/bin/python"
    c = sub("offroadvid-collect", "--after", ",".join(ids), "--vram", "0.1", "--cpu", "4", "--ram", "8", "--timeout-h", "1", "--",
            str(py), str(Path(__file__).resolve()), "collect")
    with open(O / "jobs.txt", "a") as fh:
        fh.write(f"{c} offroadvid-collect\n")
    code = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    with open(O / "stages.txt", "a") as fh:
        fh.write(f"{time.strftime('%F %T')} launch (code {code}): chunks {' '.join(ids)} collect {c}\n")
    print("chunks", *ids, "collect", c)


def cmd_wait(a):
    sys.path.insert(0, str(REPO))
    from jevdrive.cl import pool as P
    while not (O / "DONE").exists():
        last = dict(reversed(ln.split()[:2]) for ln in open(O / "jobs.txt") if ln.split())
        jobs = P.snapshot()[0]["jobs"]
        bad = [f"{n} {i} {jobs[i]['state']}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
        if bad:
            sys.exit("failed:\n  " + "\n  ".join(bad))
        time.sleep(300)
    print("DONE", (O / "DONE").read_text().strip())


def ctrl(sim, rid):
    import numpy as np
    f = Path(sim) / f"controller/alpasim_controller_{rid}.csv"
    rows = list(csv.reader(open(f)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    return A[:, h["timestamp_us"]], np.hypot(A[:, h["vx"]], A[:, h["vy"]])


def cmd_collect(a):
    import numpy as np
    import polars as pl
    S = json.loads((O / "sel.json").read_text())
    F = facts()
    V = O / "videos"
    V.mkdir(exist_ok=True)
    out = {}
    for summ in sorted(glob.glob(f"{O}/reads/*/runs/*/sim/aggregate/results-summary.json")):
        sim = Path(summ).parents[1]
        t = Path(summ).parents[4].name
        for r in json.load(open(summ))["rollouts"]:
            s, rid = r["clipgt_id"], r["rollout_id"]
            d = sim / "rollouts" / s / rid
            sm = r.get("score_metrics") or {}
            fail = next((f for f in ("collision_at_fault", "offroad", "left_corridor_laterally") if sm.get(f)), "")
            T, v = ctrl(sim, rid)
            t0 = T[0] - 100_000
            m = pl.read_parquet(d / "metrics.parquet").filter(pl.col("name") == "offroad").select(pl.col("timestamps_us").cast(pl.Int64), "values")
            on = m.filter(pl.col("values") > 0)["timestamps_us"].to_list()
            mp4 = sorted(d.glob("*camera_front_wide*.mp4"))
            dst = V / f"{s[7:15]}_{t}.mp4"
            if mp4:
                shutil.copyfile(mp4[0], dst)
            st = F[t].get(s, {})
            out[f"{s}|{t}"] = dict(scene=s, tag=t, rid=rid, score=float(r.get("score") or 0.0), fail=fail,
                                   stored=float(st["score"]) if st else None, stored_fclass=st.get("fclass"),
                                   t_off=(min(on) - t0) / 1e6 if on else None, video=dst.name if mp4 else None,
                                   dist=(r.get("metrics") or {}).get("dist_traveled_m"), gt=(r.get("metrics") or {}).get("gt_dist_traveled_m"),
                                   t=((T - t0) / 1e6).round(2).tolist(), v=np.round(v, 2).tolist(), asl=(d / "rollout.asl").exists())
    bad = [k for k, x in out.items() if x["stored"] is None or abs(x["score"] - x["stored"]) > 1e-3]
    (O / "facts.json").write_text(json.dumps(dict(rollouts=out, mismatch=bad), indent=0))
    need = {f"{e['scene']}|{t}" for e in S["events"] for t in (e["show"], NT)}
    print(len(out), "rollouts,", len(need - set(out)), "missing,", len(bad), "score mismatches:", bad)
    (O / "DONE").write_text(time.strftime("%F %T") + f" rollouts {len(out)} missing {len(need - set(out))} mismatch {len(bad)}\n")


# ---------------------------------------------------------------- Mac
W1, HD, BV, CH = 640, 64, 576, 360               # one side: width, header, bird's-eye panel (square, centred), camera height
BEV, CAM = (slice(30, 504), slice(62, 536)), (slice(517, 992), slice(29, 872))   # panels of the 900 x 1000 DEFAULT eval frame


def speed(f):
    """Fast-forward factor (jtn_diag.speed): ~30 fps, never above 3x, at least ~6 s on screen, never below 2x."""
    try:
        r, d = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate:format=duration",
                              "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout.split()
        n, m = r.split("/")
        fps, dur = float(n) / float(m), float(d)
    except (ValueError, OSError):
        return 2.0
    return round(max(2.0, min(3.0, 30.0 / fps, dur / 6.0)), 2)


def frames(f):
    import cv2
    cap, fr = cv2.VideoCapture(str(f)), []
    while True:
        ok, im = cap.read()
        if not ok:
            return fr, cap.get(cv2.CAP_PROP_FPS) or 10.0
        fr.append(im)


def clip(e, R, src, dst, pre=7.0, post=3.0):
    """MV read | NTV0 side by side, [t_ev - pre, t_ev + post] on the rollout clock, header: read, score, t, ego speed, OFFROAD from onset."""
    import cv2
    import numpy as np
    sides = [(e["show"], R[f"{e['scene']}|{e['show']}"]), (NT, R[f"{e['scene']}|{NT}"])]
    fr = [frames(src / x["video"]) for _, x in sides]
    if not all(f for f, _ in fr):
        sys.exit(f"no frames in {[x['video'] for _, x in sides]}")
    fps = fr[0][1]
    n = min(len(f) for f, _ in fr)
    t_ev = sides[0][1]["t_off"] if sides[0][1]["t_off"] is not None else e["t_ev"]
    i0, i1 = max(0, int(round((t_ev - pre) * fps))), min(n, int(round((t_ev + post) * fps)) + 1)
    W, H = 2 * W1 + 8, HD + BV + CH
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "slow", "-movflags", "+faststart", str(dst)],
                         stdin=subprocess.PIPE)
    for i in range(i0, i1):
        ti = i / fps
        im = np.full((H, W, 3), 255, np.uint8)
        for j, (t, x) in enumerate(sides):
            X0 = j * (W1 + 8)
            f = fr[j][0][i]
            im[HD:HD + BV, X0 + (W1 - BV) // 2:X0 + (W1 + BV) // 2] = cv2.resize(f[BEV], (BV, BV), interpolation=cv2.INTER_AREA)
            im[HD + BV:, X0:X0 + W1] = cv2.resize(f[CAM], (W1, CH), interpolation=cv2.INTER_AREA)
            off = x["t_off"] is not None and ti >= x["t_off"]
            col = (0, 0, 200) if off else (40, 40, 40)
            cv2.rectangle(im, (X0, 0), (X0 + W1, HD - 4), col, -1)
            why = "offroad" if x["fail"] == "offroad" else x["fail"].replace("left_corridor_laterally", "corridor").replace("collision_at_fault", "collision") or "no zero"
            cv2.putText(im, f"{SHORT[t]}  score {x['score']:.2f} ({why})", (X0 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            v = float(np.interp(ti, x["t"], x["v"]))
            tail = "  OFFROAD" if off else (f"  offroad in {x['t_off'] - ti:.1f} s" if x["t_off"] is not None else "")
            cv2.putText(im, f"t {ti:5.1f} s   v {v:4.1f} m/s{tail}", (X0 + 8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        p.stdin.write(im.tobytes())
    p.stdin.close(), p.wait()
    return dict(t0=i0 / fps, t1=(i1 - 1) / fps, fps=fps)


# One card per (geometry, context) class: the steadiest event (most MV reads offroad, then NTV0's score; an event offroad in one read only
# where NTV0 also scores 0 is not a representative), REP overrides the pick, SECOND = visibly different sub-cases with their own card.
# CAP = what to look at, written after watching the clips (speeds from facts.json at the offroad onset).
REP = {("off_straight", "cruise"): "a2b415d5"}
SECOND = {"13d646af": "计划收短、低速拐出", "7750265d": "车道平移没跟上", "bbc53aa8": "高速路上计划收短、急减速", "6e7b42a3": "高速路上加速"}
CAP = {
    "94c1d722": "夜间 T 型路口右转：两车入弯速度相近（出路时 5.7 / 4.8 m/s），MV 出弯时切向右侧（内侧）压出路沿，NTV0 沿车道转完。",
    "ea468790": "停止线起步右转进多车道路：MV 转完不回正，一直贴右侧，压上右侧路沿（出路时 6.2 m/s；NTV0 5.2 m/s，已回到车道中）。",
    "dc401449": "夜间窄街左弯：MV 弯前计划收短（橙线变短）、减速到 1.4 m/s（NTV0 10.9 m/s），低速下计划指向右侧，从弯道外侧开出。",
    "b680a689": "城区路口左转：两车轨迹和速度几乎相同（4.8 / 4.6 m/s），MV 出弯落点偏左约半个车宽，压到窄街左侧边线；差别很小，但两个 R5LC 种子都出路。",
    "abd45a30": "路口左转：NTV0 直冲过路口（路线转弯零分）；MV 转了，但减速到 4.6 m/s 后转得大，压到外侧路沿。路线转弯的修复在这里换成了出路。",
    "de9db5dc": "雪天起步右转（只有 R3MV-s1 出路）：MV 转后贴右、几乎停下，蹭上右侧交通岛和标志牌；NTV0 同一时刻 5.9 m/s，在车道内。",
    "e1c6ae9f": "夜雾里直行穿过路口之后：两车同速（13.6 / 13.8 m/s），MV 几秒内慢慢向右漂，压到右侧停车带边线；NTV0 居中。",
    "13d646af": "直行接近路口：MV 计划收短，从 16.6 m/s 降到 2.8 m/s（NTV0 保持 15.7 m/s），低速在路口里向右拐出路面。",
    "596af0c7": "傍晚快速路右弯段：两车同速（17.9 / 18.7 m/s），MV 逐渐右漂到岩壁一侧的路肩，NTV0 保持车道。",
    "7750265d": "路口前车道向左平移（路缘外凸）：MV 直着走、没跟上左移，压上右侧路缘；NTV0 跟上了左移（它的零分是冲过人类停车点，与出路无关）。",
    "a2b415d5": "乡间双车道直路：两车同速 16.6 m/s，MV 一直贴着右侧边线走，几秒后压出；NTV0 居中。最纯的侧向漂移。",
    "bbc53aa8": "高速路：MV 计划收短，从约 29 m/s 一路减速（出路时 7.1 m/s，NTV0 30.8 m/s），减速中右漂上路肩。",
    "6e7b42a3": "高速路：MV 加速到 36.5 m/s（NTV0 维持交接速度 27.4 m/s）；出路时刻俯视图上本车仍在车道线内，看不出明显越界，是否是可行驶区地图的问题要逐帧确认。",
}


def cmd_page(a):
    import numpy as np
    D = Path(a.dir)
    M, src = D / "media", D / "media/src"
    S = json.loads((src / "sel.json").read_text())
    FX = json.loads((src / "facts.json").read_text())
    R = FX["rollouts"]
    ev = S["events"]
    by = {}
    for e in ev:
        by.setdefault((e["geo"], e["ctx"]), []).append(e)
    key = lambda e: (-len(e["off"]), -(e["scores"][NT] or 0), e["s8"])  # noqa: E731
    ok = lambda e: len(e["off"]) > 1 or (e["scores"][NT] or 0) > 0  # noqa: E731
    pick = {k: next((e for e in g if e["s8"] == REP.get(k)), None) or min(filter(ok, g), key=key, default=None) for k, g in by.items()}
    show = [e for e in pick.values() if e] + [e for e in ev if e["s8"] in SECOND]
    order = list(GEO)
    show = sorted({e["s8"]: e for e in show}.values(), key=lambda e: (order.index(e["geo"]), list(CTX).index(e["ctx"]), key(e)))
    man, cards = [], []
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    for k, e in enumerate(show, 1):
        mv, nt = R.get(f"{e['scene']}|{e['show']}"), R.get(f"{e['scene']}|{NT}")
        if not (mv and nt and mv["video"] and nt["video"]):
            print("no video", e["s8"]), sys.exit(1)
        dst = M / f"{e['s8']}.mp4"
        if a.force or not dst.exists():
            w = clip(e, R, src / "videos", dst)
        else:
            w = json.loads((M / "clips.json").read_text())[e["s8"]] if (M / "clips.json").exists() else {}
        rate = speed(dst)
        t_ev = mv["t_off"] if mv["t_off"] is not None else e["t_ev"]
        v_mv, v_nt = float(np.interp(t_ev, mv["t"], mv["v"])), float(np.interp(t_ev, nt["t"], nt["v"]))
        reads = "、".join(f"{SHORT[t]}（{GEO.get(e['prims'][t], e['prims'][t])}）" for t in e["off"])
        lab = f"{GEO[e['geo']]} · {CTX[e['ctx']]}" + (f" · {SECOND[e['s8']]}" if e["s8"] in SECOND else "")
        oth = "；".join(f"{SHORT[t]} {f2(e['scores'][t])}（{FCL.get(e['fclass'][t] or '', e['fclass'][t] or '-')}）" for t in MV)
        cards.append(
            f'<section id="{e["s8"]}"><h3>{k}. {lab} <code>{e["s8"]}</code></h3>'
            f'<p>出路读数：{len(e["off"])} / 4（{reads}）。左：{SHORT[e["show"]]} {f2(mv["score"])}；右：NTV0 {f2(nt["score"])}'
            f'（{FCL.get(e["fclass"][NT], e["fclass"][NT])}）。出路时刻 {t_ev:.1f} s，此刻车速 {SHORT[e["show"]]} {v_mv:.1f} m/s、NTV0 {v_nt:.1f} m/s'
            f'（交接速度 {e["v_hand"]:.1f} m/s，路线转角 {e["turn_deg"]:.0f}°）。</p>'
            f'<p class="cap">看：{html.escape(CAP.get(e["s8"], ""))}</p>'
            f'<figure><video data-src="media/{dst.name}" data-rate="{rate}" muted loop playsinline controls preload="none"></video>'
            f'<figcaption><span class="rate">{rate:g}×</span> 片段 {w.get("t0", 0):.1f}–{w.get("t1", 0):.1f} s（rollout 时钟）；四个 MV 读数：{oth}</figcaption></figure></section>')
        man.append(dict(scene=e["scene"], s8=e["s8"], cls=lab, geo=e["geo"], ctx=e["ctx"], show=e["show"], off=e["off"], file=f"media/{dst.name}",
                        mb=round(dst.stat().st_size / 1e6, 2), rate=rate, t_ev=round(t_ev, 2), v_ev={e["show"]: round(v_mv, 2), NT: round(v_nt, 2)},
                        rerun={e["show"]: mv["score"], NT: nt["score"]}, stored={e["show"]: mv["stored"], NT: nt["stored"]}, clip=w))
        (M / "clips.json").write_text(json.dumps({**(json.loads((M / "clips.json").read_text()) if (M / "clips.json").exists() else {}), e["s8"]: w}))
        print(f"{k:2d} {e['s8']} {lab} {SHORT[e['show']]} {mv['score']:.2f} / NTV0 {nt['score']:.2f} v {v_mv:.1f} / {v_nt:.1f} rate {rate}", flush=True)
    shown = {e["s8"] for e in show}
    rest = [e for e in ev if e["s8"] not in shown]
    li = "".join(f"<li><code>{e['s8']}</code> {GEO[e['geo']]} · {CTX[e['ctx']]}；出路 {len(e['off'])} / 4（{'、'.join(SHORT[t] for t in e['off'])}）；"
                 f"NTV0 {f2(e['scores'][NT])}（{FCL.get(e['fclass'][NT], e['fclass'][NT])}）；"
                 + " / ".join(f"{SHORT[t]} {f2(e['scores'][t])}" for t in MV) + "</li>" for e in rest)
    li2 = "".join(f"<li><code>{e['s8']}</code> NTV0 {f2(e['scores'][NT])}；MV 出路 {len(e['off'])} / 4；" + " / ".join(f"{SHORT[t]} {f2(e['scores'][t])}" for t in MV)
                  + "</li>" for e in S["shared"])
    mism = [k for k in FX["mismatch"]]
    tot = sum(m["mb"] for m in man)
    (D / "manifest.json").write_text(json.dumps(dict(clips=man, rest=[e["scene"] for e in rest], shared=[e["scene"] for e in S["shared"]],
                                                     rerun_mismatch=mism), indent=1, ensure_ascii=False))
    intro = make_intro(S, FX, ev, show)
    page = PAGE.format(intro=intro, cards="".join(cards), n_ev=len(ev), n_show=len(show), n_rest=len(rest), li=li, n_sh=len(S["shared"]), li2=li2,
                       n_clip=len(man), mb=tot)
    (D / "index.html").write_text(page)
    print(len(ev), "events,", len(man), "clips,", round(tot, 1), "MB; rerun mismatches", mism)


# Offroad zero counts on the 275 common scenes, from experiments/round5/results/pai281_R5LC_ag.md (paired bootstrap vs NTV0, 95 % CI).
OFF281 = {NT: (12, None), "R3MV-L18-s0": (20, "+8 [+1, +16]"), "R3MV-L18-s1": (21, "+9 [+2, +17]"), "R5LC-L18-s0": (21, "+9 [+2, +17]"),
          "R5LC-L18-s1": (24, "+12 [+4, +20]")}
FINDINGS = """<p><b>先看什么。</b></p><ol>
<li><b>多数是侧向，不是速度。</b>21 个事件里 15 个出路时两车速度差在 3 m/s 以内：同样的速度下，MV 的计划慢慢偏向一侧边线（直路 a2b415d5、快速路 596af0c7、
穿过路口之后 e1c6ae9f），或者转弯出弯时贴内侧、不回正（94c1d722、ea468790、b680a689）；NTV0 在同一场景居中。与第 427 条「短板是侧向」一致。
直路 / 车道边是最大的一类（10 / 21）。</li>
<li><b>第二种：计划收短、减速，然后在低速下转出路面。</b>5 个事件 MV 出路时的速度不到 NTV0 的一半（dc401449 1.4 对 10.9 m/s、13d646af 2.8 对 15.7、
bbc53aa8 7.1 对 30.8、0f4b8753 4.3 对 10.3、c92137b4 3.0 对 12.8）：橙线先变短，车随之减速，NTV0 在同一场景保持原速；减速后的计划指向路边。
出路的几何分类看不出这一类，它们分散在直路 / 弯道 / 路口直行下面。</li>
<li><b>路线转弯的修复有一部分换成了出路。</b>5 个事件 NTV0 自己也是零分（路线转弯 3 个：abd45a30、68e53160、1e8e939e；冲过人类停车点 2 个：7750265d、4e4a2a51）：
MV 转了弯或跟上了路形，却出路。这 5 个不是 MV 弄坏的满分场景。abd45a30 的 NTV0 顶条也有出路倒计时：它先被走廊判零，15.0 s 才压到路外。</li>
<li><b>稳定。</b>7 个场景四个 MV 读数全出路，是配方本身的行为，不是种子噪声。只在一个读数出路的 7 个多半是噪声，只给 de9db5dc 一张卡（起步后路口转弯这一类只有它）。</li>
<li>6e7b42a3 是唯一一个 MV 明显更快的（36.5 对 27.4 m/s），出路处俯视图上看不出越界，单独标出。</li>
</ol>"""


def make_intro(S, FX, ev, show):
    from collections import Counter
    nk = Counter(len(e["off"]) for e in ev)
    nf = Counter(e["fclass"][NT] for e in ev)
    R = FX["rollouts"]
    need = [f"{e['scene']}|{t}" for e in ev for t in (e["show"], NT)]
    got = [k for k in need if k in R]
    mm = [k for k in got if k in FX["mismatch"]]
    off = "，".join(f"{SHORT[t]} {n}" + (f"（{d}）" if d else "") for t, (n, d) in OFF281.items())
    mmt = ("全部与存档读数相同（|差| &lt; 0.001）" if not mm else
           f"{len(mm)} 个与存档不同：" + "；".join(f"<code>{k.split('|')[0][7:15]}</code> {SHORT[k.split('|')[1]]} 存档 {R[k]['stored']:.2f} → 重跑 {R[k]['score']:.2f}" for k in mm))
    return (f"<p>PAI 281（ACC + GAP 提交 serve）上，含 PAI 人类行的 MV 系配方把出路零分从 NTV0 的 12 个抬到 20–24 个（275 个共同场景；{off}；"
            f"<code>experiments/round5/results/pai281_R5LC_ag.md</code>，决策 427）。第 429 条的本地 PCS 尺子说明这些新零分很贵：R3MV 的 23 个新增侧向零分里 10 个落在 NTV0 一族都能过的高区分度场景上。"
            f"这一页把这些出路逐个放出来，和 NTV0 在同一场景上的 rollout 并排看。</p>"
            f"<p><b>怎么选的。</b>四个 MV 读数（R3MV-L18-s0 / s1、R5LC-L18-s0 / s1）中至少一个出路、NTV0 不出路的场景共 {len(ev)} 个："
            f"四个读数都出路 {nk[4]} 个、三个 {nk[3]} 个、两个 {nk[2]} 个、只有一个 {nk[1]} 个。"
            f"这些场景上 NTV0 满分 {nf['ok']} 个，其余是 NTV0 自己别的失分（" + "、".join(f"{FCL.get(k, k)} {v}" for k, v in nf.items() if k != "ok") + "）。"
            f"几何类别取 <code>rf_tax.py</code> 在出路时刻的判定：路线前后各 20 m 的转角 ≥ 20° 算弯，路口标记在出路时刻附近算路口，偏向转弯一侧算内侧；"
            f"情境：交接速度 &lt; 1 m/s 记「起步后」，stop / lead 层（人类在场景里停车或跟车）记「停车 / 跟车场景」。</p>"
            f"<p><b>视频从哪来。</b>存档读数只给零分 rollout 留 rollout.asl、不留视频，所以每个场景用同一 checkpoint、同一 serve（ACC + GAP、g10、ALPASIM_FAST=1）重读一次，打开 AlpaSim 自带评测视频并保留 asl"
            f"（每个事件一个 MV 读数 + NTV0，共 {len(got)} / {len(need)} 个 rollout，经 GPU 池 PRIO 18）。重读得分{mmt}，可以当原读数的录像看。</p>" + FINDINGS)


PAGE = '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MV 系配方新增的出路零分（PAI 281，ACC + GAP）</title><style>
body{{font:15px/1.6 -apple-system,"PingFang SC",sans-serif;max-width:1340px;margin:0 auto;padding:0 .8em 3em;color:#222}}
#rb{{position:sticky;top:0;z-index:9;background:#fffffff2;padding:6px 0;border-bottom:1px solid #ddd;font-size:14px}}
#rb button{{margin:0 2px;padding:1px 8px;border:1px solid #999;border-radius:4px;background:#fff}}#rb button.on{{background:#333;color:#fff}}
#rb input{{vertical-align:middle;width:150px}}section{{border-top:1px solid #ddd;margin-top:1.2em;padding-top:.2em}}h3{{margin:.4em 0}}
figure{{margin:0}}video{{width:100%;aspect-ratio:1288/1000;max-height:86vh;background:#000;display:block}}figcaption{{font-size:13px;color:#555}}.cap{{background:#f4f4fb;padding:3px 8px}}
.rate{{background:#333;color:#fff;border-radius:4px;padding:0 5px;font-size:12px}}details{{margin:1em 0}}li{{font-size:14px}}code{{font-size:13px}}
</style></head><body>
<div id="rb">播放速度 <button data-r="0">自动</button><button data-r="1">1×</button><button data-r="2">2×</button><button data-r="3">3×</button><button data-r="5">5×</button><button data-r="10">10×</button><button data-r="16">16×</button> <input type="range" min="0.5" max="16" step="0.5"> <b id="rv"></b></div>
<h1>MV 系配方新增的出路零分长什么样（PAI 281，ACC + GAP）</h1>
{intro}
<p>{n_ev} 个候选事件（至少一个 MV 读数出路、NTV0 不出路）按「几何 × 情境」归类，每类放一段代表视频（取出路读数最多、NTV0 得分最高的那个场景），共 {n_show} 张卡片、{n_clip} 段视频、{mb:.0f} MB。
每段左边是出路的 MV 读数，右边是同一场景的 NTV0；画面是 AlpaSim 自带的评测视频（上：以本车为中心的俯视图，橙线是本车的计划轨迹，绿线是路线，浅绿框是本车；右上：指标表；下：模型输入的前视相机，叠加计划轨迹），剪到出路前 7 s 到出路后 3 s。
顶条显示读数、得分、时刻和车速，出路后变红。顶部速度条改全部视频（自动 = 每段 max(2, min(3, 30/fps, 时长/6))，选择记在浏览器里），滑到附近才加载，过半可见才播放。</p>
{cards}
<details><summary>其余 {n_rest} 个 MV 新增出路事件（不放视频）</summary><ul>{li}</ul></details>
<details><summary>NTV0 也出路的 {n_sh} 个场景（共有出路，不算新增）</summary><ul>{li2}</ul></details>
<script>
// Speed bar: 0 = per-clip auto rate (data-rate); any other value overrides every clip. localStorage throws in the reader's sandbox.
let G = 0; try {{ G = +localStorage.vidRate || 0; }} catch (e) {{}}
const R = v => {{ const r = G || +v.dataset.rate; v.defaultPlaybackRate = r; v.playbackRate = r; }};
const setG = r => {{ G = r; try {{ localStorage.vidRate = r; }} catch (e) {{}}
  document.querySelectorAll("#rb button").forEach(b => b.classList.toggle("on", +b.dataset.r === r));
  document.querySelector("#rb input").value = r || 2; document.getElementById("rv").textContent = r ? r + "×" : "自动";
  document.querySelectorAll("video[data-src]").forEach(v => {{ if (v.src) R(v); const s = v.parentNode.querySelector(".rate"); if (s) s.textContent = (G || +v.dataset.rate) + "×"; }}); }};
document.querySelectorAll("#rb button").forEach(b => b.onclick = () => setG(+b.dataset.r));
document.querySelector("#rb input").oninput = e => setG(+e.target.value);
setG(G);
// Lazy: fetch a clip within about one screen, play it only while >= 50 % visible, pause when it leaves.
const near = new IntersectionObserver(es => es.forEach(e => {{ const v = e.target;
  if (e.isIntersecting && !v.src) {{ v.src = v.dataset.src; v.preload = "auto"; v.addEventListener("loadedmetadata", () => R(v)); near.unobserve(v); }} }}),
  {{ rootMargin: "100% 0px" }});
const seen = new IntersectionObserver(es => es.forEach(e => {{ const v = e.target;
  if (e.isIntersecting) {{ if (!v.src) {{ v.src = v.dataset.src; v.addEventListener("loadedmetadata", () => R(v)); }} R(v); v.play().catch(() => {{}}); }} else v.pause(); }}),
  {{ threshold: 0.5 }});
document.querySelectorAll("video[data-src]").forEach(v => {{ near.observe(v); seen.observe(v); }});
</script></body></html>'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("select", "launch", "wait", "collect"):
        sp.add_parser(c)
    p = sp.add_parser("page")
    p.add_argument("dir"), p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dict(select=cmd_select, launch=cmd_launch, wait=cmd_wait, collect=cmd_collect, page=cmd_page)[a.cmd](a)
