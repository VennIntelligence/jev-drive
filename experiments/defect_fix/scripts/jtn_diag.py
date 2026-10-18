#!/usr/bin/env python3
"""JTN-DIAG: a video page of JTN-L18-s0's zero / low-score windows next to RSG-L18-s0 under the same serve (decision 353).

  $AP jtn_diag.py select [--thr 0.5] [--drop 0.3] [--chunk 6]   (box) the stored reads -> $O/sel.json + one rerun list per (source, shard) chunk
  $AP jtn_diag.py collect                                       (box) the camera reruns -> $O/videos/<src>_<h16>_<tag>.mp4 + $O/selection.json
  python3 jtn_diag.py page DIR                                  (Mac) DIR = research/jtn-diag (videos/ + selection.json) -> index.html + manifest.json

Sources (each JTN read paired with RSG-L18-s0's stored read under the same serve, on the windows where neither is a route-fail):
  arena  arena held split (2 246 windows), JEV_HOLD=stop3 (the deploy serve): JTN = runs/defect_fix/jt/arena/JTN-L18-s0@stop3, RSG = ARENA-HOLD 2
  held   held 460 @18 (sh30, no hold): runs/route_mem/read/{JTN,RSG}-L18-s0
  off    arena held split, no hold: JTN@off vs the arena's stored RSG read; every JTN-only zero and low score, plus seeded samples of the
         windows JTN fixes (RSG zero, JTN > 0) and of the shared zeros
Picked in arena / held: JTN score < thr, or RSG - JTN >= drop. Stored reads keep rollout.asl only for zero scores, so every clip is a camera
rerun (AlpaSim's own eval video, eval.video.render_video=true) of the same checkpoint, serve and window; the rerun's score is recorded next
to the stored one. Descriptions (speed, hold latch) come from the stored reads' controller traces and hold.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
REPO = Path(__file__).resolve().parents[3]
RUNS = DATA / "runs"
O = RUNS / "defect_fix/jtn_diag"
AR = RUNS / "alpasim_proxy/arena"
T, R = "JTN-L18-s0", "RSG-L18-s0"
READS = {  # src -> (JTN read root, RSG read root, RSG rows csv or None, RSG tag in it)
    "arena": (RUNS / f"defect_fix/jt/arena/{T}@stop3", AR / "hold/reads/stop3", AR / "hold/report2/rows.csv.gz", "stop3"),
    "held": (RUNS / f"route_mem/read/{T}", RUNS / f"route_mem/read/{R}", None, None),
    "off": (RUNS / f"defect_fix/jt/arena/{T}@off", AR / f"reads/{R}", AR / "hold/report/ref_rows.csv.gz", R),
}


def summaries(root):
    """clipgt id -> dict(run, rid, score, metrics, failure_reason) over <root>/c*/run."""
    out = {}
    for d in sorted(glob.glob(f"{root}/c*/run")):
        f = Path(d) / "aggregate/results-summary.json"
        if f.exists():
            for r in json.loads(f.read_text())["rollouts"]:
                out[r["clipgt_id"]] = dict(run=d, rid=r["rollout_id"], score=float(r.get("score") or 0.0), metrics=r.get("metrics") or {},
                                           failure_reason=r.get("failure_reason"))
    return out


def speeds(run, rid):
    """Ego speed summary from the controller trace (10 Hz): start (0.5 s), max, mean, end, time stopped (< 0.3 m/s) after 1 s."""
    import numpy as np
    sys.path.insert(0, str(REPO / "experiments/route_follow/scripts"))
    import rf_tax as TX
    C = TX.ctrl(run, rid)
    if C is None or len(C["t"]) < 3:
        return {}
    v = np.hypot(C["vx"], C["vy"]); t = (C["t"] - C["t"][0]) / 1e6
    late = t >= 1.0
    return dict(v0=round(float(np.interp(0.5, t, v)), 2), vmax=round(float(v.max()), 2), vmean=round(float(v[late].mean()) if late.any() else 0.0, 2),
                vend=round(float(v[-1]), 2), still_s=round(float((v[late] < 0.3).sum() * np.median(np.diff(t))), 1), dur=round(float(t[-1]), 1))


def holds(root):
    """session id -> hold record at the handoff (state held / free, s_lim) and whether a cut was ever served, over <root>/c*/run."""
    out = {}
    for f in glob.glob(f"{root}/c*/run/driver-logs/hold.jsonl"):
        for ln in open(f):
            r = json.loads(ln)
            s = out.setdefault(r["session"], dict(state="pre", cut=False))
            if r.get("state") in ("held", "free") and s["state"] == "pre":
                s.update(state=r["state"], kind=("stop" if "stop_s" in r else "leash") if r["state"] == "held" else None, s_lim=r.get("s_lim"))
            s["cut"] |= bool(r.get("cut"))
    return out


def cmd_select(a):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(REPO / "experiments/alpasim_proxy/scripts"))
    import arena as A
    S = pd.read_csv(AR / "set/set.csv.gz").set_index("id")
    H = S[S.split == "held"]
    rng = np.random.default_rng(0)
    sel, lists = [], {}
    for src, (jr, rr, _, _) in READS.items():
        J, Rs = summaries(jr), summaries(rr)
        hj, hr = (holds(jr), holds(rr)) if src == "arena" else ({}, {})
        ids = sorted(set(J) & set(Rs) & (set(H.index) if src != "held" else set(J)))
        rows = []
        for i in ids:
            x, y = J[i], Rs[i]
            tx, ty = A.failure_type(x), A.failure_type(y)
            if "route-fail" in (tx, ty):
                continue
            rows.append(dict(id=i, js=x["score"], rs=y["score"], jt=tx, rt=ty))
        D = pd.DataFrame(rows).set_index("id")
        if src == "off":
            pick = D[((D.js == 0) & (D.rs > 0)) | ((D.js > 0) & (D.js < a.thr)) | (D.rs - D.js >= a.drop)].assign(group="off_new")
            fx = D[(D.rs == 0) & (D.js > 0) & (H.loc[D.index, "stratum"] == "static").values]
            bz = D[(D.rs == 0) & (D.js == 0)]
            fx = fx.loc[sorted(rng.choice(fx.index, min(a.n_fixed, len(fx)), replace=False))].assign(group="off_fixed")
            bz = bz.loc[sorted(rng.choice(bz.index, min(a.n_both, len(bz)), replace=False))].assign(group="off_both")
            pool = dict(new=len(pick), fixed=len(D[(D.rs == 0) & (D.js > 0)]), fixed_static=len(D[(D.rs == 0) & (D.js > 0) & (H.loc[D.index, "stratum"] == "static").values]),
                        both=len(D[(D.rs == 0) & (D.js == 0)]), common=len(D))
            pick = pd.concat([pick, fx, bz])
        else:
            pick = D[(D.js < a.thr) | (D.rs - D.js >= a.drop)].copy()
            pick["group"] = np.where(pick.js == 0, np.where(pick.rs == 0, "zero_both", "zero_new"), "low")
            pool = dict(common=len(D), zeros=int((D.js == 0).sum()), rsg_zeros=int((D.rs == 0).sum()), low=int(((D.js > 0) & (D.js < a.thr)).sum()),
                        drop=int(((D.rs - D.js >= a.drop) & (D.js >= a.thr)).sum()))
        print(src, pool, len(pick), flush=True)
        for i, p in pick.iterrows():
            x, y = J[i], Rs[i]
            o = dict(src=src, id=i, h16=i[-16:], group=p.group, pool=pool,
                     orig={T: dict(score=x["score"], type=p.jt, dist=x["metrics"].get("dist_traveled_m"), **speeds(x["run"], x["rid"]), hold=hj.get(x["rid"])),
                           R: dict(score=y["score"], type=p.rt, dist=y["metrics"].get("dist_traveled_m"), **speeds(y["run"], y["rid"]), hold=hr.get(y["rid"]))},
                     gt=x["metrics"].get("gt_dist_traveled_m"))
            if src != "held":
                w = H.loc[i]
                o.update(shard=int(w.shard), stratum=w.stratum, hv0=round(float(w.v0), 2), hv_end=round(float(w.v_end), 2), lead=w.lead)
            sel.append(o)
            lists.setdefault((src, o.get("shard", "L")), []).append(i)
    O.mkdir(parents=True, exist_ok=True)
    (O / "lists").mkdir(exist_ok=True)
    jobs = []
    for (src, k), ids in lists.items():
        for n in range(0, len(ids), a.chunk):
            nm = f"{src}_s{k}_{n // a.chunk}"
            (O / "lists" / f"{nm}.txt").write_text("\n".join(ids[n:n + a.chunk]) + "\n")
            jobs.append(dict(name=nm, src=src, shard=k))
    (O / "sel.json").write_text(json.dumps(dict(sel=sel, jobs=jobs), indent=1, default=str))
    print(len(sel), "windows", len(jobs), "lists")


def cmd_collect(a):
    import shutil as sh
    S = json.loads((O / "sel.json").read_text())
    V = O / "videos"
    V.mkdir(exist_ok=True)
    re_ = {}
    for j in S["jobs"]:
        for tag in (T, R):
            d = O / "rerun" / f"{j['name']}_{tag}"
            f = d / "aggregate/results-summary.json"
            if not f.exists():
                print("missing", d); continue
            for r in json.loads(f.read_text())["rollouts"]:
                m = glob.glob(f"{d}/rollouts/{r['clipgt_id']}/{r['rollout_id']}/**/*.mp4", recursive=True)
                m = sorted(m, key=os.path.getsize)[-1:] if m else []
                key = (j["src"], r["clipgt_id"], tag)
                re_[key] = dict(score=float(r.get("score") or 0.0), mp4=m[0] if m else None)
    sys.path.insert(0, str(REPO / "experiments/alpasim_proxy/scripts"))
    n = 0
    for o in S["sel"]:
        o["rerun"] = {}
        for tag in (T, R):
            x = re_.get((o["src"], o["id"], tag))
            o["rerun"][tag] = None if x is None else x["score"]
            if x and x["mp4"]:
                dst = V / f"{o['src']}_{o['h16']}_{tag}.mp4"
                sh.copyfile(x["mp4"], dst); n += 1
    (O / "selection.json").write_text(json.dumps(S, indent=1, default=str))
    print(len(S["sel"]), "windows", n, "videos")


# ---------------------------------------------------------------- page (Mac)
def speed(f):
    """Fast-forward factor (nx_zvid_page.speed): ~30 fps, never above 3x, at least ~6 s on screen, never below 2x (2 fps clips look choppy at 1x)."""
    try:
        r, d = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate:format=duration",
                              "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout.split()
        n, m = r.split("/"); fps, dur = float(n) / float(m), float(d)
    except (ValueError, OSError):
        return 2.0
    return round(max(2.0, min(3.0, 30.0 / fps, dur / 6.0)), 2)


MECH = [  # (key, Chinese label, test on the JTN row of the window)
    ("af_front", "at-fault 正面碰撞（追尾前车 / 撞前方物体）", lambda t, o: t.startswith("AF-front")),
    ("af_side", "at-fault 侧向 / 后部碰撞", lambda t, o: t.startswith("AF-")),
    ("offroad", "出路（offroad）", lambda t, o: "offroad" in t),
    ("corridor", "偏出走廊（离人类轨迹 >= 4 m；静止窗里是滚出人类停车点 4 m 圆）", lambda t, o: t == "corridor"),
    ("rear", "被追尾（rear-hit，部分分）", lambda t, o: t == "rear-hit"),
    ("hit", "其他非 at-fault 碰撞（部分分）", lambda t, o: t == "hit"),
    ("nolaunch", "不起步（人类走了 > 5 m，本车 < 2 m）", lambda t, o: t == "no-launch"),
    ("creep", "爬行（本车里程 < 人类的 30%）", lambda t, o: t == "creep"),
    ("slow", "偏慢（本车里程 >= 人类的 30%）", lambda t, o: t == "slow"),
    ("ok", "JTN 不为零、不低（分组见标题）", lambda t, o: True),
]
SRC = {"arena": "A. arena held 2 228 窗，stop3（部署配置）", "held": "B. held 460 @18（sh30，无 hold）", "off": "C. arena held，不加 hold（JTN 原生）"}
GRP = {"zero_new": "JTN 零分、RSG 不为零（JTN 弄坏的）", "zero_both": "两者都零分", "low": "JTN 低分（< 0.5）或比 RSG 低 >= 0.3",
       "off_new": "JTN 新零分 / 低分（RSG 不为零或高 >= 0.3）", "off_fixed": "JTN 修好的静止窗（RSG 零分，JTN > 0；抽样）",
       "off_both": "两者都零分（抽样）"}
TYPE = {"AF-front": "正面碰撞", "AF-side": "侧碰", "AF-rear": "后部 at-fault", "offroad": "出路", "corridor": "偏出走廊", "rear-hit": "被追尾",
        "hit": "被碰", "no-launch": "不起步", "creep": "爬行", "slow": "偏慢", "ok": "满分"}


INTRO = """<p>JTN-L18-s0 是 JTB / JTN 一轮里不加弧长项的那一臂（第 353 条；结果 <code>experiments/defect_fix/results/jtb/README.md</code>）：
部署配置 stop3 下与 RSG-L18-s0 打平（arena held 0.943 vs 0.944，配对差 −0.001 [−0.009, +0.010]），held 460 零分 34 = 34（含 8 个路线失败），
不加 hold 时 +0.089 [+0.050, +0.135]，几乎全来自静止窗。这一页把它的零分和低分全部放出来，每窗并排放同一 serve 下的 RSG-L18-s0。</p>
<p><b>怎么做的。</b>A 组取 arena held 2 228 窗（stop3）里 JTN 得分 &lt; 0.5 或比 RSG 低 &gt;= 0.3 的窗，B 组同规则取 held 460 @18（sh30，无 hold），
两边都排除路线失败（Waypoint）窗。C 组是不加 hold 的 arena 读数：JTN 所有新零分 / 低分全放，另按种子 0 抽 24 个 JTN 修好的静止窗（全池 213）和 16 个两者都零分的窗（全池 225）。
存档读数只对零分保留 rollout.asl，所以每段视频都是同一 checkpoint、同一 serve、同一窗的<b>重跑</b>，打开 AlpaSim 自带的评测视频（上：BEV + 指标表，下：前视相机，2 帧/秒）。
470 段重跑得分与存档读数全部一致（差 &lt; 0.05），可以当原读数的录像看。卡片里的速度（起步 0.5 s / 峰值 / 1 s 后平均 / 末速）、里程和 hold 锁定状态来自存档读数的控制器轨迹和 hold.jsonl。
得分 = min(进度 / 0.8, 1)，被 at-fault 碰撞 / 出路 / 偏出走廊清零；没有舒适度分量，低分只来自进度损失。</p>
<p><b>先看什么。</b></p>
<ol>
<li><b>JTN 弄坏的零分很少，而且多是出路。</b>A 组 JTN 独有零分 20 个（RSG 同窗不为零），其中出路 16、正面碰撞 2、走廊 2；B 组 11 个（出路 6、正面 3、侧碰 1、走廊 1）。
A 组 20 个里 7 个、B 组 11 个里 5 个是 JTN 比 RSG 更快地进入弯道或弯后加速（峰值高 &gt;= 0.5 m/s 或末速高 &gt;= 1 m/s），然后跑宽出路：accd0462、9be7aade、e9cfc98f、9a675656（右转后冲进对向车道）。
另一种是同一条停车场回环（log 2021.10.06.08.16.17_veh-52，d0bf0c6c / ddc1271e）：JTN 提前切内侧转向，直接开出路面，RSG 沿回环走。</li>
<li><b>刹车窗的两个正面碰撞</b>：5d9fa0b5 在路口前排队，JTN 停车点比 RSG 远 0.6 m（10.6 vs 10.0 m），低速蹭上前车；b53373db 前车在汇入口转入，JTN 的 plan 向前车一侧偏、接触，RSG 同窗满分。数量小，但与 arena 表里刹车层 −0.016 同向。</li>
<li><b>两者都零分的窗基本是同一条轨迹。</b>A 组 40 个共享零分里 36 个两车里程差 &lt; 1 m，B 组 15 个里 10 个；多是弯道 / 路口出路。这些是 RSG 一脉的共同缺陷，不是 JTN 引入的。</li>
<li><b>A 组低分几乎都是 serve 造成的。</b>58 个低分里 45 个是起步窗被 stop3 的静止 leash 锁住、只能走 3 m（爬行 / 被追尾），47 个与 RSG 同分。
JTN 自己的差别在 7 个「不起步」：同样锁住，RSG 用满 3 m，JTN 只挪 0–1.8 m，对应 arena 起步层 −0.033。</li>
<li><b>C 组：JTN 原生会停。</b>抽样的 24 个修好的静止窗里 JTN 走 0–4.0 m（16 个几乎不动），RSG 走 2.1–5.8 m、滚出人类停车点的 4 m 圆被清零。代价在起步：C 组 JTN 新增 7 个不起步、2 个爬行。</li>
</ol>
<p>组内按 JTN 的失败类型分节（arena_proxy 的 failure_type：零分按 at-fault 碰撞 / 出路 / 走廊；部分分按被追尾 / 不起步 / 爬行 / 偏慢），节内按 JTN 得分升序。
重跑目录和 selection.json 在 box 的 <code>runs/defect_fix/jtn_diag/</code>；生成脚本 <code>experiments/defect_fix/scripts/jtn_diag.{py,sh}</code>。</p>
"""


def tname(t):
    return "+".join(TYPE.get(x, x) for x in t.split("+"))


def desc(r, gt):
    if not r:
        return ""
    s = [f"起 {r.get('v0', '-')} / 峰 {r.get('vmax', '-')} / 均 {r.get('vmean', '-')} / 末 {r.get('vend', '-')} m/s", f"走 {r.get('dist') or 0:.1f} m（人 {gt or 0:.1f} m）"]
    if r.get("still_s") is not None:
        s.append(f"静止 {r['still_s']} s / {r.get('dur', '-')} s")
    h = r.get("hold")
    if h:
        s.append("hold 未锁（free）" if h["state"] == "free" else f"hold 锁定（{'停车' if h.get('kind') == 'stop' else '静止'} leash，s_lim {h.get('s_lim')} m{'，有截断' if h.get('cut') else ''}）"
                 if h["state"] == "held" else "hold 未到交接")
    return "；".join(s)


def cmd_page(a):
    import html
    D = Path(a.dir)
    S = json.loads((D / "selection.json").read_text())
    man, secs, nav, tab = [], [], [], []
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731

    def card(o):
        cells = []
        for tag, lab in ((T, "JTN-L18-s0"), (R, "参照 RSG-L18-s0")):
            f = D / "videos" / f"{o['src']}_{o['h16']}_{tag}.mp4"
            r = o["orig"][tag]
            rr = (o.get("rerun") or {}).get(tag)
            warn = "" if rr is None or abs(rr - r["score"]) < 0.05 else f' <span class="warn">重跑 {rr:.2f}，与原读数不同</span>'
            cap = f"<b>{lab}</b> {f2(r['score'])}（{tname(r['type'])}）{warn}<br>{html.escape(desc(r, o.get('gt')))}"
            if f.exists():
                v = speed(f)
                man.append(dict(src=o["src"], id=o["id"], tag=tag, file=f"videos/{f.name}", mb=round(f.stat().st_size / 1e6, 2), speed=v,
                                orig=r["score"], rerun=rr))
                cells.append(f'<figure><video data-src="videos/{f.name}" data-rate="{v}" muted loop playsinline controls preload="none"></video>'
                             f'<figcaption><span class="rate">{v:g}×</span> {cap}</figcaption></figure>')
            else:
                cells.append(f'<figure class="miss"><figcaption>{cap}<br>无视频</figcaption></figure>')
        w = (f"；窗型 {o['stratum']}，人类 {o['hv0']} → {o['hv_end']} m/s{'，有前车' if str(o.get('lead')) == 'True' else ''}" if o.get("stratum") else "")
        return (f'<section><h4>{o["h16"][:8]} <small>{html.escape(o["id"])}</small></h4>'
                f'<p>JTN {f2(o["orig"][T]["score"])} vs RSG {f2(o["orig"][R]["score"])}；{GRP[o["group"]]}{w}</p><div class="row">{"".join(cells)}</div></section>')

    for src, title in SRC.items():
        os_ = [o for o in S["sel"] if o["src"] == src]
        if not os_:
            continue
        nav.append(f'<a href="#{src}">{title.split("（")[0]}</a>')
        body = f'<h2 id="{src}">{title}：{len(os_)} 窗</h2>'
        for g in GRP:
            og = [o for o in os_ if o["group"] == g]
            if not og:
                continue
            body += f'<h3>{GRP[g]}（{len(og)}）</h3>'
            left = og
            for k, lab, fn in MECH:
                om = [o for o in left if fn(o["orig"][T]["type"], o)]
                left = [o for o in left if o not in om]
                if om:
                    aid = f"{src}-{g}-{k}"
                    body += f'<h4 class="cls" id="{aid}">{lab}：{len(om)}</h4>' + "".join(card(o) for o in sorted(om, key=lambda o: o["orig"][T]["score"]))
                    tab.append(f'<tr><td>{title.split("（")[0]}</td><td>{GRP[g]}</td><td><a href="#{aid}">{lab}</a></td><td>{len(om)}</td>'
                               f'<td>{sum(o["orig"][R]["score"] == 0 for o in om)}</td></tr>')
        secs.append(body)
    (D / "manifest.json").write_text(json.dumps(man, indent=1))
    tot = sum(m["mb"] for m in man)
    intro = INTRO
    (D / "index.html").write_text(f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JTN-L18-s0 零分 / 低分视频诊断</title><style>body{{font:15px/1.55 -apple-system,sans-serif;max-width:1100px;margin:1em auto;padding:0 .8em;color:#222}}
small{{color:#888;font-weight:400;font-size:11px;word-break:break-all}}.row{{display:flex;gap:8px;flex-wrap:wrap}}figure{{margin:0;flex:1 1 320px;max-width:540px}}
video{{width:100%;background:#000}}figcaption{{font-size:13px;color:#444}}section{{border-top:1px solid #ddd;margin-top:1em;padding-top:.3em}}h4{{margin:.3em 0}}
h4.cls{{background:#eef;padding:3px 8px;margin-top:1.6em}}h3{{margin-top:2em;border-bottom:2px solid #889}}.warn{{color:#a50}}.miss{{color:#a00}}
.rate{{background:#333;color:#fff;border-radius:4px;padding:0 5px;font-size:12px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:2px 8px;text-align:left}}
nav a{{margin-right:1em}}#rb{{position:sticky;top:0;z-index:9;background:#fffffff0;padding:5px 0;border-bottom:1px solid #ddd;font-size:14px}}#rb button{{margin:0 2px;padding:1px 7px;border:1px solid #999;border-radius:4px;background:#fff}}#rb button.on{{background:#333;color:#fff}}#rb input{{vertical-align:middle;width:140px}}</style></head><body>
<div id="rb">播放速度 <button data-r="0">自动</button><button data-r="1">1×</button><button data-r="2">2×</button><button data-r="3">3×</button><button data-r="5">5×</button><button data-r="10">10×</button><button data-r="16">16×</button> <input type="range" min="0.5" max="16" step="0.5"> <b id="rv"></b></div>
<h1>JTN-L18-s0 的零分与低分长什么样（对照 RSG-L18-s0，同一 serve、同一窗）</h1>
{intro}
<table><tr><th>来源</th><th>组</th><th>机制（按 JTN 的失败类型）</th><th>窗数</th><th>其中 RSG 也零分</th></tr>{"".join(tab)}</table>
<p>{len(man)} 段视频，合计 {tot:.0f} MB。每段左上角是当前倍数；顶部速度条可改全部视频（自动 = 每段 max(2, min(3, 30/fps, 时长/6))，选择记在浏览器里），滑到附近才加载，过半可见才播放。</p>
<nav>{"".join(nav)}</nav>
{"".join(secs)}
<script>
// Speed bar: 0 = per-clip auto rate (data-rate); any other value overrides every clip. Kept in localStorage across pages where storage is allowed.
let G = 0; try {{ G = +localStorage.vidRate || 0; }} catch (e) {{}}  // the web reader's sandbox (opaque origin) throws on storage
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
</script></body></html>''')
    print(len(S["sel"]), "windows", len(man), "videos", round(tot, 1), "MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--drop", type=float, default=0.3)
    p.add_argument("--chunk", type=int, default=6)
    p.add_argument("--n-fixed", type=int, default=24)
    p.add_argument("--n-both", type=int, default=16)
    sub.add_parser("collect")
    p = sub.add_parser("page")
    p.add_argument("dir")
    a = ap.parse_args()
    dict(select=cmd_select, collect=cmd_collect, page=cmd_page)[a.cmd](a)
