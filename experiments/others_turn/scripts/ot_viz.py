#!/usr/bin/env python3
"""OTHERS-TURN comparison page: the 24 held-460 turns (results/turn24.txt) x six policies, with route_follow/scripts/plan_exec_viz.py's held
machinery (analyse / draw_bev / draw_traces): BEV of every decision's plan coloured by time (1-2 s segment thick), the executed track, the human
path and the lane polygons; traces of speed, 2 s / 4 s plan heading, the car's offset from the plan 1 s earlier and the plan-to-plan jump; plus
the front-camera clip of the frames the driver received (CAM_F0 driver_camera_image of rollout.asl, one per decision). Stored reads are read only.

  extract (box, AlpaSim venv)  ot_viz.py extract --follow <rf_follow scenes.csv> --lanes <route_follow/corridor/in/lanes_held.pkl (rf_corridor)> --out DIR
                                 --set LABEL=<run dir>[,<run dir>] ... [--frames LABEL ...] [--workers N]
      DIR/recs.pkl: {recs: {label: [plan_exec_viz held record + row]}, lanes: {scene: lanes}}; DIR/frames/<label>/<s8>/<k:02d>.jpg (480 px wide)
  render  (Mac, repo .venv)    ot_viz.py render --pkl DIR/recs.pkl --frames DIR/frames --cols LABEL[:CAMLABEL]=TITLE ... --out OUT
      OUT/<label>_<s8>_bev.png, _tr.png, _cam.mp4, index.html (Chinese legend), meta.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

R_ = Path(__file__).resolve().parents[3]
RF = R_ / "experiments/route_follow/scripts"
sys.path[:0] = [str(RF), str(R_ / "experiments/junction_eyes/lib")]
T0, DT = 17_000, 500_000
CAM = "CAM_F0"


# ---------------------------------------------------------------- extract (box)
def frames_of(job):
    """(asl path, {t0: k}, out dir) -> number of frames written (CAM_F0 at each decision's t0, 480 px wide JPEG)."""
    import asyncio
    import io
    from alpasim_utils.logs import async_read_pb_log
    from PIL import Image
    f, want, out = job
    out.mkdir(parents=True, exist_ok=True)

    async def go():
        n = 0
        async for e in async_read_pb_log(f):
            if e.WhichOneof("log_entry") == "driver_camera_image":
                c = e.driver_camera_image.camera_image
                if c.logical_id == CAM and int(c.frame_end_us) in want:
                    im = Image.open(io.BytesIO(c.image_bytes)).convert("RGB")
                    im.thumbnail((480, 480))
                    im.save(out / f"{want[int(c.frame_end_us)]:02d}.jpg", quality=80)
                    n += 1
        return n
    return asyncio.run(go())


def extract(a):
    from concurrent.futures import ProcessPoolExecutor
    import rf_tax as TX
    from rf_receding import plan_records
    from rf_pai_follow import dedup
    fo = list(csv.DictReader(open(a.follow)))
    L = pickle.load(open(a.lanes, "rb"))["scenes"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(recs={}, lanes={})
    fjobs = []
    for spec in a.set:
        lab, dirs = spec.split("=", 1)
        rows = {r["scene"]: r for r in fo if r["label"] == f"{lab}@18" and r["turn"] == "True"}
        jobs = {}
        for run in dirs.split(","):
            recs = TX.drive_recs(run)
            cfg = str(Path(run) / "generated-user-config-0.yaml")
            for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]:
                if r["clipgt_id"] in rows:
                    jobs[(r["clipgt_id"], cfg)] = [dict(rid=r["rollout_id"], run=run, score=r.get("score"), metrics=r.get("metrics") or {},
                                                        recs=recs.get(r["rollout_id"], []))]
        with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
            rolls = {o["scene"]: o for v in ex.map(TX.scene_job, [(s, c, v, False) for (s, c), v in sorted(jobs.items())]) for o in v}
        P = {}
        for run in dirs.split(","):
            P |= plan_records([os.path.join(run, "driver-logs", "drive.jsonl")], {o["rid"] for o in rolls.values()})
        out_l = []
        for s, row in rows.items():
            o, p = rolls.get(s), None
            if o is None or o.get("ctrl") is None or (p := P.get(o["rid"])) is None:
                print(lab, s, "missing", "rollout" if o is None else "ctrl" if o.get("ctrl") is None else "plans", flush=True)
                continue
            kz = int(row["kz"])
            sc = float(o["score"] or 0)
            out_l.append(dict(scene=s, rid=o["rid"], score=sc, tz=float(T0 + DT * kz) if kz >= 0 else None, zflag="", route=dedup(np.asarray(o["route"], float)),
                              ctrl={k: np.asarray(o["ctrl"][k], float) for k in ("t", "x", "y", "yaw", "vx")},
                              log=dict(T=np.asarray(o["log"]["T"], float), P=np.asarray(o["log"]["P"], float)), mp4=None,
                              kind="zero" if sc == 0 else "ok", t0=p["t0"], k=p["k"], anchor=p["anchor"], poses=p["poses"], run=o["run"],
                              asl=(o.get("files") or {}).get("asl"),
                              row=dict(entry=float(row["entry"]), exit=float(row["exit"]), side=row["side"], turn_deg=float(row["turn_deg"]),
                                       outcome=row["outcome"], completed=row["completed"] == "True", kz=kz,
                                       metrics={k: (o.get("metrics") or {}).get(k) for k in ("collision_at_fault", "offroad", "left_corridor_laterally")})))
            if lab in (a.frames or []) and out_l[-1]["asl"]:
                fjobs.append((out_l[-1]["asl"], {int(t): i for i, t in enumerate(p["t0"])}, out / "frames" / lab / s[-8:]))
        res["recs"][lab] = out_l
        print(lab, len(rows), "turns,", len(out_l), "records", flush=True)
    res["lanes"] = {s: L[s] for v in res["recs"].values() for s in [r["scene"] for r in v] if s in L}
    pickle.dump(res, open(out / "recs.pkl", "wb"), protocol=4)
    with ProcessPoolExecutor(a.workers) as ex:
        n = list(ex.map(frames_of, fjobs))
    print(len(fjobs), "clips,", sum(n), "frames ->", out, flush=True)


# ---------------------------------------------------------------- render (Mac)
CN_OUT = {"completed": "完成", "event": "零分", "off": "偏离路线", "short": "没到出口"}


def render(a):
    import plan_exec_viz as V
    D = pickle.load(open(a.pkl, "rb"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = []
    for c in a.cols:
        lab, title = c.split("=", 1)
        lab, cam = (lab.split(":") + [lab])[:2]
        cols.append((lab, cam, title))
    by = {lab: {r["scene"]: r for r in D["recs"].get(lab, [])} for lab, _, _ in cols} | {cam: {r["scene"]: r for r in D["recs"].get(cam, [])} for _, cam, _ in cols}
    scenes = [s for s in Path(a.scenes).read_text().split()]
    cells, counts = {}, {}
    for lab, cam, title in cols:
        counts[lab] = dict(done=sum(r["row"]["completed"] for r in by[lab].values()), n=len(by[lab]),
                           cam_done=sum(r["row"]["completed"] for r in by[cam].values()) if cam != lab else None)
        for s in scenes:
            r = by[lab].get(s)
            if r is None:
                continue
            td = dict(entry=r["row"]["entry"], exit=r["row"]["exit"], side=r["row"]["side"], turn_deg=r["row"]["turn_deg"])
            key = f"{lab}_{s[-8:]}"
            g = V.analyse(r, td, V.DT_HELD, 0.0, False)
            V.draw_bev(r, td, g, D["lanes"].get(s), out / f"{key}_bev.png")
            V.draw_traces(r, td, g, out / f"{key}_tr.png", "time s since decision 0 (2 Hz decisions)")
            vid, rc = None, by[cam].get(s)
            fd = Path(a.frames) / cam / s[-8:]
            if rc is not None and fd.exists() and any(fd.glob("*.jpg")):
                vid = f"{key}_cam.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "2", "-pattern_type", "glob", "-i", str(fd / "*.jpg"), "-vf",
                                "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "2", str(out / vid)], check=True)
            m = r["row"]["metrics"]
            ev = "+".join(n for k, n in (("collision_at_fault", "碰撞"), ("offroad", "出路"), ("left_corridor_laterally", "出走廊")) if (m.get(k) or 0) > 0)
            cells[key] = dict(lab=lab, scene=s, outcome=r["row"]["outcome"], cn=CN_OUT.get(r["row"]["outcome"], r["row"]["outcome"]), score=round(r["score"], 3),
                              v_ent=None if not np.isfinite(g["v_ent"]) else round(float(g["v_ent"]), 1),
                              v_ent_h=None if not np.isfinite(g["v_ent_h"]) else round(float(g["v_ent_h"]), 1),
                              t_event=None if g["t_event"] is None else round(g["t_event"], 1),
                              ev=ev,
                              vid=vid, cam_outcome=None if rc is None or cam == lab else rc["row"]["outcome"], side=td["side"], deg=round(abs(td["turn_deg"])))
            print(key, cells[key]["outcome"], cells[key]["v_ent"], "video" if vid else "-", flush=True)
    meta = dict(cols=[dict(lab=l, cam=c, title=t) for l, c, t in cols], counts=counts, scenes=scenes, cells=cells)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (out / "index.html").write_text(html(meta, a.note or ""))


def html(meta, note):
    cols, cells = meta["cols"], meta["cells"]
    head = "".join(f"<th>{c['title']}<br><span class=cnt>{meta['counts'][c['lab']]['done']} / {meta['counts'][c['lab']]['n']}</span></th>" for c in cols)
    body = []
    for s in meta["scenes"]:
        any_c = next((cells[k] for k in cells if cells[k]["scene"] == s), None)
        tds = []
        for c in cols:
            x = cells.get(f"{c['lab']}_{s[-8:]}")
            if x is None:
                tds.append("<td class=none>没有这个 rollout</td>")
                continue
            k = f"{c['lab']}_{s[-8:]}"
            cls = "ok" if x["outcome"] == "completed" else "bad" if x["outcome"] == "event" else "mid"
            ev = f"，{x['ev']} @ {x['t_event']} s" if x["outcome"] == "event" and x["t_event"] is not None else ""
            cam = ""
            if x["vid"]:
                note_c = "" if x["cam_outcome"] is None else f"<div class=cn>视频来自同一 checkpoint 的重跑（重跑结果：{CN_OUT.get(x['cam_outcome'], x['cam_outcome'])}），鸟瞰是存档 run</div>"
                cam = f"<video src='{x['vid']}' controls muted playsinline preload=none></video>{note_c}"
            else:
                cam = "<div class=cn>没有前视帧</div>"
            tds.append(f"<td class={cls}><div class=hd><b>{x['cn']}</b>{ev}<br>入口速度 {x['v_ent']} m/s（人 {x['v_ent_h']}）· 分 {x['score']}</div>"
                       f"<a href='{k}_bev.png' target=_blank><img src='{k}_bev.png' class=bev loading=lazy></a>"
                       f"<a href='{k}_tr.png' target=_blank><img src='{k}_tr.png' class=tr loading=lazy></a>{cam}</td>")
        lab = f"<b>{s[-8:]}</b><br><span class=sc>{s}</span><br>{'左转' if any_c and any_c['side'] == 'L' else '右转'} {any_c['deg'] if any_c else ''}°"
        body.append(f"<tr><td class=sn>{lab}</td>{''.join(tds)}</tr>")
    return f"""<!doctype html><html lang=zh><head><meta charset=utf-8><title>别人会转弯吗：held 24 个转弯 × 6 个策略</title><style>
body{{font:13px/1.45 -apple-system,'PingFang SC',sans-serif;margin:12px;color:#222}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;vertical-align:top;padding:4px}}
th{{position:sticky;top:0;background:#fafafa;z-index:2}} .cnt{{font-size:20px;color:#1a7f37}} .sn{{width:110px;font-size:12px}} .sc{{color:#888;font-size:10px;word-break:break-all}}
td.ok{{background:#f0faf2}} td.bad{{background:#fdf1f0}} td.mid{{background:#fff8ec}} .hd{{font-size:12px;margin-bottom:3px}} .bev{{width:250px;display:block}} .tr{{width:250px;display:block}}
video{{width:250px;display:block;margin-top:3px}} .cn{{font-size:11px;color:#777}} .none{{color:#aaa}} .legend{{max-width:1500px;background:#f6f8fa;padding:8px 12px;border-radius:6px}}
</style></head><body><h1>别人会转弯吗：held 460 的 24 个转弯 × 6 个策略（18 决策，AlpaSim nuPlan）</h1>
<div class=legend><b>每列标题下的数字 = 完成的转弯数 / 24</b>（rf_follow：零分事件前越过出口 +10 m、离路线 ≤ 2.5 m）。
格子底色：绿 = 完成，红 = 零分（碰撞 / 出路 / 出走廊，后面是零分时刻），橙 = 没零分但没完成（偏离路线 &gt; 2.5 m 或 rollout 结束时还没到出口）。<br>
<b>鸟瞰</b>（点击放大）：浅灰 = 地图车道，浅蓝 = 路线经过的车道链，蓝虚线 = 路线，粗灰线 = 人类日志路径，黑线 = 车实际走的（点 = 每秒），
彩色细线 = 每个决策的 plan（0–4 s，颜色 = 决策时间，紫早黄晚），粗段 = plan 的 1–2 s 段；黄色实线 / 虚线 = 弯的入口 / 出口，红叉 = 第一个零分时刻。朝向：接近方向朝上。<br>
<b>读数</b>（点击放大）：① 车速（黑）对人在同一路线位置的速度（灰）与 plan 头 1 s 的平均速度（蓝点）；② plan 的 2 s / 4 s 航向变化（实线，+ = 朝弯里）对人的（虚线）；
③ 车相对 1 s 前那个 plan 的横向偏差（+ = 偏弯内）；④ 相邻两个 plan 在 2 s 点的横向跳动。黄底 = 鸟瞰画的决策窗口，黄竖线 = 入口，红竖线 = 零分。<br>
<b>视频</b>：driver 自己收到的前视帧（rollout.asl 里的 CAM_F0，每个决策一帧，2 帧 / 秒）。RSG 与规则专家的鸟瞰来自存档 run（它们的 16 / 24、22 / 24 读数），存档没留帧，所以视频是同 checkpoint 在这 24 个场景上的重跑，格子里写了重跑的结果；
WA-JEPA、LTF、DiffusionDrive、garage 的鸟瞰和视频是同一次重跑。第三方策略都原样跑（自己的 driver 和环境），拿到的路线是 AlpaSim sample 的四值命令（首个 ≥ 5 m 路点，±2 m）。{note}</div>
<table><tr><th>场景</th>{head}</tr>{''.join(body)}</table></body></html>"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--follow", required=True), e.add_argument("--lanes", required=True), e.add_argument("--out", required=True)
    e.add_argument("--set", nargs="+", required=True), e.add_argument("--frames", nargs="*"), e.add_argument("--workers", type=int, default=24)
    r = sp.add_parser("render")
    r.add_argument("--pkl", required=True), r.add_argument("--frames", required=True), r.add_argument("--cols", nargs="+", required=True)
    r.add_argument("--scenes", default=str(R_ / "experiments/others_turn/results/turn24.txt")), r.add_argument("--out", required=True)
    r.add_argument("--note", default="")
    a = p.parse_args()
    {"extract": extract, "render": render}[a.cmd](a)
