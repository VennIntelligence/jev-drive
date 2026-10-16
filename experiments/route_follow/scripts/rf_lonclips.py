#!/usr/bin/env python3
"""LON-CLIPS: longitudinal loss classes of one stored PAI read (rf_tax.py extract pkl + report csv) and the clip picks that show them.

  rf_lonclips.py pick --pkl g10.pkl --csv g10.csv --out picks.json     class table (stdout, markdown) + <=5 clips per class
      [--label L --vs-csv ref.csv --vs-label R]   also the scenes zero here and non-zero in the reference csv ("newzero" picks, with clips) and the
                                                  reverse ("fixed": listed, no clips); --label names this read in the picks
  rf_lonclips.py clips --pkl x.pkl --picks picks.json --out DIR   copy each pick's rollout front-camera mp4 to DIR/<REL_DIR>/... names the page links
  rf_lonclips.py page --picks picks.json --out DIR --title T [--ref ref_picks.json ...] [--videos REL_DIR] [--novideo]
                                                         index.html (Chinese); clips link to <REL_DIR>/<class>/<nn>_<scene8>_<score>.mp4
                                                         (newzero: <REL_DIR>/newzero/<nn>_<scene8>_<g10>to<acc>.mp4); --ref adds one
                                                         count / points column group per reference read (their picks.json); video markup is
                                                         autoplay-on-screen (no controls, IntersectionObserver script); --novideo drops the video tags
Event time is on the rollout clock (the video clock). Human speed = logged ego speed at the same time. Only scenes that lost points are picked.
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# key, label, what to look for
CLASSES = [
    ("slow_rear", "慢（前方无车）且被后车追尾截断", "前方路面空着、本车比日志慢很多，后车按日志速度追上来；看慢是从哪一刻开始的、有没有任何前方原因。"),
    ("slow_free", "慢（前方无车），未被追尾", "同上但没被追尾；进度不够 80 % 就丢分。看画面里本车为什么不加速。"),
    ("slow_lead", "慢且跟在前车后面", "前车在动、本车落后；看时距是否过大。"),
    ("rear_only", "被追尾截断（速度比日志略低）", "本车比日志只慢一点，但后车撞上来、进度被截断；看追尾那一刻。"),
    ("with_log", "停在日志也慢 / 也停的地方（没起步）", "日志在这里也很慢或停着，本车起步后仍长时间不动；看人什么时候起步、本车晚了多久。"),
    ("unexpl", "无故停车", "日志在同位置有速度（>= 2 m/s）而本车停了 >= 1.5 s；看画面有没有任何停下的理由。"),
    ("late", "晚起步 / 没起步", "前车走后本车迟迟不动。"),
    ("route_end", "人停了、车没停（cor_route_end）", "人在灯口 / 停止线 / 排队处停到结束，本车以 8–16 m/s 开过去、越过日志终点被判走廊零分；看红灯、停止线、前车。"),
    ("coll_lead", "撞前车 / 静止车", "本车在慢速下撞上同向前车或静止车；看撞前的刹车。"),
]
KEY = {k: (l, w) for k, l, w in CLASSES}


def lon_class(r):
    p, sec = r.primary, str(r.secondary)
    if p == "low_slow_free":
        return "slow_rear" if "rear_ended" in sec else "slow_free"
    return dict(low_slow_behind_lead="slow_lead", low_rear_ended="rear_only", low_with_log="with_log", low_unexplained="unexpl",
                low_late_start="late", cor_route_end="route_end", coll_veh_ahead="coll_lead", coll_stationary="coll_lead").get(p)


def stop_start(stops, kind):
    m = re.search(kind + r"@([0-9.]+)s", str(stops))
    return float(m.group(1)) if m else None


def event(o, r, k):
    """(event time s on the video clock, why, human v, ego v, lag m)."""
    from rf_tax import Roll
    R = Roll(o)
    t_ev, why = None, ""
    if k in ("route_end", "coll_lead") and pd.notna(r.t_ev_s):
        t_ev, why = r.t_ev_s, "评估器判零的时刻"
    elif k == "rear_only" and pd.notna(r.rear_t_s):
        t_ev, why = r.rear_t_s, "被追尾"
    elif k in ("unexpl", "with_log"):
        kind = "unexplained" if k == "unexpl" else "with_log"
        t_ev = stop_start(r.stops, kind)
        why = "停车开始"
    if t_ev is None:
        dx = np.hypot(np.diff(R.x), np.diff(R.y))
        se = np.r_[0.0, np.cumsum(dx)]
        ts = (R.t - R.t0) * 1e-6
        sl = np.interp(R.t, R.LT + (R.t0 - R.LT[0]), R.LS[:len(R.LT)]) if False else np.interp(R.t - R.t0 + R.LT[0], R.LT, R.LS[:len(R.LT)])
        lag = sl - se
        i = np.argmax(lag >= 10) if (lag >= 10).any() else int(np.argmax(lag))
        t_ev, why = float(ts[i]), "本车落后日志 10 m 的时刻" if (lag >= 10).any() else "本车落后日志最多的时刻"
    ta = R.t0 + t_ev * 1e6
    hv = float(np.interp(ta, R.LT + (R.t0 - R.LT[0]), R.Lv[:len(R.LT)]))
    dx = np.hypot(np.diff(R.x), np.diff(R.y))
    se = np.r_[0.0, np.cumsum(dx)]
    sl = np.interp(ta, R.LT + (R.t0 - R.LT[0]), R.LS[:len(R.LT)])
    lag = float(sl - np.interp(ta, R.t, se))
    return dict(t=round(float(t_ev), 1), why=why, v_h=round(hv, 1), v_e=round(R.speed(ta), 1), lag=round(lag, 0))


def pick(a):
    d = pd.read_csv(a.csv)
    d["loss"] = (1 - d.score.fillna(0)) / len(d) * 100
    d["lon"] = d.apply(lon_class, axis=1)
    n = len(d)
    rows = [(k, l, int(((d.lon == k)).sum()), float(d.loc[d.lon == k, "loss"].sum())) for k, l, _ in CLASSES]
    tot = float(d.loss.sum())
    other = d[d.lon.isna() & (d.score.fillna(0) < 1)]
    print(f"| class | scenes | points lost | share of loss ({tot:.1f}) |\n|:--|--:|--:|--:|")
    for k, l, c, v in rows:
        print(f"| {l} | {c} | {v:.2f} | {100 * v / tot:.1f} % |")
    lo = sum(r[3] for r in rows)
    print(f"| longitudinal total | {sum(r[2] for r in rows)} | {lo:.2f} | {100 * lo / tot:.1f} % |")
    ran = other[other.secondary.fillna("").str.contains("ran_stop_line")]
    print(f"| not longitudinal (turn / lane / collision / offroad / ok) | {len(other)} | {other.loss.sum():.2f} | {100 * other.loss.sum() / tot:.1f} % |")
    print(f"(of which with a ran stop line as secondary: {len(ran)} scenes, {ran.loss.sum():.2f} points)", file=sys.stderr)
    import pickle
    O = {o["scene"]: o for o in pickle.load(open(a.pkl, "rb"))["rolls"]}
    picks = []
    for k, l, _ in CLASSES:
        c = d[d.lon == k].sort_values(["loss", "scene"], ascending=[False, True])
        got, locs = [], set()
        for _, r in c.iterrows():                              # spread over locations first
            if len(got) < a.per and str(r.loc_ if hasattr(r, "loc_") else r["loc"]) not in locs:
                got.append(r)
                locs.add(str(r["loc"]))
        for _, r in c.iterrows():
            if len(got) < a.per and all(r.scene != g.scene for g in got):
                got.append(r)
        for r in got:
            e = event(O[r.scene], r, k)
            picks.append(dict(cls=k, scene=r.scene, rid=r.rid, score=round(float(r.score), 3), loss=round(float(r.loss), 3), loc=str(r["loc"]),
                              primary=r.primary, secondary=None if pd.isna(r.secondary) else r.secondary, **e))
    fixed = []
    if a.vs_csv:
        R = pd.read_csv(a.vs_csv).set_index("scene")
        z = lambda x: pd.notna(x) and x <= 0                  # noqa: E731
        for _, r in d.iterrows():
            if r.scene not in R.index or pd.isna(r.score) or pd.isna(R.score[r.scene]):
                continue
            rs = float(R.score[r.scene])
            if z(r.score) and rs > 0:
                e = event(O[r.scene], r, "route_end")
                if pd.isna(r.t_ev_s):
                    print("no evaluator event time for", r.scene, file=sys.stderr)
                picks.append(dict(cls="newzero", scene=r.scene, rid=r.rid, score=round(float(r.score), 3), ref=round(rs, 3), loss=round(float(r.loss), 3),
                                  loc=str(r["loc"]), primary=r.primary, secondary=None if pd.isna(r.secondary) else r.secondary,
                                  zero_type=None if pd.isna(r.zero_type) else r.zero_type, **e))
            elif z(rs) and r.score > 0:
                fixed.append(dict(scene=r.scene, score=round(float(r.score), 3), ref=round(rs, 3), ref_zero_type=None if pd.isna(R.zero_type[r.scene]) else R.zero_type[r.scene],
                                  ref_primary=R.primary[r.scene], primary=r.primary))
    Path(a.out).write_text(json.dumps(dict(rows=[(k, l, c, v) for k, l, c, v in rows], total=tot, other=[len(other), float(other.loss.sum())], n=n,
                                           picks=picks, label=a.label, vs_label=a.vs_label, fixed=fixed), indent=1))


SCRIPT = """<script>
const io = new IntersectionObserver(es => { for (const e of es) { const v = e.target;
  if (e.isIntersecting) { v.play().catch(err => console.log(err)); } else { v.pause(); } } }, {threshold: 0.3});
document.querySelectorAll('video[data-t]').forEach(v => { io.observe(v);
  v.addEventListener('ended', () => { v.currentTime = +v.dataset.t; v.play(); }); });
</script>"""


def page(a):
    def vid(f, p):                                   # autoplay when on screen, no progress bar; starts 3 s before the event
        t = f"{max(p['t'] - 3, 0):.1f}"
        return "" if a.novideo else f"<video muted playsinline preload=metadata src='{f}#t={t}' data-t='{t}'></video>"
    P = json.loads(Path(a.picks).read_text())
    Rs = [json.loads(Path(f).read_text()) for f in a.ref or []]
    refs = [{r[0]: r for r in Rj["rows"]} for Rj in Rs]
    lab = P.get("label") or "this read"
    h = ["<!doctype html><html lang=zh><meta charset=utf-8><title>%s</title>" % html.escape(a.title),
         "<style>body{font:15px/1.5 sans-serif;max-width:1180px;margin:20px auto;padding:0 12px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 9px;text-align:right}"
         "td:first-child,th:first-child{text-align:left}.c{display:inline-block;vertical-align:top;margin:6px 10px 14px 0;width:560px}video{width:560px;background:#000}"
         "h2{margin-top:30px;border-bottom:2px solid #444}.m{font-size:13px;color:#333}</style>",
         "<body><h1>%s</h1>" % html.escape(a.title), "<p>%s</p>" % a.intro,
         "<table><tr><th>类别</th><th>%s scene 数</th><th>丢分（点）</th><th>占总丢分 %.1f 点</th>%s</tr>" % (html.escape(lab), P["total"], "".join(
             "<th>{0} scene 数</th><th>{0} 丢分（点）</th><th>{0} 占 {1:.1f} 点</th>".format(html.escape(R.get("label") or "ref"), R["total"]) for R in Rs))]
    cols = lambda k: "".join(f"<td>{rf[k][2]}</td><td>{rf[k][3]:.2f}</td><td>{100 * rf[k][3] / R['total']:.1f} %</td>" for rf, R in zip(refs, Rs))  # noqa: E731
    for k, l, c, v in P["rows"]:
        h.append(f"<tr><td><a href='#{k}'>{html.escape(l)}</a></td><td>{c}</td><td>{v:.2f}</td><td>{100 * v / P['total']:.1f} %</td>{cols(k)}</tr>")
    lo = sum(r[3] for r in P["rows"])
    gt = "".join(f"<th>{sum(r[2] for r in R['rows'])}</th><th>{sum(r[3] for r in R['rows']):.2f}</th><th>{100 * sum(r[3] for r in R['rows']) / R['total']:.1f} %</th>" for R in Rs)
    go = "".join(f"<td>{R['other'][0]}</td><td>{R['other'][1]:.2f}</td><td>{100 * R['other'][1] / R['total']:.1f} %</td>" for R in Rs)
    h.append(f"<tr><th>纵向合计</th><th>{sum(r[2] for r in P['rows'])}</th><th>{lo:.2f}</th><th>{100 * lo / P['total']:.1f} %</th>{gt}</tr>")
    h.append(f"<tr><td>非纵向（转弯 / 变道 / 碰撞 / 出路 / 其余轻微丢分）</td><td>{P['other'][0]}</td><td>{P['other'][1]:.2f}</td><td>{100 * P['other'][1] / P['total']:.1f} %</td>{go}</tr></table>")
    h.append("<p class=m>丢分（点）= 该类 scene 的 (1 − 分) 之和 / 281 × 100。主类按 rf_tax.py 的规则，所有读数都用日志里的交通参与者分类；「慢」类里被追尾的单列。每类下面只挂丢了分的 scene，按丢分从大到小、尽量分散在不同日志地点。</p>")
    vl = P.get("vs_label") or "ref"
    nz = [p for p in P["picks"] if p["cls"] == "newzero"]
    fx = P.get("fixed") or []
    if nz or fx:
        h.append(f"<h2 id=newzero>{html.escape(lab)} 新造的零分（{html.escape(vl)} 非零）</h2><p>{html.escape(vl)} 非零、{html.escape(lab)} 为零的 {len(nz)} 个 scene。每段标：{html.escape(vl)} 分 → {html.escape(lab)} 分，零分类型，事件时刻本车与人的速度。</p>")
        for i, p in enumerate(nz, 1):
            f = f"{a.videos}/newzero/{i:02d}_{p['scene'][7:15]}_{p['ref']:.2f}to{p['score']:.2f}.mp4"
            h.append(f"<div class=c>{vid(f, p)}<div class=m><b>{p['scene'][7:15]}</b> {html.escape(vl)} {p['ref']:.2f} → {html.escape(lab)} {p['score']:.2f}；零分类型 {html.escape(str(p['zero_type']))}"
                     f"（{p['primary']}）；事件 t = {p['t']:.1f} s（{p['why']}）；该时刻本车 {p['v_e']} m/s，人 {p['v_h']} m/s</div></div>")
        h.append(f"<h2 id=fixed>{html.escape(lab)} 修好的零分（{html.escape(vl)} 为零，{html.escape(lab)} 非零）</h2><p>共 {len(fx)} 个，不配视频：</p><table><tr><th>scene</th><th>{html.escape(vl)} 分 / 零分类型</th><th>{html.escape(lab)} 分</th><th>{html.escape(lab)} 主类</th></tr>")
        for p in fx:
            h.append(f"<tr><td>{p['scene'][7:15]}</td><td>{p['ref']:.2f} / {html.escape(str(p['ref_zero_type']))}（{p['ref_primary']}）</td><td>{p['score']:.2f}</td><td>{p['primary']}</td></tr>")
        h.append("</table>")
    for k, l, w in CLASSES:
        ps = [p for p in P["picks"] if p["cls"] == k]
        h.append(f"<h2 id={k}>{html.escape(l)}</h2><p>{html.escape(w)}</p>")
        if not ps:
            h.append("<p class=m>本次读数里这一类没有 scene。</p>")
        for i, p in enumerate(ps, 1):
            f = f"{a.videos}/{k}/{i:02d}_{p['scene'][7:15]}_{p['score']:.2f}.mp4"
            h.append(f"<div class=c>{vid(f, p)}<div class=m><b>{p['scene'][7:15]}</b> 得分 {p['score']:.2f}（丢 "
                     f"{p['loss']:.2f} 点）；事件 t = {p['t']:.1f} s（{p['why']}）；该时刻人 {p['v_h']} m/s，本车 {p['v_e']} m/s，本车落后日志 {p['lag']:.0f} m；"
                     f"主类 {p['primary']}{'+' + p['secondary'] if p['secondary'] else ''}</div></div>")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "index.html").write_text("\n".join(h) + ("\n" if a.novideo else "\n" + SCRIPT + "\n") + "</body></html>")


def clips(a):
    import pickle
    import shutil
    O = {o["scene"]: o for o in pickle.load(open(a.pkl, "rb"))["rolls"]}
    P = json.loads(Path(a.picks).read_text())
    cnt = {}
    for p in P["picks"]:
        k = p["cls"]
        cnt[k] = cnt.get(k, 0) + 1
        name = (f"{cnt[k]:02d}_{p['scene'][7:15]}_{p['ref']:.2f}to{p['score']:.2f}.mp4" if k == "newzero" else f"{cnt[k]:02d}_{p['scene'][7:15]}_{p['score']:.2f}.mp4")
        src = (O[p["scene"]].get("files") or {}).get("mp4")
        if not src:
            print("NO VIDEO", p["scene"], file=sys.stderr)
            continue
        dst = Path(a.out) / a.videos / k / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pick")
    p.add_argument("--pkl", required=True), p.add_argument("--csv", required=True), p.add_argument("--out", required=True)
    p.add_argument("--per", type=int, default=5)
    p.add_argument("--label", default=""), p.add_argument("--vs-csv"), p.add_argument("--vs-label", default="")
    c = sp.add_parser("clips")
    c.add_argument("--pkl", required=True), c.add_argument("--picks", required=True), c.add_argument("--out", required=True), c.add_argument("--videos", default="clips")
    g = sp.add_parser("page")
    g.add_argument("--picks", required=True), g.add_argument("--out", required=True), g.add_argument("--title", required=True)
    g.add_argument("--ref", nargs="*"), g.add_argument("--novideo", action="store_true")
    g.add_argument("--intro", default=""), g.add_argument("--videos", default="clips")
    a = ap.parse_args()
    {"pick": pick, "page": page, "clips": clips}[a.cmd](a)
