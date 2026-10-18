#!/usr/bin/env python3
"""Index page of RTB1 zero / low-score example scenes (held 460 @18, public 1491 @10) next to RSG-L18-s0.
Usage: nx_zvid_page.py DIR  (DIR holds <scene8>_<tag>.mp4 and selection.json from the box-side enrich step; writes manifest.json + index.html)."""
import html, json, subprocess, sys
from pathlib import Path

D = Path(sys.argv[1])
SEL = json.loads((D / "selection.json").read_text())
TAGS = {"a": ("RTB1O-L18-s0", "a: RTB1O-L18-s0"), "R": ("RSG-L18-s0", "参照 RSG-L18-s0"), "d": ("RTB1OB1-L18-s1", "d: RTB1OB1-L18-s1 (另一基座种子)")}
CLS = {  # class key -> label (group, class)
 "atfault_front": "碰撞 (at-fault)，正面 / 追尾前方物体", "atfault_lateral": "碰撞 (at-fault)，侧向", "atfault_rear": "碰撞 (at-fault)，后部", "hit_notatfault": "被碰撞 (非 at-fault)",
 "offroad": "出路 (offroad)", "corridor": "偏出路线走廊 (left_corridor_laterally)", "never_launch": "不起步 (走了不到 2 m)",
 "progress_barely_moved": "低分：进度几乎为零 (只走了人类里程的 <10%)", "progress_crawl": "低分：爬行 (人类里程的 10-30%)", "progress_slow": "低分：偏慢 (人类里程的 >=30%)", "extra": "补充场景"}
GROUPS = [("zero_only", "B. 零分：RTB1 独有 (RSG-L18-s0 同场景不为零)"), ("low", "C. 低分：RTB1 < 0.5，或比 RSG 低 >= 0.3 (均不为零)"), ("zero_both", "D. 参照：RTB1 与 RSG 都为零"),
          ("extra", "E. 补充：按经典错误类别指定的场景 (不属于比例抽样，见标签)")]
TAGS_ = {"failed_turn": "转弯失败 (未完成 / 出路 / 撞)", "never_launch": "不起步 (全程停在原地)", "late_launch": "晚起步 (起步迟)",
         "slowdown": "渐进减速 (后 30% 平均车速不到前 30% 的一半)", "straight_offroad": "直路出路 / 漂出本车道"}
TURN = {"L": "左转", "R": "右转", "straight": "直行 / 无转弯", "?": "转弯未评估"}
# the 7 first scenes (held, shared by all four RTB1 arms): notes from rtb_cont/atfault.md
NOTES = {
 "de4ace24443b5b2d": "at-fault，同向车 2.6 m/s，本车 2.4 → 1.9 m/s 接触，缓慢起步时并线；接触 6.5 s；距离人类 +0.15 m。RSG 同场景 0.48 分（未清零）。",
 "6d2b59f78d995d9c": "at-fault，同向车 9.2 m/s，本车 4.5 → 8.4 m/s 起步加速撞入；接触 6.0 s；偏离人类路径 -0.77 m，本车道内。",
 "9a675656c3c85f4c": "at-fault，被撞车航向 147°（head-on），本车在对向车道，偏离人类路径 3.0 m；RSG 同一里程留在本车道。接触 6.0 s。",
}
EXISTING = {"de4ace24443b5b2d", "6d2b59f78d995d9c", "9d1d720d0e2e511e", "9a675656c3c85f4c", "53d3787367c75240", "ddc1271ea57154bc", "1f54430427975391"}


def speed(f):
    """Fast-forward factor from the clip's frame rate and real duration: aim at ~30 fps (10 Hz -> 3x), never above 3x, and keep at least ~6 s
    on screen so sparse short clips (2 Hz, 10 s -> 1.7x) do not flash by."""
    try:
        r, d = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate:format=duration",
                              "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout.split()
        n, m = r.split("/"); fps, dur = float(n) / float(m), float(d)
    except (ValueError, OSError):
        return 2.0
    return round(max(2.0, min(3.0, 30.0 / fps, dur / 6.0)), 2)


fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
man = []


def card(o):
    s = o["scene"]; sid = s[-16:]; log = s.rsplit("-", 1)[0]; kind = o["kind"]
    ks = ["a", "R", "d"] if kind == "held" else ["a", "R"]
    cells = []
    for k in ks:
        t, lab = TAGS[k]; f = D / f"{sid[:8]}_{t}.mp4"
        rr, rc = o["rerun"].get(k), o["rerun_cls"].get(k)
        cap = f"{lab}（重跑得分 {fmt(rr)}{'，' + CLS.get(rc, rc) if rc and rc != 'progress' else ''}）"
        if f.exists():
            v = speed(f)
            man.append(dict(scene=s, tag=t, file=f.name, mb=round(f.stat().st_size / 1e6, 2), speed=v))
            cells.append(f'<figure><video data-src="{f.name}" data-rate="{v}" muted loop playsinline controls preload="none"></video><figcaption>{html.escape(cap)}（{v:g}× 快放）</figcaption></figure>')
        else:
            cells.append(f'<figure class="miss"><figcaption>{html.escape(lab)}：无视频</figcaption></figure>')
    z = [f"{'a' if a == 'a' else a}={fmt(o['orig'].get(a))}" for a in ks]
    src = "held 460 @18 (重跑同配置)" if kind == "held" else "public 1491 @10 (重跑按 public 的 10 步配置)"
    turn = TURN.get(o["turn"], o["turn"]) + (f"（路线转弯结果：{o['turn_outcome']}）" if o.get("turn_outcome") and o["turn"] in ("L", "R") else "")
    repro = "" if o["rerun"].get("a") is not None and abs((o["rerun"]["a"] or 0) - (o["orig"].get("a") or 0)) < 0.05 else \
        ' <span class="warn">重跑得分与原读数不同（rerun 非逐位复现），以原读数为准</span>'
    cls = CLS.get(o["cls"], o["cls"]); note = NOTES.get(sid, "")
    chips = "".join(f'<span class="chip">{TAGS_[t]}</span>' for t in o.get("tags", []))
    outs = o.get("turn_outs"); tp = (f"<p>转弯结果 (rf_follow)：a={outs.get('a', '-')}，d={outs.get('d', '-')}，RSG={outs.get('R', '-')}</p>" if outs else "")
    return (f'<section id="s{sid[:8]}"><h3>{sid[:8]} <span class="chips">{chips}</span> <small> {html.escape(s)}</small></h3><p>来源：{src}；<b>{cls}</b>；转弯类型：<b>{turn}</b></p>'
            f'<p>原读数得分：a(RTB1O)={fmt(o["orig"].get("a"))}，RSG={fmt(o["orig"].get("R"))}'
            f'{"，d(RTB1OB1)=" + fmt(o["orig"].get("d")) if kind == "held" else ""}{repro}</p>'
            f'{tp}{"<p>" + note + "</p>" if note else ""}<div class="row">{"".join(cells)}</div></section>')


sel = SEL["sel"]
A = [o for o in sel if o["scene"][-16:] in EXISTING]
secs = [f'<h2 id="A">A. held：7 个共同新增零分 (已有)</h2>' + "".join(card(o) for o in A)]
nav = ['<a href="#A">A. held: 7 shared zeros</a>']
for g, title in GROUPS:
    os_ = [o for o in sel if o["group"] == g and o["scene"][-16:] not in EXISTING]
    if not os_:
        continue
    nav.append(f'<a href="#{g}">{title.split("：")[0]}</a>')
    body = f'<h2 id="{g}">{title}</h2>'
    for c in CLS:
        oc = [o for o in os_ if o["cls"] == c]
        if oc:
            body += f"<h3 class='cls'>{CLS[c]}（本节 {len(oc)} 个）</h3>" + "".join(card(o) for o in sorted(oc, key=lambda o: (not o["held"], o["scene"])))
    secs.append(body)

# top table: pool vs shown (unique scenes)
rows = []
for g, title in GROUPS:
    for c in CLS:
        n = SEL["pool"].get(g, {}).get(c, 0)
        if n:
            sh = sum(1 for o in sel if o["group"] == g and o["cls"] == c)
            rows.append(f"<tr><td>{title.split('：')[0]}</td><td>{CLS[c]}</td><td>{n}</td><td>{sh}</td></tr>")
    if g == "extra":
        rows.append(f"<tr><td>E</td><td>补充场景</td><td>-</td><td>{sum(1 for o in sel if o['group'] == g)}</td></tr>"); continue
    tot = sum(SEL["pool"].get(g, {}).values()); shn = sum(1 for o in sel if o["group"] == g)
    rows.append(f"<tr class='tot'><td>{title.split('：')[0]}</td><td>合计</td><td>{tot}</td><td>{shn}</td></tr>")
tagrows = "".join(f"<tr><td>{TAGS_[t]}</td><td>{sum(t in o.get('tags', []) for o in sel)}</td><td>" + " ".join(
    f'<a href="#s{o["scene"][-16:][:8]}">{o["scene"][-16:][:4]}</a>' for o in sel if t in o.get("tags", [])) + "</td></tr>" for t in TAGS_)
(D / "manifest.json").write_text(json.dumps(man, indent=1))
total = sum(m["mb"] for m in man)
(D / "index.html").write_text(f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>RTB1 零分 / 低分示例场景</title>
<style>body{{font:15px/1.5 -apple-system,sans-serif;max-width:1500px;margin:1em auto;padding:0 1em}}small{{color:#888;font-weight:400;font-size:12px}}
.row{{display:flex;gap:8px;flex-wrap:wrap}}figure{{margin:0;flex:1 1 420px;max-width:480px}}video{{width:100%;background:#000}}figcaption{{font-size:13px;color:#555}}
section{{border-top:1px solid #ccc;padding-top:.5em;margin-top:1.5em}}.miss{{color:#a00}}.chip{{background:#fed;border:1px solid #ca8;border-radius:8px;padding:0 6px;font-size:12px;margin-left:4px;font-weight:400}}.warn{{color:#a60}}h3.cls{{background:#eef;padding:2px 8px;margin-top:2em}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:2px 10px;text-align:left}}tr.tot{{font-weight:600;background:#f4f4f4}}nav a{{margin-right:1.2em}}#rb{{position:sticky;top:0;z-index:9;background:#fffffff0;padding:5px 0;border-bottom:1px solid #ddd;font-size:14px}}#rb button{{margin:0 2px;padding:1px 7px;border:1px solid #999;border-radius:4px;background:#fff}}#rb button.on{{background:#333;color:#fff}}#rb input{{vertical-align:middle;width:140px}}</style></head><body>
<div id="rb">播放速度 <button data-r="0">自动</button><button data-r="1">1×</button><button data-r="2">2×</button><button data-r="3">3×</button><button data-r="5">5×</button><button data-r="10">10×</button><button data-r="16">16×</button> <input type="range" min="0.5" max="16" step="0.5"> <b id="rv"></b></div>
<h1>RTB1 的零分与低分长什么样：{len(sel)} 个示例场景（对照 RSG-L18-s0）</h1>
<p>目的：判断 RTB1（retain-base 线）与 RSG-L18 比，零分和低分的真实构成。每个场景并排放臂 a (RTB1O-L18-s0)、参照 RSG-L18-s0，held 场景再加臂 d (RTB1OB1-L18-s1，另一基座种子)；
每段上为 BEV 与打分面板，下为前视相机，约 2 帧/秒，每帧一个决策。视频是用同一 sh30 路径加视频<b>重跑</b>得到的，不是原读数的录像（同一输入，但每段标题注明重跑得分；与原读数不一致的场景会标出）。
held 场景按 held 460 @18 配置重跑，public 场景按 public 1491 @10 的 10 步配置重跑。</p>
<p><b>抽样方式。</b>池 = RTB1 独有零分（RSG-L18-s0 同场景不为零；held 取臂 a、d，public 取臂 a，按场景去重）与 RTB1 低分（RTB1 得分 &lt; 0.5，或比 RSG 同场景低 &gt;= 0.3）。
各类按池中占比比例抽样（最大余数法，种子 0；已有的 7 个共同零分场景固定在内）；另留少量「双方皆零」作参照；E 组是为保证经典错误类别（转弯失败、渐进减速等）至少有几例而额外指定的场景，不按比例。总共 {len(sel)} 个场景，所有视频合计 {total:.0f} MB。
本评测的得分 = min(clamp(progress)/0.8, 1)，再被 at-fault 碰撞 / 出路 / 偏出走廊清零；<b>没有舒适度或 TTC 分量</b>，所以低分只有「进度损失」一种成本，这里按相对人类里程的进度细分。
转弯类型来自路线（70 度以上的路线转角，规则同 rf_follow）；「转弯未评估」表示该场景没有可用的路线读数。</p>
<table><tr><th>组</th><th>类别</th><th>全池场景数</th><th>页面展示</th></tr>{"".join(rows)}</table>
<table><tr><th>经典错误类别 (可叠加；由速度曲线 / 转弯读数 / 类别自动打标)</th><th>页面场景数</th><th>场景</th></tr>{tagrows}</table>
<p>视频按帧率与实际时长快放（最多 3 倍，每段标注），滑到视野里才加载、播放，滑出即暂停。</p>
<nav>{"".join(nav)}</nav>
{"".join(secs)}
<script>
// Speed bar: 0 = per-clip auto rate (data-rate); any other value overrides every clip. Kept in localStorage across pages.
let G = +localStorage.vidRate || 0;
const R = v => {{ const r = G || +v.dataset.rate; v.defaultPlaybackRate = r; v.playbackRate = r; }};
const setG = r => {{ G = r; localStorage.vidRate = r;
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
print(len(sel), "scenes", len(man), "videos", round(total, 1), "MB")
