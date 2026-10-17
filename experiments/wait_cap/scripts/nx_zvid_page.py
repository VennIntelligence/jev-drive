#!/usr/bin/env python3
"""Index page of the RTB1 shared new-zero scenes (held 460 @18). Usage: nx_zvid_page.py DIR  (DIR holds <scene8>_<tag>.mp4; writes manifest.json + index.html)."""
import json, subprocess, sys
from pathlib import Path

D = Path(sys.argv[1])
TAGS = [("RTB1O-L18-s0", "a: RTB1O-L18-s0 (零分)"), ("RSG-L18-s0", "参照 RSG-L18-s0"), ("RTB1OB1-L18-s1", "d: RTB1OB1-L18-s1 (另一基座种子)")]
# scene -> (log, zero class of arm a, RSG-L18-s0 score, atfault row / note)
S = [
 ("de4ace24443b5b2d", "2021.06.28.16.57.59_veh-26", "碰撞 (at-fault)，本车同向、跟在低速车后起步", 0.482,
  "at-fault，同向车 2.6 m/s，本车 2.4 → 1.9 m/s 接触，缓慢起步时并线；接触 6.5 s；距离人类 +0.15 m。RSG 同场景 0.48 分（未清零）。"),
 ("6d2b59f78d995d9c", "2021.09.09.18.04.06_veh-40", "碰撞 (at-fault)，本车起步加速撞上同向车", 1.0,
  "at-fault，同向车 9.2 m/s，本车 4.5 → 8.4 m/s 起步加速撞入；接触 6.0 s；偏离人类路径 -0.77 m，本车道内。"),
 ("9d1d720d0e2e511e", "2021.09.09.18.04.06_veh-40", "offroad", 1.0, "非 at-fault 场景（出路），atfault.md 无此行。"),
 ("9a675656c3c85f4c", "2021.09.16.19.49.00_veh-42", "碰撞 (at-fault)，对向车道正面相撞", 1.0,
  "at-fault，被撞车航向 147°（head-on），本车在对向车道，偏离人类路径 3.0 m；RSG 同一里程留在本车道。接触 6.0 s。"),
 ("53d3787367c75240", "2021.06.28.16.29.11_veh-38", "left_corridor_laterally（偏出路线走廊）", 1.0, "非碰撞场景，atfault.md 无此行。"),
 ("ddc1271ea57154bc", "2021.10.06.08.16.17_veh-52", "offroad", 1.0, "非 at-fault 场景（出路），atfault.md 无此行。"),
 ("1f54430427975391", "2021.10.06.08.16.17_veh-52", "offroad（另 b 臂同时判 left_corridor_laterally）", 1.0, "非 at-fault 场景（出路），atfault.md 无此行。"),
]


def slow(f):
    """Playback slow-down from the clip's frame rate and real duration: sparse decision clips (<= 5 fps) or short scenes (< 15 s) 3x, else 2x."""
    try:
        r, d = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate:format=duration",
                              "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout.split()
        n, m = r.split("/"); fps, dur = float(n) / float(m), float(d)
    except (ValueError, OSError):
        return 3
    return 3 if fps <= 5 or dur < 15 else 2


man, secs = [], []
for sid, log, cls, rsg, note in S:
    cells = []
    for t, lab in TAGS:
        f = D / f"{sid[:8]}_{t}.mp4"
        if f.exists():
            k = slow(f)
            man.append(dict(scene=sid, tag=t, file=f.name, mb=round(f.stat().st_size / 1e6, 1), slow=k))
            cells.append(f'<figure><video src="{f.name}" data-rate="{1 / k:.4f}" autoplay muted loop playsinline controls preload="metadata"></video>'
                         f'<figcaption>{lab}（放慢 {k}×）</figcaption></figure>')
        else:
            cells.append(f'<figure class="miss"><figcaption>{lab}：无视频</figcaption></figure>')
    secs.append(f'<section><h2>{sid[:8]} <small>{sid}</small></h2><p>日志：{log}</p>'
                f'<p>臂 a 零分原因：<b>{cls}</b>；RSG-L18-s0 同场景得分 {rsg:.2f}</p><p>{note}</p><div class="row">{"".join(cells)}</div></section>')
(D / "manifest.json").write_text(json.dumps(man, indent=1))
(D / "index.html").write_text(f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>RTB1 共同新增零分场景</title>
<style>body{{font:15px/1.5 -apple-system,sans-serif;max-width:1500px;margin:1em auto;padding:0 1em}}small{{color:#888;font-weight:400;font-size:12px}}
.row{{display:flex;gap:8px;flex-wrap:wrap}}figure{{margin:0;flex:1 1 420px;max-width:480px}}video{{width:100%;background:#000}}figcaption{{font-size:13px;color:#555}}
section{{border-top:1px solid #ccc;padding-top:.5em;margin-top:1.5em}}.miss{{color:#a00}}</style></head><body>
<h1>RTB1 四个臂共同新增的 7 个零分场景</h1>
<p>held 460 @18 决策、单次读数（seed 0 为主）。这 7 个场景在 RTB1 的四个臂（a RTB1O-L18-s0、b RTB1N-L18-s0、c RTB1O-L18-s1、d RTB1OB1-L18-s1）里全部得零，
而参照 RSG-L18（s0 / s1）都没有清零。每节并排放臂 a、RSG-L18-s0 和臂 d（另一基座种子）的回放（上为 BEV 与打分面板，下为前视相机，约 2 帧/秒、10 秒，每帧一个决策）；视频是用同一 sh30 路径加视频重跑得到的，不是原读数的录像。
播放速度按帧率与实际时长放慢 2–3 倍（每段标注；本页的片段 2 帧/秒、10 秒，都是 3×）。先看零分发生的时刻，再看 RSG 在同一位置怎么处理。数字来源：<code>rtb_cont.md</code>、<code>rtb_cont/atfault.md</code>。</p>
{"".join(secs)}
<script>document.querySelectorAll("video[data-rate]").forEach(v=>{{const r=+v.dataset.rate;const set=()=>{{v.defaultPlaybackRate=r;v.playbackRate=r;}};
set();v.addEventListener("loadedmetadata",set);v.addEventListener("play",set);}});</script></body></html>''')
print(len(man), "videos", sum(m["mb"] for m in man), "MB")
