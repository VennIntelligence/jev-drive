"""HTML page of the LOSS-MAP (research/round7/loss_diag/index.html): the class table, the ranking, and one card per clip of loss_videos.py.
Numbers come from experiments/round7/results/loss_map.json; the lever / cost / probability columns and the captions are judgments written
here after reading the result docs and watching the clips (CAP)."""
from __future__ import annotations

import html

N = 281
PCS_PER_PT = 3.0          # d426: board PCS per local 281 scene point (0.6 x the local mean gap, ~1 400 PCS per mean)

# key -> (rank, lever, cost, p, note); p = rough chance the lever recovers the demonstrated-recoverable points in one iteration
PLAN = {
    "off-turn-in": (1, "DAgger 地图横向教师第 2 次迭代（在跑：学生 = R6DAG，入弯 / 超速行标签速度取 min(学生, 人)）；必要时第 3 次", "第 2 次已付；第 3 次约 1 h 建行 + 10 GPU job-h", 0.5,
                    "两个种子 7 个共同失败，全部有我们的读数拿到过；种子差（s0 出路 20 对 s1 12）主要在这一类"),
    "slow": (2, "数据卫生 F2（去掉无前车自发减速不停的 PAI 行，FIX 上已见 bbc53aa8 两个种子都回到 1.0）+ DAgger 行的纵向「不无故减速」标签（无前车、人保持速度时弧长取 max(学生, 人)）", "约 1–2 h 代码，搭第 3 次迭代训练，无额外卡时", 0.5,
             "均分上几乎不值钱，但 s1 在尺子上剩下的 6 个高区分度不满分场景是这里 3 个 + 近满分 2 个 + 行驶中被追尾 1 个（d90fe123，同一个慢习惯）；NTV0 / R2C 在这 6 个上全是 1.0"),
    "near1": (2, "同上（与「偏慢」同一个杠杆）", "同上", 0.5, "0.98–0.99 的微损；zoib 尺子把高区分度题上的非 1 分当失败"),
    "drift": (3, "指令规则改看路线航向变化（d432，从存档特征重算，不重建行）+ DAgger 横向教师", "约 2–3 h 代码 + 闭环小测；训练搭第 3 次迭代", 0.3,
              "28–38 m/s 的快速路上向左离开车道；4 个两种子共同，NTV0 只解 1–2 个"),
    "off-straight": (4, "DAgger 横向教师（同 1）", "同 1", 0.3, "NTV0 多数也失败；种子噪声大（共同 1 / 任一 5）"),
    "off-junc-straight": (4, "DAgger 横向教师（同 1）", "同 1", 0.3, "路口内直行段压线；NTV0 也失败"),
    "off-turn-out": (4, "DAgger 横向教师（同 1）", "同 1", 0.2, "没有任何读数解过 2 / 3；s1 只剩 1 个"),
    "overshoot": (5, "停止线纵向教师（DAgger 行上标签弧长取 min(学生, 到人停车点的距离)）；但 ACC + GAP 会把平缓减速按巡航速度送出（d406）", "约 3–4 h 代码与闸 + 10 GPU job-h；须读原生与 ACC + GAP 两种", 0.2,
                  "最大的一块（19–20 分），但在 ACC + GAP 下我们没有任何读数解过其中 19 个；4 个只在别的 serve 下解过（2 个原生 VC + LEAD、1 个只加 ACC 不加 GAP、1 个加 keywarp），15 个我们任何读数都没解过；尺子区分度≈0（全族一起失败）；历史上修停车都付了起步 / 追尾（d400 / d402 / d412）"),
    "rear-move": (6, "无专门杠杆；无故减速那一部分随 2", "—", 0.1, "多数在人的速度上被回放后车追尾（不反应的回放车），部分不可解"),
    "rear-launch": (7, "无（d419–d428 起步线已关；起步 35 / 37）", "—", 0.1, "起步慢一点就被不反应的回放后车追尾；NTV0 也一样"),
    "coll-front": (8, "无现成；PAI 行碰撞 hinge 未核实", "约 4 h + 10 GPU job-h", 0.15, "种子噪声为主（共同 2 / 任一 5）"),
    "coll-side": (8, "同上", "同上", 0.15, "共同 0 / 任一 5：全是种子噪声"),
    "route-turn": (9, "已修（NTV0 15 → 2），剩下 2 个没有任何读数解过", "—", 0.0, "—"),
}

# scene8 -> (what to look at and when, mechanism guess); written after watching the clips
CAP: dict[str, tuple[str, str]] = {
    "ea468790": ("停止线起步右转进多车道路。5–8 s 两车同速起步、同样右转；8–10 s 出弯时 R6DAG 的橙线一直贴右侧，不回到车道中间，10.3 s 以约 6.9 m/s 压上右侧路沿；NTV0 转完回正、居中。",
                 "出弯不回正：计划在入弯阶段就偏向弯内，转完仍沿内侧走；与第 5 轮 MV 系同一场景同形（第 431 条），DAgger 第 1 次没修掉。"),
    "b680a689": ("夜间窄街路口左转。7.6 s 起 R6DAG 的转弯弧比 NTV0 更早、更急（8.6 s 先压到对向车道线），9.6 s 车身左前角压上弯内路沿；NTV0 同速（7.3 m/s）走更大的弧。",
                 "转弯切内侧：弯道几何读成更急的弧，入弯点提前；速度相同，问题在横向形状。"),
    "bbc53aa8": ("高速路 35 m/s。7–10 s 左侧车道一辆车并到前方，两车都减速；NTV0 只降到约 27.5 m/s 就跟车回到 35；R6DAG 一路降到约 3 m/s（16 s），同时贴向右侧车道边，前车走远了也不回速。",
                 "无故减速：对并线车过度反应后计划一直收短（第 431 条「计划收短后右漂」的同一习惯），来自 PAI 人类行；NTV0 / R2C 不会，R6FIX（去掉无前车减速行 F2）两个种子都是 1.0。"),
    "d90fe123": ("多车道大路口左转。8–10 s R6DAG 在入口前把计划收短、降到 2.4 m/s（NTV0 保持约 8 m/s 直接转），13.3 s 转弯中被后车追尾，得 0.65。",
                 "入路口犹豫减速（无前车），被不反应的回放后车撞上；与「偏慢」同一个慢习惯。这个场景在尺子上 a = 1.96（我们族都满分）。"),
    "4cb05d65": ("夜间快速路 38.5 m/s，几乎看不清车道线。3–9 s 路线（绿虚线）轻微右弯，R6DAG 的计划一直是直线，车逐渐落到路线左边一个车道，8.5 s 离开走廊；NTV0 同速贴着路线走。",
                 "高速下不跟路线的缓弯（向左离开，与第 432 条「漂移零分 37 / 41 向左」同向）；夜间画面读不到车道线时更依赖路线输入，而路线指令语义在 PAI 行里不一致。"),
    "127c4f85": ("白天直路接近红灯路口，无前车。6–9 s 车速 10.8 → 9.2 m/s 几乎不减，计划橙线一直穿过停止线（红虚线）；10–12 s 才急刹，停在停止线之后的人行横道里，越过人类的停车点。",
                 "停车太晚：计划不含停车点，直到很近才刹（第 404 条的惯性：计划随速度前移）；ACC 把早期的平缓减速按巡航速度送出。"),
    "47db2eb6": ("夜间接近 T 型路口红灯。左 R6DAG（ACC + GAP）10.5 s 还有 8.8 m/s、计划穿过停止线，13.5 s 已压在停止线上才降到 4.5 m/s，随后右转进路口；右 NTV0 不加 ACC 的原生读数整段只开 3–3.7 m/s，场景结束时还在接近路段上。",
                 "原生 NTV0「解」这个场景靠的是慢（还没走到停车点场景就结束了），不是看见红灯停下；所以这一类在提交 serve 下没有现成的正例可学。"),
    "cbb22323": ("拥堵车流里的起步：两车前 10 s 一样慢；12–13 s 前方车队挪动时 R6DAG 只走 0.1 m/s（NTV0 0.4），与前车拉开空档，14.9 s 后车追尾。",
                 "排队起步跟得不够紧，不反应的回放后车按人的轨迹前进就撞上；差别只有零点几 m/s。"),
    "4f7cb072": ("山路 22 m/s，场景一开始前方本车道就有一辆慢卡车。1–2 s 计划向右甩、试图从右侧绕开，2.7 s 冲上右侧路肩 / 护栏。",
                 "为避前车宁可开出路面（用户 10-21 提的「躲撞撞路边」的一个实例）；计划不会跟车减速，横向避让又不看路沿；NTV0 也零分。"),
    "a5599793": ("夜间小环岛，停车让行后起步（约 3 m/s）。3–6 s 计划直指中心岛，没有沿环道向右绕，6.6 s 压上中心岛。",
                 "路口内直行段：把环岛读成直行路口，路线（右绕）没进计划；所有读数都失败，属于少见几何。"),
    "7381c9fa": ("从窄巷起步左转汇入街道（3–5 m/s），转弯半径过大，3.9 s 车身外侧压上对侧路沿，随后在路上。",
                 "转弯外侧过宽：低速大角度转弯的弧不够紧；NTV0 也失败，没有读数解过。"),
    "268949d2": ("起步右转后上直路。7–10 s 两车同样右转；10–12 s R6DAG 贴右侧走、并减速到 1.7 m/s，右侧的骑车人赶上来，12.1 s 侧面相撞；NTV0 居中、6 m/s，骑车人从旁边过去。",
                 "贴右 + 减速，把右侧留给骑车人的空间压没了；只在 s1 发生（s0 是被追尾部分分 0.61），种子噪声成分大。"),
    "3ce13eed": ("双车道同时右转：起步后 2–4 s 左侧车道的白车也右转并切到本车前方，R6DAG 照自己的弧转，5.4 s 前撞。",
                 "不让同向并行转弯的车；NTV0 与所有读数也都零分，回放车不让人，属难题。"),
    "302c5c99": ("黄昏大路口，路线（绿线）是掉头（向左绕回）。R6DAG 一直直行，8.9 s 走廊判路线该转未转。",
                 "掉头路线：路线指令在 42 m 处的横向偏移规则读不出掉头；没有任何读数解过（剩下的 2 个路线转零分之一）。"),
}


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def table(LM):
    C = LM["classes"]
    rows = []
    keys = sorted((k for k in C if k in PLAN), key=lambda k: (PLAN[k][0], list(PLAN).index(k)))
    for k in keys:
        c, (rank, lever, cost, p, note) = C[k], PLAN[k]
        a0, a1 = c["R6DAG-s0"], c["R6DAG-s1"]
        if a0["scenes"] + a1["scenes"] == 0:
            continue
        rec = (a0["ag_rec_pts"] + a1["ag_rec_pts"]) / 2
        gain = rec * p
        rows.append(
            f"<tr><td>{rank}</td><td><b>{html.escape(c['label'])}</b></td>"
            f"<td>{a0['scenes']} / {a0['zeros']} / {a0['pts']:.1f}</td><td>{a1['scenes']} / {a1['zeros']} / {a1['pts']:.1f}</td>"
            f"<td>{(a0['pts'] + a1['pts']) / 2 / N:.3f}</td>"
            f"<td>{c['both']} / {c['union']}</td>"
            f"<td>{a0['ntv0_solves']} / {a1['ntv0_solves']}</td><td>{a0['ag_solves']} / {a1['ag_solves']}（{a0['ag_rec_pts']:.1f} / {a1['ag_rec_pts']:.1f} 分）</td>"
            f"<td>{a0['never']} / {a1['never']}</td>"
            f"<td>{a0['imperfect_a1']} / {a1['imperfect_a1']}；带内满分率 {f2(a1['band_pass'] if a1['band_pass'] is not None else a0['band_pass'])}</td>"
            f"<td>{html.escape(lever)}</td><td>{html.escape(cost)}</td>"
            f"<td>{rec:.1f} × {p:.1f} = <b>{gain:.1f}</b> 分（均分 +{gain / N:.3f}，榜约 +{gain * PCS_PER_PT:.0f} PCS）</td>"
            f"<td class='note'>{html.escape(note)}</td></tr>")
    return ("<table><thead><tr><th>排名</th><th>失分类别</th><th>s0 场景 / 零分 / 丢分</th><th>s1 场景 / 零分 / 丢分</th><th>均分当量（两种子平均）</th>"
            "<th>两种子都丢 / 任一丢</th><th>NTV0 解（s0 / s1 的失分场景里）</th><th>我们任一 ACC+GAP 读数解（可收回分）</th><th>从未有读数解（任何 serve）</th>"
            "<th>尺子：a ≥ 1 的不满分场景（s0 / s1）</th><th>杠杆</th><th>成本</th><th>估计收益 = 可收回 × 概率</th><th>备注</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def card(k, s8, c, LM):
    cls = LM["classes"][c["cls"]]
    look, mech = CAP.get(s8, ("", ""))
    sides = "；".join(
        f"{'左' if i == 0 else '右'}：{s['tag'].replace('-L18', '')}{'（原生 VC + LEAD，不加 ACC / GAP）' if s['serve'] == 'native' else ''} 得分 {s['rerun']:.2f}"
        + ("（存档读数的原 rollout，离线渲染）" if s["src"] == "stored" else "（重读）" if s["src"].startswith("rerun") else "（第 5 轮页的重读视频）")
        for i, s in enumerate(c["sides"]) if True) if c["n"] > 1 else (
        f"{c['sides'][0]['tag'].replace('-L18', '')} 得分 {c['sides'][0]['rerun']:.2f}（存档读数的原 rollout，离线渲染）")
    sc = cls["scenes"].get(s8, {})
    other = f"其他读数：R6DAG-s0 {f2(sc.get('R6DAG-s0'))} / R6DAG-s1 {f2(sc.get('R6DAG-s1'))} / NTV0 {f2(sc.get('ntv0'))} / 我们 ACC+GAP 最好 {f2(sc.get('best_ag'))}（{sc.get('best_ag_read', '-')}）；尺子区分度 a = {f2(sc.get('a'))}"
    w = c["clip"]
    return (f'<section id="{s8}"><h3>{k}. {html.escape(cls["label"])} <code>{s8}</code></h3>'
            f'<p>{sides}。事件时刻 {c["t_ev"]:.1f} s（rollout 时钟）。{other}。</p>'
            f'<p class="cap"><b>看：</b>{html.escape(look)}<br><b>机理猜测：</b>{html.escape(mech)}</p>'
            f'<figure><video data-src="{c["file"]}" data-rate="{c["rate"]}" muted loop playsinline controls preload="none" '
            f'style="aspect-ratio:{1288 if c["n"] > 1 else 640}/1000"></video>'
            f'<figcaption><span class="rate">{c["rate"]:g}×</span> 片段 {w.get("t0", 0):.1f}–{w.get("t1", 0):.1f} s</figcaption></figure></section>')


def render(LM, clips, mismatch):
    m, lost = LM["mean"], LM["lost"]
    order = sorted(clips.items(), key=lambda kv: (PLAN.get(kv[1]["cls"], (99,))[0], list(clips).index(kv[0])))
    cards = "".join(card(i, s8, c, LM) for i, (s8, c) in enumerate(order, 1))
    mb = sum(c["mb"] for c in clips.values())
    mm = "全部与存档读数相同（|差| &lt; 0.001）" if not mismatch else "；".join(html.escape(x) for x in mismatch) + " 与存档不同"
    return PAGE.format(s0=m["R6DAG-s0"], s1=m["R6DAG-s1"], l0=lost["R6DAG-s0"], l1=lost["R6DAG-s1"], table=table(LM), cards=cards, n_clip=len(clips),
                       mb=mb, mm=mm, body=BODY, rec=REC)


BODY = """<h2>先看结论</h2><ol>
<li><b>还丢多少。</b>s1 丢 51.6 分、s0 丢 62.8 分；s1 丢分的 70 个场景里 67 个 s0 也丢（失分集合非常稳定，种子差几乎都是 s0 多出来的出路）。
把我们任一 ACC + GAP 读数（含另一个种子）在每个场景上取最好，两个种子都到 0.879：能「示范收回」的是 s1 17.5 分 / s0 28.7 分；35 个失分场景我们任何 serve 的任何读数都没解过。</li>
<li><b>最大的一块是停车点冲过（19–20 分，均分 0.07），但最难拿。</b>19 个两种子都丢；在提交 serve 下我们没有任何读数解过其中任何一个；尺子区分度 ≈ 0.2（全族一起失败），尺子估值 +7 / +8。
历史上每次修停车都付了起步或追尾（第 400 / 402 / 412 条）。</li>
<li><b>最值得做的是横向：转弯切内侧。</b>两种子共同失败 7 个，全部有我们的读数解过（NTV0 解 3–4 个），s0 比 s1 多的 8 个出路主要在这里；DAgger 第 2 次迭代正对着它。</li>
<li><b>尺子上 s1 比 NTV0 低的 155 分，几乎全压在 6 个「部分分」场景上，不在零分上。</b>s1 在尺子的高区分度题上没有零分；不满分的 6 个是 bbc53aa8 0.74（高速无故减速）、d90fe123 0.65（入路口减速被追尾）、a1b166cc 0.90、94c1d722 0.978、2a0efada 0.99、37d660b1 0.99。
NTV0 与 R2C 在这 6 个上全是 1.0，所有含 PAI 人类行的配方都会丢其中几个 → 这个「慢」来自 PAI 人类行，DAgger 标签保留学生速度所以没纠。均分上只值约 1 分，尺子上值很多（尺子强带是我们自己的族，要打折）。</li>
<li><b>被追尾部分分（8–9 分）多数不是我们能修的：</b>在人的速度附近被不反应的回放后车撞上，我们最好的读数也只能多拿约 2 分。碰撞零分（每个种子 6 个）以种子噪声为主；路线该转未转已修到 2 个（都是掉头 / 少见几何，没人解过）。</li>
<li>视频里看到用户 10-21 提的「为避前车开出路面」的一个实例（4f7cb072），在直路出路类里；其余出路都是转弯几何问题，不是避让。</li>
</ol>"""
REC = """<h2>建议：接下来集中在这三类</h2><ol>
<li><b>转弯切内侧（含直路 / 路口出路）——继续 DAgger 迭代。</b>示范可收回、两种子稳定、成本已付（第 2 次迭代在跑）。判第 2 次迭代就看这 7 个两种子共同失败的场景和 s0 的出路数。</li>
<li><b>无故减速的部分分——下一次建 DAgger 行时顺手加。</b>F2（去掉无前车自发减速不停的 PAI 行，R6FIX 已证明能救 bbc53aa8）加一个纵向「不无故减速」标签（无前车、人保持速度时弧长取 max(学生, 人)）。代码约 1–2 h，训练搭第 3 次迭代，不另开臂；它保护的是尺子（和榜上的 2PL 权重），不是均分。</li>
<li><b>高速漂移——指令规则改看路线航向变化（第 432 条），并入同一个臂。</b>4 个两种子共同失败、示范可收回 3–4 分；改动小（从存档特征重算），先在这 4 个场景 + 9 个稳定出路场景上用 RECORDED 路线做闭环小测。</li>
</ol>
<p><b>暂不开停止线臂。</b>它是最大的一块，但在提交 serve 下我们没有任何正例，ACC 会把平缓减速按巡航速度送出，过去的修法都拿起步换；尺子上几乎不值分。若要做，应该作为 DAgger 行上的纵向标签（标签弧长取 min(学生, 到人停车点的距离)），并且原生与 ACC + GAP 两种 serve 都读。</p>"""

PAGE = '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R6DAG 在 PAI 281 上还在哪里丢分</title><style>
body{{font:15px/1.6 -apple-system,"PingFang SC",sans-serif;max-width:1500px;margin:0 auto;padding:0 .8em 3em;color:#222}}
#rb{{position:sticky;top:0;z-index:9;background:#fffffff2;padding:6px 0;border-bottom:1px solid #ddd;font-size:14px}}
#rb button{{margin:0 2px;padding:1px 8px;border:1px solid #999;border-radius:4px;background:#fff}}#rb button.on{{background:#333;color:#fff}}
#rb input{{vertical-align:middle;width:150px}}section{{border-top:1px solid #ddd;margin-top:1.2em;padding-top:.2em}}h3{{margin:.4em 0}}
figure{{margin:0}}video{{width:100%;max-height:86vh;background:#000;display:block;margin:0 auto}}figcaption{{font-size:13px;color:#555}}.cap{{background:#f4f4fb;padding:3px 8px}}
.rate{{background:#333;color:#fff;border-radius:4px;padding:0 5px;font-size:12px}}li{{font-size:14.5px}}code{{font-size:13px}}
table{{border-collapse:collapse;font-size:12.5px;margin:.6em 0}}td,th{{border:1px solid #ccc;padding:3px 5px;vertical-align:top}}th{{background:#f3f3f3}}td.note{{color:#555}}
</style></head><body>
<div id="rb">播放速度 <button data-r="0">自动</button><button data-r="1">1×</button><button data-r="2">2×</button><button data-r="3">3×</button><button data-r="5">5×</button><button data-r="10">10×</button><button data-r="16">16×</button> <input type="range" min="0.5" max="16" step="0.5"> <b id="rv"></b></div>
<h1>R6DAG 在 PAI 281 上还在哪里丢分（ACC + GAP，2026-10-21）</h1>
<p>对象：当前最好的臂 R6DAG（DAgger + NuRec 地图横向教师，第 434 条），PAI 281 提交 serve（ACC + GAP，g10）下 s1 均分 {s1:.3f}、s0 {s0:.3f}（281 个场景；丢分 {l1:.1f} / {l0:.1f} 个场景分），
参照是已提交的 NTV0（<code>p1021a</code>，榜 PCS 2154）0.744。1 分 = 1 个场景，部分分按 1 − 得分计。全部来自已存读数，没有新的闭环读数；记录 <code>experiments/round7/results/loss_map.md</code>，逐场景表 <code>loss_map.csv</code>。</p>
{body}
<h2>失分地图</h2>
<p>「我们已有读数」= 我们自己 checkpoint 在 281 上的全部 ACC + GAP 存档读数（NTV0 两种子、R2C / R2J、R3MV / R5LC / R6FIX 两种子、R5A、JTN / JTXS、NTVX / NTVXS、STAND 三份、BNC2K、RSG，共 21 份，再加 R6DAG 的另一个种子）；「从未有读数解」再加上原生 / 只 ACC / g5 / g20 / holdlead 等其他 serve 的读数，第三方 garage / go-straight 不算。
尺子 = 第 392 / 429 条的本地 PCS 尺子（zoib 2PL）的题目区分度 a：a ≥ 1 的题是配池「强带」都满分的题，在上面拿非 1 分会压低能力估计；带内满分率 = 13 份强带读数在这一类场景上满分的比例。
<b>强带就是我们自己的族</b>（NTV0 / JTN / RSG / STAND 系），别的强队未必都过这些题，尺子的权重只能当「我们族都会、你却丢了」来读，不是榜的定标预测。估计收益的「榜约 PCS」按第 426 条的均分换算（本地 1 个场景分 ≈ 3 PCS），不含 2PL 区分度的放大。</p>
{table}
{rec}
<h2>视频（每类 1–2 段，按上表排名）</h2>
<p>画面是 AlpaSim 自带的评测视频：上为以本车为中心的俯视图（橙线 = 本车计划，绿线 = 路线，浅绿框 = 本车，蓝线 = 车道），下为模型输入的前视相机。剪到事件前 8 s 到事件后 3 s；顶条显示读数、得分、时刻、车速和事件倒计时，事件后变红。
零分的 R6DAG rollout 直接用存档读数留下的 rollout.asl 在 CPU 上离线渲染（就是计分的那一次）；部分分和对照读数没有 asl，用同一 checkpoint、同一 serve 在 GPU 池重读一次（低优先级），重读得分{mm}。
共 {n_clip} 段、{mb:.0f} MB。顶部速度条改全部视频（自动 = 每段 max(2, min(3, 30/fps, 时长/6))，选择记在浏览器里），滑到附近才加载，过半可见才播放。</p>
{cards}
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
