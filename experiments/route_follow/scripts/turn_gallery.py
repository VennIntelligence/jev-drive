#!/usr/bin/env python3
"""Zero-score gallery of RSG-L18-s0 on PAI 281 for watching on the Tokyo desktop (layout of p4_videos.py, no simulation, no GPU).

Sections: A = the route-following corridor zeros (cor_straight_at_turn, cor_turn_under, cor_wrong_exit, cor_lane_change, cor_drift),
B = every other zero (cor_route_end, offroad, collision), C = strict turns RSG completed with score 1.0 where a straight alternative exists
(plan topo straight_exit / straight_road). Groups = taxonomy primary class; inside a group: straight ahead available first, then score
ascending, then the event decision. Row = scene; header = side, topo, "straight ahead available", RSG outcome, taxonomy class, human vs RSG
speed at the strict entry (else at the bend entry), P4 clip status. Cells: HUMAN (log-following render, pait/pass), RSG (r281/rsg, the policy
alone), the best completing P4 branch cell (omitted when the scene has none); the other rendered P4 cells sit behind a per-row "more" toggle.

Overlays (ffmpeg re-encode of every shown clip, originals untouched): corner label (HUMAN / RSG / cell name); P4 branch clips get a coloured
border + "PUSH" from the handover decision h for 3 s (onp_p4.LAT_N = 30 decisions) and "SELF" after it ("LOG" before h for a log-source cell);
a short "ENTRY" flag at the strict-entry decision where the P4 manifest has it (self-source entry only).
Decision -> video time (verified): decisions are 10 Hz, k = 0..n_dec-1 (`now` steps 100 ms); the eval video is 10 fps with n_dec + 2 frames (252 of 252
r281/rsg rollouts) and the BEV panel prints the frame's sim time: frame 0 is the pre-roll step, frame 1 carries exactly decision 0's `now`, frame 2
decision 1. So decision k is at clip time (k + 1) * 0.1 s.

Inputs (repo-relative, scp'd to a scratch tree on Tokyo): results/taxonomy/pai281_rsg.csv, results/pai_follow/scenes.csv,
results/turn_gallery/plan_strict36.json (rf_p4.py plan of the 36 strict turns), results/p4_self/read.json (handover h per cell where a
manifest has no h), and the P4 video manifests (--p4, several roots, each manifest.json + its mp4s; the first root that has a (scene, cell)
wins): the box camera re-runs of the 27 train turns, the box fill (P4 prereg Amendment 5), the Tokyo fill of the exam turns (re-runs of
p4_exam) and the Tokyo own job of 302c5c99. When no cell of a scene completes, D40 g1.25-raw is the shown cell, marked "no cell completes".
A box-run identity (the P4 branch's own root state; box vs Tokyo rollouts are not bit-identical) sits behind "more".
Usage: turn_gallery.py [OUT] [--jobs N] [--p4 ROOT ...] (default OUT ~/Desktop/turn_gallery).
"""
import argparse, csv, glob, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "experiments/route_follow/results"
ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default=str(Path.home() / "Desktop" / "turn_gallery"))
ap.add_argument("--rsg", default="/data/runs/route_night/r281/rsg")
ap.add_argument("--human", default="/data/runs/route_night/pait/pass")
ap.add_argument("--p4", nargs="+", default=["/data/runs/route_follow/p4_video", "/data/runs/route_follow/p4_video_fill",   # manifest.json + mp4s
                                             "/data/runs/route_night/p4_fill/exam_video", "/data/runs/route_night/p4_fill/new"])  # (root prefix replaced)
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--threads", type=int, default=4)
A = ap.parse_args()
OUT = Path(A.out)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OFF_DEC, DT, PUSH_S = 1, 0.1, 3.0       # clip time of decision k = (k + OFF_DEC) * DT; push window = 30 decisions
STRAIGHT = ("straight_exit", "straight_road")
FAIL = {"collision_at_fault": "coll", "offroad": "offr", "left_corridor_laterally": "corr"}
CN = {"cor_straight_at_turn": "该转没转", "cor_turn_under": "转了但不够", "cor_wrong_exit": "转错出口", "cor_lane_change": "没变道", "cor_drift": "漂移",
      "cor_route_end": "开过日志终点", "off_junc_outside": "出路:路口外侧过宽", "off_junc_inside": "出路:路口内侧切角", "off_junc_straight": "出路:路口内直行段",
      "off_curve_outside": "出路:弯道外侧", "off_curve_inside": "出路:弯道内侧", "off_straight": "出路:直路/车道边",
      "coll_veh_side": "碰撞:侧擦同向车", "coll_veh_oncoming": "碰撞:对向车", "coll_veh_crossing": "碰撞:横穿车", "coll_cyclist": "碰撞:骑车人", "coll_ped": "碰撞:行人"}
SEC_A = ["cor_straight_at_turn", "cor_turn_under", "cor_wrong_exit", "cor_lane_change", "cor_drift"]
SEC_B = ["cor_route_end", "off_junc_outside", "off_junc_inside", "off_junc_straight", "off_curve_outside", "off_curve_inside", "off_straight",
         "coll_veh_side", "coll_veh_oncoming", "coll_veh_crossing", "coll_cyclist", "coll_ped"]
PREF = [f"g{g}-{lon}" for g in ("1.25", "1", "1.5") for lon in ("raw", "ve", "ve08")]


def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- clips
def rollouts(root, pattern):
    """{scene: dict(mp4, score, outcome)} of every results-summary.json under root/<pattern>."""
    d = {}
    for summ in sorted(glob.glob(str(Path(root) / pattern))):
        run = Path(summ).parents[1]
        for r in json.loads(Path(summ).read_text())["rollouts"]:
            mp4 = sorted((run / "rollouts" / r["clipgt_id"] / r["rollout_id"]).glob("*camera_front_wide*.mp4"))
            if mp4 and r.get("score") is not None:
                kind = next((v for k, v in FAIL.items() if (r.get("score_metrics") or {}).get(k)), "ok")
                d[r["clipgt_id"]] = dict(mp4=mp4[0], score=float(r["score"]), outcome=kind)
    return d


rsg = rollouts(A.rsg, "runs/*/sim/aggregate/results-summary.json")
human = rollouts(A.human, "runs/*/sim/aggregate/results-summary.json")
read = {(x["scene"], x["tag"]): x for x in json.loads((RES / "p4_self/read.json").read_text())["rows"]}
P4S, MS = {}, {}                     # scene -> cells (first root wins per cell), scene -> manifest scene facts
for root in map(Path, A.p4):
    if not (root / "manifest.json").exists():
        print("no manifest in", root, file=sys.stderr)
        continue
    m = json.loads((root / "manifest.json").read_text())
    box = m.get("backend", "box" if m["root"].startswith("/root/") else "tokyo") == "box"
    for c in m["cells"]:
        if c["mp4"] and not any(x["tag"] == c["tag"] for x in P4S.get(c["scene"], [])):
            P4S.setdefault(c["scene"], []).append(dict(c, path=root / Path(c["mp4"]).relative_to(m["root"]), box=box,
                                                       h=c["h"] if "h" in c else (read.get((c["scene"], c["tag"])) or {}).get("h")))
    for sc, v in m["scenes"].items():
        MS.setdefault(sc, v)
plan = json.loads((RES / "turn_gallery/plan_strict36.json").read_text())

# ---------------------------------------------------------------- facts
tax = {r["scene"]: r for r in csv.DictReader(open(RES / "taxonomy/pai281_rsg.csv"))}
pf = {r["scene"]: r for r in csv.DictReader(open(RES / "pai_follow/scenes.csv")) if r["label"] == "rsg"}


def facts(sc, section, group):
    t, f = tax[sc], pf[sc]
    pl = plan["scenes"].get(sc)
    topo, tsrc = (f["topo"], "ego lane at the failure") if f["topo"] in ("straight_exit", "straight_road", "curve_single", "no_straight_exit", "lane_change") \
        else (pl["topo"], "human path") if pl else ("", "")
    if section == "C":
        topo, tsrc = pl["topo"], "human path"
    straight = "n/a" if not topo else "yes" if topo in STRAIGHT else "no"
    side = f["side"] or ("L" if (num(f["z_turn_deg"]) or 0) > 0 else "R" if (num(f["z_turn_deg"]) or 0) < 0 and f["z_turn"] == "True" else (pl or {}).get("side", ""))
    if f["turn"] == "True" and num(f["v_entry"]) is not None:
        vr, vh, vk = num(f["v_entry"]), num(f["v_log_entry"]), "strict entry"
    elif f["z_turn"] == "True" and num(f["v_ego_0"]) is not None:
        vr, vh, vk = num(f["v_ego_0"]), num(f["v_log_0"]), "bend entry"
    else:
        vr = vh = None
        vk = "strict entry, ego left the route before it" if f["turn"] == "True" else "no route turn"
    return dict(scene=sc, s8=sc[7:15], section=section, group=group, side=side, topo=topo, topo_src=tsrc, straight=straight, score=float(t["score"]),
                zero=t["zero_type"], primary=t["primary"], secondary=t["secondary"], cn=CN.get(t["primary"], ""), k_ev=num(t["k_ev"]),
                strict=f["turn"] == "True", v_rsg=vr, v_hum=vh, vkind=vk, turn_deg=num(f["turn_deg"]) or num(f["z_turn_deg"]))


rows = []
for sc, t in tax.items():
    if float(t["score"]) == 0:
        g = t["primary"]
        rows.append(facts(sc, "A" if g in SEC_A else "B", g))
for sc, f in pf.items():            # section C: strict turn completed with score 1.0, straight alternative by the plan topo
    pl = plan["scenes"].get(sc)
    if f["turn"] == "True" and f["completed"] == "True" and float(f["score"]) >= 0.9999 and pl and pl["topo"] in STRAIGHT:
        rows.append(facts(sc, "C", "completed"))
order = {g: i for i, g in enumerate(SEC_A + SEC_B + ["completed"])}
rows.sort(key=lambda r: (r["section"], order[r["group"]], r["straight"] != "yes", r["score"], r["k_ev"] if r["k_ev"] is not None else 1e9, r["s8"]))

# ---------------------------------------------------------------- jobs
jobs, missing = [], []
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "thumbs").mkdir(exist_ok=True)


def job(src, name, label, ent=None, p4=None):
    jobs.append(dict(src=str(src), dst=str(OUT / name), label=label, ent=ent, p4=p4))
    return name


for r in rows:
    sc, s8 = r["scene"], r["s8"]
    cells = []
    ms = MS.get(sc)
    ent_t = round((ms["entry_k"] + OFF_DEC) * DT, 2) if ms and ms["entry_src"] == "self" else None
    h = human.get(sc)
    if h:
        cells.append(dict(kind="human", label="HUMAN", file=job(h["mp4"], f"{s8}_human.mp4", "HUMAN"), outcome=h["outcome"], score=round(h["score"], 3)))
    else:
        missing.append(f"{s8} human")
    g = rsg.get(sc)
    if g:
        cells.append(dict(kind="rsg", label="RSG", file=job(g["mp4"], f"{s8}_rsg.mp4", "RSG", ent=ent_t), outcome=g["outcome"], score=round(g["score"], 3)))
    else:
        missing.append(f"{s8} rsg")
    pc = [c for c in P4S.get(sc, []) if c["tag"] != "id" and c["score"] is not None]
    r["p4_n"], r["p4_done"], r["p4"] = len(pc), sum(bool(c["done"]) for c in pc), "none"
    best = None
    if pc:
        key = lambda c: (c["D"] != 40, PREF.index(c["tag"].split("-", 2)[2]) if c["tag"].split("-", 2)[2] in PREF else 9, -c["D"])  # noqa: E731
        done = [c for c in pc if c["done"]]
        raw = [c for c in done if c["tag"].endswith("-raw")]   # speed profiles were dropped (Amendment 5): a completing raw cell first
        fb = next((c for c in pc if c["tag"] == "D40-self-g1.25-raw"), None)
        best = min(raw, key=key) if raw else fb or min(done or pc, key=key)
        note = "" if best["done"] else "no cell completes" if not done else "no raw cell completes (a speed-profile cell does: more)"
        if not raw and best["done"]:
            note = "no raw cell rendered completes; speed-profile cell"
        r["p4"] = f"{len(pc)} cells rendered, {len(done)} complete ({len(raw)} raw)" + (f" ({note})" if note else "")
        bid = next((c for c in P4S.get(sc, []) if c["tag"] == "id" and c["box"] and c["score"] is not None), None)
        if bid:                                  # the box identity the branches start from (the RSG cell is the Tokyo r281 rollout)
            cells.append(dict(kind="p4id", label="P4 id (box)", file=job(bid["path"], f"{s8}_id_box.mp4", "P4 id (box)", ent=ent_t),
                              outcome=bid["outcome"], score=round(bid["score"], 3), main=False))
        for c in sorted(pc, key=lambda c: (c is not best, not c["done"], key(c))):
            D, src, gp = c["tag"].split("-", 2)
            hh = c["h"]
            if hh is None:
                missing.append(f"{s8} {c['tag']} (no handover decision)")
                continue
            t0 = round((hh + OFF_DEC) * DT, 2)
            label = f"{D} {src} {gp}"
            p4 = dict(t0=t0, t1=round(t0 + PUSH_S, 2), src=src)
            cells.append(dict(kind="p4", label=label, file=job(c["path"], f"{s8}_{c['tag']}.mp4", label, ent=ent_t, p4=p4),
                              outcome=c["outcome"], score=round(c["score"], 3), main=c is best, h=hh, t0=t0, done=bool(c["done"]), v=c["v_entry"],
                              note=note if c is best else "", box=c["box"]))
    r["ent_t"] = ent_t
    for c in cells:
        c.setdefault("main", c["kind"] != "p4")
    r["cells"] = cells
    if g:
        thumb = OUT / "thumbs" / f"{s8}.jpg"
        if not thumb.exists():
            jobs.append(dict(thumb=str(thumb), src=str(g["mp4"])))


def esc(s):
    return s.replace(":", r"\:").replace("'", "")


def run(j):
    if "thumb" in j:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", j["src"], "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "6", j["thumb"]], check=False)
        return
    if Path(j["dst"]).exists() and Path(j["dst"]).stat().st_size > 0:
        return
    tx = lambda txt, x, y, size, col, en="": (f"drawtext=fontfile={FONT}:text='{esc(txt)}':x={x}:y={y}:fontsize={size}:fontcolor={col}"  # noqa: E731
                                              f":box=1:boxcolor=black@0.55:boxborderw=5" + (f":enable='{en}'" if en else ""))
    vf = [tx(j["label"], 8, 8, 26, "white")]
    p = j["p4"]
    if p:
        a, b = p["t0"], p["t1"]
        if p["src"] == "log":
            vf.append(tx("LOG", "(w-text_w)/2", 8, 34, "white", f"lt(t,{a})"))
        vf += [f"drawbox=x=0:y=0:w=iw:h=ih:color=orange@0.95:t=10:enable='between(t,{a},{b})'", tx("PUSH", "(w-text_w)/2", 8, 34, "orange", f"between(t,{a},{b})"),
               f"drawbox=x=0:y=0:w=iw:h=ih:color=cyan@0.9:t=10:enable='gte(t,{b})'", tx("SELF", "(w-text_w)/2", 8, 34, "cyan", f"gte(t,{b})")]
    if j["ent"] is not None:
        vf.append(tx("ENTRY", "w-text_w-8", 8, 26, "yellow", f"between(t,{j['ent']},{j['ent'] + 1.0})"))
    tmp = j["dst"] + ".part.mp4"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", j["src"], "-vf", ",".join(vf), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                        "-pix_fmt", "yuv420p", "-threads", str(A.threads), "-movflags", "+faststart", tmp], capture_output=True, text=True)
    if r.returncode:
        print("ffmpeg failed", j["dst"], r.stderr[-300:], file=sys.stderr)
    else:
        Path(tmp).rename(j["dst"])


with ThreadPoolExecutor(A.jobs) as ex:
    list(ex.map(run, jobs))

SECT = {"A": "A: route-following corridor zeros", "B": "B: every other zero (past the log end, offroad, collision)", "C": "C: completed (score 1.0), straight ahead was available"}
cnt = {s: sum(r["section"] == s for r in rows) for s in "ABC"}
html = r"""<!doctype html><meta charset=utf-8><title>RSG-L18-s0 PAI 281: zeros and straight-available turns</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#222;padding:8px 12px;display:flex;gap:16px;flex-wrap:wrap;align-items:center;z-index:2}
#legend{padding:8px 14px;font-size:13px;line-height:1.5;color:#bbb;border-bottom:1px solid #333}
#legend b{color:#ddd}
#foot{padding:10px 14px 40px;font-size:12px;line-height:1.5;color:#999}
h2{margin:0;padding:8px 14px;background:#2a2a1a;color:#fd8;font-size:16px;position:sticky;top:38px;z-index:1}
h3{margin:0;padding:5px 14px;background:#1c1c28;color:#9bf;font-size:14px}
.scene{display:flex;gap:8px;padding:6px 10px;border-bottom:1px solid #333;align-items:flex-start}
.scene.cur{background:#1d2a1d}
.head{width:210px;flex:none;font-family:monospace;font-size:12px;padding-top:4px;line-height:1.45}
.head .y{color:#8d8}.head .n{color:#fa4}.head .na{color:#888}
.cells{flex:1;display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:6px}
.cell{min-width:0;cursor:pointer}
.cell video,.cell img{width:100%;aspect-ratio:9/10;background:#000;display:block;object-fit:cover}
.cell .lab{font-family:monospace;font-size:12px;padding:2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.done,.ok{color:#8d8}.coll{color:#f66}.offr{color:#fa4}.corr{color:#fc6}.rear{color:#c8f}.end,.oth{color:#aaa}
.human .lab{color:#8cf}
.none{aspect-ratio:9/10;background:#181818;color:#777;display:flex;align-items:center;justify-content:center;text-align:center;font-family:monospace;font-size:12px;padding:6px}
.more{grid-column:1/-1;font-family:monospace;font-size:12px;color:#8cf;cursor:pointer;padding:2px 0}
.extra{display:none}.scene.open .extra{display:block}
#big{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;flex-direction:column;z-index:9}
#big.on{display:flex}
#big video{height:88vh;max-width:96vw;background:#000}
#big .lab{font-family:monospace;font-size:16px;padding:8px}
</style>
<div id=bar>
 <b>RSG-L18-s0 on PAI 281</b> <span id=n></span>
 <span>scene <input id=q placeholder="8 hex" style="width:7em"></span>
 <span>straight ahead available <input id=st type=checkbox></span>
 <span>only rows with a P4 clip <input id=p4 type=checkbox></span>
 <span>speed <select id=rate><option>0.5</option><option>1</option><option selected>1.5</option><option>2</option><option>3</option></select>x</span>
 <span style="color:#888">j / k next / prev scene and play the row, space play / pause, . , step 0.1 s, 0 rewind, click a cell to play it alone, double-click to enlarge, Esc closes, [ ] speed</span>
</div>
<div id=legend>
<b>Row</b> = one PAI 281 scene driven by RSG-L18-s0. Sections: __SECTIONS__. Groups = taxonomy main class (<code>research/zero-taxonomy.md</code>); inside a group the scenes with a straight alternative come first, then score ascending, then the event decision.<br>
<b>Header</b>: side of the turn, topo class (yes / no / n/a = "straight ahead available"; n/a when the scene has no route turn), RSG outcome and score, taxonomy class and sub class, v human / v RSG at the strict turn entry (or at the bend entry when the turn is not strict; "-" when the ego left the route before it), P4 branch clips.<br>
<b>Cells</b>: HUMAN = the human log rendered by the sim (log-following driver; missing for some scenes); RSG = the policy alone (r281/rsg, Tokyo); the P4 cell = the best completing raw branch rollout of the scene (D40 first, g1.25 preferred), or D40 g1.25-raw marked "no cell completes" / "no raw cell completes" (a completing speed-profile cell from the first video batch then sits behind "more"); "more" shows the other rendered P4 cells (train turns: run on the GPU box, label "box", with "P4 id (box)" = the box identity they branch from, which is not bit-identical to the Tokyo RSG cell; exam turns: Tokyo, whose identity is bit-identical to RSG). Fill rows (P4 prereg Amendment 5) carry identity + g1 / g1.25 / g1.5 raw at the D40 and D0 handovers only.
Clip overlays: corner label; on P4 branch clips an orange border + PUSH from the handover decision h for 3 s, then a cyan border + SELF (the policy drives itself); a log-source cell shows LOG before h. A yellow ENTRY flag (1 s) marks the strict-entry decision where the P4 manifest has it.
</div>
<div id=list></div>
<div id=foot>__FOOT__</div>
<div id=big><video loop muted></video><div class=lab></div></div>
<script>
for(const id of ["q","st","p4","n","list","big","rate"]) window[id+"_"]=document.getElementById(id);
const R=__ROWS__, SEC=__SEC__;
const f1=v=>v==null?"-":(+v).toFixed(1);
let cur=-1, scenes=[];
function render(){
  const q=q_.value.trim();
  scenes=R.filter(r=>!q||r.s8.startsWith(q)).filter(r=>!st_.checked||r.straight=="yes").filter(r=>!p4_.checked||r.p4_n>0);
  n_.textContent=scenes.length+" scenes";
  let last="", lastg="", html="";
  scenes.forEach((r,i)=>{
    if(r.section!=last){html+=`<h2>${SEC[r.section]}</h2>`; last=r.section; lastg=""}
    if(r.group!=lastg){const k=scenes.filter(x=>x.group==r.group).length; html+=`<h3>${r.group=="completed"?"strict turns completed with score 1.0":r.group+" "+r.cn} (${k})</h3>`; lastg=r.group}
    const out=r.section=="C"?`<span class=done>done ${r.score.toFixed(2)}</span>`:`<span class=${r.zero.slice(0,4)=="corr"?"corr":r.zero.slice(0,3)=="off"?"offr":"coll"}>zero: ${r.zero}</span>`;
    const cs=r.cells, main=cs.filter(c=>c.main), ex=cs.filter(c=>!c.main);
    const cell=c=>`<div class="cell${c.kind=="human"?" human":""}${c.main?"":" extra"}"><img loading=lazy src="thumbs/${r.s8}.jpg" data-src="${c.file}"><div class=lab title="${c.file}">${c.note?"<b style='color:#f88'>"+c.note+":</b> ":""}${c.label} <b class=${c.outcome}>${c.outcome} ${c.score.toFixed(2)}</b>${c.kind=="p4"?" h "+c.h+" ("+c.t0.toFixed(1)+" s)"+(c.box?" box":""):""}</div></div>`;
    const none=r.p4_n>0&&!main.some(c=>c.kind=="p4")?`<div class=cell><div class=none>no P4 cell completes this scene (${r.p4_n} rendered, see more)</div></div>`:r.p4_n==0?`<div class=cell><div class=none>no P4 clip for this scene</div></div>`:"";
    html+=`<div class=scene data-i=${i}><div class=head><b>${r.s8}</b> ${r.side||"-"}${r.turn_deg?" "+Math.abs(r.turn_deg).toFixed(0)+" deg":""}<br>topo ${r.topo||"-"}<br>straight ahead: <b class=${r.straight=="yes"?"y":r.straight=="no"?"n":"na"}>${r.straight}</b><br><span style="color:#888">${r.topo_src}</span><br>${out} ${r.section=="C"?"":r.score.toFixed(2)}<br>${r.primary} ${r.cn}<br><span style="color:#888">${r.secondary||"-"}</span><br>v human ${f1(r.v_hum)} / RSG ${f1(r.v_rsg)}<br><span style="color:#888">${r.vkind}</span><br>P4: ${r.p4}${r.ent_t!=null?"<br>entry flag at "+r.ent_t.toFixed(1)+" s":""}</div><div class=cells>`+
      main.map(cell).join("")+none+(ex.length?`<div class=more>&#9656; more (${ex.length})</div>`:"")+ex.map(cell).join("")+`</div></div>`});
  list_.innerHTML=html; cur=-1;
}
function toVideo(im){const v=document.createElement("video");v.src=im.dataset.src;v.playbackRate=+rate_.value;v.poster=im.src;v.loop=true;v.muted=true;v.dataset.src=im.dataset.src;v.dataset.thumb=im.src;im.replaceWith(v);return v}
function toPh(v){const i=document.createElement("img");i.loading="lazy";i.src=v.dataset.thumb;i.dataset.src=v.dataset.src;v.pause();v.removeAttribute("src");v.load();v.replaceWith(i)}
function go(i,scroll){
  if(i<0||i>=scenes.length) return;
  const old=document.querySelector(".scene.cur"); if(old){old.querySelectorAll("video").forEach(toPh); old.classList.remove("cur")}
  cur=i; const d=document.querySelector(`.scene[data-i="${i}"]`); d.classList.add("cur"); if(scroll) d.scrollIntoView({block:"center"});
  d.querySelectorAll(d.classList.contains("open")?".cell img":".cell:not(.extra) img").forEach(im=>{const v=toVideo(im); v.play()});
}
const bigV=big_.querySelector("video");
function openBig(cell){const src=(cell.querySelector("img,video")||{}).dataset.src; if(!src) return; bigV.src=src; bigV.playbackRate=+rate_.value; bigV.currentTime=0; bigV.play(); big_.querySelector(".lab").textContent=cell.querySelector(".lab").title; big_.classList.add("on")}
function closeBig(){big_.classList.remove("on"); bigV.pause(); bigV.removeAttribute("src"); bigV.load()}
big_.addEventListener("click",e=>{if(e.target===big_) closeBig()}); bigV.addEventListener("dblclick",closeBig);
list_.addEventListener("dblclick",e=>{const c=e.target.closest(".cell"); if(c) openBig(c)});
addEventListener("keydown",e=>{
  if(e.target.tagName=="INPUT"||e.target.tagName=="SELECT") return;
  if(e.key=="Escape") return closeBig();
  const vs=big_.classList.contains("on")?[bigV]:[...document.querySelectorAll(".scene.cur video")];
  if(e.key=="j") go(cur+1,true); else if(e.key=="k") go(cur-1,true);
  else if(e.key==" "){e.preventDefault(); vs.forEach(v=>v.paused?v.play():v.pause())}
  else if(e.key=="."||e.key==","){vs.forEach(v=>{v.pause(); v.currentTime+= e.key=="."?0.1:-0.1})}
  else if(e.key=="0"){vs.forEach(v=>{v.currentTime=0})}
  else if(e.key=="["||e.key=="]"){const o=[...rate_.options].map(x=>+x.value); let i=o.indexOf(+rate_.value)+(e.key=="]"?1:-1); i=Math.max(0,Math.min(o.length-1,i)); rate_.value=o[i]; setRate()}
});
list_.addEventListener("click",e=>{const m=e.target.closest(".more"); if(m){const d=m.closest(".scene"); const o=d.classList.toggle("open"); m.innerHTML=(o?"&#9662; less":"&#9656; more")+` (${d.querySelectorAll(".extra").length})`; if(o&&+d.dataset.i===cur) d.querySelectorAll(".extra img").forEach(im=>toVideo(im).play()); return}
  const t=e.target.closest(".cell"); if(!t||t.querySelector(".none")) return; const d=t.closest(".scene"); const cs=[...d.querySelectorAll(".cell")].filter(c=>!c.querySelector(".none")); const k=cs.indexOf(t);
  if(+d.dataset.i!==cur) go(+d.dataset.i); const im=t.querySelector("img"); if(im) toVideo(im); cs.map(c=>c.querySelector("video")).forEach((x,j)=>{if(!x) return; if(j===k){x.currentTime=0; x.play()} else x.pause()})});
function setRate(){document.querySelectorAll("video").forEach(v=>v.playbackRate=+rate_.value)}
rate_.addEventListener("change",setRate);
for(const id of ["q","st","p4"]) document.getElementById(id).addEventListener(id=="q"?"input":"change",render);
render();
</script>"""
nh, nr = sum(any(c["kind"] == "human" for c in r["cells"]) for r in rows), sum(any(c["kind"] == "rsg" for c in r["cells"]) for r in rows)
n4 = sum(any(c["kind"] == "p4" and c["main"] for c in r["cells"]) for r in rows)
n4d = sum(any(c["kind"] == "p4" and c["main"] and c["done"] for c in r["cells"]) for r in rows)
foot = (f"Rows: A {cnt['A']}, B {cnt['B']}, C {cnt['C']} ({len(rows)}); cells: HUMAN {nh}, RSG {nr}, P4 shown cell {n4} (completing {n4d}); "
        f"scenes with any P4 clip {sum(r['p4_n'] > 0 for r in rows)}.<br>"
        f"Decision-to-video mapping: decisions are 10 Hz (k = 0..n_dec-1); the eval video is 10 fps with n_dec + 2 frames (201 for n_dec 199). The BEV panel prints each frame's sim time: "
        f"frame 0 is the pre-roll step, frame 1 carries exactly decision 0's timestamp, frame 2 decision 1 (checked on r281/rsg; the last metrics timestamp is the last decision + 100 ms in 252 of 252 rollouts). "
        f"So decision k is at clip time (k + {OFF_DEC}) x {DT} s. PUSH = decisions [h, h + 30) = clip time [(h+{OFF_DEC})x0.1 s, +{PUSH_S:.0f} s), SELF after it. The lateral gain acts for exactly this window; "
        f"the speed profiles ve / ve08 act from h until turn exit + 10 m (so they run on in the SELF phase). A log-source cell (D0-log) drives the human log before h. "
        f"ENTRY flag at (entry_k + {OFF_DEC}) x 0.1 s.<br>"
        f"Topo: for corridor zeros the ego lane at the failure (rf_pai_follow); otherwise, and for section C, the human path (rf_p4.py plan). v at the strict entry: RSG from rf_pai_follow, human from the log.")
(OUT / "index.html").write_text(html.replace("__ROWS__", json.dumps(rows)).replace("__SEC__", json.dumps(SECT)).replace("__FOOT__", foot)
                                .replace("__SECTIONS__", "; ".join(f"<b>{SECT[s]}</b> ({cnt[s]})" for s in "ABC")))
(OUT / "README.txt").write_text(
    f"Zero-score gallery of RSG-L18-s0 on PAI 281 (turn_gallery.py in experiments/route_follow/scripts). Open index.html.\n"
    f"Section A (route-following corridor zeros): {cnt['A']} rows. Section B (every other zero): {cnt['B']} rows. Section C (strict turn completed with score 1.0, straight ahead "
    f"available): {cnt['C']} rows.\nCells: HUMAN {nh}, RSG {nr}, P4 shown cell {n4} (completing {n4d}; else D40 g1.25-raw marked 'no cell completes'). Every mp4 is a re-encode of the original with an overlay "
    f"(label; PUSH / SELF border on P4 branch clips); originals are untouched. thumbs/ = first frames.\n"
    f"Clip time of decision k = (k + {OFF_DEC}) x {DT} s; PUSH window = 3 s from the handover decision h.\n"
    f"Missing: {', '.join(missing) or 'none'}\n")
print(f"{len(rows)} rows (A {cnt['A']}, B {cnt['B']}, C {cnt['C']}); {len([j for j in jobs if 'thumb' not in j])} clips; human {nh}, rsg {nr}, p4 shown {n4} (completing {n4d}); "
      f"missing: {', '.join(missing) or 'none'}; index {OUT / 'index.html'}")
