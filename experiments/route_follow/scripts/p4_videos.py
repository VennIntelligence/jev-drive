#!/usr/bin/env python3
"""Gallery of the P4 self-branch camera re-runs for watching on the Tokyo desktop (same layout as experiments/alpasim/scripts/pai_videos.py).

Input: the box video root's manifest.json (rf_p4.sh vcollect) and its mp4s rsynced to LOCAL (the box root prefix is replaced by LOCAL), plus the
human log-following rollout of each scene (onp_log_driver in log mode, rendered by the sim) from HUMAN if it has the scene's mp4.
Output: ~/Desktop/p4_videos/<scene8>_<side>_<D><src>_<cell>_<outcome>_<score>.mp4 (symlinks; identity = `id`, human = `human`) and
~/Desktop/p4_videos.html: one row per scene (header: side, topo, entry decision, identity outcome, class v_e, entry speed of the identity, the
best completing cell and the human), three cells (human, identity, the best completing cell or g1.25-raw at D40 when none completes) and the
other rendered cells behind a per-row "more" toggle; rows: the late turns first (LATE), then the 4 push-completes scenes (PUSH), then the
rest, each by entry decision descending. Cells show the scene's first frame until the row is active (only the active row holds <video>).
Usage: p4_videos.py MANIFEST LOCAL [HUMAN] (HUMAN default /data/runs/route_night/pait/pass).
"""
import glob, json, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAN, LOCAL = Path(sys.argv[1]), Path(sys.argv[2])
HUMAN = Path(sys.argv[3] if len(sys.argv) > 3 else "/data/runs/route_night/pait/pass")
OUT = Path.home() / "Desktop" / "p4_videos"
THUMBS = Path.home() / ".cache" / "p4_thumbs"
INDEX = Path.home() / "Desktop" / "p4_videos.html"
PREF = [f"g{g}-{lon}" for g in ("1.25", "1", "1.5") for lon in ("raw", "ve", "ve08")]
LATE = ("213dfdac", "a4d2f7cf", "1e8e939e", "3232fdb7", "98694f91", "dc1966f4", "a174faed", "ea468790")   # entry at decision ~96-165
PUSH = ("8a365ac2", "232d240e", "37d660b1", "e5e47fd2")      # a push completes, the identity fails
FAIL = {"collision_at_fault": "coll", "offroad": "offr", "left_corridor_laterally": "corr"}

M = json.loads(MAN.read_text())
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
THUMBS.mkdir(parents=True, exist_ok=True)
human = {}
for summ in sorted(glob.glob(str(HUMAN / "runs/*/sim/aggregate/results-summary.json"))):
    run = Path(summ).parents[1]
    for r in json.loads(Path(summ).read_text())["rollouts"]:
        mp4 = sorted((run / "rollouts" / r["clipgt_id"] / r["rollout_id"]).glob("*camera_front_wide*.mp4"))
        if mp4 and r.get("score") is not None:
            kind = next((v for k, v in FAIL.items() if (r.get("score_metrics") or {}).get(k)), "ok")
            human[r["clipgt_id"]] = dict(mp4=mp4[0], score=float(r["score"]), outcome=kind)

rows, scenes, todo, missing = [], {}, [], []
order = {t: i for i, t in enumerate(PREF)}
for c in M["cells"]:
    sc, s8 = c["scene"], c["scene"][7:15]
    P = M["scenes"][sc]
    if not c["mp4"] or c["score"] is None:                   # not rendered or not finished
        missing.append(f"{s8} {c['tag']}")
        continue
    src = LOCAL / Path(c["mp4"]).relative_to(M["root"])
    if c["tag"] == "id":
        cell, key = "id", (0, 0, 0)
    else:
        D, s, g = c["tag"].split("-", 2)
        cell, key = f"{D}{s}_{g}", (1, -int(D[1:]), order.get(g, 9))
    name = f"{s8}_{P['side']}_{cell}_{c['outcome']}_{c['score']:.2f}"
    (OUT / f"{name}.mp4").symlink_to(src)
    rows.append(dict(scene=s8, file=f"{name}.mp4", label=cell, outcome=c["outcome"], score=round(c["score"], 3), v=c["v_entry"],
                     ident=c["identical"], key=key, done=bool(c["done"])))
    if c["tag"] == "id" and not (THUMBS / f"{s8}.jpg").exists():
        todo.append((src, THUMBS / f"{s8}.jpg"))
    h = scenes.setdefault(s8, dict(scene=s8, side=P["side"], topo=P["topo"], v_e=P["v_e"], v_log=P.get("v_log_entry"), entry_k=P["entry_k"], entry_src=P["entry_src"]))
    if c["tag"] == "id":
        h.update(id_out=c["outcome"], id_v=c["v_entry"])
    elif c["done"] and (key[1] != -40, key[2], key[1]) < h.get("best_key", (9,)):
        h.update(best_key=(key[1] != -40, key[2], key[1]), best=cell, best_v=c["v_entry"])
for s8, h in scenes.items():
    sc = next(c["scene"] for c in M["cells"] if c["scene"][7:15] == s8)
    hu = human.get(sc)
    if hu:
        name = f"{s8}_{h['side']}_human_{hu['outcome']}_{hu['score']:.2f}"
        (OUT / f"{name}.mp4").symlink_to(hu["mp4"])
        rows.append(dict(scene=s8, file=f"{name}.mp4", label="human", outcome=hu["outcome"], score=round(hu["score"], 3), v=h["v_log"],
                         ident=True, key=(-1, 0, 0), done=hu["outcome"] == "ok"))
    else:
        missing.append(f"{s8} human")
    h.pop("best_key", None)
for s8 in scenes:                                            # main cells: human, identity, the best completing cell (rf_p4.best_cell's order)
    rr = [r for r in rows if r["scene"] == s8 and r["label"] not in ("id", "human")]
    done = [r for r in rr if r["done"]]
    best = min(done, key=lambda r: (r["key"][1] != -40, r["key"][2], r["key"][1])) if done else next((r for r in rr if r["label"] == "D40self_g1.25-raw"), None)
    for r in rows:
        if r["scene"] == s8:
            r["main"] = r["label"] in ("id", "human") or r is best
rows.sort(key=lambda r: (r["scene"], not r["main"], r["key"]))
for r in rows:
    r.pop("key")
rank = lambda h: (0 if h["scene"] in LATE else 1 if h["scene"] in PUSH else 2, -h["entry_k"])  # noqa: E731
srank = [h["scene"] for h in sorted(scenes.values(), key=rank)]


def render_thumb(a):
    src, dst = a
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "6", str(dst)], check=False)


with ThreadPoolExecutor(16) as ex:
    list(ex.map(render_thumb, todo))
(OUT / "thumbs").symlink_to(THUMBS)
html = r"""<!doctype html><meta charset=utf-8><title>P4 self-branch rollouts</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#222;padding:8px 12px;display:flex;gap:16px;flex-wrap:wrap;align-items:center;z-index:2}
#bar label{margin-right:6px;white-space:nowrap}
#legend{padding:8px 14px;font-size:13px;line-height:1.5;color:#bbb;border-bottom:1px solid #333}
#legend b{color:#ddd}
.scene{display:flex;gap:8px;padding:6px 10px;border-bottom:1px solid #333;align-items:flex-start}
.scene.cur{background:#1d2a1d}
.head{width:150px;flex:none;font-family:monospace;font-size:12px;padding-top:4px;line-height:1.45}
.cells{flex:1;display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:6px}
.cell{min-width:0;cursor:pointer}
.cell video,.cell img{width:100%;aspect-ratio:9/10;background:#000;display:block;object-fit:cover}
.cell .lab{font-family:monospace;font-size:12px;padding:2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.done,.ok{color:#8d8}.coll{color:#f66}.offr{color:#fa4}.corr{color:#fc6}.rear{color:#c8f}.end,.oth{color:#aaa}
.human .lab{color:#8cf}
.more{grid-column:1/-1;font-family:monospace;font-size:12px;color:#8cf;cursor:pointer;padding:2px 0}
.extra{display:none}.scene.open .extra{display:block}
#big{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;flex-direction:column;z-index:9}
#big.on{display:flex}
#big video{height:88vh;max-width:96vw;background:#000}
#big .lab{font-family:monospace;font-size:16px;padding:8px}
</style>
<div id=bar>
 <b>P4 self-branch rollouts</b> <span id=n></span>
 <span>outcome: <span id=kinds></span></span>
 <span>D: <span id=ds></span></span>
 <span>scene <input id=q placeholder="8 hex" style="width:7em"></span>
 <span>only rows with a completing cell <input id=dn type=checkbox></span>
 <span>speed <select id=rate><option>0.5</option><option>1</option><option selected>1.5</option><option>2</option><option>3</option></select>x</span>
 <span style="color:#888">j / k next / prev scene and play the row, space play / pause, . , step 0.1 s, 0 rewind, click a cell to play it alone, double-click to enlarge, Esc closes, [ ] speed</span>
</div>
<div id=legend>
<b>Row</b> = one turn scene; RSG-L18-s0 drives from the start (P4 run, re-run with the camera: bit-identical). Order: late turns first (long straight approach), then the 4 scenes where a push completes and the identity fails, then the rest, each by entry decision. Header: side, topo class, entry = the decision (and sim second) at the turn entry, identity outcome, class v_e (human entry-speed median), entry speed of the identity / best cell / human. Three cells: human, id, the best completing cell (D40 first, g1.25-raw preferred; g1.25-raw at D40 if none completes); "more" shows the other rendered cells.<br>
<b>Cell</b> <code>&lt;scene8&gt;_&lt;side&gt;_&lt;D&gt;&lt;src&gt;_&lt;cell&gt;_&lt;outcome&gt;_&lt;score&gt;</code>: handover h = first decision with the ego's route station within D m of the turn entry (self = the policy's own state; log = never reached, handover on the human log); <code>id</code> = the policy alone, no push; <code>human</code> = the human's logged path, rendered by the sim (log-following driver).<br>
<b>gain</b> g1 / g1.25 / g1.5: from h the served plan's yaw rate is multiplied by g for 3 s (g1 = no lateral push).<br>
<b>profile</b> raw / ve / ve08: raw = the plan's own speed; ve / ve08 = the plan's speed re-sampled to the curvature-limited profile capped at v_e / 0.8 v_e, from h until turn exit + 10 m.<br>
<b>outcome</b>: done = turn completed (P2 rule); corr / offr / coll = zeroed by leaving the corridor / offroad / at-fault collision; rear = rear-ended (the log's following car does not react); end = run ended short of the target; human: ok or the zero kind. Score = AlpaSim score; v = realized entry speed (m/s).
</div>
<div id=list></div>
<div id=big><video loop muted></video><div class=lab></div></div>
<script>
for(const id of ["q","dn","n","list","big","rate"]) window[id+"_"]=document.getElementById(id);
const R=__ROWS__, S=__SCENES__, KINDS=["done","corr","offr","coll","rear","end","oth","ok"], ORDER=__ORDER__, DS=["human","id","D40","D30","D20","D10","D0"];
const byScene={}; for(const r of R)(byScene[r.scene]??=[]).push(r);
const chk=(id,xs)=>document.getElementById(id).innerHTML=xs.map(x=>`<label><input type=checkbox value="${x}" checked> ${x}</label>`).join("");
chk("kinds",KINDS); chk("ds",DS);
const sel=id=>new Set([...document.querySelectorAll(`#${id} input:checked`)].map(i=>i.value));
const dOf=r=>r.label=="human"||r.label=="id"?r.label:r.label.match(/^D\d+/)[0];
const f1=v=>v==null?"-":(+v).toFixed(1);
let cur=-1, scenes=[];
function render(){
  const K=sel("kinds"), D=sel("ds"), q=q_.value.trim(), dn=dn_.checked;
  scenes=ORDER.filter(s=>byScene[s]).filter(s=>!q||s.startsWith(q)).filter(s=>!dn||byScene[s].some(r=>r.done&&r.label!="human"))
    .map(s=>({s,cells:byScene[s].filter(r=>K.has(r.outcome)&&D.has(dOf(r)))})).filter(x=>x.cells.length);
  n_.textContent=`${scenes.length} scenes / ${scenes.reduce((n,x)=>n+x.cells.length,0)} rollouts`;
  list_.innerHTML=scenes.map((x,i)=>{const h=S[x.s];
    return `<div class=scene data-i=${i}><div class=head><b>${x.s}</b> ${h.side}<br>${h.topo}<br>entry k ${h.entry_k} (${(h.entry_k/10).toFixed(1)} s)${h.entry_src=="log"?" log":""}<br>id <span class=${h.id_out}>${h.id_out}</span><br>v_e ${f1(h.v_e)}<br>v id ${f1(h.id_v)}<br>v best ${f1(h.best_v)}${h.best?`<br><span style="color:#888">${h.best}</span>`:""}<br>v human ${f1(h.v_log)}</div><div class=cells>`+
    x.cells.filter(r=>r.main).concat(x.cells.some(r=>!r.main)?[{more:x.cells.filter(r=>!r.main).length}]:[],x.cells.filter(r=>!r.main)).map(r=>r.more?`<div class=more>&#9656; more (${r.more})</div>`:`<div class="cell${r.label=="human"?" human":""}${r.main?"":" extra"}"><img loading=lazy src="p4_videos/thumbs/${x.s}.jpg" data-src="p4_videos/${r.file}"><div class=lab title="${r.file}">${r.label} <b class=${r.outcome}>${r.outcome} ${r.score.toFixed(2)}</b> v ${f1(r.v)}${r.ident?"":" <span style='color:#f66'>NOT IDENTICAL</span>"}</div></div>`).join("")+`</div></div>`}).join("");
  cur=-1;
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
  const t=e.target.closest(".cell"); if(!t) return; const d=t.closest(".scene"); const cs=[...d.querySelectorAll(".cell")]; const k=cs.indexOf(t);
  if(+d.dataset.i!==cur) go(+d.dataset.i); const im=t.querySelector("img"); if(im) toVideo(im); cs.map(c=>c.querySelector("video")).forEach((x,j)=>{if(!x) return; if(j===k){x.currentTime=0; x.play()} else x.pause()})});
function setRate(){document.querySelectorAll("video").forEach(v=>v.playbackRate=+rate_.value)}
rate_.addEventListener("change",setRate);
for(const id of ["kinds","ds","q","dn"]) document.getElementById(id).addEventListener(id=="q"?"input":"change",render);
render();
</script>"""
INDEX.write_text(html.replace("__ROWS__", json.dumps(rows)).replace("__SCENES__", json.dumps(scenes)).replace("__ORDER__", json.dumps(srank)))
print(f"{len(rows)} videos linked under {OUT} ({sum(r['label'] == 'human' for r in rows)} human), gallery {INDEX}; "
      f"not identical: {sum(not r['ident'] for r in rows)}; missing: {', '.join(missing) or 'none'}")
