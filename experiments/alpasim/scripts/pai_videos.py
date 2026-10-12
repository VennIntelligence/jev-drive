#!/usr/bin/env python3
"""Flat symlink index + HTML gallery of PAI rollout videos for watching on the Tokyo desktop.

One flat folder ~/Desktop/pai_videos/<scene8>_<arm>_<score>_<zero kind>.mp4 -> the front-wide mp4 of that rollout
(sorted by name, the arms of one scene sit next to each other; mpv plays the folder as a playlist), plus ~/Desktop/pai_videos.html:
one row per scene, one column per arm, score and zero kind on each cell, filters by arm / kind / score / scene,
j / k move between scenes and play the row, click a cell to play it, double-click to enlarge it; each row shows the scene's first frame (~/.cache/pai_thumbs, one JPEG per scene, all arms start identical), cells are text until the row is active and holds the <video> elements. Idempotent: rerun to pick up new chunks.
Usage: pai_videos.py [ROOT] (default /data/runs/onpol_long/pai281).
"""
import glob, json, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/runs/onpol_long/pai281")
OUT = Path.home() / "Desktop" / "pai_videos"
THUMBS = Path.home() / ".cache" / "pai_thumbs"  # persistent, one first frame per scene (all arms start identical)
INDEX = Path.home() / "Desktop" / "pai_videos.html"
FAIL = ["collision_at_fault", "offroad", "left_corridor_laterally"]
SHORT = {"collision_at_fault": "crash", "offroad": "offroad", "left_corridor_laterally": "corridor"}

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
THUMBS.mkdir(parents=True, exist_ok=True)
rows, todo = [], []
for summ in sorted(glob.glob(str(ROOT / "*/runs/*/sim/aggregate/results-summary.json"))):
    summ = Path(summ)
    run_dir, arm = summ.parents[2], summ.parents[4].name
    for r in json.loads(summ.read_text())["rollouts"]:
        scene, rid, score = r["clipgt_id"], r["rollout_id"], r["score"]
        mp4 = sorted((run_dir / "sim/rollouts" / scene / rid).glob("*camera_front_wide*.mp4"))
        if not mp4 or score is None:
            continue
        kind = next((SHORT[f] for f in FAIL if (r.get("score_metrics") or {}).get(f)), "ok" if score > 0 else "zero")
        s8 = scene.replace("clipgt-", "")[:8]
        name = f"{s8}_{arm}_{score:.2f}_{kind}.mp4"
        (OUT / name).symlink_to(mp4[0])
        thumb = THUMBS / f"{s8}.jpg"
        if not thumb.exists() and all(t[1] != thumb for t in todo):
            todo.append((mp4[0], thumb))
        m = r["metrics"]
        rows.append(dict(scene=s8, arm=arm, score=round(score, 3), kind=kind, file=name, chunk=run_dir.name,
                         dist=round(m.get("dist_traveled_m", 0)), gt=round(m.get("gt_dist_traveled_m", 0))))

def render_thumb(a):
    src, dst = a  # first frame, 240 px wide, q 6: about 10 KB each
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-frames:v", "1", "-vf", "scale=240:-2",
                    "-q:v", "6", str(dst)], check=False)

with ThreadPoolExecutor(16) as ex:
    list(ex.map(render_thumb, todo))
(OUT / "thumbs").symlink_to(THUMBS)
arms = sorted({r["arm"] for r in rows})
html = r"""<!doctype html><meta charset=utf-8><title>PAI rollouts</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#222;padding:8px 12px;display:flex;gap:16px;flex-wrap:wrap;align-items:center;z-index:2}
#bar label{margin-right:6px;white-space:nowrap}
.scene{display:flex;gap:6px;padding:8px 12px;border-bottom:1px solid #333;align-items:flex-start}
.scene.cur{background:#1d2a1d}
.head{width:240px;flex:none;font-family:monospace}
.head img{width:240px;height:267px;object-fit:cover;display:block;background:#000}
.cell{width:240px;flex:none}
.scene:not(.cur) .cell{width:150px}
.cell video{width:240px;height:267px;background:#000;display:block;object-fit:cover}
.scene:not(.cur) .cell .ph{height:30px}
.cell .lab{font-family:monospace;font-size:12px;padding:2px 0}
.ok{color:#8d8}.crash{color:#f66}.offroad{color:#fa4}.corridor{color:#fc6}.zero{color:#aaa}
.empty{color:#444;font-family:monospace;padding-top:6px}
#big{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;flex-direction:column;z-index:9}
#big.on{display:flex}
#big video{height:88vh;max-width:96vw;background:#000}
#big .lab{font-family:monospace;font-size:16px;padding:8px}
</style>
<div id=bar>
 <b>PAI rollouts</b> <span id=n></span>
 <span>arms: <span id=arms></span></span>
 <span>kind: <span id=kinds></span></span>
 <span>score &le; <input id=smax type=number step=0.05 value=1 style="width:4em"></span>
 <span>scene <input id=q placeholder="8 hex" style="width:7em"></span>
 <span>sort <select id=sort><option value=scene>scene</option><option value=spread>arm spread</option><option value=min>min score</option></select></span>
 <span>only scenes where arms disagree <input id=dis type=checkbox></span>
 <span style="color:#888">j / k next / prev scene, space play / pause row, . , frame step, click a cell to play it, double-click to enlarge, Esc closes</span>
</div>
<div id=list></div>
<div id=big><video loop muted></video><div class=lab></div></div>
<script>
for(const id of ["smax","q","dis","sort","n","list","big"]) window[id+"_"]=document.getElementById(id);
const R=__ROWS__, ARMS=__ARMS__, KINDS=["ok","crash","offroad","corridor","zero"];
const byScene={}; for(const r of R){(byScene[r.scene]??={})[r.arm]=r}
const chk=(id,xs,on=true)=>document.getElementById(id).innerHTML=xs.map(x=>`<label><input type=checkbox value="${x}" ${on?"checked":""}> ${x}</label>`).join("");
chk("arms",ARMS); chk("kinds",KINDS);
const sel=id=>[...document.querySelectorAll(`#${id} input:checked`)].map(i=>i.value);
let cur=-1, scenes=[];
function render(){
  const A=sel("arms"), K=new Set(sel("kinds")), smax=+smax_.value, q=q_.value.trim(), dis=dis_.checked;
  scenes=Object.keys(byScene).filter(s=>!q||s.startsWith(q)).map(s=>{
    const cells=A.map(a=>byScene[s][a]).filter(r=>r&&K.has(r.kind)&&r.score<=smax);
    const sc=cells.map(r=>r.score); return {s,cells,min:Math.min(...sc),spread:sc.length?Math.max(...sc)-Math.min(...sc):0};
  }).filter(x=>x.cells.length&&(!dis||x.spread>0.1));
  const o=sort_.value; scenes.sort((a,b)=>o=="scene"?a.s.localeCompare(b.s):o=="min"?a.min-b.min:b.spread-a.spread);
  n_.textContent=`${scenes.length} scenes / ${scenes.reduce((n,x)=>n+x.cells.length,0)} rollouts`;
  list_.innerHTML=scenes.map((x,i)=>`<div class=scene data-i=${i}><div class=head><img loading=lazy src="pai_videos/thumbs/${x.s}.jpg">${x.s} <small style="color:#888">spread ${x.spread.toFixed(2)}</small></div>`+
    A.map(a=>{const r=byScene[x.s][a]; if(!r||!K.has(r.kind)||r.score>smax) return `<div class=cell><div class=empty>${a}</div></div>`;
      return `<div class=cell><div class=ph data-src="pai_videos/${r.file}"></div><div class=lab>${a} <b class=${r.kind}>${r.score.toFixed(2)} ${r.kind}</b> ${r.dist}/${r.gt} m <span style="color:#666">${r.chunk}</span></div></div>`}).join("")+`</div>`).join("");
  cur=-1;
}
function toVideo(ph){const v=document.createElement("video");v.src=ph.dataset.src;v.loop=true;v.muted=true;v.dataset.src=ph.dataset.src;ph.replaceWith(v);return v}
function toPh(v){const p=document.createElement("div");p.className="ph";p.dataset.src=v.dataset.src;v.pause();v.removeAttribute("src");v.load();v.replaceWith(p)}
function go(i){
  if(i<0||i>=scenes.length) return;
  const old=document.querySelector(".scene.cur"); if(old){old.querySelectorAll("video").forEach(toPh); old.classList.remove("cur")}
  cur=i; const d=document.querySelector(`.scene[data-i="${i}"]`); d.classList.add("cur"); d.scrollIntoView({block:"center"});
  d.querySelectorAll(".ph").forEach(ph=>{const v=toVideo(ph); v.play()});
}
const bigV=big_.querySelector("video");
function openBig(cell){const src=(cell.querySelector(".ph,video")||{}).dataset.src; if(!src) return; bigV.src=src; bigV.currentTime=0; bigV.play(); big_.querySelector(".lab").textContent=cell.querySelector(".lab").textContent; big_.classList.add("on")}
function closeBig(){big_.classList.remove("on"); bigV.pause(); bigV.removeAttribute("src"); bigV.load()}
big_.addEventListener("click",e=>{if(e.target===big_) closeBig()});
list_.addEventListener("dblclick",e=>{const c=e.target.closest(".cell"); if(c) openBig(c)});
addEventListener("keydown",e=>{
  if(e.target.tagName=="INPUT"||e.target.tagName=="SELECT") return;
  if(e.key=="Escape") return closeBig();
  const vs=big_.classList.contains("on")?[bigV]:[...document.querySelectorAll(".scene.cur video")];
  if(e.key=="j") go(cur+1); else if(e.key=="k") go(cur-1);
  else if(e.key==" "){e.preventDefault(); vs.forEach(v=>v.paused?v.play():v.pause())}
  else if(e.key=="."||e.key==","){vs.forEach(v=>{v.pause(); v.currentTime+= e.key=="."?0.1:-0.1})}
  else if(e.key=="0"){vs.forEach(v=>{v.currentTime=0})}
});
list_.addEventListener("click",e=>{const t=e.target.closest(".cell,.head"); if(!t) return; const d=t.closest(".scene"); const k=[...d.querySelectorAll(".cell")].indexOf(t); go(+d.dataset.i); d.querySelectorAll("video").forEach((x,j)=>{if(j!==k)x.pause()})});
for(const id of ["arms","kinds","smax","q","sort","dis"]) document.getElementById(id).addEventListener(id=="q"||id=="smax"?"input":"change",render);
render();
</script>"""
html = html.replace("__ROWS__", json.dumps(rows)).replace("__ARMS__", json.dumps(arms))
INDEX.write_text(html)
print(f"{len(rows)} videos linked under {OUT}, gallery {INDEX}")
