"""Replay D3b B/C recorders with target/command overlays and contact sheets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import carla
from PIL import Image, ImageDraw

os.environ.setdefault('DATA_DIR','/data')
from b2d_run import Server
from b2d_tfv6_d2_render import _align, _camera_frames, _overlay, contact_sheet, FONT, SMALL

ROOT=Path('/data/runs/b2d/tfv6-d3')
SHEETS=Path(__file__).resolve().parents[1]/'todos/2026-09-23-tfv6-controller/results/diagnosis/d3'
GROUPS=(('2','2084',0,16.0),('2','2084',1,19.35),('2','2084',2,21.75),
        ('2','27529',0,18.7),('2','27529',1,21.4),('2','27529',2,22.05),
        ('1','2091',0,10.),('1','2091',1,10.),('1','2091',2,10.))


def load_case(level,route,seed,arm):
    case=ROOT/'factorial/cases'/level/f'route-{route}'/f'seed-{seed}'/arm
    done=json.loads((case/'done.json').read_text())
    attempt=case/f"attempt-{done['attempt']}"
    frames=[json.loads(line) for line in (attempt/'frames.jsonl').open()]
    frames=[f for f in frames if f.get('truth')]
    recorder=next((Path(done['run_dir'])/'attempts'/route/'1'/'recorder').glob('*.log'))
    return frames,recorder


def overlay(src,frame,title):
    image=_overlay(src,frame,title)
    draw=ImageDraw.Draw(image)
    nav=frame.get('d3_nav') or {}
    target=nav.get('target_point')
    if target is not None:
        # Redraw the D2 ego-plane inset with a scale that includes the sparse
        # target (often 30-50 m ahead), plus both TFv6 prediction streams.
        x0,y0,panel=image.width-255,38,242
        draw.rectangle((x0,y0,x0+panel,y0+panel),fill=(15,23,31),outline=(220,220,220),width=1)
        origin=(x0+panel/2,y0+panel-22)
        routes=(frame.get('route_prediction') or [],frame.get('waypoint') or [])
        forward=max([20.,abs(float(target[0]))]+[abs(float(p[0])) for line in routes for p in line])
        scale=min(7.,170./forward)
        project=lambda p:(origin[0]+float(p[1])*scale,origin[1]-float(p[0])*scale)
        def path(points,color):
            pts=[project(p) for p in points if p is not None and len(p)>=2]
            if len(pts)>=2:draw.line(pts,fill=color,width=3)
            for px,py in pts:draw.ellipse((px-2,py-2,px+2,py+2),fill=color)
        path(routes[0],(0,158,115));path(routes[1],(55,175,235))
        draw.polygon([(origin[0],origin[1]-10),(origin[0]-7,origin[1]+7),
                      (origin[0]+7,origin[1]+7)],fill=(255,255,255))
        x,y=project(target)
        draw.ellipse((x-7,y-7,x+7,y+7),fill=(255,60,215),outline=(255,255,255),width=2)
        command=nav.get('command') or []
        next_command=nav.get('next_command') or []
        names=['LEFT','RIGHT','STRAIGHT','LANE','CHG L','CHG R']
        label=lambda seq:names[seq.index(max(seq))] if seq else '?'
        draw.rectangle((x0+2,y0+2,x0+panel-2,y0+37),fill=(15,23,31))
        draw.text((x0+8,y0+4),f'TFv6 target {float(target[0]):.1f},{float(target[1]):+.1f} m',
                  font=SMALL,fill=(255,60,215))
        draw.text((x0+8,y0+19),f'cmd {label(command)} -> {label(next_command)}',
                  font=SMALL,fill=(255,255,255))
        draw.text((x0+8,y0+panel-18),'route=green  wp=blue  target=pink',
                  font=SMALL,fill=(220,220,220))
    return image


def render_one(client,level,route,seed,arm,center):
    frames,recorder=load_case(level,route,seed,arm)
    root=ROOT/'render'/f'{level}-{route}-{seed}-{arm}'
    raw=root/'raw';out=root/'overlay';out.mkdir(parents=True,exist_ok=True)
    for old in out.glob('*.jpg'):
        old.unlink()
    locations,images=_camera_frames(client,recorder,frames,center,raw)
    mapped,fit=_align(locations,frames)
    selected=[]
    for image_id,path in sorted(images.items()):
        index=mapped.get(image_id)
        if index is None or not 0<=index<len(frames):continue
        f=frames[index];t=f['sim_time']
        if center-2.2<=t<=center+2.2:selected.append((t,path,f))
    if len(selected)<5:raise RuntimeError(f'{route}/{seed}/{arm}: only {len(selected)} aligned frames')
    for n,(t,path,frame) in enumerate(selected):
        with Image.open(path) as src:
            overlay(src,frame,f'L{level} route {route} seed {seed} {arm}  t={t:.2f}s').save(
                out/f'{n:06d}.jpg',quality=86,optimize=True)
    mp4=root/'window.mp4'
    subprocess.run(['ffmpeg','-loglevel','error','-y','-framerate','10','-i',str(out/'%06d.jpg'),
                    '-c:v','libx264','-pix_fmt','yuv420p','-crf','24',str(mp4)],check=True)
    result={'level':level,'route':route,'seed':seed,'arm':arm,'center_s':center,
            'mp4':str(mp4),'recorder':str(recorder),'frames':len(selected),
            'frame_times_s':[v[0] for v in selected],'alignment_residual_m':fit[0]}
    (root/'render.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--group',help='2-2084-0; omit for all nine')
    parser.add_argument('--route',help='render one route in a fresh CARLA server')
    args=parser.parse_args()
    groups=[g for g in GROUPS if (args.group is None or f'{g[0]}-{g[1]}-{g[2]}'==args.group)
            and (args.route is None or g[1]==args.route)]
    if not groups:parser.error('unknown group')
    if args.group is None and args.route is None:
        # CARLA recorder replay does not reliably recover from a map switch in
        # the same server process. Each route gets a fresh server.
        for route in dict.fromkeys(g[1] for g in groups):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--route',route],check=True)
        return
    server=Server(99,ROOT/'render-server','Epic',gpu_rank=0)
    server.start()
    try:
        client=carla.Client('127.0.0.1',server.port);client.set_timeout(120.)
        for level,route,seed,center in groups:
            sheet=SHEETS/f'd3b-{level}-{route}-{seed}-contact-sheet.png'
            def current_render(arm):
                root=ROOT/'render'/f'{level}-{route}-{seed}-{arm}'
                meta=root/'render.json'
                return (meta.exists() and (root/'window.mp4').exists() and
                        abs(json.loads(meta.read_text())['center_s']-center)<1e-9)
            if sheet.exists() and sheet.stat().st_size<500*1024 and all(
                current_render(arm) for arm in 'BC') and sheet.stat().st_mtime_ns>=max(
                (ROOT/'render'/f'{level}-{route}-{seed}-{arm}'/'render.json').stat().st_mtime_ns
                for arm in 'BC'):
                print('already rendered',level,route,seed,flush=True)
                continue
            town='Town04' if route=='27529' else 'Town12'
            if not client.get_world().get_map().name.endswith(town):
                client.load_world(town)
            rows=[]
            for arm in 'BC':
                print('replay',level,route,seed,arm,flush=True)
                rows.append(render_one(client,level,route,seed,arm,center))
            size=contact_sheet(rows,sheet)
            print('sheet',sheet,size,flush=True)
    finally:
        server.stop()


if __name__=='__main__':main()
