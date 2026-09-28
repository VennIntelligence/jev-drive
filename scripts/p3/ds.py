"""nq4 P3: drivestudio (OmniRe) glue -- preprocess, sky masks, train, render x+ / x- / real.
Registration: todos/2026-09-26-night-queue-4.md, P section, [P3] 18:50 entry. Runs from the drivestudio checkout.

  prep    (envs/p3-wodprep) scene-flow tfrecords -> drivestudio's processed layout, scene k = k-th picked segment
  sky     (envs/drivestudio) sky masks with SegFormer-B5 Cityscapes (HF port of the checkpoint drivestudio uses)
  train   (envs/drivestudio) OmniRe (configs/omnire.yaml, dataset waymo/3cams) without SMPL: pedestrians become
          DeformableNodes; every pedestrian gets a node (only_moving off), so standing ones can be deleted too
  render  (envs/drivestudio) the three worlds along the logged poses at 5 Hz inside [f0 - 3 s, f0 + 2 s]:
          real (the loaded, undistorted log image), plus (re-render), minus (the corridor pedestrians' nodes hidden)
"""
import argparse, json, os, sys, time
from pathlib import Path

DS = Path(os.environ.get("DATA_DIR", Path.home() / "data")) / "third_party/drivestudio"
CAMS = ("front", "front_left", "front_right")        # drivestudio waymo/3cams order: 0 FRONT, 1 FRONT_LEFT, 2 FRONT_RIGHT
HZ, STEP, PRE, POST = 10, 2, 30, 20                  # log rate, 5 Hz stride, window in log frames


def prep(a):
    os.chdir(DS)
    sys.path.insert(0, str(DS))
    from datasets.waymo.waymo_preprocess import WaymoProcessor
    names = json.loads(Path(a.scenes).read_text())["segments"]
    ids = list(range(len(names))) if a.ids is None else a.ids
    p = WaymoProcessor(a.raw, a.out, "training", process_id_list=ids, workers=a.workers)
    p.tfrecord_pathnames = [f"{a.raw}/segment-{n}_with_camera_labels.tfrecord" for n in names]
    p.convert()


def sky(a):
    import numpy as np, torch
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation
    revision = "2c6f153e4c23c229e2fa2b188eb250607e030cd8"
    cached = DS.parents[1] / "ckpt/p3-segformer" / revision
    if (cached / "DONE").exists():
        m = SegformerForSemanticSegmentation.from_pretrained(str(cached), local_files_only=True)
    else:
        m = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", revision=revision)
    print(f"sky model revision={revision}, local_cache={cached if (cached / 'DONE').exists() else None}", flush=True)
    m = m.cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    for sd in map(Path, a.scene_dirs):
        (sd / "sky_masks").mkdir(exist_ok=True)
        files = sorted(f for f in (sd / "images").glob("*.jpg") if int(f.stem.split("_")[1]) in a.cams)
        for f in files:
            out = sd / "sky_masks" / f"{f.stem}.png"
            if out.exists():
                continue
            im = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).cuda().permute(2, 0, 1)[None].float() / 255
            H, W = im.shape[-2:]
            x = torch.nn.functional.interpolate((im - mean) / std, size=(1024, int(1024 * W / H) // 32 * 32), mode="bilinear")
            with torch.no_grad():
                lg = m(pixel_values=x.half()).logits.float()
            lab = torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear").argmax(1)[0]
            Image.fromarray(((lab == 10).cpu().numpy() * 255).astype(np.uint8)).save(out)   # cityscapes 10 = sky
        print(f"sky masks: {sd} {len(files)} images", flush=True)


def config(a) -> Path:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(DS / "configs/omnire.yaml")
    del cfg.model["SMPLNodes"]
    cfg.model.DeformableNodes.init.only_moving = False
    if getattr(a, "rigid_all", False):          # vehicle pairs: stopped vehicles must be nodes too, to be deletable
        cfg.model.RigidNodes.init.only_moving = False
    if a.iters:
        cfg.trainer.optim.num_iters = a.iters
    p = Path(a.out_root) / f"omnire_nosmpl_{a.iters or 'default'}{'_rigidall' if getattr(a, 'rigid_all', False) else ''}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, p)
    return p


def train(a):
    import subprocess
    cfg = config(a)
    cmd = [sys.executable, "tools/train.py", "--config_file", str(cfg), "--output_root", a.out_root, "--project", "p3",
           "--run_name", f"{a.scene:03d}", "dataset=waymo/3cams", f"data.data_root={a.data_root}", f"data.scene_idx={a.scene}",
           "data.pixel_source.load_smpl=False", "data.preload_device=cuda"]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=DS, env={**os.environ, "PYTHONPATH": str(DS)}))


def _box_mask(pose, size, K, c2w, H, W, pad=12):
    """Pixel rectangle (expanded by pad) around the projection of a 3D box; None when behind the camera."""
    import numpy as np
    c = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * size
    pw = (pose[:3, :3] @ c.T).T + pose[:3, 3]
    w2c = np.linalg.inv(c2w)
    pc = (w2c[:3, :3] @ pw.T).T + w2c[:3, 3]
    if (pc[:, 2] < 0.1).all():
        return None
    pc = pc[pc[:, 2] >= 0.1]
    uv = (K @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:]
    x0, y0 = np.floor(uv.min(0) - pad).astype(int)
    x1, y1 = np.ceil(uv.max(0) + pad).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)


def _load(run: Path, ckpt: str | None, node_type: str = "DeformableNodes"):
    """Dataset, trainer (eval mode, checkpoint loaded), pixel source, deformable node model, node -> dataset instance
    keys and node -> Waymo laser id, from a finished OmniRe run."""
    import torch
    from omegaconf import OmegaConf
    os.chdir(DS)
    sys.path.insert(0, str(DS))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str
    cfg = OmegaConf.load(run / "config.yaml")
    ds = DrivingDataset(data_cfg=cfg.data)
    tr = import_str(cfg.trainer.type)(**cfg.trainer, num_timesteps=ds.num_img_timesteps, model_config=cfg.model,
                                      num_train_images=len(ds.train_image_set), num_full_images=len(ds.full_image_set),
                                      test_set_indices=ds.test_timesteps, scene_aabb=ds.get_aabb().reshape(2, 3), device="cuda")
    ckpt = ckpt or str(sorted(run.glob("checkpoint_*.pth"))[-1])
    # Upstream checkpoints contain NumPy scalars, so PyTorch >=2.6's new
    # weights-only default rejects our own locally trained checkpoint.
    tr.load_state_dict(torch.load(ckpt, weights_only=False), load_only_model=True, strict=True)
    tr.set_eval()
    ps = ds.pixel_source
    # node instance k <-> dataset instance <-> Waymo laser_object_id (instances_info.json "id")
    info = json.loads((Path(cfg.data.data_root) / f"{cfg.data.scene_idx:03d}" / "instances/instances_info.json").read_text())
    true2wid = {int(k): v["id"] for k, v in info.items()}
    peds = [k for k in range(len(ps.instances_true_id)) if info[str(int(ps.instances_true_id[k]))]["class_name"] == "Pedestrian"]
    dcfg = cfg.model[node_type].init
    kw = {"exclude_smpl": False} if node_type == "DeformableNodes" else {}
    keys = list(ds.get_init_objects(cur_node_type=node_type, instance_max_pts=dcfg.instance_max_pts,
                                    only_moving=dcfg.only_moving, traj_length_thres=dcfg.traj_length_thres, **kw))
    node = tr.models[node_type]
    assert len(keys) == node.instances_fv.shape[1], (len(keys), node.instances_fv.shape)
    wid_of = [true2wid[int(ps.instances_true_id[k])] for k in keys]
    return cfg, ds, tr, ps, node, keys, wid_of, peds, ckpt


def render(a):
    import numpy as np, torch
    from PIL import Image
    t_start = time.time()
    run = Path(a.run)
    sel = json.loads(Path(a.target).read_text())
    cfg, ds, tr, ps, node, keys, wid_of, peds, ckpt = _load(run, a.ckpt, sel.get("node_type", "DeformableNodes"))
    delete = [wid_of.index(w) for w in sel["delete_tracks"] if w in wid_of]
    missing = [w for w in sel["delete_tracks"] if w not in wid_of]
    ds_del = [keys[k] for k in delete]
    nf = ps.num_timesteps if hasattr(ps, "num_timesteps") else ds.num_img_timesteps
    f0 = int(sel["f0"])
    frames = list(range(max(f0 - PRE, 0), min(f0 + POST, nf - 1) + 1, STEP))
    out = Path(a.out)
    for w in ("real", "plus", "minus"):
        for c in CAMS:
            (out / w / "cams" / c).mkdir(parents=True, exist_ok=True)
    fv = node.instances_fv.clone()
    hide = fv.clone()
    hide[:, delete] = False
    stats, calib, jl = [], {}, {w: [] for w in ("real", "plus", "minus")}
    to8 = lambda x: (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
    with torch.no_grad():
        for t in frames:
            rec = {w: {} for w in jl}
            for ci, c in enumerate(CAMS):
                ii, ci_ = ds.full_image_set.get_image(t * ps.num_cams + ci, 1)
                ii = {k: v.cuda() if torch.is_tensor(v) else v for k, v in ii.items()}
                ci_ = {k: v.cuda() if torch.is_tensor(v) else v for k, v in ci_.items()}
                node.instances_fv.copy_(fv)
                plus = tr(ii, ci_)["rgb"]
                if t == frames[len(frames) // 2] and ci == 0:        # determinism: the same view twice
                    again = tr(ii, ci_)["rgb"]
                    det = float((plus - again).abs().max())
                node.instances_fv.copy_(hide)
                minus = tr(ii, ci_)["rgb"]
                node.instances_fv.copy_(fv)
                real = ii["pixels"]
                H, W = real.shape[:2]
                K = ci_["intrinsics"].cpu().numpy().astype(np.float64)
                c2w = ci_["camera_to_world"].cpu().numpy().astype(np.float64)
                if c not in calib:
                    calib[c] = {"K": K.tolist(), "wh": [W, H]}
                imgs = {"real": to8(real), "plus": to8(plus), "minus": to8(minus)}
                fn = f"{2 * t:07d}.jpg"                       # 20 Hz tick units: 5 Hz frames are 4 apart
                for w, im in imgs.items():
                    Image.fromarray(im).save(out / w / "cams" / c / fn, quality=95)
                    rec[w][c] = f"cams/{c}/{fn}"
                # deleted-region masks: projected boxes of the deleted pedestrians present at t
                inbox = np.zeros((H, W), bool)
                for k in ds_del:
                    if ps.per_frame_instance_mask[t, k]:
                        r = _box_mask(ps.instances_pose[t, k].cpu().numpy(), ps.instances_size[k].cpu().numpy(), K, c2w, H, W)
                        if r:
                            inbox[r[1]:r[3], r[0]:r[2]] = True
                ped = np.zeros((H, W), bool)                  # every pedestrian box (0 px pad) for the in-box PSNR
                for k in peds:
                    if ps.per_frame_instance_mask[t, k]:
                        r = _box_mask(ps.instances_pose[t, k].cpu().numpy(), ps.instances_size[k].cpu().numpy(), K, c2w, H, W, 0)
                        if r:
                            ped[r[1]:r[3], r[0]:r[2]] = True
                d = np.abs(imgs["plus"].astype(np.int16) - imgs["minus"].astype(np.int16)).max(-1)
                e = (imgs["plus"].astype(np.float64) - imgs["real"].astype(np.float64)) / 255
                mse = (e ** 2).mean()
                stats.append({"t": t, "cam": c, "psnr": float(-10 * np.log10(mse)),
                              "psnr_ped": float(-10 * np.log10((e[ped] ** 2).mean())) if ped.any() else np.nan,
                              "ped_px": int(ped.sum()), "del_px": int(inbox.sum()),
                              "diff_px_in": int((d[inbox] > 8).sum()), "diff_px_out": int((d[~inbox] > 8).sum()),
                              "diff_any_out": int((d[~inbox] > 0).sum())})
            for w in jl:
                jl[w].append({"frame": 2 * t, "t": t / HZ, "files": rec[w]})
    for w, rows in jl.items():
        (out / w / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext = {c: np.loadtxt(Path(cfg.data.data_root) / f"{cfg.data.scene_idx:03d}" / "extrinsics" / f"{i}.txt").tolist()
           for i, c in enumerate(CAMS)}
    intr = {c: np.loadtxt(Path(cfg.data.data_root) / f"{cfg.data.scene_idx:03d}" / "intrinsics" / f"{i}.txt").tolist()
            for i, c in enumerate(CAMS)}
    meta = {**sel, "ckpt": ckpt, "frames": frames, "calib": calib, "extrinsics_cam_to_ego": ext, "intrinsics_raw": intr,
            "node_mapping": [{"node_index": i, "dataset_instance": int(k),
                              "true_id": int(ps.instances_true_id[k]), "waymo_id": wid_of[i],
                              "deleted": i in delete} for i, k in enumerate(keys)],
            "deleted_node_instances": delete, "deleted_missing": missing, "determinism_max_abs": det,
            "n_deformable_instances": len(keys), "render_s": round(time.time() - t_start, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    import pandas as pd
    pd.DataFrame(stats).to_csv(out / "render_stats.csv", index=False)
    print(json.dumps({k: meta[k] for k in ("deleted_node_instances", "deleted_missing", "determinism_max_abs", "render_s")}))


def clip(a):
    """Review clip of one scene: FRONT camera at 10 Hz over [f0 - pre, f0 + post]; 2 x 2 grid of the log image, x+,
    x- and |x+ - x-| (deleted pedestrians' boxes outlined there only, red on the filter's react frames), with a caption
    from the exam-item filter labels (jevdrive.nq4_p3_filter scenes -> clip_labels/<key>.json). H.264 MP4, real time."""
    import imageio.v2 as imageio
    import numpy as np, torch
    from PIL import Image, ImageDraw, ImageFont
    run = Path(a.run)
    cfg, ds, tr, ps, node, keys, wid_of, peds, ckpt = _load(run, a.ckpt)
    sel = json.loads(Path(a.target).read_text())
    lab = json.loads(Path(a.labels).read_text())
    delete = [wid_of.index(w) for w in sel["delete_tracks"] if w in wid_of]
    ds_del = {keys[k]: wid_of[k] for k in delete}
    nf = ps.num_timesteps if hasattr(ps, "num_timesteps") else ds.num_img_timesteps
    f0 = int(sel["f0"])
    frames = list(range(max(f0 - int(a.pre * HZ), 0), min(f0 + int(a.post * HZ), nf - 1) + 1))
    fv = node.instances_fv.clone()
    hide = fv.clone()
    hide[:, delete] = False
    ttf = Path(sys.executable).parents[1] / "lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
    font, big = ImageFont.truetype(str(ttf), 17), ImageFont.truetype(str(ttf), 26)
    names = {w: chr(65 + i) for i, w in enumerate(sel["delete_tracks"])}
    to8 = lambda x: (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(out, fps=HZ, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    with torch.no_grad():
        for t in frames:
            ii, ci_ = ds.full_image_set.get_image(t * ps.num_cams + 0, 1)
            ii = {k: v.cuda() if torch.is_tensor(v) else v for k, v in ii.items()}
            ci_ = {k: v.cuda() if torch.is_tensor(v) else v for k, v in ci_.items()}
            node.instances_fv.copy_(fv)
            plus = to8(tr(ii, ci_)["rgb"])
            node.instances_fv.copy_(hide)
            minus = to8(tr(ii, ci_)["rgb"])
            node.instances_fv.copy_(fv)
            real = to8(ii["pixels"])
            H, W = real.shape[:2]
            K = ci_["intrinsics"].cpu().numpy().astype(np.float64)
            c2w = ci_["camera_to_world"].cpu().numpy().astype(np.float64)
            d = np.abs(plus.astype(np.int16) - minus.astype(np.int16)).max(-1)
            dimg = np.stack([np.clip(d * 4, 0, 255).astype(np.uint8)] * 3, -1)
            fl = lab["frames"].get(str(t), {})
            panels = [Image.fromarray(x) for x in (real, plus, minus, dimg)]
            dr = ImageDraw.Draw(panels[3])
            for k, w in ds_del.items():
                if ps.per_frame_instance_mask[t, k]:
                    r = _box_mask(ps.instances_pose[t, k].cpu().numpy(), ps.instances_size[k].cpu().numpy(), K, c2w, H, W, 0)
                    if r:
                        react = fl.get(w, {}).get("react", False)
                        dr.rectangle(r, outline=(255, 40, 40) if react else (255, 210, 0), width=3)
                        dr.text((r[0], max(r[1] - 22, 0)), names[w], fill=(255, 255, 255), font=font)
            for p, txt in zip(panels, ("log", "x+ (re-render)", "x- (deleted)", "|x+ - x-| x4, deleted boxes")):
                ImageDraw.Draw(p).text((10, 8), txt, fill=(255, 255, 255), font=big, stroke_width=2, stroke_fill=(0, 0, 0))
            grid = Image.new("RGB", (2 * W, 2 * H + 112), (20, 20, 20))
            for i, p in enumerate(panels):
                grid.paste(p, ((i % 2) * W, (i // 2) * H))
            g = ImageDraw.Draw(grid)
            react_any = any(v.get("react") for v in fl.values())
            head = (f"{sel['key']}  t - f0 = {(t - f0) / HZ:+.1f} s  ego {lab['v'][t]:.1f} m/s   "
                    f"{'REACT (filter: should-react frame)' if react_any else 'no react'}"
                    f"{'   [scored 5 Hz frame]' if (t - f0) % 2 == 0 and -24 <= t - f0 <= 20 else ''}")
            g.text((12, 2 * H + 8), head, fill=(255, 90, 90) if react_any else (230, 230, 230), font=big)
            parts = []
            for w in sel["delete_tracks"]:
                v = fl.get(w)
                parts.append(f"{names[w]}: -" if v is None else
                             f"{names[w]}: d {v['d']:.0f} m, lat {v['L']:+.1f} m, ttr {v['ttr']:.1f} s, "
                             f"{'lane' if v['in_lane'] else 'off-lane'}, {'LEAD' if v['lead'] else 'no lead'}, "
                             + (f"label {v['lab_h']:.0f} px" if v["lab_h"] is not None else "no label"))
            for j in range(0, len(parts), 2):
                g.text((12, 2 * H + 44 + 30 * (j // 2)), "   |   ".join(parts[j:j + 2]), fill=(210, 210, 210), font=font)
            wr.append_data(np.asarray(grid))
    wr.close()
    print(json.dumps({"clip": str(out), "frames": len(frames), "deleted": len(delete)}))


def main():
    # Absolute interpreter invocation does not activate its console tools.
    # nvdiffrast JIT needs the ninja executable installed in this same env.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--raw"); p.add_argument("--out"); p.add_argument("--scenes")
    p.add_argument("--ids", type=int, nargs="*"); p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("sky"); p.add_argument("scene_dirs", nargs="+"); p.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    p = sp.add_parser("train"); p.add_argument("--scene", type=int); p.add_argument("--data-root"); p.add_argument("--out-root")
    p.add_argument("--iters", type=int, default=0)
    p.add_argument("--rigid-all", action="store_true", help="every vehicle a rigid node (vehicle deletion pairs)")
    p = sp.add_parser("render"); p.add_argument("--run"); p.add_argument("--target"); p.add_argument("--out"); p.add_argument("--ckpt")
    p = sp.add_parser("clip"); p.add_argument("--run"); p.add_argument("--target"); p.add_argument("--labels"); p.add_argument("--out")
    p.add_argument("--ckpt"); p.add_argument("--pre", type=float, default=4.0); p.add_argument("--post", type=float, default=4.0)
    a = ap.parse_args()
    {"prep": prep, "sky": sky, "train": train, "render": render, "clip": clip}[a.cmd](a)


if __name__ == "__main__":
    main()
