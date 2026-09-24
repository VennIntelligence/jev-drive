"""Driving backbones in the P3 ladder: Alpamayo 1.5 prefill-only features (alpamayo1.5 venv).
Pre-registration: todos/2026-09-24-driving-backbones/README.md. Targets from `python -m jevdrive.drive_backbones
--steps prepare` (alp_targets.npz, op_plan.json for the JPEG spans).

Input per target frame f, as the WOD exam (scripts/wod_zeroshot_alpamayo.py) except for the cameras: the four
f-theta views are rendered from FRONT / FRONT_LEFT / FRONT_RIGHT only (the box's slim shards; uncovered = black),
frames f-3..f, egomotion history from the WOD past states, the no-nav prompt. The full prompt (with the history
tokens fused) goes through `vlm.model` once -- no lm_head, no reasoning decode, no flow-matching expert -- and
forward hooks pool it the way `jevdrive.features.QwenFeatures` does:

  vit_mean     ViT tokens before the merger, mean over every patch of the 16 images
  vis_mean     merger output (the image-token embeddings), mean
  L{k}_mean    decoder layer k's output, mean over the image-token positions
  L{k}_last    decoder layer k's output at the last prompt token

Steps:
  extract   the subset targets (this process takes every n-th of them with --shard i/n), batched; writes
            $DATA_DIR/processed/drive_backbones/alp/<set>/part<i>of<n>_<chunk>.npz (resumable per chunk)
  equiv     batched hook features vs one-sample `output_hidden_states` on --n frames; renderer vs the exam's
  rater7    features of the 479 rater frames from the exam's 7-camera packages and from their front three alone
  native    Alpamayo's own K=6 no-nav trajectories from front-three input on the rater frames (exam config)
"""
import argparse, io, json, queue, sys, threading, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import drive_backbones as D  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.alpamayo import infer as I  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

LAYERS = (9, 18, 27, 36)
FRONT3 = (1, 2, 3)
DEV = "cuda"


# ---------------------------------------------------------------- rendering

class Renderer:
    """The exam's GPU renderer (wod_zeroshot_alpamayo.render) with the per-calibration sampling grids cached.

    The exam recomputed the rays and the source choice for every target; they depend only on the calibration,
    which is fixed per sequence, so here they are built once per calibration and reused. Same float math."""

    def __init__(self):
        self.rays = {v: G.ftheta_rays(torch, G.PAI_RIG[v], 1.0, device=DEV) for v in G.PAI_ORDER}
        self.cache, self.key = None, None

    def maps(self, key, cal):
        """Per view and source camera: the flat indices of the output pixels it covers and their sampling grid."""
        if key != self.key:
            out = []
            for v in G.PAI_ORDER:
                src, U, V = G.choose_sources(torch, self.rays[v], cal)
                per = []
                for i, c in enumerate(cal):
                    idx = (src == i).flatten().nonzero().squeeze(1)
                    if not len(idx):
                        per.append(None)
                        continue
                    w, h = cal[c]["width"], cal[c]["height"]
                    grid = torch.stack([(U + 0.5) / w * 2 - 1, (V + 0.5) / h * 2 - 1], -1).float()
                    per.append((idx, grid.view(-1, 2)[idx][None, None]))
                out.append(per)
            self.cache, self.key = out, key
        return self.cache

    def __call__(self, key, cal, jpgs) -> torch.Tensor:
        """jpgs: {camera id: [4 encoded JPEG uint8 tensors, oldest first]} -> (4 views, 4, 3, 1080, 1920) uint8 CPU.

        Bilinear sampling is per pixel, so sampling only the covered pixels of each source camera and scattering
        them gives exactly the exam's full-frame grid_sample + where (checked bit for bit by `equiv`)."""
        from torchvision.io import decode_jpeg
        cams = list(cal)
        dec = decode_jpeg([j for c in cams for j in jpgs[c]], device=DEV)
        imgs = [torch.stack(dec[i * 4:(i + 1) * 4]).float() for i in range(len(cams))]
        H, W = self.rays[G.PAI_ORDER[0]].shape[:2]
        out = []
        for per in self.maps(key, cal):
            acc = torch.zeros(4, 3, H * W, device=DEV)
            for i, ig in enumerate(per):
                if ig is not None:
                    idx, grid = ig
                    acc[:, :, idx] = F.grid_sample(imgs[i], grid.expand(4, -1, -1, -1), mode="bilinear",
                                                   padding_mode="border", align_corners=False)[:, :, 0]
            out.append(acc.view(4, 3, H, W).round().clamp(0, 255).to(torch.uint8))
        return torch.stack(out).cpu()


def slim_jpegs(name: str, spans: dict, shard_dir: Path) -> dict:
    """{camera id: 4 JPEG byte tensors} of frames f-3..f from the slim shards (FRONT, FRONT_LEFT, FRONT_RIGHT)."""
    out = {c: [] for c in FRONT3}
    fh = {}
    for hn in Z.history_names(name, 3):
        sp = spans[hn]
        f = fh.get(sp[0]) or fh.setdefault(sp[0], open(shard_dir / sp[0], "rb"))
        for k, c in enumerate(FRONT3):
            f.seek(sp[1 + 2 * k])
            out[c].append(torch.frombuffer(bytearray(f.read(sp[2 + 2 * k])), dtype=torch.uint8))
    for f in fh.values():
        f.close()
    return out


def op_calib(seq: str, cal_json: dict) -> dict:
    return {int(c): {"intrinsic": np.array(d["intrinsic"]), "extrinsic": np.array(d["extrinsic"]),
                     "width": int(d["width"]), "height": int(d["height"])} for c, d in cal_json[seq].items()
            if int(c) in FRONT3}


# ---------------------------------------------------------------- model inputs and pooled features

def tokenize(frames: torch.Tensor, past: np.ndarray, processor) -> dict:
    """CPU tensors of one target, the exam's message and tokenization (no nav)."""
    from alpamayo1_5 import helper
    msgs = helper.create_message(frames=frames.flatten(0, 1), camera_indices=torch.tensor([0, 1, 2, 6]), nav_text=None)
    tok = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                        return_dict=True, return_tensors="pt")
    xyz, rot = Z.alpamayo_history(past)
    return {"tok": tok, "xyz": torch.from_numpy(xyz).float(), "rot": torch.from_numpy(rot).float()}


def collate(items: list[dict]) -> dict:
    ids = torch.cat([x["tok"]["input_ids"] for x in items])          # identical length: asserted by torch.cat
    return {"input_ids": ids, "attention_mask": torch.cat([x["tok"]["attention_mask"] for x in items]),
            "pixel_values": torch.cat([x["tok"]["pixel_values"] for x in items]),
            "image_grid_thw": torch.cat([x["tok"]["image_grid_thw"] for x in items]),
            "xyz": torch.stack([x["xyz"] for x in items])[:, None], "rot": torch.stack([x["rot"] for x in items])[:, None]}


class Pooler:
    """Forward hooks that pool inline, so nothing but a few vectors per layer leaves the GPU."""

    def __init__(self, model, layers=LAYERS):
        self.vlm, self.layers, self.out, self.ipos = model.vlm.model, layers, {}, None
        self.image_token_id = model.vlm.config.image_token_id
        lm, vis = self.vlm.language_model, self.vlm.visual
        for k in layers:
            lm.layers[k - 1].register_forward_hook(self._layer(k))
        vis.merger.register_forward_pre_hook(lambda m, a: self._vis("vit_mean", a[0]))
        vis.merger.register_forward_hook(lambda m, a, o: self._vis("vis_mean", o))

    def _vis(self, key, x):
        self.out[key] = x.float().view(self.b, -1, x.shape[-1]).mean(1)

    def _layer(self, k):
        def f(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            self.out[f"L{k:02d}_mean"] = h[:, self.ipos].sum(1, dtype=torch.float32) / self.ipos.numel()
            self.out[f"L{k:02d}_last"] = h[:, -1].float()
        return f

    @torch.inference_mode()
    def __call__(self, model, batch: dict) -> dict:
        b = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}
        ids = model.fuse_traj_tokens(b["input_ids"], {"ego_history_xyz": b["xyz"], "ego_history_rot": b["rot"]})
        assert bool(b["attention_mask"].all()), "padding in a batch: prompts of different length"
        self.b, self.out = len(ids), {}
        self.ipos = (ids[0] == self.image_token_id).nonzero().squeeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.vlm(input_ids=ids, attention_mask=b["attention_mask"], pixel_values=b["pixel_values"],
                     image_grid_thw=b["image_grid_thw"], use_cache=False)
        return dict(self.out)


# ---------------------------------------------------------------- pipelines

def producer(targets, prep, q, n_threads):
    """Prepare targets in `n_threads` threads (JPEG read, GPU render, processor on the CPU), in order."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n_threads) as ex:
        futs = []
        for t in targets:
            futs.append(ex.submit(prep, t))
            while len(futs) > 4 * n_threads:
                q.put(futs.pop(0).result())
        for f in futs:
            q.put(f.result())
    q.put(None)


def _vision_attention(self, hidden_states, cu_seqlens, position_embeddings, **_):
    """jevdrive.features._vision_attention (that module needs the project venv): one batched SDPA call for a batch
    of same-size images instead of the stock per-image split, which syncs the host in every block."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
    L, b = hidden_states.shape[0], cu_seqlens.numel() - 1
    q, k, v = self.qkv(hidden_states).reshape(L, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    q, k, v = (t.reshape(b, L // b, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
    return self.proj(o.transpose(1, 2).reshape(L, -1))


def load_model(log, fast_vision: bool = True):
    """fast_vision: the vision attention of `jevdrive.features._vision_attention` -- every image here has the same
    size, so the stock per-image SDPA loop (16 x batch calls per block, each after a host sync) becomes one batched
    call with the same math. Checked against the stock path by `equiv`."""
    import types
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    model = Alpamayo1_5.from_pretrained(I.REPO, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    if fast_vision:
        for blk in model.vlm.model.visual.blocks:
            blk.attn.forward = types.MethodType(_vision_attention, blk.attn)
    log.info(f"model loaded (fast_vision={fast_vision})")
    return model, helper.get_processor(model.tokenizer)


def cmd_extract(a, log):
    from jevdrive.common import data_dir
    si, sn = map(int, a.shard.split("/"))
    tg = np.load(D.root() / "alp_targets.npz")
    plan = json.loads((D.root() / "op_plan.json").read_text())
    spans = plan["spans"]
    cal_json = json.loads((Z.root() / "op_calib.json").read_text())
    names = tg["name"].astype(str)
    ok = np.array([all(h in spans for h in Z.history_names(n, 3)) and len(Z.history_names(n, 3)) == 4 for n in names])
    idx = np.flatnonzero(ok)[si::sn][: a.limit or None]
    log.info(f"{int(ok.sum())}/{len(names)} targets have f-3..f; this shard {len(idx)}, batch {a.batch}")
    out = D.root("alp", a.set_name)
    chunk = a.batch * 32
    chunks = [idx[i:i + chunk] for i in range(0, len(idx), chunk)]
    todo = [(j, c) for j, c in enumerate(chunks) if not (out / f"part{si}of{sn}_{j:04d}.npz").exists()]
    model, processor = load_model(log)
    pool, rend = Pooler(model), Renderer()
    shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
    lock = threading.Lock()

    def prep(i):
        n = names[i]
        seq = n.rsplit("-", 1)[0]
        cal = op_calib(seq, cal_json)
        jp = slim_jpegs(n, spans, shard_dir)
        with lock:                                  # one renderer (cached grids) shared by the threads
            frames = rend(seq, cal, jp)
        return i, tokenize(frames, tg["past"][i], processor)

    t0, n_done, t_gpu = time.time(), 0, 0.0
    for j, c in todo:
        q = queue.Queue(maxsize=4 * a.threads)
        threading.Thread(target=producer, args=(list(c), prep, q, a.threads), daemon=True).start()
        rows, buf = {}, []
        while True:
            item = q.get()
            if item is not None:
                buf.append(item)
            if buf and (len(buf) == a.batch or item is None):
                tg0 = time.perf_counter()
                feats = pool(model, collate([x[1] for x in buf]))
                torch.cuda.synchronize()
                t_gpu += time.perf_counter() - tg0
                for k, v in feats.items():
                    rows.setdefault(k, []).append(v.cpu().numpy().astype(np.float16))
                rows.setdefault("_i", []).extend(x[0] for x in buf)
                n_done += len(buf)
                buf = []
            if item is None:
                break
        np.savez(out / f"part{si}of{sn}_{j:04d}.npz", name=names[np.array(rows.pop("_i"))],
                 **{k: np.concatenate(v) for k, v in rows.items()})
        el = time.time() - t0
        log.info(f"chunk {j + 1}/{len(chunks)}: {n_done} frames, {1e3 * el / n_done:.0f} ms/frame wall, "
                 f"{1e3 * t_gpu / n_done:.0f} ms/frame GPU, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, "
                 f"ETA {(len(idx) - n_done - (len(idx) - sum(len(c) for _, c in todo))) * el / max(n_done, 1) / 60:.0f} min")
        log.event("chunk", chunk=j, frames=n_done, wall_s=el, gpu_s=t_gpu)
    log.event("end", frames=n_done, wall_s=time.time() - t0, gpu_s=t_gpu,
              ms_per_frame_wall=1e3 * (time.time() - t0) / max(n_done, 1), ms_per_frame_gpu=1e3 * t_gpu / max(n_done, 1),
              peak_gb=torch.cuda.max_memory_allocated() / 2**30)


def cmd_equiv(a, log):
    """(1) batched + hook-pooled == one-sample output_hidden_states; (2) cached renderer == the exam renderer."""
    import wod_zeroshot_alpamayo as WA
    from jevdrive.common import data_dir
    sets = Z.load_sets()
    names = [str(n) for n in sets["rater"]["name"][: a.n]]
    past = sets["rater"]["past"][: a.n]
    model, processor = load_model(log)
    rend = Renderer()
    diffs = []
    items = []
    for n, p in zip(names, past):
        z = np.load(Z.root("packages") / f"{n}.npz")
        cal = Z.read_calib(z, Z.ALP_SRC)
        jp = {c: [torch.from_numpy(z[f"jpg_{k}_{c}"]) for k in range(4)] for c in Z.ALP_SRC}
        mine, ref = rend(n, cal, jp), WA.render(z)
        diffs.append(int((mine.int() - ref.int()).abs().max()))
        items.append(tokenize(mine, p, processor))
    log.info(f"renderer vs exam: max |diff| per frame {diffs}")
    pool = Pooler(model)
    batched = pool(model, collate(items))
    single = [pool(model, collate([x])) for x in items]
    import types
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention
    for blk in model.vlm.model.visual.blocks:                      # the stock per-image attention, for reference
        blk.attn.forward = types.MethodType(Qwen3VLVisionAttention.forward, blk.attn)
    stock = pool(model, collate(items))
    for k in batched:
        d = (batched[k] - stock[k]).norm(dim=1) / stock[k].norm(dim=1)
        log.info(f"fast vision vs stock, {k}: rel L2 max {float(d.max()):.2e}")
    rows = []
    for k in batched:
        s = torch.cat([x[k] for x in single])
        d = (batched[k] - s).norm(dim=1) / s.norm(dim=1)
        cos = F.cosine_similarity(batched[k], s, dim=1)
        rows.append({"array": k, "batched_vs_single_rel_l2_max": float(d.max()), "cos_min": float(cos.min())})
    # hook pooling vs output_hidden_states on one sample (hidden_states[k] = output of layer k)
    x = collate(items[:1])
    b = {k: v.to(DEV) for k, v in x.items()}
    ids = model.fuse_traj_tokens(b["input_ids"], {"ego_history_xyz": b["xyz"], "ego_history_rot": b["rot"]})
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hs = model.vlm.model(input_ids=ids, attention_mask=b["attention_mask"], pixel_values=b["pixel_values"],
                             image_grid_thw=b["image_grid_thw"], output_hidden_states=True, use_cache=False).hidden_states
    ipos = (ids[0] == model.vlm.config.image_token_id).nonzero().squeeze(1)
    for k in LAYERS[:-1]:
        ref = hs[k][:, ipos].float().mean(1)
        got = single[0][f"L{k:02d}_mean"]
        rows.append({"array": f"L{k:02d}_mean vs output_hidden_states", "rel_l2": float((got - ref).norm() / ref.norm())})
    for r in rows:
        log.info(str(r))
    log.event("equiv", renderer_max_abs=diffs, rows=rows)


def cmd_rater7(a, log):
    """Check 1 of the pre-registration: 7-camera vs front-three-only features on the exam's rater packages."""
    sets = Z.load_sets()
    names, past = [str(n) for n in sets["rater"]["name"]], sets["rater"]["past"]
    model, processor = load_model(log)
    pool, rend = Pooler(model), Renderer()
    out = {"7cam": {}, "front3": {}}
    for i0 in range(0, len(names), a.batch):
        for mode in out:
            items = []
            for n, p in zip(names[i0:i0 + a.batch], past[i0:i0 + a.batch]):
                z = np.load(Z.root("packages") / f"{n}.npz")
                cams = Z.ALP_SRC if mode == "7cam" else FRONT3
                cal = Z.read_calib(z, cams)
                jp = {c: [torch.from_numpy(z[f"jpg_{k}_{c}"]) for k in range(4)] for c in cams}
                items.append(tokenize(rend((n, mode), cal, jp), p, processor))
            for k, v in pool(model, collate(items)).items():
                out[mode].setdefault(k, []).append(v.cpu().numpy())
        log.info(f"{min(i0 + a.batch, len(names))}/{len(names)}")
    res = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in out.items()}
    np.savez(D.root() / "alp_rater7.npz", name=np.array(names), **{f"{m}/{k}": v for m, d in res.items() for k, v in d.items()})
    rows = []
    for k in res["7cam"]:
        x, y = res["7cam"][k], res["front3"][k]
        cos = (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
        rel = np.linalg.norm(x - y, axis=1) / np.linalg.norm(x, axis=1)
        spread = np.linalg.norm(x - x.mean(0), axis=1).mean()        # typical between-frame distance to the mean
        rows.append({"array": k, "cos_median": float(np.median(cos)), "rel_l2_median": float(np.median(rel)),
                     "diff_over_between_frame": float(np.linalg.norm(x - y, axis=1).mean() / spread)})
        log.info(str(rows[-1]))
    (D.root() / "alp_rater7.json").write_text(json.dumps(rows, indent=2))
    log.event("rater7", rows=rows)


def cmd_native(a, log):
    """Check 2: the model's own trajectories from front-three input, the exam's K=6 no-nav config and seed."""
    import wod_zeroshot_alpamayo as WA
    sets = Z.load_sets()
    names, past = [str(n) for n in sets["rater"]["name"]], sets["rater"]["past"]
    model, processor = load_model(log)
    I.apply(model, WA.CFG)
    rend = Renderer()
    out = D.root("alp_native_front3")
    t0 = time.time()
    for i, (n, p) in enumerate(zip(names, past)):
        if (out / f"{n}.npz").exists():
            continue
        z = np.load(Z.root("packages") / f"{n}.npz")
        cal = Z.read_calib(z, FRONT3)
        jp = {c: [torch.from_numpy(z[f"jpg_{k}_{c}"]) for k in range(4)] for c in FRONT3}
        frames = rend((n, "front3"), cal, jp)
        xyz, rot = Z.alpamayo_history(p)
        data = {"image_frames": frames, "camera_indices": torch.tensor([0, 1, 2, 6]),
                "ego_history_xyz": torch.from_numpy(xyz)[None, None], "ego_history_rot": torch.from_numpy(rot)[None, None]}
        r = I.run(model, I.build_inputs(data, processor, nav_text=None), WA.CFG, None, seed=WA.SEED)
        np.savez(out / f"{n}.npz", xyz=r["xyz"].astype(np.float32), wall=r["wall"], cot=np.array(json.dumps(r["cot"])))
        if (i + 1) % 20 == 0:
            log.info(f"[{i + 1}/{len(names)}] {(time.time() - t0) / (i + 1):.1f} s/frame")
    log.event("end", seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("extract", "equiv", "rater7", "native"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=8, help="equiv: frames")
    ap.add_argument("--set-name", default=D.ALP_SET)
    a = ap.parse_args()
    log = RunLog("drive_backbones", f"alp-{a.step}")
    log.event("start", args=vars(a), layers=LAYERS)
    {"extract": cmd_extract, "equiv": cmd_equiv, "rater7": cmd_rater7, "native": cmd_native}[a.step](a, log)
    log.info("done")


if __name__ == "__main__":
    main()
