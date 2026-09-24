"""WOD-E2E zero-shot exam: Alpamayo 1.5 runner (alpamayo1.5 venv). Pre-registration:
todos/2026-09-24-zeroshot-exam/wod-e2e.md.

Per target frame: the package's JPEGs (7 WOD cameras x frames f-3..f) are decoded on the GPU, re-projected into
Alpamayo's four f-theta cameras (jevdrive.camgeom, torch path, bilinear, uncovered = black), and fed with the
converted egomotion history; one K=6 rollout per variant (nav / no-nav). A background thread prepares the next
frames while the GPU runs the current one. Predictions go to $DATA_DIR/processed/wod_zeroshot/preds/alpamayo_<v>/
<frame>.npz (resumable), the run log to $DATA_DIR/runs/wod_zeroshot/alpamayo/<stamp>/.

  python scripts/wod_zeroshot_alpamayo.py --set rater --variants nav nonav
"""
import argparse, io, json, queue, sys, threading, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.alpamayo import infer as I  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

CFG = I.Config(name="wod", attn="sdpa", n_samples=6, compile=("visual", "expert"))
SEED = 42


def render(z, dev="cuda") -> torch.Tensor:
    """(4 cameras, 4 frames, 3, 1080, 1920) uint8 on the CPU, cameras in loader order (index 0, 1, 2, 6)."""
    from torchvision.io import decode_jpeg
    cal = Z.read_calib(z, Z.ALP_SRC)
    jpg = [torch.from_numpy(z[f"jpg_{k}_{c}"]) for c in Z.ALP_SRC for k in range(4)]
    dec = decode_jpeg(jpg, device=dev)
    imgs = [torch.stack(dec[i * 4:(i + 1) * 4]).float() for i in range(len(Z.ALP_SRC))]  # per camera (4, 3, H, W)
    out = []
    for v in G.PAI_ORDER:
        rays = G.ftheta_rays(torch, G.PAI_RIG[v], 1.0, device=dev)
        src, U, V = G.choose_sources(torch, rays, cal)
        acc = torch.zeros(4, 3, *src.shape, device=dev)
        for i, c in enumerate(Z.ALP_SRC):
            m = src == i
            if not m.any():
                continue
            w, h = cal[c]["width"], cal[c]["height"]
            grid = torch.stack([(U + 0.5) / w * 2 - 1, (V + 0.5) / h * 2 - 1], -1).float()[None].expand(4, -1, -1, -1)
            s = F.grid_sample(imgs[i], grid, mode="bilinear", padding_mode="border", align_corners=False)
            acc = torch.where(m[None, None], s, acc)
        out.append(acc.round().clamp(0, 255).to(torch.uint8))
    return torch.stack(out).cpu()


def prepare(name, past, intent, processor, variants):
    z = np.load(Z.root("packages") / f"{name}.npz")
    frames = render(z)
    xyz, rot = Z.alpamayo_history(past)
    data = {"image_frames": frames, "camera_indices": torch.tensor([0, 1, 2, 6]),
            "ego_history_xyz": torch.from_numpy(xyz)[None, None], "ego_history_rot": torch.from_numpy(rot)[None, None]}
    nav = Z.NAV_TEXT.get(int(intent))
    return frames, {v: I.build_inputs(data, processor, nav_text=nav if v == "nav" else None) for v in variants}


def load_model(log):
    try:  # official path: the Cosmos-Reason2-8B gate is accepted now
        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        model = Alpamayo1_5.from_pretrained(I.REPO, dtype=torch.bfloat16, attn_implementation=CFG.attn).to("cuda").eval()
        log.info("model loaded through the official (online) path")
        return model, helper.get_processor(model.tokenizer), "online"
    except Exception as e:  # noqa: BLE001
        log.info(f"official path failed ({e!r}); falling back to the offline path")
        m, p = I.load(CFG.attn)
        return m, p, "offline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", default=["rater"])
    ap.add_argument("--variants", nargs="+", default=["nav", "nonav"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th frame starting at i")
    ap.add_argument("--save-views", type=int, default=0, help="save the rendered newest frames of the first N targets")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    log = RunLog("wod_zeroshot", "alpamayo")
    log.event("start", args=vars(a), config=CFG.__dict__, seed=SEED)
    sets = Z.load_sets()
    todo = [(str(s["name"][i]), s["past"][i], int(s["intent"][i])) for w in a.set for s in [sets[w]]
            for i in range(len(s["name"]))]
    outdir = {v: Z.root("preds", f"alpamayo_{v}") for v in a.variants}
    todo = [t for t in todo if not all((outdir[v] / f"{t[0]}.npz").exists() for v in a.variants)]
    todo = todo[si::sn][: a.limit or None]
    log.info(f"{len(todo)} frames to run, variants {a.variants}, shard {a.shard}")
    model, processor, path = load_model(log)
    I.apply(model, CFG)
    log.event("model", path=path, versions=I.versions())

    q = queue.Queue(maxsize=3)

    def producer():
        left = list(todo)
        while left:  # take whatever is packaged; the packager may still be waiting on the record fetch
            ready = [t for t in left if (Z.root("packages") / f"{t[0]}.npz").exists()]
            if not ready:
                time.sleep(20)
                continue
            t = ready[0]
            left.remove(t)
            try:
                q.put((t, *prepare(t[0], t[1], t[2], processor, a.variants)))
            except Exception as e:  # noqa: BLE001
                q.put((t, e, None))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t_start, n, walls = time.time(), 0, []
    while (item := q.get()) is not None:
        (name, _, _), frames, inputs = item
        if inputs is None:
            log.info(f"{name}: prepare failed: {frames!r}")
            log.event("error", frame=name, error=repr(frames))
            continue
        if n < a.save_views:
            from PIL import Image
            grid = torch.cat([torch.cat(list(frames[:, 3, :, ::4, ::4]), 2)], 1)  # newest frame, 4 cams side by side
            Image.fromarray(grid.permute(1, 2, 0).numpy()).save(log.dir / f"views_{name}.png")
        rec = {}
        for v, inp in inputs.items():
            r = I.run(model, inp, CFG, None, seed=SEED)
            np.savez(outdir[v] / f"{name}.npz", xyz=r["xyz"].astype(np.float32), wall=r["wall"],
                     cot=np.array(json.dumps(r["cot"])), n_prompt=r["n_prompt"])
            rec[v] = r["wall"]
        walls.append(sum(rec.values()))
        n += 1
        log.event("frame", frame=name, **{f"wall_{k}": round(w, 3) for k, w in rec.items()},
                  mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        if n % 20 == 0 or n == len(todo):
            el = time.time() - t_start
            log.info(f"[{n}/{len(todo)}] {el / n:.2f} s/frame wall ({np.mean(walls[-20:]):.2f} s GPU), "
                     f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, ETA {(len(todo) - n) * el / n / 60:.0f} min")
            log.scalar("s_per_frame", el / n, n)
    log.event("end", frames=n, seconds=time.time() - t_start)
    log.info(f"done: {n} frames in {time.time() - t_start:.0f} s -> {list(outdir.values())}")


if __name__ == "__main__":
    main()
