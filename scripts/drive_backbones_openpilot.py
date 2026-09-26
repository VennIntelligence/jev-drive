"""Driving backbones in the P3 ladder: openpilot feature runner (openpilot venv).
Pre-registration: todos/2026-09-24-driving-backbones/README.md. Plan from `python -m jevdrive.drive_backbones
--steps prepare` (op_plan.json).

Inputs are exactly the WOD exam's (scripts/wod_zeroshot_openpilot.py: front three cameras -> road / wide calib
frames, every WOD frame fed twice for 20 Hz, desire 0, traffic [1, 0]); only the temporal context differs:

  stream  (default) each stream -- one val sequence from its first indexed frame to its last subset frame, split
          at index gaps -- runs once from a zero state and every subset frame on it is read in passing, like a car
          that has been driving since engagement. Lebowski runs one context-rate stream per frame parity (exact
          at the output phase, exam deviation 1).
  exam    the exam's own protocol, per target: zero state, warm-up from max(f - 100, first frame). Used only to
          check the stream against the exam's saved predictions and to measure how far the definition moved.

Per subset frame and model: taps `temporal` and `vision` (intermediate ONNX values exposed as extra outputs),
`hidden` (the queued feature, from the output vector), `plan` (MDN mean, 33 x 15), and the native plan converted
to WOD rear-axle waypoints (`wod`). Output: $DATA_DIR/processed/drive_backbones/op/<model>/<stream>.npz
(resumable; `exam` mode writes op_exam/<model>/<target>.npz).

  python scripts/drive_backbones_openpilot.py --shard 0/2      # on GPU 0, and --shard 1/2 on GPU 1
"""
import argparse, io, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wod_zeroshot_openpilot as WZ  # noqa: E402  (the exam runner: renderer and model settings)
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import drive_backbones as D  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402


def render(names: list[str]) -> np.ndarray:
    """Packed (n, 2, 6, 128, 256) [road, wide] model frames, the exam's renderer (WZ.model_frames) on any names."""
    from PIL import Image
    spans = WZ._ctx["spans"]
    idx = WZ._maps(names[0].rsplit("-", 1)[0])
    out = np.empty((len(names), 2, 6, 128, 256), np.uint8)
    fh = {}
    for j, n in enumerate(names):
        sp = spans[n]
        f = fh.get(sp[0]) or fh.setdefault(sp[0], open(WZ._ctx["shard_dir"] / sp[0], "rb"))
        planes = []
        for k in range(3):
            f.seek(sp[1 + 2 * k])
            im = Image.open(io.BytesIO(f.read(sp[2 + 2 * k])))
            im.draft("YCbCr", im.size)
            planes.append(np.asarray(im.convert("YCbCr")).reshape(-1, 3))
        cat = np.concatenate(planes)
        for m, k in enumerate(("road", "wide")):
            out[j, m] = WZ._pack(cat[idx[k]].reshape(G.OP_H, G.OP_W, 3))
    for f in fh.values():
        f.close()
    return out


def job(item):
    key, names, targets = item
    return key, names, targets, render(names)


def bounded_map(ex, fn, items, depth):
    """ex.map without submitting everything up front: a stream renders to ~85 MB, so at most `depth` in flight."""
    from collections import deque
    it, q = iter(items), deque()
    for x in it:
        q.append(ex.submit(fn, x))
        if len(q) >= depth:
            break
    while q:
        yield q.popleft().result()
        x = next(it, None)
        if x is not None:
            q.append(ex.submit(fn, x))


class Recorder:
    """Collects the per-target arrays of one model (`keep`: the arrays to store; default all)."""

    def __init__(self, model, dev_xy, keep=None):
        from jevdrive.openpilot.model import T_IDXS, decode, mdn_mu
        self.m, self.dev, self.T, self.decode, self.mdn = model, dev_xy, T_IDXS, decode, mdn_mu
        self.keep = keep
        self.rows = []

    def add(self, name, raw, hist):
        s, tv = self.m.slices, self.m.tap_values
        taps = D.OP_TAPS[self.m.name]
        d = self.decode(raw, s, 10.0, WZ.ACTION_T)
        r = {"name": name, "temporal": tv[taps["temporal"]], "vision": tv[taps["vision"]],
             "hidden": raw[s["hidden_state"]], "plan": self.mdn(raw[s["plan"]], (33, 15)).ravel(),
             "wod": Z.openpilot_to_wod(d["plan_pos"], d["plan_yaw"], self.T, self.dev), "hist": hist}
        if self.keep is not None and "lead" in self.keep:     # raw output slices, decoded as fusion_q4c.outputs
            r |= {"lead": raw[s["lead"]].copy(), "lead_prob": raw[s["lead_prob"]].copy()}
        self.rows.append(r if self.keep is None else {k: r[k] for k in ("name", "hist", *self.keep)})

    def save(self, path):
        k = self.rows[0].keys()
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{c: np.stack([r[c] for r in self.rows]) if c != "name" else np.array([r[c] for r in self.rows])
                         for c in k})
        tmp.replace(path)


def run_stream(m, frames, names, targets, dev_xy, desire=None, keep=None) -> Recorder:
    """One stream through one model; records every target on it. `hist` = frames of history before the target.
    desire: {frame name: openpilot desire index} (absent = none); the one-hot is fed every step and OPModel turns
    it into modeld's rising-edge pulse."""
    rec, tset = Recorder(m, dev_xy, keep), set(targets)
    eye = np.eye(8, dtype=np.float32)
    des = [eye[desire.get(n, 0)] if desire else eye[0] for n in names]
    if m.skip == 1:   # context rate: one stream per parity, one step per 0.2 s
        for p in (0, 1):
            m.reset()
            for j in range(p, len(names), 2):
                raw = m.step(frames[j], desire=des[j], action_t=WZ.ACTION_T)
                if j in tset:
                    rec.add(names[j], raw, j)
    else:
        m.reset()
        for j in range(len(names)):
            for _ in range(2):
                raw = m.step(frames[j], desire=des[j], action_t=WZ.ACTION_T)
            if j in tset:
                rec.add(names[j], raw, j)
    rec.rows.sort(key=lambda r: r["name"])
    return rec


def run_exam(m, frames, names, target, dev_xy) -> Recorder:
    """The exam's per-target protocol (WZ.run_one) on the frames max(f-100, first)..f, recording taps too."""
    rec = Recorder(m, dev_xy)
    m.reset()
    steps = [(s, 1) for s in frames[::-1][::2][::-1]] if m.skip == 1 else [(s, 2) for s in frames]
    for s, rep in steps:
        for _ in range(rep):
            raw = m.step(s, action_t=WZ.ACTION_T)
    rec.add(target, raw, len(names) - 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(D.OP_MODELS))
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th stream")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="streams (stream mode) or targets (exam mode)")
    ap.add_argument("--mode", choices=("stream", "exam"), default="stream")
    ap.add_argument("--split", default="subset", choices=("subset", "trainval"), help="stream mode: which plan")
    ap.add_argument("--targets", default="rater", help="exam mode: 'rater' (the exam's frames on the subset) or a file")
    ap.add_argument("--desire", action="store_true", help="stream mode: desire pulses from the WOD routing intent "
                    "(op_desire_<split>.json from `python -m jevdrive.op_route desire`); stores temporal + native plan")
    ap.add_argument("--out-sub", default="", help="override the output directory under processed/drive_backbones")
    ap.add_argument("--lead", action="store_true", help="stream mode: store temporal + the raw lead / lead_prob slices")
    ap.add_argument("--only", default="", help="stream mode: a file of frame names; keep only the streams holding one "
                    "and only those targets on them (real-data transfer G1: lead outputs on the E1 eval frames)")
    a = ap.parse_args()
    from jevdrive.common import data_dir
    from jevdrive.openpilot.model import OPModel
    si, sn = map(int, a.shard.split("/"))
    log = RunLog("drive_backbones", f"op-{a.mode}")
    log.event("start", args=vars(a), backends=WZ.MODELS, taps=D.OP_TAPS)
    plan = json.loads((D.root() / D.plan_name(a.split)).read_text())
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    if a.split != "subset":
        op_calib |= json.loads((D.root() / f"op_calib_{a.split}.json").read_text())
    spans = plan["spans"]
    sub = ("op" if a.split == "subset" else f"op_{a.split}") if a.mode == "stream" else "op_exam"
    sub = a.out_sub or sub + ("_desire" if a.desire else "")
    desire = json.loads((D.root() / f"op_desire_{a.split}.json").read_text()) if a.desire else None
    keep = ("temporal", "wod") if a.desire else ("temporal", "lead", "lead_prob") if a.lead else None
    outdir = {k: D.root(sub, k) for k in a.models}
    if a.mode == "stream":
        items = [(f"{i:04d}_{s['sequence']}", s["names"], s["targets"]) for i, s in enumerate(plan["streams"])]
        if a.only:
            only = set(Path(a.only).read_text().split())
            items = [(k, nm, t) for k, nm, t in ((k, nm, [j for j in t if nm[j] in only]) for k, nm, t in items) if t]
            items = [(k, nm[: t[-1] + 1], t) for k, nm, t in items]     # the stream stops at its last kept target
        items = [it for it in items[si::sn] if not all((outdir[k] / f"{it[0]}.npz").exists() for k in a.models)]
    else:
        want = ([str(n) for n in Z.load_sets()["rater"]["name"]] if a.targets == "rater"
                else Path(a.targets).read_text().split())
        items = []
        for n in want:
            if n not in spans:
                continue
            h = [x for x in Z.history_names(n, Z.OP_WARMUP) if x in spans]
            items.append((n, h, [len(h) - 1]))
        items = items[si::sn]
    items = items[: a.limit or None]
    shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
    with ProcessPoolExecutor(a.workers, initializer=WZ._init, initargs=(spans, op_calib, str(shard_dir))) as ex:
        # fork the render workers before the TensorRT sessions exist: forked after, every worker carried the
        # parent's ~27 GB of engine and CUDA mappings, and six runners pushed the box's cgroup into OOM kills
        list(ex.map(int, range(a.workers)))
        del plan
        models = {k: OPModel(k, WZ.MODELS[k], context_rate=(k == "lebowski"), taps=list(D.OP_TAPS[k].values()))
                  for k in a.models}
        n_frames = sum(len(it[1]) for it in items)
        log.info(f"{len(items)} {a.mode} items, {n_frames} frames, models {list(models)}, {a.workers} render workers")
        t0, tm, n, nf = time.time(), {k: 0.0 for k in models}, 0, 0
        for key, names, targets, frames in bounded_map(ex, job, items, 2 * a.workers):
            seq = names[0].rsplit("-", 1)[0]
            dev = np.array(op_calib[seq]["1"]["extrinsic"]).reshape(4, 4)[:2, 3]
            for k, m in models.items():
                t = time.perf_counter()
                rec = (run_stream(m, frames, names, targets, dev, desire, keep) if a.mode == "stream"
                       else run_exam(m, frames, names, key, dev))
                tm[k] += time.perf_counter() - t
                rec.save(outdir[k] / f"{key}.npz")
            n, nf = n + 1, nf + len(names)
            if n % 20 == 0 or n == len(items):
                el = time.time() - t0
                log.info(f"[{n}/{len(items)}] {nf} frames, {1e3 * el / nf:.2f} ms/frame wall; model ms/frame "
                         + ", ".join(f"{k} {1e3 * v / nf:.2f}" for k, v in tm.items())
                         + f"; ETA {(n_frames - nf) * el / nf / 60:.0f} min")
                log.event("progress", n=n, frames=nf, wall_s=el, **{f"gpu_s_{k}": v for k, v in tm.items()})
    el = time.time() - t0
    timing = {"items": n, "frames": nf, "wall_s": el, "ms_per_frame_wall": 1e3 * el / max(nf, 1),
              **{f"gpu_ms_per_frame_{k}": 1e3 * v / max(nf, 1) for k, v in tm.items()}}
    log.event("end", **timing)
    if a.mode == "stream":
        for k in models:
            (outdir[k] / f"timing_{si}of{sn}.json").write_text(json.dumps(timing))
    log.info(f"done: {timing}")


if __name__ == "__main__":
    main()
