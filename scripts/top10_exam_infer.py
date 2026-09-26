"""Top-10 exam runner (executor T1): SparseDriveV2 or ZTRS on the frames of a jevdrive.top10_exam plan.
Pre-registration: todos/2026-09-26-top10-intersection.md, [T1] 10:05. Runs in the model's own env, from its repo:

  cd ~/data/third_party/sparsedrivev2 && OPENBLAS_CORETYPE=Haswell $DATA_DIR/envs/sparsedrivev2/bin/python \
      ~/data/jev-drive/scripts/top10_exam_infer.py --model sparsedrivev2 --set p5
  cd ~/data/third_party/ztrs && NAVSIM_DEVKIT_ROOT=$PWD ... envs/gtrs/bin/python ... --model ztrs --set p5
  ... --check 256        # adapter equivalence on navtest tokens, before any exam number

Per frame: the three current source JPEGs are rendered into virtual nuPlan cameras (jevdrive.navsim_rig), then the
model's own feature code runs unchanged from its image-loading step on (SparseDriveV2: get_camera_params -> resize /
crop -> normalise -> data_adapter; ZTRS: HydraFeatureBuilder's stitching), and the model's forward picks the plan.
Output: processed/top10_exam/<set>/<model>.npz (frame_name, raw trajectory, 0.25 s grid in the exam's frame).
"""
import argparse, copy, functools, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from jevdrive import navsim_rig as R  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

DATA = Path(os.environ["DATA_DIR"])
NAV = DATA / "datasets/navsim"
cv2.setNumThreads(1)


# ---------------------------------------------------------------- models

def sparsedrivev2():
    """SparseDriveV2 NAVSIM v1 checkpoint with run_pdm_score_navtest_v1.sh's inference settings (as the smoke)."""
    from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
    W = DATA / "models/sparsedrivev2"
    ck = W / "sparsedrive_navsimv1_92p2.ckpt"
    cfg = SparseDriveConfig(bkb_path=str(W / "resnet34.bin"), path_anchor=str(W / "kmeans/path_1024.npy"),
                            velocity_anchor=str(W / "kmeans/velocity_256.npy"),
                            trajectory_anchor=str(W / "kmeans/trajectory_1024_256.npz"), dataset_version="v1",
                            metrics=["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
                                     "time_to_collision_within_bound", "comfort", "ego_progress"],
                            velocity_filter_num=[64, 20])
    agent = SparseDriveAgent(cfg, lr=1e-4, checkpoint_path=str(ck))
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    agent.load_state_dict({k.replace("agent.", ""): v for k, v in sd.items()}, strict=True)
    fb = agent.get_feature_builders()[0]

    def feats(imgs: dict, cams: dict, ego: np.ndarray):
        """imgs / cams keyed cam_l0 / cam_f0 / cam_r0; the builder's pipeline minus its load_images (in memory)."""
        info = {k: {"image_path": None, **cams[k]} for k in cfg.cams}
        r = fb.get_camera_params(info)
        r["imgs"] = [cv2.cvtColor(imgs[k], cv2.COLOR_RGB2BGR) if cfg.to_bgr else imgs[k] for k in cfg.cams]
        r["img_shape"] = [x.shape[:2] for x in r["imgs"]]
        r = fb.resize_crop_flip_img(r, True)
        r, _ = fb.ego_rotation(r, {}, True)
        r = fb.normalize_img(fb.photo_metric_distortion(r, True))
        r = fb.data_adapter(r)
        return {"camera_feature": r, "status_feature": torch.tensor(np.r_[ego[28:32], ego[18:20], ego[26:28]], dtype=torch.float32)}

    def native(agent_input):
        f = fb.compute_features(agent_input)
        return fb.pipeline(f, {}, None, test_mode=True)[0]

    def forward(batch):
        return agent.forward(batch, {})[0]["trajectory"][..., :3], None

    from torch.utils.data import default_collate
    return SimpleNamespace(agent=agent, feats=feats, native=native, forward=forward, collate=default_collate, dt=0.5,
                           sensor_config=agent.get_sensor_config())


def ztrs():
    """ZTRS (ztrs_vov.ckpt, 8192 vocabulary as docs/ztrs_inference.md); ec_target off (loss-only pass, [T1] 10:05 (3))."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    W = DATA / "models/gtrs"
    root = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", Path.cwd()))
    cfg = OmegaConf.load(root / "navsim/planning/script/config/common/agent/ztrs_vov.yaml")
    cfg.pdm_gt_path = None
    cfg.checkpoint_path = str(W / "ztrs_vov.ckpt")
    cfg.config.vocab_path = str(root / "traj_final/8192.npy")
    cfg.config.vov_ckpt = str(W / "dd3d_det_final.pth")
    agent = instantiate(cfg)
    agent.initialize()
    agent._config.ec_target = False
    fb = agent.get_feature_builders()[0]
    fb1 = copy.copy(fb)
    fb1._config = copy.copy(fb._config)
    fb1._config.seq_len = 1                                   # only the current frame is read with ec_target off

    def status(ego, j):
        return torch.tensor(np.r_[ego[28:32], ego[12 + 2 * j: 14 + 2 * j], ego[20 + 2 * j: 22 + 2 * j]], dtype=torch.float32)

    def feats(imgs: dict, cams: dict, ego: np.ndarray):
        cam = SimpleNamespace(**{k: SimpleNamespace(image=imgs[k]) for k in R.NAMES})
        img = fb1._get_camera_feature(SimpleNamespace(cameras=[cam]))[-1]
        return {"camera_feature": img, "status_feature": status(ego, 3)}

    def native(agent_input):
        f = fb.compute_features(agent_input)
        return {"camera_feature": f["camera_feature"][-1], "status_feature": f["status_feature"][0]}

    def forward(batch):
        o = agent.forward({"camera_feature": [batch["camera_feature"]], "status_feature": [batch["status_feature"]]})
        return o["trajectory"], o["selected_indices"]

    def collate(xs):
        return {k: torch.stack([x[k] for x in xs]) for k in xs[0]}
    return SimpleNamespace(agent=agent, feats=feats, native=native, forward=forward, collate=collate, dt=0.1,
                           sensor_config=agent.get_sensor_config())


# ---------------------------------------------------------------- data

def maps_path(set_: str, key: str) -> Path:
    return DATA / "processed/top10_exam" / set_ / "maps" / f"{key}.npz"


def rig_of(rig: dict) -> tuple:
    virt = [{k: np.asarray(v, np.float64) for k, v in c.items()} for c in rig["virt"]]
    return rig["src"], virt, rig["primary"]


def make_maps(args):
    set_, key, rig = args
    p = maps_path(set_, key)
    if not p.exists():
        src, virt, prim = rig_of(rig)
        mp = R.maps(src, virt, prim)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **{f"{n}{i}": a for i, m in enumerate(mp) for n, a in zip("suv", m)})
        tmp.rename(p)
    return key


@functools.lru_cache(maxsize=2)
def load_maps(set_: str, key: str):
    z = np.load(maps_path(set_, key))
    return [(z[f"s{i}"], z[f"u{i}"], z[f"v{i}"]) for i in range(3)]


class Frames(torch.utils.data.Dataset):
    def __init__(self, plan, feats):
        self.p, self.f = plan, feats
        self.cams = {k: dict(zip(R.NAMES, [{n: np.asarray(v) for n, v in c.items()} for c in r["virt"]]))
                     for k, r in plan["rigs"].items()}

    def __len__(self):
        return len(self.p["frames"]["frame_name"])

    def __getitem__(self, i):
        fr = self.p["frames"]
        key = fr["rig"][i]
        src = [np.asarray(Image.open(f).convert("RGB")) for f in fr["files"][i]]
        out = R.render(src, load_maps(self.p["set"], key))
        return i, self.f(dict(zip(R.NAMES, out)), self.cams[key], np.asarray(fr["ego"][i], np.float32))


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, list):
        return [to_dev(v, dev) for v in x]
    return x


def infer(rl, M, plan, workers: int, batch: int):
    from concurrent.futures import ProcessPoolExecutor
    set_ = plan["set"]
    maps_path(set_, "x").parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, len(plan["rigs"]))) as ex:
        list(ex.map(make_maps, [(set_, k, r) for k, r in plan["rigs"].items()]))
    rl.log.info("maps for %d rigs in %.0f s", len(plan["rigs"]), time.time() - t0)
    ds = Frames(plan, M.feats)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=lambda xs: (
        [x[0] for x in xs], M.collate([x[1] for x in xs])), prefetch_factor=4, persistent_workers=False)
    dev = torch.device("cuda")
    M.agent.to(dev).eval()
    n = len(ds)
    raw, sel, idx = [None] * n, np.full(n, -1), []
    from tqdm import tqdm
    t0, done = time.time(), 0
    with torch.no_grad():
        for ids, b in tqdm(dl, total=(n + batch - 1) // batch, mininterval=30):
            tr, s = M.forward(to_dev(b, dev))
            tr = tr.float().cpu().numpy()
            for j, i in enumerate(ids):
                raw[i] = tr[j]
                if s is not None:
                    sel[i] = int(s[j])
            done += len(ids)
            if done % (batch * 200) < batch:
                rl.event("progress", done=done, n=n, fps=done / (time.time() - t0))
    raw = np.stack(raw)
    grid = R.spline_grid(raw, M.dt) - np.asarray(plan["offset"], np.float32)
    out = DATA / "processed/top10_exam" / set_ / f"{rl.model}.npz"
    np.savez_compressed(out, frame_name=np.asarray(plan["frames"]["frame_name"]), raw=raw, grid=grid, selected=sel)
    dt = time.time() - t0
    rl.log.info("%s on %s: %d frames in %.0f s (%.1f fps) -> %s; v(2 s) median %.2f m/s", rl.model, set_, n, dt, n / dt, out,
                float(np.median(np.linalg.norm(grid[:, 7] - grid[:, 6], axis=-1) / 0.25)))
    rl.event("done", frames=n, seconds=dt)


# ---------------------------------------------------------------- adapter check on navtest

def check(rl, M, n_tok: int, batch: int):
    """Native pipeline vs our render path with nuPlan's own F0 / L0 / R0 as the sources, on n_tok navtest tokens.
    Arms: (a) virtual = the frame's real nuPlan cameras (render ~ identity), (b) virtual = navsim_rig.virtual at the
    source's yaw and position (the path the exams take)."""
    import yaml
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    split = yaml.safe_load(open(Path.cwd() / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                     log_names=split["log_names"], tokens=split.get("tokens"))
    loader = SceneLoader(NAV / "navsim_logs/test", NAV / "sensor_blobs/test", sf, sensor_config=M.sensor_config)
    toks = np.random.default_rng(0).choice(sorted(loader.tokens), n_tok, replace=False)
    dev = torch.device("cuda")
    M.agent.to(dev).eval()
    names = {"cam_l0": "cam_l0", "cam_f0": "cam_f0", "cam_r0": "cam_r0"}
    rows, ego_rows = [], []
    for k0 in range(0, n_tok, batch):
        nat, ad_a, ad_b = [], [], []
        for tok in toks[k0: k0 + batch]:
            ai = loader.get_agent_input_from_token(tok) if hasattr(loader, "get_agent_input_from_token") else \
                loader.get_scene_from_token(tok).get_agent_input()
            nat.append(M.native(ai))
            cur = ai.cameras[-1]
            cams = {k: {"sensor2lidar_rotation": np.asarray(getattr(cur, k).sensor2lidar_rotation, np.float64),
                        "sensor2lidar_translation": np.asarray(getattr(cur, k).sensor2lidar_translation, np.float64),
                        "intrinsics": np.asarray(getattr(cur, k).intrinsics, np.float64)[:3, :3],
                        "distortion": np.asarray(getattr(cur, k).distortion, np.float64)} for k in names}
            imgs = {k: (np.asarray(getattr(cur, k).image) if getattr(cur, k).image is not None
                        else np.asarray(Image.open(getattr(cur, k).image_path).convert("RGB"))) for k in names}
            src_keys = ("cam_f0", "cam_l0", "cam_r0")                 # (front, front_left, front_right) as the plans
            src = [R.as_camgeom(cams[k]) for k in src_keys]
            es = ai.ego_statuses
            ego = np.zeros(32, np.float32)
            ego[28:32] = es[-1].driving_command
            for j, e in enumerate(es[-4:]):
                ego[12 + 2 * j: 14 + 2 * j], ego[20 + 2 * j: 22 + 2 * j] = e.ego_velocity, e.ego_acceleration
            for arm, lst in (("a", ad_a), ("b", ad_b)):
                if arm == "a":
                    virt = [cams[k] for k in R.NAMES]
                else:
                    yaw = lambda k: float(np.degrees(np.arctan2(cams[k]["sensor2lidar_rotation"][1, 2],  # noqa: E731
                                                                cams[k]["sensor2lidar_rotation"][0, 2])))
                    virt = [R.virtual(yaw(k), cams[k]["sensor2lidar_translation"]) for k in R.NAMES]
                mp = R.maps(src, virt, [1, 0, 2])
                out = R.render([imgs[k] for k in src_keys], mp)
                lst.append(M.feats(dict(zip(R.NAMES, out)), dict(zip(R.NAMES, virt)), ego))
        with torch.no_grad():
            res = {k: M.forward(to_dev(M.collate(v), dev)) for k, v in (("native", nat), ("a", ad_a), ("b", ad_b))}
        for j, tok in enumerate(toks[k0: k0 + batch]):
            r = {"token": tok}
            tn = res["native"][0][j].float().cpu().numpy()
            for arm in ("a", "b"):
                ta = res[arm][0][j].float().cpu().numpy()
                r[f"ade_{arm}"] = float(np.linalg.norm(ta[:, :2] - tn[:, :2], axis=-1).mean())
                r[f"same_{arm}"] = bool(np.allclose(ta, tn, atol=1e-3)) if res[arm][1] is None else \
                    bool(int(res[arm][1][j]) == int(res["native"][1][j]))
            rows.append(r)
        rl.log.info("check %d / %d tokens", len(rows), n_tok)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(rl.dir / "adapter_check.csv", index=False)
    s = {f"{k}_{arm}": float(df[f"{k}_{arm}"].mean()) for k in ("same", "ade") for arm in "ab"} | \
        {f"ade_p95_{arm}": float(df[f"ade_{arm}"].quantile(0.95)) for arm in "ab"} | {"n": len(df)}
    (rl.dir / "adapter_check.json").write_text(json.dumps(s, indent=1))
    rl.log.info("adapter check: %s", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("sparsedrivev2", "ztrs"), required=True)
    ap.add_argument("--set", default="p5")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--check", type=int, default=0, help="adapter equivalence on this many navtest tokens instead")
    a = ap.parse_args()
    M = {"sparsedrivev2": sparsedrivev2, "ztrs": ztrs}[a.model]()
    rl = RunLog("top10_exam", f"{'check' if a.check else 'infer-' + a.set}-{a.model}")
    rl.model = a.model
    if a.check:
        check(rl, M, a.check, a.batch)
    else:
        plan = json.loads((DATA / "processed/top10_exam" / a.set / "plan.json").read_text())
        infer(rl, M, plan, a.workers, a.batch)
    rl.close()


if __name__ == "__main__":
    main()
