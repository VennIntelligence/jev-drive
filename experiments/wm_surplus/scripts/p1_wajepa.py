"""P1 level (a) on WA-JEPA: project the scene features its planner reads onto their top-k navtrain principal directions.

Feature site: MultiViewCausalFutureMaskedJEPA._encode_scene_context (models/multiview_causal_future_jepa.py): V-JEPA 2.1
ViT-L history tokens -> SceneTokenProjector (LayerNorm + Linear 1024 -> 512) -> context_scene [B, 4096, 512]
(4 cameras x 2 tubelets x 16 x 32 patches). It is the only image input of the flow planner (SceneTrajectoryFlowPredictor),
which reads it at every one of its 4 denoising steps. Third-party code runs as shipped; the hook (lib/usage_probe.Probe)
wraps that method on the model instance. Inference = the repo's NAVSIM agent (eval/navsim_agent.py): its feature builder,
fp32, batch 1, config wa_jepa_hugsim.yaml (= the EPDMS preset without the pretrained-encoder load), the checkpoint that
gave navtest EPDMS 91.71 (experiments/top10/lib/top10_t2/navsim_repro.sh).

  gate  --n N                identity gate on N navtest tokens (see `gate`), timing; writes gate.json
  fit   --shard I N          navtrain moments of the feature (tokens on disk, --max spread over logs) -> fit/m_sIofN.npz
  pca                        merge the moments -> pca.npz (+ explained variance per k)
  sweep --shard I N          navtest plans for k = off and every --ks (encoder once per token) -> sweep/sIofN/c*.npz
  merge                      -> bench pred files PROBE-wajepa-ctx-<k> (scored by `python -m jevdrive.bench run`), ev.json

Run with $DATA_DIR/envs/wajepa/bin/python (the lane script p1_lane.sh submits every stage to the pool).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
TP = DATA / "third_party"
WJ = TP / "wajepa"
sys.path[:0] = [str(REPO / "experiments/wm_surplus/lib"), str(WJ), str(TP / "navsim"), str(REPO)]
import usage_probe as UP  # noqa: E402

OUT = Path(os.environ.get("P1_OUT", DATA / "runs/wm_surplus/p1_usage"))   # P1_OUT: smoke dirs
CFG = WJ / "configs/wa_jepa_hugsim.yaml"
CKPT = DATA / "models/wajepa/model_state_dict.pt"
REF = DATA / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/navtest_trajectories.pkl"   # the 91.71 plans
NAV = DATA / "datasets/navsim"
SITE = "_encode_scene_context"
KS = [8, 16, 32, 64, 128, 256, 512]
TAG = "wajepa-ctx"
for k, v in dict(OPENSCENE_DATA_ROOT=NAV, NUPLAN_MAPS_ROOT=NAV / "maps", NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
                 OPENBLAS_CORETYPE="Haswell").items():
    os.environ.setdefault(k, str(v))


def kname(k) -> str:
    return "off" if k is None else f"k{k}"


def agent():
    from eval.navsim_agent import WorldModelNavsimAgent
    a = WorldModelNavsimAgent(config_path=str(CFG), checkpoint_path=str(CKPT), device="cuda")
    a.initialize()
    return a


def loader(split: str, sensor_config, max_tokens: int = 0):
    """navtest: the navtest scene filter as the WA-JEPA export uses it. navtrain: the navtrain filter restricted to logs whose
    camera blobs are on disk (a few navtrain archives are downloaded for the fit) and tokens whose 4 history frames exist."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    cfg = TP / "navsim/navsim/planning/script/config/common/train_test_split/scene_filter"
    sf = instantiate(OmegaConf.load(cfg / f"{split}.yaml"))
    sub = "test" if split == "navtest" else "trainval"
    blobs = NAV / "sensor_blobs" / sub
    if split != "navtest":
        have = {p.name for p in blobs.iterdir() if (p / "CAM_F0").is_dir()}
        sf.log_names = [lg for lg in sf.log_names if lg in have]
    L = SceneLoader(original_sensor_path=blobs, data_path=NAV / "navsim_logs" / sub, scene_filter=sf, sensor_config=sensor_config)
    toks = list(L.tokens)
    if split != "navtest":
        def ok(t):
            fr = L.scene_frames_dicts[t][:sf.num_history_frames][-4:]
            return all((blobs / f["cams"][c]["data_path"]).exists() for f in fr for c in ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0"))
        toks = [t for t in toks if ok(t)]
        if max_tokens and len(toks) > max_tokens:
            toks = toks[:: len(toks) // max_tokens][:max_tokens]
    return L, toks


class Feats(torch.utils.data.Dataset):
    """The agent's own feature builder on the agent input of each token (what compute_trajectory does before the model)."""

    def __init__(self, L, toks, builder):
        self.L, self.toks, self.b = L, toks, builder

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        t = self.toks[i]
        return t, self.b.compute_features(self.L.get_agent_input_from_token(t))


def batches(A, L, toks, workers):
    dl = torch.utils.data.DataLoader(Feats(L, toks, A.get_feature_builders()[0]), batch_size=1, num_workers=workers,
                                     prefetch_factor=4 if workers else None, persistent_workers=False)
    for (t,), f in dl:
        yield t, {k: v.to("cuda") for k, v in f.items()}


def plan(A, f) -> np.ndarray:
    """= WorldModelNavsimAgent.compute_trajectory after the feature builder (batch 1, squeeze, float32)."""
    with torch.no_grad():
        return A.model.predict_trajectory(f).squeeze(0).detach().float().cpu().numpy().astype(np.float32)


def workers() -> int:
    from jevdrive.common import n_cpus
    return max(1, n_cpus() - 2)


# ---------------------------------------------------------------- commands
def cmd_gate(a):
    """Identity gate (all on the same N navtest tokens):
    G1 hook installed but off == no hook, bit for bit (DataLoader path);  G2 DataLoader path == agent.compute_trajectory, bit
    for bit;  G3 sweep k=off (feature cached) == no hook, bit for bit;  G4 determinism: two plain calls identical;
    G5 the stored 91.71 plans: max abs diff (reported; different run / card, so not required to be 0);
    R  full-rank projection (k=512) vs plain: max abs diff (fp32 rounding only; needs --pca)."""
    A = agent()
    L, toks = loader("navtest", A.get_sensor_config())
    toks = toks[:: max(1, len(toks) // a.n)][: a.n]
    ref = pickle.load(open(REF, "rb"))
    ref = ref.get("trajectories", ref)
    plain, hooked, swept, again, direct, full, t_enc = [], [], [], [], [], [], []
    pca = UP.PCA.load(a.pca) if a.pca else None
    for t, f in batches(A, L, toks, min(8, workers())):
        plain.append(plan(A, f))
        again.append(plan(A, f))
        direct.append(A.compute_trajectory(L.get_agent_input_from_token(t)).poses.astype(np.float32))
        with UP.Probe(A.model, SITE) as p:
            hooked.append(plan(A, f))
            p.use_pca(pca, "cuda") if pca else p.use_pca(UP.PCA(np.zeros(512), np.eye(512), np.ones(512), 1), "cuda")
            t0 = time.time()
            r = p.sweep(lambda: plan(A, f), [None, 512] if pca else [None])
            t_enc.append(time.time() - t0)
            swept.append(r[None])
            if pca:
                full.append(r[512])
    g = dict(n=len(toks), G1_off_vs_nohook=UP.identity_gate(plain, hooked), G2_loader_vs_agent=UP.identity_gate(direct, plain),
             G3_sweep_off_vs_nohook=UP.identity_gate(plain, swept), G4_determinism=UP.identity_gate(plain, again),
             G5_vs_stored_9171_max_abs_m=max(UP.max_abs_diff(p, ref[t]) for p, t in zip(plain, toks)))
    if pca:
        g["R_fullrank_vs_plain_max_abs_m"] = max(UP.max_abs_diff(x, y) for x, y in zip(plain, full))
    g["passed"] = all(g[k]["bit_identical"] for k in ("G1_off_vs_nohook", "G2_loader_vs_agent", "G3_sweep_off_vs_nohook",
                                                       "G4_determinism"))
    out = OUT / "gate.json"
    out.write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))
    if not g["passed"]:
        sys.exit("identity gate failed")


def cmd_fit(a):
    from jevdrive import par
    A = agent()
    L, toks = loader("navtrain", A.get_sensor_config(), a.max)
    mine = par.shards(toks, a.shard[1], a.shard[0])
    d = OUT / "fit"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"tokens_s{a.shard[0]}of{a.shard[1]}.txt").write_text("\n".join(mine) + "\n")
    M = UP.Moments()
    t0 = time.time()
    with UP.Probe(A.model, SITE) as p:
        p.capture(M, stop=True)
        for i, (t, f) in enumerate(batches(A, L, mine, workers()), 1):
            try:
                A.model.predict_trajectory(f)
            except UP.Captured:
                pass
            if i % 100 == 0:
                print(f"[fit] {i}/{len(mine)} {(time.time() - t0) / i:.2f} s/token", flush=True)
    np.savez(d / f"m_s{a.shard[0]}of{a.shard[1]}.npz", tokens=len(mine), **M.state())
    print(f"[fit] {len(mine)} tokens, {M.n} rows in {time.time() - t0:.0f} s", flush=True)


def cmd_pca(a):
    fs = sorted((OUT / "fit").glob("m_s*of*.npz"))
    st = UP.Moments.merge(dict(np.load(f)) for f in fs)
    pca = UP.Moments.fit(st)
    pca.save(OUT / "pca.npz")
    ntok = sum(int(np.load(f)["tokens"]) for f in fs)
    ev = {kname(k): pca.explained(k) for k in KS}
    (OUT / "pca.json").write_text(json.dumps(dict(shards=len(fs), navtrain_tokens=ntok, rows=pca.n, dim=pca.dim,
                                                  explained_navtrain=ev), indent=1))
    print(ev)


def cmd_sweep(a):
    from jevdrive import par
    A = agent()
    L, toks = loader("navtest", A.get_sensor_config())
    mine = par.shards(toks, a.shard[1], a.shard[0])
    d = OUT / "sweep" / f"s{a.shard[0]}of{a.shard[1]}"
    d.mkdir(parents=True, exist_ok=True)
    pca = UP.PCA.load(OUT / "pca.npz")
    ks = [None] + KS
    t0, n = time.time(), 0
    with UP.Probe(A.model, SITE) as p:
        p.use_pca(pca, "cuda")
        for c0 in range(0, len(mine), a.chunk):
            f_out = d / f"c{c0:06d}.npz"
            if f_out.exists():
                continue
            p.energy = {}
            T, P = [], []
            for t, f in batches(A, L, mine[c0:c0 + a.chunk], workers()):
                r = p.sweep(lambda: plan(A, f), ks)
                T.append(t)
                P.append(np.stack([r[k] for k in ks]))
            n += len(T)
            e = p.energy
            np.savez(d / "tmp.npz", tokens=np.array(T), poses=np.stack(P, 1), ks=np.array([kname(k) for k in ks]),
                     e_total=e["total"], e_res=np.array([e[k] for k in KS]))
            os.replace(d / "tmp.npz", f_out)
            print(f"[sweep] {c0 + len(T)}/{len(mine)} {(time.time() - t0) / max(n, 1):.2f} s/token", flush=True)
    (d / "DONE").write_text(json.dumps(dict(tokens=len(mine), wall_s=time.time() - t0)))


def cmd_merge(a):
    from jevdrive.bench.navsim import pred_file
    from jevdrive.bench.models import resolve
    import yaml
    toks = yaml.safe_load(open(TP / "navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))["tokens"]
    fs = sorted((OUT / "sweep").glob("s*of*/c*.npz"))
    zs = [np.load(f) for f in fs]
    T = np.concatenate([z["tokens"] for z in zs])
    P = np.concatenate([z["poses"] for z in zs], 1)
    names = [str(x) for x in zs[0]["ks"]]
    assert len(set(T)) == len(T) == len(toks) and set(T) == set(toks), f"{len(T)} plans for {len(toks)} navtest tokens"
    ref = pickle.load(open(REF, "rb"))
    ref = ref.get("trajectories", ref)
    off = P[names.index("off")]
    for i, nm in enumerate(names):
        pf = pred_file(resolve(f"PROBE-{TAG}-{nm}"), "navtest")
        pf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pf, tokens=T, poses=P[i])
    tot = sum(float(z["e_total"]) for z in zs)
    res = sum(z["e_res"] for z in zs)
    pca = UP.PCA.load(OUT / "pca.npz")
    ev = dict(tokens=len(T), explained_navtrain={f"k{k}": pca.explained(k) for k in KS},
              explained_navtest={f"k{k}": 1 - float(r) / tot for k, r in zip(KS, res)},
              plan_shift_vs_off_m={nm: float(np.linalg.norm(P[i][..., :2] - off[..., :2], axis=-1).mean()) for i, nm in enumerate(names)},
              off_vs_stored_9171_max_abs_m=float(max(np.abs(off[j] - ref[t]).max() for j, t in enumerate(T))))
    (OUT / "ev.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "fit", "pca", "sweep", "merge"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pca", default="")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--max", type=int, default=3000, help="navtrain tokens for the fit (spread over the logs on disk)")
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(WJ)
    from jevdrive.run import Run
    with Run("wm_surplus", f"p1-{a.cmd}" + (f"-s{a.shard[0]}of{a.shard[1]}" if a.cmd in ("fit", "sweep") else ""),
             config=vars(a)) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("navsim/navtrain" if a.cmd in ("fit", "pca") else "navsim/navtest"))
        globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
