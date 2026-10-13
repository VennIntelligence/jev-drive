"""ONPOL-NUPLAN trainer: experiments/alpasim/scripts/joint_train.py with the S flags, plus a start from an archived checkpoint and on-policy rows
(lane ONPOL-NUPLAN, plans/2026-10-11-onpol-prereg.md "Arms"; the pool's file format: plans/on_pool_format.md).

Nothing it imports is edited: joint_train's layout / cmd_ident, pp_train's Store / PModel / Prefetch / dev_eval, loss43's Losses43 (subclassed) /
OffAgentHinge, ot_rows' off_hinge / offsets. Only joint_train's S subset is carried: NAVSIM input standard, no yaw-rate rows, no memory bank. With
--init and --on off every operation is joint_train's in joint_train's order (gate T1: bit-identical weights).

  --init TAG        start from $DATA_DIR/runs/op_parity/runs/<TAG>/ckpt-final.pt: every trained tensor of the net and the adapter (the key sets must
                    match exactly); fresh optimiser, scaler and schedule; dev metrics are also logged at step 0. The encoder stays frozen as before.
  --on POOL --on-k 13 --on-mode imit|self|both
                    k rows of every batch are drawn uniformly (without replacement inside a batch) from the pool cache dir POOL with the rng stream
                    [seed, 0, 53]: the same pool indices at every step in every mode (a running sha1 of them is logged as `on_hash`, also at step 60). The normal
                    rows shrink by k; the hinge-only rows of --ho stay. Batch order: normal, on-policy, hinge-only. On-policy rows are never anchor
                    rows, never dev rows, and never carry P2H10's own drivable hinge or the imitation rows' agent term A.
      imit          imitation rows on the pool's 8-pose target (tab `fut`, the row's own frame): they enter pp_train.Losses like any imitation row
                    (the mean `imit` over all imitation rows, lam_i 1; counted in n_imit) with their non-plan heads in `distill` (lam_d 30, to the
                    shipped teacher on the same tokens). No hinge: the pool's label files are not opened.
      self          hinge-only rows through Losses43's own hinge-only path (family `on`): agent hinge on the own plan against the row's actor boxes
                    and the road-and-lane raster hinge, both shape-only with --shape, each summed over the hinge-only rows / n_imit x lam x ho_w;
                    non-plan heads in `distill_ho` (summed over the hinge-only rows / number of other rows, lam_d 30); no imitation target.
                    As bd4_train's note (x) does for S's rows, a row whose footprint at t0 is off the road-and-lane raster takes the pool's
                    drivable raster (<POOL>/drivable.npz, P2H10's layers) if the pool has one.
      both          the imit row plus the two hinges of the on-policy rows, summed / n_imit x lam x ho_w (n_imit then counts the on-policy rows).
  mixed         per row (turning school, 2026-10-13): pool rows whose tab `mode` is "self" are `self` rows, every other row is a `both` row
                (one merged pool of S's on-policy rows and demonstration rows, each with its own recipe). Pool without `mode`: = both.
  --yr-mass F       round 2 (plans/2026-10-12-onpol2-prereg.md, B: the YR10m10-F base): joint_train's yaw-rate rows, operation for operation:
                    round(batch x F) imitation rows of every batch from the caches yr1_* out of the normal rows' stream [seed, 0], never anchor rows,
                    P2H10's drivable hinge in each row's own frame (ot_rows.off_hinge), dev metrics `ot_*` on the yaw-rate dev rows. Batch order:
                    normal, yaw-rate, on-policy, hinge-only; the on-policy rows come out of the normal rows. With --init / --on off and no S flag
                    the run is `joint_train.py --yr-mass F` bit for bit (gate Y1). 0 = off: no operation is added.
  --agent-on-only   self / both: the agent hinge exists on the on-policy rows only (the pool's actor boxes); no offline row carries it (a base
                    without lesson S: --ho "" and this flag). Logged: `on/off_agent_rows` = offline rows of the batch with an agent label (0).
  --ego-drop P      onpol_long Amendment 11: the ego status of the fed ego vector (parity_adapter.EGO_STATE: vx, vy, ax, ay, the 4 history poses;
                    not `present`, not the command) zeroed on a fraction P of the non-anchor rows of every batch, drawn from its own stream
                    [seed, 0, 71] (every row stream unchanged). P = 1: the adapter's input transform zeroes it (TR1 key `noego`, saved with the
                    weights, so the served driver gets no ego status either). 0 = off: no operation is added.
  --on-road FILE    onpol_long Amendment 13: the pool rows' road raster from FILE instead of <POOL>/road.npz (same tokens, layout); '' = the pool's.
  --on-lon          extra per-row terms of the on-policy rows in self / both from experiments/onpol_nuplan/lib/on_lon.py (plans/on_pool_format.md):
                    each term summed / n_imit x --on-lon-lam 10 x ho_w. Off: no operation is added.

Logged per step under `on/` (on-policy rows of the batch): n_imit / n_hinge = rows that carry an imitation term / any hinge; imit, ade = their imitation
loss and mean displacement to the target (m); agent_* / road_* (+ every --on-lon term) = _mean / _pos / _posmean as loss43 logs a family.

  train    --tag T [flags]                 checkpoint $DATA_DIR/runs/op_parity/runs/<T>/ckpt-final.pt (+ adapter.pt; NAVSIM-standard record, served by
                                           sh30_driver.py through SH30_TAG; an existing checkpoint is never overwritten), run dir runs/<exp>/on-train-<T>/
  ident    --a TAG --b TAG [--out F.json]  joint_train.py's: trained weights equal bit for bit
  standin  --data navtrain_full.s2of12     pool `onstand_<data>` = the bd4 rows of that shard in the pool format (CPU)
  known    --on onstand_<data> --init TAG --out F.json   known-answer test on the stand-in: `self` = S's hinge-only path on the same bd4 rows; `imit`
                                           = plain imitation rows; `both` = imit + the hinges x ho_w / n_imit (exit 2 on a mismatch)
  t3       --tags IMIT SELF BOTH --out F.json            gate T3 from three logged runs: equal on_hash, per-mode row accounting
  t2       --tags T .. [--ref 0.545] --out F.json        gate T2 from logged --init runs (key match, dev ADE at step 0) + steps / s, peak GB
  y3       --tags SELF SELF2 --out F.json                gate Y3 (round 2) from two logged `self` runs of the Y recipe: row accounting, no agent
                                           hinge on offline rows, equal on_hash

  $DATA_DIR/envs/op-train/bin/python experiments/onpol_nuplan/scripts/on_train.py train --tag SSELF-ON-s0 --init P2H10S-F-s0 --on on1 --on-mode self \
      --ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib"), str(_R / "experiments/alpasim/scripts"), str(_R / "experiments/body1/lib"),
                 str(_R / "experiments/onpol_nuplan/lib")]
import argparse, hashlib, json, time  # noqa: E401,E402
from dataclasses import asdict, replace  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import joint_train as JT  # noqa: E402

MODES = ("imit", "self", "both")
ALL_MODES = MODES + ("mixed",)                                       # `mixed` stays out of the known-answer loop over MODES
NAVSIM_RASTER = "runs/op_probe/labels/navtrain_all.npz"             # pp_train.Cfg.hinge_labels: P2H10's drivable raster


def losses_on():
    """-> LossesOn (built lazily: loss43 imports torch and pp_train)."""
    import torch
    import pp_train as T
    from loss43 import Losses43, row_stats, shape_only

    class LossesOn(Losses43):
        """Losses43 + on-policy rows. on (n_store,) bool marks them. `self`: the caller flags them hinge-only (ho, family `on`), so Losses43 itself
        carries them; `imit` / `both`: plain imitation rows without a label in `hinge` and with agent2.ok False, so Losses43 adds the imitation
        term and nothing else; `both` then adds the two hinges here from on_aok (the pool's agent ok) and road.ok."""

        def __init__(self, *a, on, mode, on_aok=None, on0=0, lon=None, pack=None, lon_lam=10.0, agent_off=False, **k):
            super().__init__(*a, **k)
            self.on, self.mode, self.on_aok, self.on0, self.lon, self.pack, self.lon_lam = on, mode, on_aok, on0, lon, pack, lon_lam
            self.agent_off = agent_off

        def __call__(self, out, S, rows, anchor):
            total, Ls = super().__call__(out, S, rows, anchor)
            m = self.on[rows]
            if not m.any():
                return total, Ls
            c = self.cfg
            Ls = {("on/" + q[7:] if q.startswith("fam/on/") else q): v for q, v in Ls.items()}
            h = self.ho[rows] if self.ho is not None else torch.zeros_like(m)
            if self.mode == "mixed":                                     # the `self` rows are Losses43's hinge-only rows; the rest is `both`
                m = m & ~h
                if not m.any():
                    return total, Ls
            imit = ~h & ~anchor & S.has_fut[rows]
            n_imit = imit.sum().clamp_min(1)                             # Losses43's n_imit
            r = rows[m]
            x, y, psi = T.rear(out.float()[m][:, self.pi].view(-1, 33, 15), S.cam_x[r], self.W)
            hg = imit & self.hinge.ok[rows] if self.hinge is not None else torch.zeros_like(m)       # rows a hinge is evaluated on, as Losses43 picks them
            if self.agent2 is not None:
                hg = hg | ((imit | h) & self.agent2.ok[rows])
            if self.road is not None:
                hg = hg | (h & self.road.ok[rows])
            Ls["on/n_imit"] = (m & imit).sum().float()
            if self.mode != "self":                                      # logging only: the on-policy rows' share of `imit`
                with torch.no_grad():
                    f = S.fut[r]
                    Ls["on/imit"] = self.dist(out.float()[m][:, self.pi].view(-1, 33, 15), S.cam_x[r], f[..., 0], f[..., 1], f[..., 2]).mean()
                    Ls["on/ade"] = torch.hypot(x - f[..., 0], y - f[..., 1]).mean()
            if self.mode in ("both", "mixed"):                           # the hinge-only rows' two hinges, on rows that are imitation rows
                xs, ys = shape_only(x, y, psi) if self.shape else (x, y)
                av, rv = torch.zeros(len(r), device=x.device), torch.zeros(len(r), device=x.device)
                ok = self.on_aok[r]
                if ok.any():
                    av[ok] = self.agent2.per_step(xs[ok], ys[ok], psi[ok], r[ok]).mean(1)
                ko = self.road.ok[r]
                if ko.any():
                    rv[ko] = torch.relu(self.road.margin - self.road.margins(xs[ko], ys[ko], psi[ko], r[ko])).mean(1)
                Ls["on/agent_ho"], Ls["on/road_ho"] = av.sum() / n_imit, rv.sum() / n_imit
                total = total + self.ho_w * (c.agent_lam * Ls["on/agent_ho"] + self.road_lam * Ls["on/road_ho"])
                Ls |= row_stats(av, "on/agent") | row_stats(rv, "on/road")
                hg = hg.clone()
                hg[m] |= ok | ko
            Ls["on/n_hinge"] = (m & hg).sum().float()
            if self.agent_off:                                           # --agent-on-only: offline rows that would enter the agent hinge
                Ls["on/off_agent_rows"] = (~m & self.agent2.ok[rows]).sum().float()
            if self.lon is not None and self.mode != "imit":
                for q, v in self.lon.loss(torch.stack([x, y, psi], -1), r - self.on0, self.pack).items():
                    Ls[f"on/{q}_ho"] = v.sum() / n_imit
                    total = total + self.ho_w * self.lon_lam * Ls[f"on/{q}_ho"]
                    Ls |= row_stats(v, f"on/{q}")
            return total, Ls
    return LossesOn


def load_init(model, tag) -> dict:
    """Every trained tensor of run `tag` into `model` (net and adapter); the key sets must match exactly. -> the key-match record."""
    import torch
    import pp_train as T
    st = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"]
    net = {k for k, p in model.net.params.items() if p.requires_grad}
    ad = set(model.adapter.state_dict())
    res = dict(init=tag, arm=st["arm"], net_trainable=len(net), net_loaded=len(set(st["net"]) & net), net_missing=sorted(net - set(st["net"])),
               net_unexpected=sorted(set(st["net"]) - net), adapter_tensors=len(ad), adapter_loaded=len(set(st["parity"]) & ad),
               adapter_missing=sorted(ad - set(st["parity"])), adapter_unexpected=sorted(set(st["parity"]) - ad), tr1=st.get("tr1") or {})
    assert st["arm"] == model.arm and not (res["net_missing"] or res["net_unexpected"] or res["adapter_missing"] or res["adapter_unexpected"]), res
    assert not res["tr1"], "the S recipe has no TR1 input reduction"
    for k, v in st["net"].items():
        model.net.params[k].data.copy_(v)
    model.adapter.load_state_dict(st["parity"])
    dn = max(float((model.net.params[k].detach().cpu().float() - v.float()).abs().max()) for k, v in st["net"].items())
    da = max(float((p.detach().cpu().float() - st["parity"][k].float()).abs().max()) for k, p in model.adapter.state_dict().items())
    assert dn == 0 and da == 0, (dn, da)
    return res | dict(max_abs_diff_after_load=max(dn, da))


def setup(a, dev):
    """joint_train.cmd_train up to its training loop for the S flags, plus --init and the pool. -> namespace; ns.losses(mode) builds the loss
    (mode None = Losses43 as joint_train builds it, the on-policy rows then being plain rows without any label)."""
    import torch
    import b1 as B1
    import ot_rows as OR
    import pp_train as T
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    from loss43 import Losses43, OffAgentHinge
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    ho = [(f, int(k)) for f, k in (x.split(":") for x in a.ho.split(","))] if a.ho else []
    fams, kk = [f for f, _ in ho], [k for _, k in ho]
    mode = a.on_mode if a.on else None
    assert mode in ALL_MODES + (None,), "--on needs --on-mode imit | self | both | mixed"
    hing = mode in ("self", "both", "mixed")
    nB, k_ho, k_on = a.batch, sum(kk), a.on_k if a.on else 0
    k_yr = int(round(nB * a.yr_mass))                                # joint_train --yr-mass
    nN = nB - k_yr - k_ho - k_on                                     # normal rows per batch
    assert nN > 0 and not (hing and not (a.agent_lam > 0 and a.road_lam > 0)), "self / both need --agent-lam and --road-lam"
    assert not a.agent_on_only or hing, "--agent-on-only: with --on-mode self | both"
    yr = tuple(f"yr1_{d}" for d in a.data) if k_yr else ()
    hod = tuple(f"{f}_{d}" for f in fams for d in a.data)
    cfg = T.Cfg(arm="P2", seed=a.seed, steps=a.steps, batch=nB, data=tuple(a.data) + yr + hod + ((a.on,) if a.on else ()), split=a.split, frames="warp", host=True,
                warmup=a.warmup, eval_every=a.eval_every, hinge_lam=a.hinge_lam, hinge_margin=a.hinge_margin, agent_lam=a.agent_lam,
                agent_margin=a.agent_margin, agent_side_margin=a.agent_side_margin, agent_labels=a.agent_labels)
    torch.manual_seed(cfg.seed)
    S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
    names, logs = S.tab["names"], S.tab["log"]
    nd = [len(np.load(OR.CR / d / "tab.npz")["names"]) for d in cfg.data]
    n_base, n_on = sum(nd[:len(a.data)]), nd[-1] if a.on else 0
    on0 = S.n - n_on
    is_yr, fam, src = JT.layout(list(a.data), bool(k_yr), fams, nd[:len(a.data) + len(yr) + len(hod)])
    fam, is_on = np.concatenate([fam, np.full(n_on, -1)]), np.arange(S.n) >= on0
    is_yr, is_ho = np.concatenate([is_yr, np.zeros(n_on, bool)]), fam >= 0
    tr_rows, dv_rows, sp = T.split_rows(dict(names=np.where(is_ho | is_on, "", names), log=logs, is_b2d=S.is_b2d, is_wod=S.is_wod), cfg.split)
    assert not (is_ho | is_on)[tr_rows].any() and not (is_ho | is_on)[dv_rows].any()
    tr_o, tr_t, dv_o, dv_t = tr_rows[~is_yr[tr_rows]], tr_rows[is_yr[tr_rows]], dv_rows[~is_yr[dv_rows]], dv_rows[is_yr[dv_rows]]
    assert k_yr == 0 or len(tr_t) >= k_yr
    off = OR.offsets(cfg.data)
    trl = splits.load(B1.TRAIN)
    in_tr = trl.mask(logs) & ~is_on
    excl = splits.load(a.ho_excl) if a.ho_excl else None
    in_ho = in_tr & ~excl.mask(logs) if excl is not None else in_tr
    pools = [np.flatnonzero((fam == j) & in_ho) for j in range(len(fams))]
    pd, nv, onrec = OR.CR / a.on if a.on else None, None, {}
    if a.on:                                                         # the pool's contract (plans/on_pool_format.md)
        pt, tz = np.load(pd / "tab.npz"), np.load(OR.CR / f"{a.on}@warp" / "teacher.npz")
        pn = pt["names"]
        assert len(set(pn.tolist())) == n_on >= 1 and all(t.startswith("on:") for t in pn.tolist()), "pool names: unique, prefix on:"
        assert np.isfinite(pt["fut"]).all() and pt["fut"].shape == (n_on, 8, 3) and pt["ego"].shape == (n_on, S.tb["ego"].shape[1])
        assert np.array_equal(tz["di"], S.di) and np.array_equal(tz["pi"], S.pi) and tz["out"].shape == (n_on, len(S.di)), "pool teacher: di / pi / out"
        assert S.front.nslot[-1] == 8 and len(S.front.mms[-1]) == n_on, "pool front: (N, 8, 32, 512)"
        assert mode == "imit" or "off" not in pt.files or a.on.startswith("onstand_"), "a real pool stores its labels in the row's own frame (no `off`)"
        if "nv" in pt.files and (pt["nv"] < 8).any():                # cold-start rows fed fewer than 8 slots (PModel.forward nv)
            nv = torch.full((S.n,), 8, device=dev)
            nv[on0:] = torch.as_tensor(pt["nv"].astype(np.int64), device=dev)
        onrec = dict(pool=a.on, rows=n_on, logs=len(set(pt["log"].tolist())), nv_lt8=int((pt["nv"] < 8).sum()) if "nv" in pt.files else 0, has_off="off" in pt.files)
    self_on = np.zeros(S.n, bool)                                    # --on-mode mixed: the pool rows tagged mode == "self"
    if mode == "mixed" and "mode" in pt.files:
        self_on[on0:] = pt["mode"].astype(str) == "self"
    if mode == "mixed":
        onrec |= dict(mixed_self_rows=int(self_on.sum()), mixed_both_rows=int(n_on - self_on.sum()))
    model = T.PModel(cfg.arm).to(dev)
    model.set_tr1(dict(noax=False, hist_cv=False))
    init = load_init(model, a.init) if a.init else None
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_o, device=dev)].float().std(0).clamp_min(1e-3)
    bk = {"bank": True} if a.label_bank else {}
    hinge = None
    if cfg.hinge_lam > 0:                                            # with yaw-rate rows: in each row's own frame, as joint_train
        hinge = (OR.off_hinge(cfg, np.where(is_on, "", names), off, dev, **bk) if k_yr else
                 Hinge([data_dir() / f for f in cfg.hinge_labels], np.where(is_on, "", names), dev, cfg.hinge_margin, list(cfg.hinge_footprint), **bk))
    agent2 = road = on_aok = None
    if cfg.agent_lam > 0:                                            # A: imitation rows of the train logs + every hinge-only row, each in its own frame
        agent2 = OffAgentHinge(data_dir() / cfg.agent_labels, np.where(in_tr, names, ""), off, dev, cfg.agent_margin,
                               None if cfg.agent_side_margin < 0 else cfg.agent_side_margin)
        if a.agent_on_only:                                          # no offline row carries the agent hinge; the pool's rows are set below
            agent2.ok[:] = False
    if (k_ho or hing) and a.road_lam > 0:                            # C: the road-and-lane raster on the hinge-only rows (+ the pool's own file)
        rc = replace(cfg, hinge_labels=(a.road_labels,) + ((getattr(a, "on_road", "") or str(pd / "road.npz"),) if hing else ()), hinge_footprint=("pacifica",) * (1 + hing),
                     hinge_margin=a.road_margin)
        road = OR.off_hinge(rc, np.where(is_ho | (is_on & hing), names, ""), off, dev, **bk)
    n_fb = 0
    if road is not None and hinge is not None and k_ho:              # bd4_train note (x): a row not on the scorer-layer road at t0 keeps the NAVSIM raster
        hr = torch.nonzero(torch.as_tensor(is_ho, device=dev) & road.ok)[:, 0]
        fb = []
        with torch.no_grad():
            for i in range(0, len(hr), 4096):
                r = hr[i:i + 4096]
                z = torch.zeros(len(r), 8, device=dev)
                fb.append(r[road.margins(z, z, z, r).amin(1) < 0])
        fb = torch.cat(fb)
        road.sdf[fb] = hinge.sdf[fb]
        road.ok[fb] = hinge.ok[fb]
        n_fb = int(len(fb))
    if hing:                                                         # the pool's actor boxes, already in each row's own frame
        ah = AgentHinge(pd / "agent.npz", pn, dev, cfg.agent_margin, None if cfg.agent_side_margin < 0 else cfg.agent_side_margin)
        assert ah.box.shape[1:] == agent2.box.shape[1:], f"pool agent boxes {tuple(ah.box.shape)}: (N, 9, {agent2.box.shape[2]}, 5) expected"
        agent2.box[on0:], agent2.val[on0:] = ah.box, ah.val
        on_aok = torch.zeros(S.n, dtype=torch.bool, device=dev)
        on_aok[on0:] = ah.ok
        with torch.no_grad():                                         # note (x) for the pool: rows off the road raster at t0 take <POOL>/drivable.npz
            r = torch.arange(on0, S.n, device=dev)
            r = r[road.ok[r]]
            z = torch.zeros(len(r), 8, device=dev)
            fb = torch.cat([r[i:i + 4096][road.margins(z[i:i + 4096], z[i:i + 4096], z[i:i + 4096], r[i:i + 4096]).amin(1) < 0] for i in range(0, len(r), 4096)] or [r])
        onrec |= dict(agent_rows=int(ah.ok.sum()), road_rows=int(road.ok[on0:].sum()), offroad_at_t0=int(len(fb)), drivable_raster_kept=0)
        if len(fb) and (pd / "drivable.npz").exists():
            dr = Hinge(pd / "drivable.npz", pn, dev, a.road_margin, bank=True)
            road.sdf[fb], road.ok[fb] = dr.sdf[fb - on0], dr.ok[fb - on0]
            onrec |= dict(drivable_raster_kept=int(len(fb)), road_rows=int(road.ok[on0:].sum()))
            del dr
        del ah
    lon = pack = None
    if a.on_lon:
        assert hing, "--on-lon: self / both only"
        import importlib
        lon = importlib.import_module(a.on_lon)
        pack = lon.load(pd, pn, dev) if hasattr(lon, "load") else dict(dir=pd, names=pn, dev=dev)
    t = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731

    def losses(md):
        k43 = dict(agent2=agent2, road=road, road_lam=a.road_lam, ho_w=a.ho_w, shape=a.shape)
        sfr = is_on & (self_on if md == "mixed" else np.full(S.n, md == "self"))   # pool rows Losses43 carries as hinge-only rows
        if agent2 is not None:
            agent2.ok[on0:] = on_aok[on0:] & t(sfr[on0:]) if md in ("self", "mixed") else False
        if md is None:
            return Losses43(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None, ho=t(is_ho) if k_ho else None, fam=t(fam), fams=fams, **k43)
        sf = bool(sfr.any())                                         # family `on`: Losses43's own hinge-only rows
        return losses_on()(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None, ho=t(is_ho | sfr) if k_ho or sf else None,
                           fam=t(np.where(sfr, len(fams), fam)), fams=fams + ["on"] * sf, on=t(is_on), mode=md, on_aok=on_aok, on0=on0,
                           lon=lon, pack=pack, lon_lam=a.on_lon_lam, agent_off=a.agent_on_only, **k43)
    return SimpleNamespace(cfg=cfg, S=S, model=model, base=base, new=new, hinge=hinge, agent2=agent2, road=road, losses=losses, mode=mode, ho=ho, fams=fams,
                           kk=kk, pools=pools, nN=nN, k_ho=k_ho, k_on=k_on, n_on=n_on, on0=on0, nv=nv, tr_o=tr_o, dv_o=dv_o, sp=sp, trl=trl, excl=excl,
                           n_fb=n_fb, init=init, onrec=onrec, names=names, fam=fam, is_on=is_on, k_yr=k_yr, tr_t=tr_t, dv_t=dv_t, dv_rows=dv_rows)


def cmd_train(a):
    import torch
    import parity_adapter as PA
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    d = T.proot("runs", a.tag)
    assert not (d / "ckpt-final.pt").exists(), f"{d / 'ckpt-final.pt'} exists: a tag is never overwritten"
    ns = setup(a, dev)
    cfg, S, model, base, new, pools, kk = ns.cfg, ns.S, ns.model, ns.base, ns.new, ns.pools, ns.kk
    nB, nN, k_ho, k_on, tr_o, dv_o = cfg.batch, ns.nN, ns.k_ho, ns.k_on, ns.tr_o, ns.dv_o
    k_yr, tr_t, dv_t = ns.k_yr, ns.tr_t, ns.dv_t
    LS = ns.losses(ns.mode)
    pp_lr = not (k_yr or k_ho or k_on or a.agent_lam > 0)            # pp_train's own recipe: its cosine form
    rng = np.random.default_rng([cfg.seed, 0])                       # pp_train / bd4_train's row stream
    hrng = np.random.default_rng([cfg.seed, 0, 43])                  # bd4_train's hinge-only rows
    orng = np.random.default_rng([cfg.seed, 0, 53])                  # the on-policy rows
    erng = np.random.default_rng([cfg.seed, 0, 71])                  # --ego-drop's rows
    if a.ego_drop >= 1.0:                                            # ego status removed, at training and at serving (saved TR1 key)
        model.set_tr1({"noego": True})
    sha = hashlib.sha1()
    lb, ln = (cfg.lr, cfg.lr_new) if a.lr_scale == 1.0 else (cfg.lr * a.lr_scale, cfg.lr_new * a.lr_scale)   # onpol_long Amendment 6: gentle continuation
    opt = torch.optim.AdamW([{"params": base, "lr": lb, "base": lb}, {"params": new, "lr": ln, "base": ln}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()

    def draw():
        r = rng.choice(tr_o, nN, replace=len(tr_o) < nN)
        if k_yr:                                                     # joint_train: the yaw-rate rows from the same stream, right after the normal rows
            r = np.concatenate([r, rng.choice(tr_t, k_yr, replace=False)])
        an = rng.random(nB - k_ho - k_on) < cfg.d_frac
        an[nN:] = False                                              # yaw-rate rows: imitation rows only (no row without --yr-mass)
        if k_on:                                                     # after the normal rows; never anchor rows
            i = orng.choice(ns.n_on, k_on, replace=ns.n_on < k_on)
            sha.update(i.astype("<i8").tobytes())
            r, an = np.concatenate([r, ns.on0 + i]), np.concatenate([an, np.zeros(k_on, bool)])
        if k_ho:                                                     # bd4_train: the hinge-only rows last
            r = np.concatenate([r] + [hrng.choice(p, q, replace=False) for p, q in zip(pools, kk)])
            an = np.concatenate([an, np.zeros(k_ho, bool)])
        return r, an, sha.hexdigest()[:16]

    def fetch(dr):
        return dr, S.front[dr[0]]
    S.front.pin = a.prefetch > 1
    Sd = T.dev_store(S, ns.dv_rows, a.dev_card_gb)

    def dev_ev():                                                    # joint_train's dev metrics (ot_* = the yaw-rate dev rows)
        ev = T.dev_eval(model.eval(), Sd, dv_o, LS.W)
        if len(dv_t):
            ev |= {"ot_" + q: v for q, v in T.dev_eval(model, S, dv_t, LS.W).items()}
        model.train()
        return ev
    with Run(a.exp, f"on-train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"onpol": vars(a)}) as run:
        for x in ns.sp + ((ns.trl,) if k_ho or ns.agent2 is not None else ()) + ((ns.excl,) if ns.excl is not None and k_ho else ()) + \
                ((splits.load(a.on_split),) if a.on and a.on_split else ()):
            run.use_split(x)
        run.info(f"{a.tag}: navsim input standard; train {len(tr_o)} normal" + (f" + {len(tr_t)} yaw-rate rows, dev {len(dv_o)} + {len(dv_t)}" if k_yr else f" rows, dev {len(dv_o)}")
                 + f"; per batch of {nB}: {nN} normal, " + (f"{k_yr} yaw-rate, " if k_yr else "") + f"{k_ho} hinge-only ("
                 + (", ".join(f"{f} {q} of {len(p)}" for (f, q), p in zip(ns.ho, pools)) or "none") + f"), {k_on} on-policy (mode {ns.mode or 'off'}; "
                 f"{json.dumps(ns.onrec)}; lon {bool(a.on_lon)}); weight {a.ho_w}, shape-only {a.shape}, excluded logs {a.ho_excl or 'none'}; hinge {cfg.hinge_lam} / "
                 f"{cfg.hinge_margin} m coverage {ns.hinge.coverage if ns.hinge else 0:.4f}{' (own frame)' if k_yr else ''}; agent {cfg.agent_lam} / {cfg.agent_margin} m"
                 f"{' (on-policy rows only)' if a.agent_on_only else ''} rows "
                 f"{int(ns.agent2.ok.sum()) if ns.agent2 is not None else 0}; road {a.road_lam if ns.road is not None else 0} rows "
                 f"{int(ns.road.ok.sum()) if ns.road is not None else 0} (NAVSIM raster kept on {ns.n_fb}); label bank {a.label_bank}, "
                 f"cosine form {'pp_train' if pp_lr else 'pi * step / steps'}")
        if ns.init:
            run.info(f"init {json.dumps(ns.init)}")
            ev = dev_ev()
            run.scalars({f"dev/{q}": v for q, v in ev.items()}, 0)
            run.info("dev @ 0: " + ", ".join(f"{q} {v:.3f}" for q, v in ev.items()))
            run.summary.update({f"dev0_{q}": v for q, v in ev.items()} | {"init": ns.init})
        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)
        t0, hist, hsh = time.time(), [], ""
        for step in range(cfg.steps):
            (r, an, hsh), front = pre.get()
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            for g in opt.param_groups:
                cosf = 0.5 * (1 + np.cos(np.pi * (step / cfg.steps))) if pp_lr else 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * cosf
            ego = S.ego[rows] * (~anchor)[:, None].float()
            if 0 < a.ego_drop < 1:
                dm = torch.as_tensor(erng.random(len(r)) < a.ego_drop, device=dev)
                ego[:, PA.EGO_STATE] = ego[:, PA.EGO_STATE] * (~dm)[:, None].float()
            out = model(front, ego, S.tc[rows], None, None, nv=None if ns.nv is None else ns.nv[rows])
            total, Ls = LS(out, S, rows, anchor)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {q: float(v) for q, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append({q: float(v) for q, v in Ls.items()})
            if (step + 1) % 25 == 0 or step + 1 == cfg.steps:
                keys = sorted({q for h in hist for q in h})
                mm = {q: float(np.mean([h[q] for h in hist if q in h])) for q in keys}
                hist = []
                el = time.time() - t0
                run.scalars({f"loss/{q}": v for q, v in mm.items()} | {"throughput/steps_per_s": (step + 1) / el, "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in mm.items() if "/" not in q)
                             + ("; on: " + ", ".join(f"{q[3:]} {v:.4f}" for q, v in mm.items() if q.startswith("on/")) + f"; on_hash {hsh}" if k_on else "")
                             + f"; {(step + 1) / el:.2f} it/s, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {step + 1}/{cfg.steps}")
            if k_on and (step + 1 == 60 or (step + 1) % 1000 == 0 or step + 1 == cfg.steps):    # gate T3: the on-policy indices drawn so far
                run.info(f"on_hash @ {step + 1}: {hsh}")
                run.summary.update({f"on_hash_{step + 1}": hsh})
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = dev_ev()
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.3f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
        torch.save({"model": model.state(), "cfg": asdict(cfg), "onpol": vars(a)}, d / "ckpt-final.pt")
        torch.save(model.adapter.state_dict(), d / "adapter.pt")
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, it_s=cfg.steps / (time.time() - t0), ckpt=str(d / "ckpt-final.pt"), n_train=len(tr_o),
                           ho_per_batch=k_ho, on_per_batch=k_on, on_mode=ns.mode or "off", on_hash=hsh, on=ns.onrec,
                           **(dict(yr_per_batch=k_yr, n_train_yr=len(tr_t), agent_on_only=a.agent_on_only) if k_yr or a.agent_on_only else {}),
                           gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


def cmd_standin(a):
    """The bd4 rows of one shard as a pool (plans/on_pool_format.md): inputs and teacher linked, actor boxes moved into the row's own frame, the road
    raster S trains them with (the logged frame + the tab's `off`; rows off the scorer-layer road at t0 keep the NAVSIM raster, bd4_train note x)."""
    import torch
    import ot_rows as OR
    import pp_train as T
    from drivable_hinge import Hinge
    from loss43 import OffAgentHinge
    from jevdrive.common import data_dir
    src, dst = f"bd4_{a.data}", a.out or f"onstand_{a.data}"
    z = np.load(OR.CR / src / "tab.npz")
    tok, off = z["names"], z["off"]
    names = np.array([f"on:bd4:{t}" for t in tok.tolist()])
    (OR.CR / dst).mkdir(exist_ok=True)
    (OR.CR / f"{dst}@warp").mkdir(exist_ok=True)
    np.savez(OR.CR / dst / "tab.npz", names=names, **{k: z[k] for k in ("log", "ego", "pose", "vel", "acc", "cmd", "fut", "cam", "lht", "speed", "off")})
    for f in ("front.npy", "teacher.npz"):
        p = OR.CR / f"{dst}@warp" / f
        p.unlink(missing_ok=True)
        p.symlink_to(OR.CR / f"{src}@warp" / f)
    H = OffAgentHinge(data_dir() / a.agent_labels, tok, off, "cpu", a.agent_margin, a.agent_side_margin)
    np.savez(OR.CR / dst / "agent.npz", tokens=names, ok=H.ok.numpy(), box=H.box.numpy(), valid=H.val.numpy())
    cfg = T.Cfg(arm="P2", hinge_labels=(a.road_labels,), hinge_margin=a.road_margin)
    road, nav = OR.off_hinge(cfg, tok, off, "cpu"), Hinge(data_dir() / NAVSIM_RASTER, tok, "cpu")
    r = torch.nonzero(road.ok)[:, 0]
    zz = torch.zeros(len(r), 8)
    fb = r[road.margins(zz, zz, zz, r).amin(1) < 0]
    road.sdf[fb], road.ok[fb] = nav.sdf[fb], nav.ok[fb]
    road.sdf[~road.ok] = 0
    np.savez(OR.CR / dst / "road.npz", tokens=names, ok=road.ok.numpy(), sdf=road.sdf[:, 0].numpy())
    print(json.dumps(dict(pool=dst, rows=len(names), agent_ok=int(H.ok.sum()), agent_moved=H.moved, road_ok=int(road.ok.sum()), navsim_raster_kept=int(len(fb)))))


def cmd_known(a):
    """Known-answer test on the stand-in pool (rows = the bd4 rows of the shard, same order): batch A = normal rows + 13 bd4 rows as S's hinge-only
    rows (plain Losses43); batch B = the same normal rows + the stand-in copies of those 13 rows as on-policy rows, in the three modes."""
    import torch
    import pp_train as T
    dev = torch.device("cuda")
    a.batch, a.on_mode, a.on_k, a.on_lon, a.yr_mass, a.agent_on_only = 128, "self", 13, "", 0.0, False
    ns = setup(a, dev)
    S, model, cfg, k = ns.S, ns.model, ns.cfg, ns.k_on
    jb = ns.fams.index("bd4")
    b0 = int(np.flatnonzero(ns.fam == jb)[0])                        # first bd4 Store row: stand-in row i = bd4 row b0 + i
    assert ns.n_on == int((ns.fam == jb).sum())
    rng = np.random.default_rng([a.seed, 0, 99])
    nN = cfg.batch - k
    acc, gmax, act = {}, 0.0, dict(agent=0, road=0, rows=0)

    def run(LS, rows, anchor, grad=False):
        with torch.no_grad():
            out = model(S.front[rows], S.ego[rows] * (~anchor)[:, None].float(), S.tc[rows], None, None, nv=None)
        out = out.requires_grad_(grad)
        total, Ls = LS(out, S, rows, anchor)
        g = torch.autograd.grad(total, out)[0] if grad else None
        return out.detach(), float(total), {q: float(v) for q, v in Ls.items()}, g

    def upd(q, x, y):
        acc[q] = max(acc.get(q, 0.0), abs(x - y))
    for b in range(a.batches):
        rn = rng.choice(ns.tr_o, nN, replace=False)
        x = rng.choice(ns.pools[jb], k, replace=False)
        assert all(ns.names[ns.on0 + (x - b0)] == np.char.add("on:bd4:", ns.names[x]))
        an = torch.as_tensor(np.concatenate([rng.random(nN) < cfg.d_frac, np.zeros(k, bool)]), device=dev)
        rA, rB = (torch.as_tensor(np.concatenate([rn, q]), device=dev) for q in (x, ns.on0 + (x - b0)))
        L43 = ns.losses(None)
        oA, tA, lA, gA = run(L43, rA, an, True)                      # S: the 13 bd4 rows as hinge-only rows
        _, tR, lR, _ = run(L43, rB, an)                              # reference of `imit`: plain imitation rows without any label
        res = {}
        for md in MODES:
            LS = ns.losses(md)
            res[md] = run(LS, rB, an, md == "self")
        oB, tS, lS, gS = res["self"]
        _, tI, lI, _ = res["imit"]
        _, tB, lB, _ = res["both"]
        upd("inputs: max |out_A - out_B|", float((oA - oB).abs().max()), 0.0)
        upd("self: total vs S", tS, tA)
        gmax = max(gmax, float(gA.abs().max()))
        upd("self: max |d total / d out| difference vs S", float((gS - gA).abs().max()), 0.0)
        for q in ("agent_ho", "road_ho", "distill_ho", "imit", "hinge", "agent", "cons", "distill"):
            upd(f"self: {q} vs S", lS.get(q, 0.0), lA.get(q, 0.0))
        for q in ("agent_mean", "agent_pos", "agent_posmean", "road_mean", "road_pos", "road_posmean"):
            upd(f"self: on/{q} vs S fam/bd4/{q}", lS.get(f"on/{q}", 0.0), lA.get(f"fam/bd4/{q}", 0.0))
            upd(f"both: on/{q} vs S fam/bd4/{q}", lB.get(f"on/{q}", 0.0), lA.get(f"fam/bd4/{q}", 0.0))
        upd("imit: total vs plain imitation rows", tI, tR)
        n_im = float((~an & S.has_fut[rB]).sum())
        upd("imit: imit vs (n0 x imit_S + k x on/imit) / (n0 + k)", lI["imit"], ((n_im - k) * lA["imit"] + k * lI["on/imit"]) / n_im)
        upd("both: total vs imit total + ho_w (agent_lam sum agent + road_lam sum road) / n_imit", tB,
            tI + a.ho_w * k * (cfg.agent_lam * lA["fam/bd4/agent_mean"] + a.road_lam * lA["fam/bd4/road_mean"]) / n_im)
        upd("self: on/n_imit", lS["on/n_imit"], 0)
        upd("self: on/n_hinge", lS["on/n_hinge"], k)
        upd("imit: on/n_imit", lI["on/n_imit"], k)
        upd("imit: on/n_hinge", lI["on/n_hinge"], 0)
        upd("both: on/n_imit", lB["on/n_imit"], k)
        upd("both: on/n_hinge", lB["on/n_hinge"], k)
        bad = [q for q in lI if q.startswith(("on/agent", "on/road"))] + [q for q in lS if q in ("on/imit", "on/ade")]
        assert not bad, f"terms that must not exist: {bad}"
        act["agent"] += round(lA["fam/bd4/agent_pos"] * k)
        act["road"] += round(lA["fam/bd4/road_pos"] * k)
        act["rows"] += k
    with torch.no_grad():                                            # PModel.forward: nv = 8 everywhere is the nv = None pass
        o8 = model(S.front[rB], S.ego[rB], S.tc[rB], None, None, nv=torch.full((len(rB),), 8, device=dev))
        o0 = model(S.front[rB], S.ego[rB], S.tc[rB], None, None, nv=None)
    acc["forward: max |out(nv = 8) - out(nv = None)|"] = float((o8 - o0).abs().max())
    tol = {q: (0.0 if "/n_" in q else a.tol) for q in acc}
    fail = [q for q, v in acc.items() if not v <= tol[q]]
    if min(act["agent"], act["road"]) < 5:
        fail.append(f"vacuous: active hinge rows {act}")
    res = dict(pool=a.on, init=a.init, batches=a.batches, on_rows_per_batch=k, active=act, tol=a.tol, max_abs_diff=acc, max_abs_grad_S=gmax, on=ns.onrec,
               fail=fail, **{"pass": not fail})
    print(json.dumps(res, indent=1))
    if a.out:
        _pl.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _pl.Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    raise SystemExit(0 if not fail else 2)


def run_scalars(exp, tag) -> tuple:
    """Latest run dir of on-train-<tag> -> ({scalar tag: {step: value}}, log text)."""
    from jevdrive.common import data_dir
    d = sorted((data_dir() / "runs" / exp).glob(f"on-train-{tag}/*/events.jsonl"))[-1]
    sc = {}
    for ln in d.read_text().splitlines():
        e = json.loads(ln)
        if e.get("kind") == "scalar":
            sc.setdefault(e["tag"], {})[int(e["step"])] = e["value"]
    return sc, (d.parent / "log.txt").read_text()


def cmd_t3(a):
    """Gate T3 from the logs of three runs (imit, self, both; same seed, pool and steps): identical on-policy indices (running sha1), and per mode
    which on-policy rows carried an imitation term / a hinge at every logged step."""
    import re
    res, bad, want = {}, [], dict(imit=(1, 0), self=(0, 1), both=(1, 1))
    for md, tag in zip(MODES, a.tags):
        sc, log = run_scalars(a.exp, tag)
        hs = dict(re.findall(r"on_hash @ (\d+): ([0-9a-f]+)", log))
        k = int(re.search(r"(\d+) on-policy \(mode (\w+)", log).group(1))
        mode = re.search(r"(\d+) on-policy \(mode (\w+)", log).group(2)
        ni, nh = sc.get("loss/on/n_imit", {}), sc.get("loss/on/n_hinge", {})
        keys = sorted(q[8:] for q in sc if q.startswith("loss/on/"))
        res[md] = dict(tag=tag, mode=mode, k=k, hash=hs, n_imit=sorted(set(ni.values())), n_hinge=sorted(set(nh.values())), on_terms=keys)
        if mode != md or set(ni.values()) != {want[md][0] * k} or set(nh.values()) != {want[md][1] * k}:
            bad.append(f"{md}: mode {mode}, n_imit {res[md]['n_imit']}, n_hinge {res[md]['n_hinge']} (expected {want[md][0] * k} / {want[md][1] * k})")
        hk = [q for q in keys if q.startswith(("agent", "road"))]
        ik = [q for q in keys if q in ("imit", "ade")]
        if bool(hk) != bool(want[md][1]) or bool(ik) != bool(want[md][0]):
            bad.append(f"{md}: logged on-policy terms {keys}")
    hs = [res[md]["hash"] for md in MODES]
    if not hs[0] or not (hs[0] == hs[1] == hs[2]) or str(a.step) not in hs[0]:
        bad.append(f"on_hash differs or is missing at step {a.step}: {hs}")
    res |= {"hash_step": a.step, "hash": hs[0].get(str(a.step)), "fail": bad, "pass": not bad}
    print(json.dumps(res, indent=1))
    if a.out:
        _pl.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _pl.Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    raise SystemExit(0 if not bad else 2)


def cmd_t2(a):
    """Gate T2 and the throughput table from logged --init runs: the key match of the load, dev ADE at step 0 against the archived value to 3
    decimals, steps / s and peak card memory at the last logged step."""
    import re
    res, bad = {}, []
    for tag in a.tags:
        sc, log = run_scalars(a.exp, tag)
        m = re.search(r"init (\{.*\})", log)
        init = json.loads(m.group(1)) if m else None
        its, gb = sc["throughput/steps_per_s"], sc["gpu/peak_gb"]
        res[tag] = dict(init=init, dev0={q[4:]: v[0] for q, v in sc.items() if q.startswith("dev/") and 0 in v}, steps=max(its), it_s=its[max(its)],
                        peak_gb=gb[max(gb)], composition=re.search(r"per batch of \d+: (.*?\(mode \w+)", log).group(1))
        ade = res[tag]["dev0"].get("ade")
        if init is None or init["net_loaded"] != init["net_trainable"] or init["adapter_loaded"] != init["adapter_tensors"] or init["max_abs_diff_after_load"] != 0:
            bad.append(f"{tag}: load {init}")
        if ade is None or round(ade, 3) != round(a.ref, 3):
            bad.append(f"{tag}: dev ADE at step 0 {ade} != {a.ref}")
    res |= {"ref_ade": a.ref, "fail": bad, "pass": not bad}
    print(json.dumps(res, indent=1))
    if a.out:
        _pl.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _pl.Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    raise SystemExit(0 if not bad else 2)


def cmd_y3(a):
    """Gate Y3 (round 2) from the logs of two `self` runs of the Y recipe (same seed, pool and flags): at every logged step the on-policy rows
    carry a hinge and no imitation term, no offline row enters the agent hinge (`agent` = 0, on/off_agent_rows = 0, no imit/agent_* statistic)
    while the offline rows keep their imitation term and P2H10's drivable hinge, both on-policy hinges fire somewhere, and the two runs drew the
    same on-policy indices (running sha1 at --step)."""
    import re
    res, bad = {}, []
    for tag in a.tags:
        sc, log = run_scalars(a.exp, tag)
        comp = re.search(r"per batch of \d+: (.*?\(mode \w+)", log).group(1)
        k, mode = re.search(r"(\d+) on-policy \(mode (\w+)", log).groups()
        k, L = int(k), {q[5:]: v for q, v in sc.items() if q.startswith("loss/")}
        on = sorted(q[3:] for q in L if q.startswith("on/"))
        r = res[tag] = dict(composition=comp, mode=mode, hash=dict(re.findall(r"on_hash @ (\d+): ([0-9a-f]+)", log)), on_terms=on,
                            n_imit=sorted(set(L["on/n_imit"].values())), n_hinge=sorted(set(L["on/n_hinge"].values())),
                            off_agent_rows=sorted(set(L.get("on/off_agent_rows", {None: None}).values())), offline_agent=sorted(set(L["agent"].values())),
                            offline_agent_stats=sorted(q for q in L if q.startswith("imit/agent")), agent_only_on="(on-policy rows only)" in log,
                            on_agent_pos_max=max(L["on/agent_pos"].values()), on_road_pos_max=max(L["on/road_pos"].values()),
                            on_agent_ho_max=max(L["agent_ho"].values()), on_road_ho_max=max(L["road_ho"].values()),
                            offline_terms={q: L[q][max(L[q])] for q in ("imit", "hinge", "distill", "distill_ho") if q in L})
        for ok, msg in ((mode == "self" and "yaw-rate" in comp and "0 hinge-only" in comp, f"composition {comp}"),
                        (r["n_imit"] == [0.0] and r["n_hinge"] == [float(k)], f"n_imit {r['n_imit']}, n_hinge {r['n_hinge']} (expected 0 / {k})"),
                        (not {"imit", "ade"} & set(on) and {"agent_mean", "road_mean"} <= set(on), f"on-policy terms {on}"),
                        (r["off_agent_rows"] == [0.0] and r["offline_agent"] == [0.0] and not r["offline_agent_stats"] and r["agent_only_on"],
                         f"offline agent hinge: rows {r['off_agent_rows']}, term {r['offline_agent']}, stats {r['offline_agent_stats']}"),
                        (r["on_agent_pos_max"] > 0 and r["on_road_pos_max"] > 0, "vacuous: an on-policy hinge never fires"),
                        ({"imit", "hinge", "distill"} <= set(r["offline_terms"]), f"offline terms {sorted(r['offline_terms'])}")):
            if not ok:
                bad.append(f"{tag}: {msg}")
    hs = [res[t]["hash"].get(str(a.step)) for t in a.tags]
    if not hs[0] or len(set(hs)) != 1:
        bad.append(f"on_hash at step {a.step}: {hs}")
    res |= {"hash_step": a.step, "hash": hs[0], "fail": bad, "pass": not bad}
    print(json.dumps(res, indent=1))
    if a.out:
        _pl.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _pl.Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    raise SystemExit(0 if not bad else 2)


def recipe_args(p):
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", nargs="+", default=JT.FULL)
    p.add_argument("--split", default="navsim/op-parity-full")
    p.add_argument("--hinge-lam", type=float, default=10.0, help="P2H10's drivable hinge on the imitation rows (NAVSIM raster)")
    p.add_argument("--hinge-margin", type=float, default=0.3)
    p.add_argument("--ho", default="", help='hinge-only rows per batch by family, e.g. "ot1:4,yr1:4,bd4:5"; "" = none')
    p.add_argument("--ho-w", type=float, default=1.0)
    p.add_argument("--ho-excl", default="", help="split whose logs give no hinge-only row (navsim/body1-val-logs)")
    p.add_argument("--shape", action="store_true", help="the new hinges get no along-heading pose gradient (body1 Amendment 6)")
    p.add_argument("--agent-lam", type=float, default=0.0)
    p.add_argument("--agent-margin", type=float, default=0.3)
    p.add_argument("--agent-side-margin", type=float, default=0.0)
    p.add_argument("--agent-labels", default="runs/op_parity/agent_labels/navtrain_all-k32.npz")
    p.add_argument("--road-lam", type=float, default=10.0, help="drivable hinge of the hinge-only rows (only with --ho or --on-mode self | both)")
    p.add_argument("--road-margin", type=float, default=0.3)
    p.add_argument("--road-labels", default=JT.ROAD)
    p.add_argument("--label-bank", action="store_true")
    p.add_argument("--init", default="", help="run tag whose trained net and adapter weights the run starts from")
    p.add_argument("--yr-mass", type=float, default=0.0, help="joint_train's: share of every batch drawn from the yaw-rate rows (yr1_*) as imitation rows; 0 = off")
    p.add_argument("--agent-on-only", action="store_true", help="the agent hinge on the on-policy rows only (self / both); offline rows carry none")
    p.add_argument("--on", default="", help="on-policy pool: a cache dir under runs/op_parity/cache (plans/on_pool_format.md); '' = no on-policy rows")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap_.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--exp", default="onpol_nuplan", help="experiment name of the run dir (runs/<exp>/on-train-<tag>/)")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--lr-scale", type=float, default=1.0, help="both learning rates x this (1.0 = the recipe's; onpol_long Amendment 6)")
    p.add_argument("--eval-every", type=int, default=1000)
    recipe_args(p)
    p.add_argument("--on-k", type=int, default=13)
    p.add_argument("--on-mode", default="", choices=("",) + ALL_MODES)
    p.add_argument("--on-split", default="", help="registered split of the pool's scenes (alpasim_nuplan/onpol-train), recorded with the run")
    p.add_argument("--on-lon", nargs="?", const="on_lon", default="", metavar="MODULE",
                   help="extra on-policy terms (self / both) from experiments/onpol_nuplan/lib/on_lon.py (or another module with its load / loss)")
    p.add_argument("--on-lon-lam", type=float, default=10.0)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--fetch-workers", type=int, default=2)
    p.add_argument("--dev-card-gb", type=float, default=4.0)
    p.add_argument("--on-road", default="", help="road-and-lane raster of the pool's rows in place of <POOL>/road.npz (same layout and names; onpol_long Amendment 13)")
    p.add_argument("--ego-drop", type=float, default=0.0, help="share of rows with the ego status zeroed; 1 = removed, also when served (onpol_long Amendment 11)")
    p = sub.add_parser("ident")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", default="")
    p = sub.add_parser("standin")
    p.add_argument("--data", default="navtrain_full.s2of12")
    p.add_argument("--out", default="", help="pool name (default onstand_<data>)")
    p.add_argument("--agent-labels", default="runs/op_parity/agent_labels/navtrain_all-k32.npz")
    p.add_argument("--agent-margin", type=float, default=0.3)
    p.add_argument("--agent-side-margin", type=float, default=0.0)
    p.add_argument("--road-labels", default=JT.ROAD)
    p.add_argument("--road-margin", type=float, default=0.3)
    p = sub.add_parser("known")
    recipe_args(p)
    p.add_argument("--batches", type=int, default=24)
    p.add_argument("--tol", type=float, default=1e-4, help="max |difference| of a loss value against its S counterpart")
    p.add_argument("--out", default="")
    p.set_defaults(steps=1, warmup=1, eval_every=1, on_lon_lam=10.0)
    p = sub.add_parser("t3")
    p.add_argument("--tags", nargs=3, required=True, metavar=("IMIT", "SELF", "BOTH"))
    p.add_argument("--exp", default="onpol_nuplan")
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--out", default="")
    p = sub.add_parser("t2")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--exp", default="onpol_nuplan")
    p.add_argument("--ref", type=float, default=0.545, help="archived dev ADE of the --init checkpoint")
    p.add_argument("--out", default="")
    p = sub.add_parser("y3")
    p.add_argument("--tags", nargs=2, required=True, metavar=("SELF", "SELF2"))
    p.add_argument("--exp", default="onpol_nuplan")
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--out", default="")
    a = ap_.parse_args()
    {"train": cmd_train, "ident": JT.cmd_ident, "standin": cmd_standin, "known": cmd_known, "t3": cmd_t3, "t2": cmd_t2, "y3": cmd_y3}[a.cmd](a)
