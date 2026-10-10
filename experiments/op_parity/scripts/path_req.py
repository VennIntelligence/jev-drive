"""op_parity path-req (plans/2026-10-09-path-req-prereg.md, decision 204): the accuracy-requirement curve of a path-predictive memory branch.
The field of the LOGGED FUTURE trajectory (decision 200's arm JP, here with an arrival-time channel) is degraded in controlled ways and fed through
the adapter memory channel (pp_train.py --mem-e2e q<kind>), tokenizer trained jointly with the adapter from a tokenizer pre-trained with its own
head. EVERY q* ARM EXCEPT qh READS THE LOGGED FUTURE: leaked-label ORACLE PROBES, never a method and never a reportable inference path. No WA-JEPA
weights or features. qh reads the path predicted by a thin head on frozen Cinque tokens (non-privileged; a reference point, not a method).

Kinds (--mem-e2e q<kind>; fields are 2 x 128 x 96 at 0.5 m, x -8..56 m, y +-24 m: distance to the polyline clip(6 m) / 3, arrival time / 4 s of the
nearest polyline point faded out over 3 m; kinds without timing leave the second channel zero):
  qf        the logged trajectory: origin + 8 poses at 0.5 .. 4 s (path and timing); the undegraded oracle
  qx        qf of another log's row (geo_x's fixed derangement): the shuffled-content control
  qn<s>     cross-track error eps * (t / 4 s)^1.5, eps ~ N(0, s^2) per row (fixed per row; s = error std at 4 s, m)
  ql<s>     along-track error of the same shape (the poses slide along the logged path; s = error std at 4 s, m)
  qt<T>     only the first T seconds of the trajectory
  qs        path shape only: the logged path extended straight to 64 m, no arrival times (no speed, no path length)
  qv        speed profile only: the arc lengths at 0.5 .. 4 s laid on a straight line ahead, with arrival times (no lateral geometry)
  qch       heading at 4 s only: a 20 m constant-curvature arc that ends at the logged heading change (no speed, no shape)
  qc7       exit class only: the same arc for the centre of the signed turn bucket (straight, +-5-20, +-20-45, > 45 deg)
  qh        qf of the path predicted by `head` (thin head on frozen Cinque tokens + the P2 ego input): NOT privileged
  qp        HEAD1b step B / C (experiments/corridor, amendment of plans/2026-10-10-head1-prereg.md; EXPLORATORY): the heading-versus-arc-length
            profile PREDICTED by the HEAD1 head (frozen vision tokens + ego; NOT privileged), integrated to a polyline over 0 .. 40 m and
            extended straight to 64 m along its heading at 40 m; the qs encoding (no arrival times). Profiles are read per data dir from the
            first directory of $H1B_PROF (colon-separated; default the head's final out-of-fold predictions) that holds <data>.npy (N, 22)
  qpx       qp of another log's row (geo_x's fixed derangement): its shuffled-content control

  selftest (CPU)              shapes, finiteness and the realised error statistics of every kind on navtest rows
  tok   --kind K (GPU, pool)  pre-train the tokenizer of a kind with its own thin plan head on navsim/op-parity-geotok-train (outside the pilot's
                              shards) -> $OUT/tok/<kind>.pt + .json (head dev ADE with the row's own / another log's field)
  head  (GPU, pool)           the non-privileged reference predictor: thin head on 2 of the 8 cached Cinque policy slots + ego, fitted on the same
                              rows -> $OUT/head/<data>.npy (N, 8, 3) for the pilot shards and navtest, head.json
  gate  --stage dev|navtest   the pre-registered reliability gate of the undegraded oracle (exit 2: not met; exit 3: oracle itself below the line)
  report (op-train, CPU)      tables, the curve, reference predictors, the registered reading -> $OUT/report/
  fig   (any env)             the curve figure from the committed report files
  ident --tag T (GPU, pool)   HEAD1b identity check of the memory channel: the no-memory checkpoint T moved into the memory arm's model, memory
                              masked on every row, must give T's own outputs (dev rows + navtest) -> $OUT/ident_<T>.json (exit 2: not met)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "research"), str(_pl.Path(__file__).parent), str(_R / "experiments/op_probe/scripts")]
import argparse, json, zlib  # noqa: E401,E402

import numpy as np  # noqa: E402

import geo_oracle as G  # noqa: E402

OUT = G.D / "runs" / "op_parity" / "path_req"
RES = _R / "experiments/op_parity/results/path_req"
GAMMA = 1.5                                  # error growth with horizon: SH30-pilot plan errors grow as t^1.6 (cross-track) / t^1.45 (along-track)
ARC_LEN, EXT_LEN, TAU_REACH = 20.0, 64.0, 3.0
C7_EDGES, C7_CENTRE = (5.0, 20.0, 45.0), (0.0, 12.5, 32.5, 70.0)
PROBE = "LEAKED-LABEL ORACLE PROBES (the memory input is computed from the logged future; QH excepted): not a method, not a reportable inference path"
# arm = checkpoint stem (tag <arm>-F-s<seed>): kind, seeds, how the tokenizer starts
ARMS = {"QF": ("qf", (0, 1, 2), "warm"), "QFC": ("qf", (0, 1, 2), "cold"), "QFL": ("qf", (0, 1, 2), "cold-lr"), "QX": ("qx", (0, 1), "warm:qf"),
        "QN025": ("qn0.25", (0, 1), "warm"), "QN050": ("qn0.5", (0, 1), "warm"), "QN100": ("qn1.0", (0, 1), "warm"), "QN200": ("qn2.0", (0, 1), "warm"),
        "QL075": ("ql0.75", (0, 1), "warm"), "QL150": ("ql1.5", (0, 1), "warm"), "QL300": ("ql3.0", (0, 1), "warm"),
        "QT1": ("qt1", (0, 1), "warm"), "QT2": ("qt2", (0, 1), "warm"), "QS": ("qs", (0, 1), "warm"), "QV": ("qv", (0, 1), "warm"),
        "QCH": ("qch", (0, 1), "warm"), "QC7": ("qc7", (0, 1), "warm"), "QH": ("qh", (0, 1), "warm:qf")}
G.STEM.update({a: a for a in ARMS})
AXES = {"lat": ("cross-track error std at 4 s (m)", [("QF", 0.0), ("QN025", 0.25), ("QN050", 0.5), ("QN100", 1.0), ("QN200", 2.0)]),
        "lon": ("along-track error std at 4 s (m)", [("QF", 0.0), ("QL075", 0.75), ("QL150", 1.5), ("QL300", 3.0)]),
        "horizon": ("horizon kept (s)", [("QF", 4.0), ("QT2", 2.0), ("QT1", 1.0)]),
        "content": ("content kept", [("QF", "path + timing"), ("QS", "path shape"), ("QV", "speed profile"), ("QCH", "heading at 4 s"),
                                     ("QC7", "exit class")])}
LINES = (0.5, 0.3)
P_DS, P_N = 2.5, 17                          # qp: the profile's grid points at 0, 2.5 .. 40 m (the first 17 of the head's 22)
H0_DEV_ADE = 0.6305                          # GH0-F-s0 / s1 dev ADE 0.632 / 0.629 (decision 200)


# ---------------------------------------------------------------- degradations and the field
def parse(kind):
    assert kind[:1] == "q", kind
    for n in ("c7", "ch", "px", "p", "f", "x", "n", "l", "t", "s", "v", "h"):
        if kind[1:].startswith(n):
            r = kind[1 + len(n):]
            return n, (float(r) if r else 0.0)
    raise ValueError(f"unknown path-req kind {kind!r}")


def noise(n, dom):
    """(n, 2) standard normals, fixed per row of a row domain (column 0 cross-track, 1 along-track): the error of a row is part of the data."""
    import torch
    return torch.randn((n, 2), generator=torch.Generator().manual_seed(zlib.crc32(f"path_req/{dom}".encode())))


def poly(kind, fut, z=None):
    """fut (B, 8, 3) poses x, y, yaw at 0.5 .. 4 s (rear axle at t0, y left), z (B, 2) -> polyline vertices Q (B, K + 1, 2) starting at the origin and
    their times T (B, K + 1) as fractions of 4 s (None: the kind carries no timing)."""
    import torch
    name, lv = parse(kind)
    B, dev = len(fut), fut.device
    if name in ("p", "px"):                                                       # fut = (B, 22) heading (rad) on the head's arc-length grid
        h = fut[:, :P_N]
        hm = 0.5 * (h[:, 1:] + h[:, :-1])                                         # heading of a 2.5 m step = the mean of its two grid points
        V = torch.cat([fut.new_zeros(B, 1, 2), (P_DS * torch.stack([torch.cos(hm), torch.sin(hm)], -1)).cumsum(1)], 1)
        tan = torch.stack([torch.cos(h[:, -1]), torch.sin(h[:, -1])], -1)
        return torch.cat([V, (V[:, -1] + (EXT_LEN - P_DS * (P_N - 1)) * tan)[:, None]], 1), None
    p, psi, o = fut[..., :2], fut[..., 2], fut.new_zeros(len(fut), 1, 2)
    k8 = torch.arange(1, 9, device=dev) / 8.0
    T = torch.cat([fut.new_zeros(B, 1), k8.expand(B, 8)], 1)
    g = k8 ** GAMMA
    if name in ("f", "x", "h"):
        return torch.cat([o, p], 1), T
    if name == "n":
        nrm = torch.stack([-torch.sin(psi), torch.cos(psi)], -1)
        return torch.cat([o, p + (lv * z[:, :1] * g)[..., None] * nrm], 1), T
    if name == "t":
        n = int(round(2 * lv))
        assert 1 <= n <= 8
        return torch.cat([o, p[:, :n]], 1), T[:, :n + 1]
    V = torch.cat([o, p], 1)
    s = (V[:, 1:] - V[:, :-1]).norm(dim=-1).cumsum(1)                              # (B, 8) arc length at the 8 poses
    tan8 = torch.stack([torch.cos(psi[:, -1]), torch.sin(psi[:, -1])], -1)
    if name == "l":                                                               # the poses slide along the logged path (straight beyond its end)
        s2 = (s + lv * z[:, 1:2] * g).clamp_min(0).cummax(1).values
        V = torch.cat([V, (p[:, -1] + 200.0 * tan8)[:, None]], 1)                 # (B, 10, 2)
        A = torch.cat([s.new_zeros(B, 1), s, s[:, -1:] + 200.0], 1).contiguous()
        i = (torch.searchsorted(A, s2.contiguous(), right=True) - 1).clamp(0, 8)
        a0, a1 = A.gather(1, i), A.gather(1, i + 1)
        w = ((s2 - a0) / (a1 - a0).clamp_min(1e-6)).clamp(0, 1)[..., None]
        v0, v1 = (V.gather(1, j[..., None].expand(-1, -1, 2)) for j in (i, i + 1))
        return torch.cat([o, v0 + w * (v1 - v0)], 1), T
    if name == "s":
        return torch.cat([V, (p[:, -1] + (EXT_LEN - s[:, -1:]).clamp_min(1.0) * tan8)[:, None]], 1), None
    if name == "v":
        return torch.cat([o, torch.stack([s, torch.zeros_like(s)], -1)], 1), T
    d = torch.atan2(torch.sin(psi[:, -1]), torch.cos(psi[:, -1]))                 # c7 / ch: the heading change at 4 s
    if name == "c7":
        c = torch.tensor(C7_CENTRE, device=dev)[torch.bucketize(torch.rad2deg(d).abs(), torch.tensor(C7_EDGES, device=dev))]
        d = torch.deg2rad(c) * torch.sign(d)
    sa = (ARC_LEN * k8).expand(B, 8)
    kap = (d / ARC_LEN)[:, None]
    small = kap.abs() < 1e-4
    kk = torch.where(small, torch.ones_like(kap), kap)
    x = torch.where(small, sa, torch.sin(kap * sa) / kk)
    y = torch.where(small, 0.5 * kap * sa ** 2, (1 - torch.cos(kap * sa)) / kk)
    return torch.cat([o, torch.stack([x, y], -1)], 1), None


def field(Q, T, gx, gy):
    """Q (B, K + 1, 2), T (B, K + 1) | None -> (B, 2, 128, 96): distance of every cell centre to the polyline, clip(6 m) / 3; arrival time of the
    nearest polyline point (fraction of 4 s) x max(0, 1 - distance / 3 m)."""
    import torch
    a, d = Q[:, None, None, :-1], (Q[:, 1:] - Q[:, :-1])[:, None, None]             # (B, 1, 1, K, 2)
    cx, cy = gx.reshape(1, -1, 1, 1) - a[..., 0], gy.reshape(1, 1, -1, 1) - a[..., 1]
    t = ((cx * d[..., 0] + cy * d[..., 1]) / d.pow(2).sum(-1).clamp_min(1e-6)).clamp(0, 1)
    dist, ix = torch.hypot(cx - t * d[..., 0], cy - t * d[..., 1]).min(-1)
    c0 = dist.clamp(max=G.CLIP) / G.SCALE
    if T is None:
        return torch.stack([c0, torch.zeros_like(c0)], 1)
    tau = (T[:, None, None, :-1] + t * (T[:, 1:] - T[:, :-1])[:, None, None]).gather(-1, ix[..., None])[..., 0]
    return torch.stack([c0, tau * (1 - dist / TAU_REACH).clamp(0, 1)], 1)


def grid(dev):
    import torch
    from drivable_hinge import NH, NW, X0, Y0
    return X0 + (torch.arange(NH, device=dev) + 0.5) * 0.5, Y0 + (torch.arange(NW, device=dev) + 0.5) * 0.5


class PathMem:
    """Memory tokens (B, 32, 512) fp32 of store rows from the trainable tokenizer (geo_oracle.build_net, 2 input channels). src (N, 8, 3): the
    poses the field is drawn from (the logged future; kind qh: the thin head's prediction); z (N, 2) the rows' fixed error draws; perm (N,): row r
    reads the field of row perm[r] (kind qx)."""

    def __init__(self, kind, src, dev, z=None, perm=None, net=None, init=""):
        import torch
        self.kind, self.src, self.z = kind, src, z
        self.gx, self.gy = grid(dev)
        self.perm = None if perm is None else torch.as_tensor(np.asarray(perm), device=dev)
        self.net = net if net is not None else G.build_net(2).to(dev)
        if init:
            self.net.load_state_dict(torch.load(init, map_location="cpu"))
        self.params = list(self.net.enc.parameters())                             # the thin head of build_net is used by `tok` only

    def raster(self, rows):
        r = rows if self.perm is None else self.perm[rows]
        return field(*poly(self.kind, self.src[r], None if self.z is None else self.z[r]), self.gx, self.gy)

    def __getitem__(self, rows):
        return self.net.tokens(self.raster(rows))


def prof_dirs():
    import os
    d = os.environ.get("H1B_PROF") or f"{G.D / 'runs/corridor/head1/final/prof'}:{G.D / 'runs/corridor/head1b/prof_ot'}"
    return [_pl.Path(x) for x in d.split(":") if x]


def prof(data):
    """The fed heading profile (N, 22) fp32 of a data dir (kind qp), tab order."""
    fs = [d / f"{data}.npy" for d in prof_dirs() if (d / f"{data}.npy").exists()]
    assert fs, f"no heading profile of {data} under {[str(d) for d in prof_dirs()]}"
    x = np.load(fs[0])
    assert x.ndim == 2 and x.shape[1] == 22 and np.isfinite(x).all(), (str(fs[0]), x.shape)
    print(f"path-req qp: profile of {data} from {fs[0]} {x.shape}", flush=True)
    return x.astype(np.float32)


def _xperm(datas, logs):
    """geo_x's fixed derangement of every data dir (decision 197), as rows of the concatenated dirs."""
    ps, off = [], 0
    for d in datas:
        q = np.load(G.MEM / "geo_x" / f"{d}.perm.npy")
        ps.append(q + off)
        off += len(q)
    perm = np.concatenate(ps)
    assert len(perm) == len(logs) and not (logs[perm] == logs).any()
    return perm


def attach(cfg, S, hinge, dev):
    """The PathMem of a pp_train store (rows = the concatenated data dirs); called by pp_train.py --mem-e2e q<kind>."""
    import torch
    name = parse(cfg.mem_e2e)[0]
    src = S.fut
    if name in ("h", "p", "px"):
        src = torch.as_tensor(np.concatenate([np.load(OUT / "head" / f"{d}.npy") if name == "h" else prof(d) for d in cfg.data]), dtype=torch.float32, device=dev)
        assert len(src) == S.n
    return PathMem(cfg.mem_e2e, src, dev, z=noise(S.n, "+".join(cfg.data)).to(dev),
                   perm=_xperm(cfg.data, S.tab["log"]) if name in ("x", "px") else None, init=cfg.mem_init)


def finish(om, model, S, dv_rows, W, rear, hinge, run, tag, ckpt_dir):
    """After training: tokenizer weights, dev diagnostics (ADE and raw-footprint departures with memory on / masked / mismatched / fp16), the
    navtest bank of the tag in tab order (what jevdrive.bench reads)."""
    import torch
    dev = S.ego.device
    torch.save(om.net.state_dict(), ckpt_dir / "geotok.pt")
    model.eval()
    with torch.no_grad():
        pi = torch.as_tensor(S.pi, device=dev)
        r = torch.as_tensor(dv_rows, device=dev)
        mis = r[torch.as_tensor(G.derange(S.tab["log"][dv_rows], np.random.default_rng(1)), device=dev)]

        def read(mode):
            ade, out, rms = [], [], []
            for i in range(0, len(r), 128):
                q = r[i:i + 128]
                tok = om[mis[i:i + 128] if mode == "mismatched" else q]
                tok = tok.half().float() if mode == "fp16" else tok
                mask = torch.zeros(len(q), 1, dtype=torch.bool, device=dev) if mode == "masked" else None
                p = model(S.front[q], S.ego[q], S.tc[q], tok, mask).float()[:, pi].view(-1, 33, 15)
                x, y, psi = rear(p, S.cam_x[q], W)
                ade.append(torch.hypot(x - S.fut[q][..., 0], y - S.fut[q][..., 1]).mean(1)[S.has_fut[q]])
                out.append((hinge.margins(x, y, psi, q).amin(1) < 0)[hinge.ok[q]].float())
                rms.append(tok.pow(2).mean((1, 2)))
            return float(torch.cat(ade).mean()), float(torch.cat(out).mean()), float(torch.cat(rms).mean().sqrt())
        dg = {}
        for mode in ("on", "masked", "mismatched", "fp16"):
            dg[f"e2e_dev_ade_{mode}"], dg[f"e2e_dev_out_{mode}"], rms = read(mode)
            if mode == "on":
                dg["e2e_token_rms"] = rms
        name = parse(om.kind)[0]
        tab = np.load(G.CR / G.TEST / "tab.npz")
        n = len(tab["names"])
        src = np.load(OUT / "head" / f"{G.TEST}.npy") if name == "h" else prof(G.TEST) if name in ("p", "px") else tab["fut"]
        assert src.shape == ((n, 22) if name in ("p", "px") else (n, 8, 3)) and np.isfinite(src).all(), "the path field needs finite inputs on every navtest row"
        te = PathMem(om.kind, torch.as_tensor(src, dtype=torch.float32, device=dev), dev, z=noise(n, G.TEST).to(dev),
                     perm=_xperm([G.TEST], tab["log"]) if name in ("x", "px") else None, net=om.net)
        bank = torch.cat([te[torch.arange(i, min(i + 512, n), device=dev)].half() for i in range(0, n, 512)]).cpu().numpy()
        assert bank.shape == (n, G.NTOK, G.DTOK) and np.isfinite(bank).all()
        G._save(f"ge_{tag}", G.TEST, bank)
        dg["e2e_bank_rms"] = float(np.sqrt((bank[::6].astype(np.float32) ** 2).mean()))
    run.info(f"path-req {om.kind}: " + json.dumps(dg) + f"; navtest bank {bank.shape} -> {G.MEM / f'ge_{tag}'}")
    run.summary.update(dg)
    if name in ("p", "px"):
        run.summary.update(prof_dirs=[str(d) for d in prof_dirs()])


# ---------------------------------------------------------------- error statistics of a set of predicted poses
def err_stats(P, fut):
    """P, fut (N, 8, 3) numpy -> the statistics the degradation axes are measured in (metres at 4 s along / across the logged heading)."""
    e, h = P[..., :2] - fut[..., :2], fut[..., 2]
    lat, lon = -np.sin(h) * e[..., 0] + np.cos(h) * e[..., 1], np.cos(h) * e[..., 0] + np.sin(h) * e[..., 1]
    d = np.linalg.norm(e, axis=-1)
    wrap = lambda a: np.degrees(np.arctan2(np.sin(a), np.cos(a)))  # noqa: E731
    c7 = lambda a: np.sign(wrap(a)) * np.digitize(np.abs(wrap(a)), C7_EDGES)  # noqa: E731
    dh = wrap(P[:, 7, 2] - fut[:, 7, 2])
    rms = lambda x: float(np.sqrt((x ** 2).mean()))  # noqa: E731
    rob = lambda x: float(1.4826 * np.median(np.abs(x)))  # noqa: E731
    return dict(n=len(P), ade=float(d.mean()), ade_1s=float(d[:, :2].mean()), ade_2s=float(d[:, :4].mean()), fde=float(d[:, 7].mean()),
                lat_rms4=rms(lat[:, 7]), lat_rob4=rob(lat[:, 7]), lon_rms4=rms(lon[:, 7]), lon_rob4=rob(lon[:, 7]),
                lat_rms_t=[rms(lat[:, k]) for k in range(8)], lon_rms_t=[rms(lon[:, k]) for k in range(8)],
                head_rms4_deg=rms(dh), head_rob4_deg=rob(dh), c7_acc=float((c7(P[:, 7, 2]) == c7(fut[:, 7, 2])).mean()))


def degraded_stats(kind, fut):
    """Realised error statistics of a noise kind's degraded poses on navtest (its fixed error draws)."""
    import torch
    Q, _ = poly(kind, torch.as_tensor(fut, dtype=torch.float32), noise(len(fut), G.TEST))
    P = np.concatenate([Q[:, 1:].numpy().astype(np.float64), fut[..., 2:]], -1)
    return err_stats(P, fut)


def cmd_selftest(a):
    import torch
    tab = np.load(G.CR / G.TEST / "tab.npz")
    fut = torch.as_tensor(tab["fut"][:2048], dtype=torch.float32)
    z = noise(len(tab["names"]), G.TEST)[:2048]
    gx, gy = grid("cpu")
    for arm, (kind, _, _) in ARMS.items():
        Q, T = poly(kind, fut, z)
        Fd = field(Q[:64], None if T is None else T[:64], gx, gy)
        assert Fd.shape == (64, 2, 128, 96) and torch.isfinite(Q).all() and torch.isfinite(Fd).all(), kind
        assert (T is None) == (parse(kind)[0] in ("s", "ch", "c7")) and (T is None or (T.diff(dim=1) >= 0).all())
        msg = f"{arm:6s} {kind:7s} vertices {tuple(Q.shape[1:])} field mean {Fd[:, 0].mean():.3f} / {Fd[:, 1].mean():.4f}"
        if parse(kind)[0] in ("n", "l"):
            st = degraded_stats(kind, tab["fut"].astype(np.float64))
            lv = parse(kind)[1]
            got = st["lat_rms4" if parse(kind)[0] == "n" else "lon_rms4"]
            msg += f"; navtest lat / lon rms at 4 s {st['lat_rms4']:.3f} / {st['lon_rms4']:.3f} m, ADE {st['ade']:.3f} m"
            assert (abs(got / lv - 1) < 0.05) if parse(kind)[0] == "n" else (0.75 < got / lv < 1.05), (kind, got)   # along-track: clipped at standstill
            assert parse(kind)[0] == "l" or st["lon_rms4"] < 0.02, (kind, st)       # cross-track errors are normal to the logged heading
        print(msg)
    s = torch.arange(P_N) * P_DS                                                  # qp: a constant-curvature profile integrates to its arc; a zero profile is qs of a straight log
    kap = torch.tensor([0.0, 0.02, -0.05, 0.1])[:, None]
    hp = torch.cat([kap * s, (kap * s)[:, -1:].expand(-1, 22 - P_N)], 1)
    Q, T = poly("qp", hp)
    kk = torch.where(kap == 0, torch.ones_like(kap), kap)
    ex = torch.stack([torch.where(kap == 0, s.expand(4, -1), torch.sin(kap * s) / kk), torch.where(kap == 0, torch.zeros(4, P_N), (1 - torch.cos(kap * s)) / kk)], -1)
    assert T is None and Q.shape == (4, P_N + 1, 2) and (Q[:, :P_N] - ex).norm(dim=-1).max() < 0.15, (Q[:, :P_N] - ex).norm(dim=-1).max()
    assert torch.allclose((Q[:, -1] - Q[:, -2]).norm(dim=-1), torch.tensor(EXT_LEN - 40.0)) and parse("qp")[0] == "p" and parse("qpx")[0] == "px"
    st8 = torch.zeros(1, 8, 3)
    st8[0, :, 0] = torch.arange(1, 9) * 5.0
    assert torch.allclose(field(*poly("qp", hp[:1]), gx, gy), field(*poly("qs", st8), gx, gy), atol=1e-5)
    print(f"qp     polyline vertices {tuple(Q.shape[1:])}; arc error of the integration at kappa 0.1 / m: {float((Q[3, :P_N] - ex[3]).norm(dim=-1).max()):.4f} m")
    f0 = field(*poly("qf", fut[:8]), gx, gy)
    xs = field(*poly("qx", fut[8:16]), gx, gy)
    assert not torch.allclose(f0, xs)
    print("selftest ok")


# ---------------------------------------------------------------- tokenizer pre-training (its own thin head) and the non-privileged thin head
def _navtrain():
    datas = [f"navtrain_full.s{i}of{G.K_SH}" for i in range(G.K_SH)]
    tabs = [np.load(G.CR / d / "tab.npz") for d in datas]
    cat = lambda k: np.concatenate([t[k] for t in tabs])  # noqa: E731
    return datas, tabs, cat("names"), cat("log"), cat("fut"), cat("ego")


def _imit(P, Y):
    import torch.nn.functional as F
    return F.huber_loss(P[..., :2], Y[..., :2], delta=1.0) + 3.0 * F.huber_loss(P[..., 2], Y[..., 2], delta=0.1)


def cmd_tok(a):
    import time
    import torch
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    (OUT / "tok").mkdir(parents=True, exist_ok=True)
    with Run("op_parity", f"path-req-tok-{a.kind}" + ("-smoke" if a.smoke else ""), seed=0, config=vars(a)) as run:
        _, _, names, log, fut, ego = _navtrain()
        has = ~np.isnan(fut[:, 0, 0])
        s_tk, s_dv = splits.load("navsim/op-parity-geotok-train"), splits.load("navsim/op-parity-s234-dev")
        run.use_split(s_tk), run.use_split(s_dv)
        fit, dv = np.flatnonzero(s_tk.mask(names) & has), np.flatnonzero(s_dv.mask(names) & has)
        ft = torch.as_tensor(np.nan_to_num(fut), dtype=torch.float32, device=dev)
        eg = torch.as_tensor(ego, dtype=torch.float32, device=dev)
        torch.manual_seed(0)
        qp = parse(a.kind)[0] == "p"                                              # HEAD1b: the field of the head's out-of-fold profile; the target stays the logged future
        src = torch.as_tensor(np.concatenate([prof(d) for d in _navtrain()[0]]), dtype=torch.float32, device=dev) if qp else ft
        assert len(src) == len(names)
        om = PathMem(a.kind, src, dev, z=noise(len(names), "navtrain12").to(dev))
        net = om.net
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        steps, wu = (30 if a.smoke else a.steps), (5 if a.smoke else 150)
        g = torch.Generator(device=dev).manual_seed(0)
        fit_r, t0, hist = torch.as_tensor(fit, device=dev), time.time(), []
        for step in range(steps):
            for q in opt.param_groups:
                q["lr"] = 1e-3 * min(1.0, (step + 1) / wu) * 0.5 * (1 + np.cos(np.pi * step / steps))
            r = fit_r[torch.randint(0, len(fit_r), (a.batch,), device=dev, generator=g)]
            loss = _imit(net.plan(om[r], eg[r]), ft[r])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            hist.append(float(loss))
            if (step + 1) % 250 == 0 or step + 1 == steps:
                run.scalars({"loss/imit": float(np.mean(hist))}, step + 1)
                run.info(f"step {step + 1}: imit {np.mean(hist):.4f}; {(step + 1) / (time.time() - t0):.1f} it/s")
                hist = []
        net.eval()

        @torch.no_grad()
        def ade(rows, src=None):
            r, s = torch.as_tensor(rows, device=dev), torch.as_tensor(rows if src is None else src, device=dev)
            P = torch.cat([net.plan(om[s[i:i + 512]], eg[r[i:i + 512]]) for i in range(0, len(r), 512)])
            return float((P[..., :2] - ft[r][..., :2]).norm(dim=-1).mean())
        st = dict(kind=a.kind, steps=steps, fit_rows=len(fit), train_s=time.time() - t0, gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30,
                  head_dev_ade=ade(dv), head_dev_ade_mismatched=ade(dv, dv[G.derange(log[dv], np.random.default_rng(1))]),
                  head_fit_ade=ade(fit[:: max(1, len(fit) // 4000)])) | (dict(prof_dirs=[str(d) for d in prof_dirs()]) if qp else {})
        run.info("tokenizer head: " + json.dumps(st))
        run.summary.update(st)
        if not a.smoke:
            torch.save(net.state_dict(), OUT / "tok" / f"{a.kind}.pt")
            (OUT / "tok" / f"{a.kind}.json").write_text(json.dumps(st, indent=1))


def cmd_head(a):
    """Thin head on frozen Cinque tokens: the newest and the oldest of the 8 cached policy slots (protocol W) + the P2 ego input -> 8 poses."""
    import time
    import torch
    import torch.nn as nn
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    (OUT / "head").mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "path-req-head" + ("-smoke" if a.smoke else ""), seed=0, config=vars(a)) as run:
        datas, tabs, names, log, fut, ego = _navtrain()
        if a.smoke:
            datas, tabs = datas[:1] + list(G.PILOT[:1]), [tabs[0], tabs[2]]
            cat = lambda k: np.concatenate([t[k] for t in tabs])  # noqa: E731
            names, log, fut, ego = cat("names"), cat("log"), cat("fut"), cat("ego")
        has = ~np.isnan(fut[:, 0, 0])
        s_tk, s_tr, s_dv, s_te = (splits.load(n) for n in ("navsim/op-parity-geotok-train", "navsim/op-parity-s234-train", "navsim/op-parity-s234-dev",
                                                          "navsim/navtest"))
        for s in (s_tk, s_tr, s_dv, s_te):
            run.use_split(s)
        splits.check_disjoint(s_tk, s_tr), splits.check_disjoint(s_tk, s_dv)

        def tokens(d):                                                           # (n, 2, 32, 512) fp16: slots 7 (t0) and 0 (-1.4 s); only their pages are read
            mm = np.load(G.CR / f"{d}@warp" / "front.npy", mmap_mode="r")
            return torch.from_numpy(np.stack([np.ascontiguousarray(mm[:, 7]), np.ascontiguousarray(mm[:, 0])], 1)).to(dev)
        X = torch.cat([tokens(d) for d in datas])
        ft = torch.as_tensor(np.nan_to_num(fut), dtype=torch.float32, device=dev)
        eg = torch.as_tensor(ego, dtype=torch.float32, device=dev)
        fit, dv, ptr = (np.flatnonzero(s.mask(names) & has) for s in (s_tk, s_dv, s_tr))
        run.info(f"thin head: {len(names)} rows, fit {len(fit)}, pilot train {len(ptr)}, dev {len(dv)}")

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 64))
                self.mlp = nn.Sequential(nn.Dropout(0.1), nn.Linear(64 * 64 + 20, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 24))

            def forward(self, x, e):
                return self.mlp(torch.cat([self.proj(x.float()).flatten(1), e], 1)).view(-1, 8, 3)
        torch.manual_seed(0)
        net = Head().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        steps, wu = (30 if a.smoke else a.steps), (5 if a.smoke else 300)
        g = torch.Generator(device=dev).manual_seed(0)
        fit_r, t0, hist = torch.as_tensor(fit, device=dev), time.time(), []
        for step in range(steps):
            for q in opt.param_groups:
                q["lr"] = 1e-3 * min(1.0, (step + 1) / wu) * 0.5 * (1 + np.cos(np.pi * step / steps))
            r = fit_r[torch.randint(0, len(fit_r), (a.batch,), device=dev, generator=g)]
            loss = _imit(net(X[r], eg[r]), ft[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            hist.append(float(loss))
            if (step + 1) % 500 == 0 or step + 1 == steps:
                run.scalars({"loss/imit": float(np.mean(hist))}, step + 1)
                run.info(f"step {step + 1}: imit {np.mean(hist):.4f}; {(step + 1) / (time.time() - t0):.1f} it/s")
                hist = []
        net.eval()

        @torch.no_grad()
        def pred(x, e):
            return torch.cat([net(x[i:i + 1024], e[i:i + 1024]) for i in range(0, len(x), 1024)]).cpu().numpy()
        P = pred(X, eg)
        f64 = fut.astype(np.float64)
        st = dict(steps=steps, train_s=time.time() - t0, params=sum(p.numel() for p in net.parameters()),
                  fit=err_stats(P[fit[::8]], f64[fit[::8]]), pilot_train=err_stats(P[ptr], f64[ptr]), dev=err_stats(P[dv], f64[dv]))
        if not a.smoke:
            off = 0
            for d, t in zip(datas, tabs):
                if d in G.PILOT:
                    np.save(OUT / "head" / f"{d}.npy", P[off:off + len(t["names"])].astype(np.float32))
                off += len(t["names"])
            del X
            tab = np.load(G.CR / G.TEST / "tab.npz")
            Pt = pred(tokens(G.TEST), torch.as_tensor(tab["ego"], dtype=torch.float32, device=dev))
            np.save(OUT / "head" / f"{G.TEST}.npy", Pt.astype(np.float32))
            st["navtest"] = err_stats(Pt, tab["fut"].astype(np.float64))
            torch.save(net.state_dict(), OUT / "head" / "head.pt")
            (OUT / "head" / "head.json").write_text(json.dumps(st, indent=1))
        run.info("thin head: " + json.dumps({k: (v if not isinstance(v, dict) else {m: v[m] for m in ("ade", "lat_rms4", "lon_rms4", "c7_acc")})
                                              for k, v in st.items()}))
        run.summary.update(dev_ade=st["dev"]["ade"], pilot_train_ade=st["pilot_train"]["ade"])


def cmd_ident(a):
    """HEAD1b identity check. The memory arm's adapter has the side branch (side_in, cam / time / slot embeddings), created between the ego MLP and
    the queries / decoder, so its init draws differ from the no-memory adapter's and a memory-masked TRAINING run cannot reproduce the baseline bit
    for bit. What the code allows: the baseline's weights moved into the memory arm's model (the side branch at its own init), every row's memory
    masked -> the outputs must be the baseline's (dev rows of the pilot split and every 6th navtest row); with the memory on they must differ.
    Read at two places: the adapter's bias (fp32: the channel itself) and the plan after the fp16 policy pathway, whose rounding turns any
    last-bit difference of the bias into plan differences; the reference for that is the baseline's own plan with noise of the size of the
    measured bias difference added to its bias (PModel.bias_sub)."""
    import torch
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_parity", f"path-req-ident-{a.tag}", seed=0, config=vars(a)) as run:
        base = T.load_pmodel(a.tag, dev)
        assert base.arm == "P2" and base.mem is None, base.arm
        ck = torch.load(T.proot("runs", a.tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"]
        torch.manual_seed(0)
        m = T.PModel("P2+ge_ident").to(dev).eval()
        for k, v in ck["net"].items():
            m.net.params[k].data.copy_(v)
        r = m.adapter.load_state_dict(ck["parity"], strict=False)
        assert not r.unexpected_keys and all(k.split(".")[0] in ("side_in", "cam_emb", "t_emb", "s_emb") for k in r.missing_keys), r
        res = dict(tag=a.tag, side_branch_keys_at_init=list(r.missing_keys), sets={})
        g = torch.Generator(device=dev).manual_seed(0)
        pi = None
        for name, datas in (("dev", list(G.PILOT)), ("navtest", [G.TEST])):
            S = T.Store(datas, dev, need_side=False, frames="warp", host=True)
            pi = torch.as_tensor(S.pi, device=dev) if S.pi is not None else pi
            W = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
            rows = (np.flatnonzero(splits.load("navsim/op-parity-s234-dev").mask(S.tab["names"])) if name == "dev" else np.arange(0, S.n, 6))
            d_out, d_xy, d_on, n = 0.0, 0.0, 0.0, 0
            d_b, s_b, d_ref, m_xy, m_ref = 0.0, 0.0, 0.0, [], []
            with torch.no_grad():
                for i in range(0, len(rows), 128):
                    q = torch.as_tensor(rows[i:i + 128], device=dev)
                    fr, eg, tc = S.front[q], S.ego[q], S.tc[q]
                    mem = torch.randn((len(q), G.NTOK, G.DTOK), device=dev, generator=g)
                    o0 = base(fr, eg, tc).float()
                    o1 = m(fr, eg, tc, mem, torch.zeros(len(q), 1, dtype=torch.bool, device=dev)).float()
                    o2 = m(fr, eg, tc, mem, None).float()
                    xy = lambda o: torch.stack(T.rear(o[:, pi].view(-1, 33, 15), S.cam_x[q], W)[:2], -1)  # noqa: E731
                    d_out, d_xy = max(d_out, float((o1 - o0).abs().max())), max(d_xy, float((xy(o1) - xy(o0)).norm(dim=-1).max()))
                    d_on, n = max(d_on, float((xy(o2) - xy(o0)).norm(dim=-1).max())), n + len(q)
                    b0 = base.adapter(eg)
                    b1 = m.adapter(eg, mem[:, None, None], torch.zeros(len(q), 1, dtype=torch.bool, device=dev))
                    d_b, s_b = max(d_b, float((b1 - b0).abs().max())), max(s_b, float(b0.abs().max()))
                    base.bias_sub = (b1 - b0).pow(2).mean().sqrt() * torch.randn((G.NTOK, G.DTOK), device=dev, generator=g)
                    o3 = base(fr, eg, tc).float()
                    base.bias_sub = None
                    d_ref = max(d_ref, float((xy(o3) - xy(o0)).norm(dim=-1).max()))
                    m_xy.append((xy(o1) - xy(o0)).norm(dim=-1).mean(1)), m_ref.append((xy(o3) - xy(o0)).norm(dim=-1).mean(1))
            res["sets"][name] = dict(rows=n, max_abs_output_diff_masked=d_out, max_plan_xy_diff_masked_m=d_xy, max_plan_xy_diff_memory_on_m=d_on,
                                     mean_plan_xy_diff_masked_m=float(torch.cat(m_xy).mean()), adapter_bias_max_abs_diff_masked=d_b, adapter_bias_max_abs=s_b,
                                     reference_max_plan_xy_diff_m=d_ref, reference_mean_plan_xy_diff_m=float(torch.cat(m_ref).mean()))
            del S
        res["bit_identical"] = all(v["max_abs_output_diff_masked"] == 0.0 for v in res["sets"].values())
        res["line_1mm_met"] = all(v["max_plan_xy_diff_masked_m"] <= 1e-3 for v in res["sets"].values())
        res["bias_identical_to_fp32_rounding"] = all(v["adapter_bias_max_abs_diff_masked"] <= 1e-4 * v["adapter_bias_max_abs"] for v in res["sets"].values())
        res["plan_at_fp16_floor"] = all(v["mean_plan_xy_diff_masked_m"] <= 3 * v["reference_mean_plan_xy_diff_m"] and v["max_plan_xy_diff_masked_m"] <= 3 * v["reference_max_plan_xy_diff_m"]
                                        for v in res["sets"].values())
        res["identical"] = bool(res["line_1mm_met"] or (res["bias_identical_to_fp32_rounding"] and res["plan_at_fp16_floor"]))
        res["mask_matters"] = all(v["max_plan_xy_diff_memory_on_m"] > 1e-3 for v in res["sets"].values())
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"ident_{a.tag}.json").write_text(json.dumps(res, indent=1) + "\n")
        run.info("identity: " + json.dumps(res))
        run.summary.update({k: v for k, v in res.items() if k != "side_branch_keys_at_init"})
    raise SystemExit(0 if res["identical"] and res["mask_matters"] else 2)


# ---------------------------------------------------------------- gate / report
def done(arm, seed):
    import glob
    fs = sorted(glob.glob(str(G.D / "runs" / "op_parity" / f"train-{arm}-F-s{seed}" / "*" / "DONE")))
    return json.loads(open(fs[-1]).read()) if fs else {}


def dev_read(arm, seed, strict=False):
    """Per-seed channel-read check on the 408 dev rows: the plan must depend on the row's own field. strict (the undegraded oracle's gate):
    dev ADE <= 0.8 x the no-memory arm's and ADE with another log's field >= 1.5 x; else ADE with another log's field >= 1.05 x."""
    d = done(arm, seed)
    on, mis, off = (d.get(f"e2e_dev_ade_{k}") for k in ("on", "mismatched", "masked"))
    if on is None:
        return dict(on=None, masked=None, mismatched=None, read=False)
    return dict(on=on, masked=off, mismatched=mis, read=bool(on <= 0.8 * H0_DEV_ADE and mis >= 1.5 * on) if strict else bool(mis >= 1.05 * on))


def scored(arm, seed):
    from jevdrive.bench import tables as BT
    try:
        return BT.load("navtest", G.spec(arm, seed))[0] is not None
    except Exception:
        return False


def frame(arm, seeds):
    return G.frames(list(seeds), [arm])[0][arm]


def cmd_gate(a):
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "dev":
        res = {arm: [dev_read(arm, s, strict=True) | dict(seed=s) for s in ARMS[arm][1]] for arm in ("QF", "QFC", "QFL")}
        ok = all(r["read"] for r in res["QF"])
        out = dict(rule=f"seed reads iff dev ADE <= 0.8 x {H0_DEV_ADE} m and dev ADE with another log's field >= 1.5 x its own", arms=res,
                   reads={arm: f"{sum(r['read'] for r in v)} / {len(v)}" for arm, v in res.items()}, gate_met=ok)
    else:
        F, sets, tok, log = G.frames([0, 1], ["H0", "QX"])
        m = sets["all"]
        F["QF"], F["QF:off"] = frame("QF", (0, 1, 2)), frame("QF:off", (0, 1, 2))
        per = [G.paired(frame("QF", (s,)), F["H0"], "EPDMS", m, log) for s in (0, 1, 2)]
        g, vx, off = (G.paired(F[x], F[y], "EPDMS", m, log) for x, y in (("QF", "H0"), ("QF", "QX"), ("QF:off", "QF")))
        ok = bool(all(p["lo"] > 0 for p in per) and vx["lo"] > 0 and off["hi"] < 0)
        out = dict(rule="every QF seed - H0 EPDMS CI above 0, QF - QX CI above 0, QF memory-masked - QF CI below 0; stop (exit 3) if QF - H0 CI high < +0.5",
                   per_seed_minus_H0=per, QF_minus_H0=g, QF_minus_QX=vx, masked_minus_QF=off, gate_met=ok, oracle_below_line=bool(g["hi"] < LINES[0]))
    (OUT / f"gate_{a.stage}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    raise SystemExit(0 if out["gate_met"] and not out.get("oracle_below_line") else 3 if out.get("oracle_below_line") else 2)


def tolerance(points, line):
    """points [(level, CI low)] in order of increasing degradation -> (last level whose CI low and every earlier one's are >= line, first level
    below it); None where there is none."""
    last = None
    for lv, lo in points:
        if lo >= line:
            last = lv
        else:
            return last, lv
    return last, None


def cmd_report(a):
    import pandas as pd
    import turn_oracle as TO
    from jevdrive.bench.compat import pred_file
    from jevdrive.run import Run
    out = OUT / "report"
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "path-req-report", seed=0, config=vars(a)) as run:
        F, sets, tok, log = G.frames([0, 1], ["H0"])
        arms = [x for x in ARMS if all(scored(x, s) for s in ARMS[x][1])]
        offs = {x: [s for s in ARMS[x][1] if scored(f"{x}:off", s)] for x in arms}
        Dt = TO.Data(a.replays)
        assert (Dt.tok == tok).all()
        TURN = ("inside-cut %", "cannot-make-turn %")
        for arm in ["H0"] + arms:
            seeds = (0, 1) if arm == "H0" else ARMS[arm][1]
            if arm != "H0":
                F[arm] = frame(arm, seeds)
            fs = [Dt.one(G.spec(arm, s))[0] for s in seeds]
            for c in TURN:
                F[arm][c] = (sum(fs) / len(fs))[c].to_numpy()
        for arm in arms:
            if offs[arm]:
                F[f"{arm}:off"], F[f"{arm}@off-seeds"] = frame(f"{arm}:off", offs[arm]), frame(arm, offs[arm])
        BK = ("< 5 deg", "5-20 deg", "20-45 deg", "> 45 deg")
        TABLE = ([("EPDMS", s) for s in ("all", *BK)] + [("DAC fail %", s) for s in ("all", "> 20 deg", "> 45 deg")] +
                 [("inside-cut %", "> 45 deg"), ("cannot-make-turn %", "> 45 deg"), ("NC+TTC fail %", "all")] +
                 [(c, "all") for c in ("NC", "DAC", "EP", "TTC", "LK", "DDC", "TLC", "HC") if c in F["H0"]])
        rows = []
        for col, sn in TABLE:
            m = sets[sn]
            rows.append(dict(metric=col, stratum=sn, n=int(m.sum()), arm="H0", value=float(F["H0"][col].to_numpy()[m].mean())))
            for arm in arms:
                p = G.paired(F[arm], F["H0"], col, m, log)
                r = dict(metric=col, stratum=sn, n=int(m.sum()), arm=arm, value=p["arm"], diff_vs_H0=p["diff"], lo_vs_H0=p["lo"], hi_vs_H0=p["hi"])
                if "QX" in arms and arm != "QX":
                    q = G.paired(F[arm], F["QX"], col, m, log)
                    r |= dict(diff_vs_QX=q["diff"], lo_vs_QX=q["lo"], hi_vs_QX=q["hi"])
                if offs[arm] and col not in TURN:
                    q = G.paired(F[f"{arm}:off"], F[f"{arm}@off-seeds"], col, m, log)
                    r |= dict(diff_masked=q["diff"], lo_masked=q["lo"], hi_masked=q["hi"])
                rows.append(r)
        T = pd.DataFrame(rows)
        T.to_csv(out / "arms.csv", index=False)
        cell = lambda arm, col, sn, k: (lambda q: [float(q[f"diff_{k}"]), float(q[f"lo_{k}"]), float(q[f"hi_{k}"])] if f"diff_{k}" in q and pd.notna(q[f"diff_{k}"])  # noqa: E731
                                        else None)(T[(T.arm == arm) & (T.metric == col) & (T.stratum == sn)].iloc[0])
        # ---- reference predictors (non-privileged) and the realised errors of the noise arms, navtest
        tab = np.load(G.CR / G.TEST / "tab.npz")
        fut = tab["fut"].astype(np.float64)

        def plan_stats(specs):
            st = []
            for sp in specs:
                z = np.load(pred_file(sp))
                pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
                st.append(err_stats(z["poses"][[pos[t] for t in tok]].astype(np.float64), fut))
            return {k: (float(np.mean([s[k] for s in st])) if not isinstance(st[0][k], list) else np.mean([s[k] for s in st], 0).tolist()) for k in st[0]}
        ref = {"H0 (the pilot policy's own plan, GH0-F-s0/s1)": plan_stats(["GH0-F-s0", "GH0-F-s1"]),
               "SH30 (full-data policy's own plan, SH30-F-s0/s1)": plan_stats(["SH30-F-s0", "SH30-F-s1"])}
        if (OUT / "head" / "head.json").exists():
            ref["thin head on frozen Cinque tokens (held-out logs)"] = err_stats(np.load(OUT / "head" / f"{G.TEST}.npy").astype(np.float64), fut)
        for arm in arms:
            if all(scored(arm, s) for s in ARMS[arm][1]) and ARMS[arm][0] not in ("qx",):
                ref[f"{arm} plan (reads the field)"] = plan_stats([G.spec(arm, s) for s in ARMS[arm][1]])
        deg = {arm: degraded_stats(ARMS[arm][0], fut) for arm in arms if parse(ARMS[arm][0])[0] in ("n", "l")}
        (out / "ref.json").write_text(json.dumps(dict(predictors=ref, degraded_inputs=deg, head=json.loads((OUT / "head" / "head.json").read_text())
                                                      if (OUT / "head" / "head.json").exists() else None), indent=1))
        # ---- one row per arm: the curve
        H0e = float(F["H0"]["EPDMS"].mean())
        cur = []
        for arm in arms:
            kind, seeds, start = ARMS[arm]
            g = cell(arm, "EPDMS", "all", "vs_H0")
            dr = [dev_read(arm, s) for s in seeds]
            tk = OUT / "tok" / f"{kind}.json"
            r = dict(arm=arm, kind=kind, seeds=len(seeds), start=start, EPDMS=H0e + g[0], gain=g[0], lo=g[1], hi=g[2],
                     per_seed_gain=[float(frame(arm, (s,))["EPDMS"].mean() - H0e) for s in seeds],
                     vs_QX=cell(arm, "EPDMS", "all", "vs_QX") if arm != "QX" and "QX" in arms else None, masked=cell(arm, "EPDMS", "all", "masked"),
                     masked_seeds=offs[arm], dev_ade_on=[d["on"] for d in dr], dev_ade_masked=[d["masked"] for d in dr],
                     dev_ade_mismatched=[d["mismatched"] for d in dr], dev_read=[d["read"] for d in dr],
                     tok_head_dev_ade=json.loads(tk.read_text())["head_dev_ade"] if tk.exists() else None,
                     tok_head_dev_ade_mismatched=json.loads(tk.read_text())["head_dev_ade_mismatched"] if tk.exists() else None)
            vx = r["vs_QX"]
            r["read"] = bool(all(r["dev_read"]) or (vx is not None and vx[1] > 0))
            for nm, (col, sn) in {"gt45": ("EPDMS", "> 45 deg"), "t2045": ("EPDMS", "20-45 deg"), "t520": ("EPDMS", "5-20 deg"), "lt5": ("EPDMS", "< 5 deg"),
                                  "nc_ttc_fail": ("NC+TTC fail %", "all"), "cannot_turn_gt45": ("cannot-make-turn %", "> 45 deg"),
                                  "inside_cut_gt45": ("inside-cut %", "> 45 deg"), "dac_fail_gt20": ("DAC fail %", "> 20 deg"), "NC": ("NC", "all"),
                                  "DAC": ("DAC", "all"), "EP": ("EP", "all"), "TTC": ("TTC", "all")}.items():
                r[nm] = cell(arm, col, sn, "vs_H0")
            if arm in deg:
                r |= {f"input_{k}": deg[arm][k] for k in ("ade", "lat_rms4", "lon_rms4")}
            cur.append(r)
        C = {r["arm"]: r for r in cur}
        (out / "curve.json").write_text(json.dumps(cur, indent=1, default=float))
        # ---- the registered reading
        vd = dict(probe=PROBE, lines=list(LINES), H0_EPDMS=H0e, axes={}, predictors={k: {m: v[m] for m in ("ade", "ade_1s", "ade_2s", "fde", "lat_rms4",
                  "lat_rob4", "lon_rms4", "lon_rob4", "head_rms4_deg", "head_rob4_deg", "c7_acc")} for k, v in ref.items()}, replay_checks=Dt.check)
        for ax, (unit, pts) in AXES.items():
            pts = [(arm, lv) for arm, lv in pts if arm in C]
            e = dict(unit=unit, points=[dict(arm=arm, level=lv, gain=C[arm]["gain"], lo=C[arm]["lo"], hi=C[arm]["hi"], read=C[arm]["read"]) for arm, lv in pts])
            if ax != "content":
                for line in LINES:
                    e[f"tolerance_{line}"] = dict(zip(("last_level_at_or_above", "first_level_below"), tolerance([(lv, C[arm]["lo"]) for arm, lv in pts], line)))
            vd["axes"][ax] = e
        vd["reliability"] = {arm: dict(start=ARMS[arm][2], dev=[dev_read(arm, s, strict=True) for s in ARMS[arm][1]]) for arm in ("QF", "QFC", "QFL")}
        for k in ("dev", "navtest"):
            if (OUT / f"gate_{k}.json").exists():
                vd[f"gate_{k}"] = json.loads((OUT / f"gate_{k}.json").read_text())
        nonpriv = {k: v for k, v in ref.items() if k.startswith(("H0", "SH30", "thin head"))}
        for ax, st in (("lat", "lat_rms4"), ("lon", "lon_rms4")):
            if ax not in vd["axes"]:
                continue
            for line in LINES:
                t = vd["axes"][ax][f"tolerance_{line}"]
                last, first = t["last_level_at_or_above"], t["first_level_below"]
                judge = lambda v: ("oracle itself below the line" if last is None else "within" if v <= last else  # noqa: E731
                                   "outside" if first is not None and v >= first else "between tested levels")
                t["predictors"] = {k: dict(rms=v[st], robust=v[st.replace("rms", "rob")], by_rms=judge(v[st]), by_robust=judge(v[st.replace("rms", "rob")]))
                                   for k, v in nonpriv.items()}
        (out / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
        # ---- tables
        f3 = lambda v: "-" if v is None else f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"  # noqa: E731
        fl = lambda v: " / ".join("-" if x is None else f"{x:.3f}" for x in v)  # noqa: E731
        L = [f"**{PROBE}.**\n", f"navtest 12 146 tokens, per-token seed means x 100; differences to H0 = GH0-F-s0/s1 (no memory, {H0e:.2f}) with 95% CI "
             f"(log-cluster paired bootstrap, B {G.NB}). Pilot recipe of decisions 197 / 200.\n",
             "| arm | input | seeds | EPDMS - H0 | per seed | - QX (shuffled) | memory masked - on (seeds) | dev ADE on / masked / mismatched, per seed (m) | read |",
             "|:--|:--|--:|:--|:--|:--|:--|:--|:--|"]
        for r in cur:
            L.append(f"| {r['arm']} | {r['kind']} ({r['start']}) | {r['seeds']} | {f3([r['gain'], r['lo'], r['hi']])} | " +
                     " / ".join(f"{x:+.2f}" for x in r["per_seed_gain"]) + f" | {f3(r['vs_QX'])} | {f3(r['masked'])} ({r['masked_seeds']}) | " +
                     " ; ".join(f"{fl(x)}" for x in zip(r["dev_ade_on"], r["dev_ade_masked"], r["dev_ade_mismatched"])) + f" | {'yes' if r['read'] else 'NO'} |")
        L += ["", f"**Where the gain is (difference to H0; {PROBE.split(':')[0]}).**\n",
              "| arm | < 5 deg | 5-20 deg | 20-45 deg | > 45 deg | NC + TTC fail % | cannot-make-turn %, > 45 | inside-cut %, > 45 | DAC fail %, > 20 | NC | DAC | EP | TTC |",
              "|:--|" + ":--|" * 12]
        for r in cur:
            L.append(f"| {r['arm']} | " + " | ".join(f3(r[k]) for k in ("lt5", "t520", "t2045", "gt45", "nc_ttc_fail", "cannot_turn_gt45", "inside_cut_gt45",
                                                                         "dac_fail_gt20", "NC", "DAC", "EP", "TTC")) + " |")
        L += ["", "**Reference predictors on navtest (error to the logged future) and the plans of the arms.**\n",
              "| predictor | ADE | ADE <= 1 s | ADE <= 2 s | FDE | cross-track 4 s rms / robust | along-track 4 s rms / robust | heading 4 s rms / robust (deg) | exit-class acc |",
              "|:--|--:|--:|--:|--:|:--|:--|:--|--:|"]
        for k, v in ref.items():
            L.append(f"| {k} | {v['ade']:.3f} | {v['ade_1s']:.3f} | {v['ade_2s']:.3f} | {v['fde']:.3f} | {v['lat_rms4']:.2f} / {v['lat_rob4']:.2f} | "
                     f"{v['lon_rms4']:.2f} / {v['lon_rob4']:.2f} | {v['head_rms4_deg']:.1f} / {v['head_rob4_deg']:.1f} | {v['c7_acc']:.3f} |")
        L.append("\n```json\n" + json.dumps({k: v for k, v in vd.items() if k != "replay_checks"}, indent=1, default=float) + "\n```")
        (out / "tables.md").write_text("\n".join(L) + "\n")
        run.info("\n".join(L))
        run.summary.update(arms=arms, out=str(out))


def cmd_fig(a):
    """The curve (run on the Mac from the committed report files): gain over the no-memory baseline per degradation axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as PS
    PS.apply()
    vd = json.loads((RES / "verdict.json").read_text())
    C = {r["arm"]: r for r in json.loads((RES / "curve.json").read_text())}
    fig, axs = plt.subplots(1, 4, figsize=(PS.DOUBLE_COLUMN_IN, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 0.75, 1.55]))
    col = PS.PALETTE
    pk = [k for k in vd["predictors"] if k.startswith(("H0", "thin head"))]
    pcol = {pk[0]: PS.BASELINE, **({pk[1]: col["vermillion"]} if len(pk) > 1 else {})}
    for ax, key, lab, st in ((axs[0], "lat", "cross-track error std at 4 s (m)", "lat"), (axs[1], "lon", "along-track error std at 4 s (m)", "lon"),
                             (axs[2], "horizon", "horizon kept (s)", None)):
        e = vd["axes"].get(key)
        if not e:
            continue
        x = [p["level"] for p in e["points"]]
        y, lo, hi = (np.array([p[k] for p in e["points"]]) for k in ("gain", "lo", "hi"))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=col["blue"], marker="o", ms=3, capsize=2, lw=1.05)
        for p in e["points"]:
            if not p["read"]:
                ax.plot([p["level"]], [p["gain"]], marker="o", ms=5, mfc="none", mec=col["black"], ls="none")
        if st:
            for k in pk:
                ax.axvline(vd["predictors"][k][f"{st}_rms4"], color=pcol[k], lw=0.9, ls="-")
                ax.axvline(vd["predictors"][k][f"{st}_rob4"], color=pcol[k], lw=0.9, ls=":")
        ax.set_xlabel(lab)
    e = vd["axes"]["content"]["points"] + [dict(arm=x, level=n, gain=C[x]["gain"], lo=C[x]["lo"], hi=C[x]["hi"], read=C[x]["read"])
                                           for x, n in (("QH", "thin-head path\n(not privileged)"), ("QX", "shuffled")) if x in C]
    xs = np.arange(len(e))
    y, lo, hi = (np.array([p[k] for p in e]) for k in ("gain", "lo", "hi"))
    cc = [col["vermillion"] if p["arm"] == "QH" else PS.BASELINE if p["arm"] == "QX" else col["blue"] for p in e]
    axs[3].bar(xs, y, color=cc, width=0.62)
    axs[3].errorbar(xs, y, yerr=[y - lo, hi - y], color=col["black"], ls="none", capsize=2, lw=0.7)
    short = {"QF": "path +\ntiming", "QS": "shape", "QV": "speed", "QCH": "heading\nat 4 s", "QC7": "exit\nclass", "QH": "thin\nhead", "QX": "shuf-\nfled"}
    axs[3].set_xticks(xs, [short[p["arm"]] for p in e], fontsize=6.5)
    PS.bars(axs[3])
    for i, ax in enumerate(axs):
        for line in vd["lines"]:
            ax.axhline(line, color=col["orange"], lw=0.7, ls="--")
        PS.zero_line(ax)
        PS.panel(ax, "abcd"[i])
    axs[0].set_ylabel("navtest EPDMS gain over no memory")
    fig.tight_layout(pad=0.4, w_pad=0.6)
    dst = _R / "experiments/op_parity/figs/path_req"
    dst.mkdir(parents=True, exist_ok=True)
    print(PS.save(fig, dst / "curve"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    p = sp.add_parser("tok")
    p.add_argument("--kind", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--smoke", action="store_true")
    p = sp.add_parser("head")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--smoke", action="store_true")
    p = sp.add_parser("gate")
    p.add_argument("--stage", required=True, choices=["dev", "navtest"])
    p = sp.add_parser("report")
    p.add_argument("--replays", nargs="+", default=["geo_s0", "geo_e2e", "path_req"])
    sp.add_parser("fig")
    p = sp.add_parser("ident")
    p.add_argument("--tag", default="GH0-F-s0")
    a = ap.parse_args()
    {"selftest": cmd_selftest, "tok": cmd_tok, "head": cmd_head, "gate": cmd_gate, "report": cmd_report, "fig": cmd_fig, "ident": cmd_ident}[a.cmd](a)
