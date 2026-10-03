"""Route overlays drawn into openpilot's packed model frames (road + wide), consistent across history frames.

Geometry lives in the t0 rear-axle frame (x fwd, y left, z up, road plane z = 0): the sample's approach path and its
branches (img_geom_nav.py). A step frame with ego pose (x, y, yaw) at its source time is drawn by moving every 3-D
primitive into that pose's frame, clipping at a near plane and projecting through the model frame's virtual camera
(origin `cam` on the vehicle, NAVSIM ego axes, OP_K road / wide), so the paint stays fixed on the road while the ego moves.
Rasterised at 4x supersampling, alpha-blended in full-range YCbCr (chroma from the 2x2 mean of the coverage).

Families (`FAMILIES`): what the image says about the commanded branch `cmd`.
  band        green semi-transparent ribbon (1.8 m) along the approach and the commanded branch
  lines       solid white lane lines on both boundaries of the commanded branch (connector + next lanes)
  arrow_road  white painted arrow (8 m) on the ground along the commanded path, 8-16 m ahead of the t0 ego
  sign        blue sign (1.6 m) with a white arrow, right side 12 m past the branching point, facing the t0 ego
  cones       traffic cones across every other branch where it is >= 4 m from the commanded path
  barrier     red / white striped boards (1 m high) across every other branch
  wall        grey 3 m wall across every other branch
  fill_grey   every other branch's road surface (from where it is >= 4 m from the commanded path, 45 m) painted flat grey
  fill_grass  same, grass texture
  band_all    control: the band on every branch (paint without route information)
  combo       added after the first readout (not pre-registered): band + fill_grass + barrier together (the bluntest)

Sky families (Q3 course change, plans/2026-10-04-img-cmd-ft2-prereg.md addendum; no occlusion of the road scene, nothing taken
from the map at test time): a screen-fixed magenta arrow with a black outline above the horizon of both views (HUD, not
projected), shape = command (left / straight / right), size from the distance to the junction (full at <= 10 m, 0.55 at >= 60 m;
every history frame uses its own distance = t0 distance + its distance behind t0).
  sky         the arrow for command `cmd`
  sky_disc    control: a magenta disc at the same place (no direction)
  sky_wrong   control: the arrow of an exit class the junction does not have (junctions with a missing class only)
"""
import numpy as np

W, H, SS = 512, 256, 4
OP_K = {"road": np.array([[910.0, 0, 256.0], [0, 910.0, 47.6], [0, 0, 1]]),
        "wide": np.array([[455.0, 0, 256.0], [0, 455.0, 0.5 * (256 + 47.6)], [0, 0, 1]])}
NEAR = 0.5
DIVERGE_M = 4.0     # blocks / fills start where the other branch centreline is this far from the commanded path
MAX_EGO_LANE_M = 1.5  # samples whose t0 ego is farther from the approach centreline are dropped (bad lane match)
FAMILIES = ("band", "lines", "arrow_road", "sign", "cones", "barrier", "wall", "fill_grey", "fill_grass", "band_all", "combo")
ROUTE_FREE = ("band_all",)          # carry no route information (one run per frame, not per command)
SKY_FAMILIES = ("sky",)
SKY_FREE = ("sky_disc", "sky_wrong")  # sky controls: one run per frame
MAGENTA, BLACK = (255, 0, 255), (0, 0, 0)
SKY_BOX = {"road": (2, 38), "wide": (6, 54)}     # (top row, full height in px) of the arrow per view; horizon rows 47.6 / 151.8
SKY_SHAPES = {  # x right, y up, height 1, centred on x = 0
    "straight": [(-0.15, 0), (0.15, 0), (0.15, 0.55), (0.45, 0.55), (0, 1), (-0.45, 0.55), (-0.15, 0.55)],
    "right": [(-0.45, 0), (-0.15, 0), (-0.15, 0.45), (0.15, 0.45), (0.15, 0.2), (0.6, 0.6), (0.15, 1.0), (0.15, 0.75), (-0.45, 0.75)]}
SKY_SHAPES["left"] = [(-x, y) for x, y in SKY_SHAPES["right"]]
SKY_SHAPES["disc"] = [(0.45 * np.cos(a), 0.5 + 0.45 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]


def sky_scale(d):
    d = 60.0 if d is None or not np.isfinite(d) else float(d)
    return float(np.interp(d, [10.0, 60.0], [1.0, 0.55]))


def sky_missing(sample):
    have = {b["cls"] for b in sample.get("branches", [])}
    return [c for c in ("left", "straight", "right") if c not in have]

GREEN, WHITE, BLUE, ORANGE, RED, CONCRETE, GREY, GRASS = (
    (0, 200, 83), (235, 235, 235), (20, 70, 200), (255, 110, 0), (210, 20, 20), (150, 150, 145), (105, 105, 105), (70, 120, 40))


def ycc(rgb):
    r, g, b = rgb
    return np.array([0.299 * r + 0.587 * g + 0.114 * b, 128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                     128 + 0.5 * r - 0.418688 * g - 0.081312 * b])


# ---------------------------------------------------------------- paths

def resample(p, step=0.5):
    p = np.asarray(p, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-3]
    p = p[keep]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < step:
        return p
    u = np.arange(0, s[-1], step)
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1)


def chain(segs, key="c"):
    return resample(np.concatenate([s[key] for s in segs]))


def arc(p):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def normals(p):
    d = np.gradient(p, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return np.stack([-d[:, 1], d[:, 0]], 1)          # left normal


def at(p, s):
    a = arc(p)
    return np.stack([np.interp(s, a, p[:, 0]), np.interp(s, a, p[:, 1])], -1)


def ribbon(p, half, z=0.02, d0=0.0):
    """Ground quads along polyline p, lateral offset d0, width 2*half."""
    n = normals(p)
    L, R = p + n * (d0 + half), p + n * (d0 - half)
    return [np.c_[np.stack([L[i], L[i + 1], R[i + 1], R[i]]), np.full(4, z)] for i in range(len(p) - 1)]


def strip(left, right, z=0.03):
    """Ground quads between two boundary polylines, matched by relative arc length."""
    left, right = resample(left), resample(right)
    n = max(len(left), len(right))
    u = np.linspace(0, 1, n)
    li = np.stack([np.interp(u * arc(left)[-1], arc(left), left[:, k]) for k in range(2)], 1)
    ri = np.stack([np.interp(u * arc(right)[-1], arc(right), right[:, k]) for k in range(2)], 1)
    return [np.c_[np.stack([li[i], li[i + 1], ri[i + 1], ri[i]]), np.full(4, z)] for i in range(n - 1)]


def cross_line(branch, s):
    """Left / right boundary points across a branch at arc s of its centreline (nearest boundary points)."""
    c = chain(branch["segs"])
    m = at(c, s)
    ends = []
    for key in ("l", "r"):
        b = chain(branch["segs"], key)
        ends.append(b[np.argmin(np.linalg.norm(b - m, axis=1))])
    return m, ends[0], ends[1]


def diverge_s(b, cmd_c, thr=DIVERGE_M):
    """Arc length along branch b's centreline where it is first >= thr from the commanded path (None: never)."""
    c = chain(b["segs"])
    d = np.linalg.norm(c[:, None] - cmd_c[None], axis=-1).min(1)
    k = np.flatnonzero(d >= thr)
    return None if not len(k) else float(arc(c)[k[0]])


def vertical(a, b, z0, z1):
    return np.array([[*a, z0], [*b, z0], [*b, z1], [*a, z1]], float)


def cmd_path(sample, cls):
    """Approach + the first branch of class cls (centre / left / right polylines) and its branch dict."""
    br = next(b for b in sample["branches"] if b["cls"] == cls)
    return br


def full_centre(sample, br):
    return chain(sample["approach"] + br["segs"])


def valid(sample):
    """Lane match check: the t0 ego is on the approach centreline (within MAX_EGO_LANE_M)."""
    return float(np.linalg.norm(chain(sample["approach"]), axis=1).min()) <= MAX_EGO_LANE_M


def ego_s(p):
    """Arc length of the point of p nearest to the t0 ego (origin)."""
    return arc(p)[np.argmin(np.linalg.norm(p, axis=1))]


# ---------------------------------------------------------------- primitives per family

ARROW = {  # (s, d) outline in metres, s along the path from the tail, d left; straight arrow, bent by the path itself
    "body": [(0, 0.25), (5.0, 0.25), (5.0, 0.8), (8.0, 0.0), (5.0, -0.8), (5.0, -0.25), (0, -0.25)]}


def path_shape(p, s0, pts, z=0.03):
    """A 2-D shape in (s, d) path coordinates mapped onto the ground along polyline p, starting at arc s0."""
    a, n = arc(p), normals(p)
    out = []
    for s, d in pts:
        q = s0 + s
        c = at(p, q)
        k = min(np.searchsorted(a, q), len(p) - 1)
        out.append([*(c + d * n[k]), z])
    return np.array(out)


def sign_quads(sample, br, cls):
    """Blue board (1.6 m) + white arrow facing the t0 ego, on the right 12 m beyond the branching point (far corner)."""
    D = max(float(np.nan_to_num(sample["dist"], nan=8.0)), 0.0) + 12.0
    t, nl = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    base = np.array([D, -4.5])
    half = 0.8 * nl
    z0, z1 = 1.6, 3.2
    board = np.array([[*(base + half), z0], [*(base - half), z0], [*(base - half), z1], [*(base + half), z1]])
    # arrow in board coordinates (u right, v up), unit 1.3 m
    if cls == "straight":
        a2 = [(-0.1, -0.45), (0.1, -0.45), (0.1, 0.1), (0.3, 0.1), (0.0, 0.5), (-0.3, 0.1), (-0.1, 0.1)]
    else:
        a2 = [(-0.1, -0.45), (0.1, -0.45), (0.1, 0.1), (0.2, 0.1), (0.2, -0.1), (0.5, 0.2), (0.2, 0.5), (0.2, 0.3), (-0.1, 0.3)]
        if cls == "left":
            a2 = [(-u, v) for u, v in a2]
    zc = 0.5 * (z0 + z1)
    right = -nl                                        # ego's right as seen when facing the board
    arrow = np.array([[*(base + u * 1.6 * right), zc + v * 1.6] for u, v in a2])
    arrow[:, :2] -= t * 0.02                           # in front of the board
    return board, arrow


def primitives(sample, fam, cmd):
    """Layers [(polys, rgb, alpha, noise)] in the t0 frame, painter's order (ground first)."""
    if fam == "none":
        return []
    if fam in SKY_FAMILIES + SKY_FREE:
        shape = "disc" if fam == "sky_disc" else sky_missing(sample)[0] if fam == "sky_wrong" else (cmd or "straight")
        d0 = sample.get("dist", np.nan) if sample.get("kind", "junction") == "junction" else np.nan
        return [("SKY", shape, d0)]
    if fam == "combo":
        return primitives(sample, "band", cmd) + primitives(sample, "fill_grass", cmd) + primitives(sample, "barrier", cmd)
    L = []
    if fam in ("band", "band_all"):
        brs = sample["branches"] if fam == "band_all" else [cmd_path(sample, cmd)]
        for br in brs:
            p = full_centre(sample, br)
            s = ego_s(p)
            p = p[arc(p) >= s - 15]
            p = p[arc(p) <= 15 + 70]
            L.append((ribbon(p, 0.9), GREEN, 0.55, 0))
        return L
    br = cmd_path(sample, cmd)
    others = [b for b in sample["branches"] if b["cls"] != cmd]
    if fam == "lines":
        polys = []
        for key in ("l", "r"):
            polys += ribbon(chain(br["segs"], key), 0.1)
        return [(polys, WHITE, 0.9, 0)]
    if fam == "arrow_road":
        p = full_centre(sample, br)
        s0 = ego_s(p) + 8.0
        # the arrow follows the path (a bent arrow at a turn) and its head points along the commanded branch
        return [([path_shape(p, s0, ARROW["body"])], WHITE, 0.9, 0)]
    if fam == "sign":
        board, arrow = sign_quads(sample, br, cmd)
        return [([board], BLUE, 1.0, 0), ([arrow], WHITE, 1.0, 0)]
    cmd_c = full_centre(sample, br)
    if fam in ("fill_grey", "fill_grass"):
        polys = []
        for b in others:
            s0 = diverge_s(b, cmd_c)
            if s0 is None:
                continue
            m = at(chain(b["segs"]), s0)
            l, r = chain(b["segs"], "l"), chain(b["segs"], "r")
            l = l[np.argmin(np.linalg.norm(l - m, axis=1)):][:90]
            r = r[np.argmin(np.linalg.norm(r - m, axis=1)):][:90]
            if len(l) >= 2 and len(r) >= 2:
                polys += strip(l, r)
        return [(polys, GREY if fam == "fill_grey" else GRASS, 1.0, 0 if fam == "fill_grey" else 22)]
    polys_a, polys_b = [], []
    for b in others:
        s0 = diverge_s(b, cmd_c)
        if s0 is None:
            continue
        m, lp, rp = cross_line(b, s0 + 0.5)
        u = (lp - rp) / max(np.linalg.norm(lp - rp), 1e-9)
        lp, rp = lp + 0.3 * u, rp - 0.3 * u
        wdt = np.linalg.norm(lp - rp)
        if fam == "wall":
            polys_a.append(vertical(rp, lp, 0.0, 3.0))
        elif fam == "barrier":
            n = max(int(wdt / 0.6), 2)
            for i in range(n):
                a, c = rp + (lp - rp) * i / n, rp + (lp - rp) * (i + 1) / n
                (polys_a if i % 2 == 0 else polys_b).append(vertical(a, c, 0.25, 1.25))
        elif fam == "cones":
            n = max(int(wdt / 0.9), 2)
            for i in range(n + 1):
                c = rp + (lp - rp) * i / n
                polys_a.append(np.array([[*(c - 0.2 * u), 0], [*(c + 0.2 * u), 0], [*c, 0.75]]))
                polys_b.append(np.array([[*(c - 0.11 * u), 0.33], [*(c + 0.11 * u), 0.33], [*(c + 0.07 * u), 0.46], [*(c - 0.07 * u), 0.46]]))
    if fam == "wall":
        return [(polys_a, CONCRETE, 1.0, 10)]
    if fam == "barrier":
        return [(polys_a, RED, 1.0, 0), (polys_b, WHITE, 1.0, 0)]
    return [(polys_a, ORANGE, 1.0, 0), (polys_b, WHITE, 1.0, 0)]


# ---------------------------------------------------------------- rendering

_NOISE = np.random.default_rng(0).normal(0, 1, (H, W)).astype(np.float32)


def project(poly, pose, cam, K):
    """t0-frame 3-D polygon -> model-frame pixel polygon (or None), near-plane clipped in the camera frame."""
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    d = poly[:, :2] - (x, y)
    v = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], poly[:, 2]] - cam      # vehicle frame at the camera
    # Sutherland-Hodgman against x >= NEAR
    out, n = [], len(v)
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        ia, ib = a[0] >= NEAR, b[0] >= NEAR
        if ia:
            out.append(a)
        if ia != ib:
            t = (NEAR - a[0]) / (b[0] - a[0])
            out.append(a + t * (b - a))
    if len(out) < 3:
        return None
    v = np.array(out)
    u = K[0, 0] * (-v[:, 1]) / v[:, 0] + K[0, 2]
    w = K[1, 1] * (-v[:, 2]) / v[:, 0] + K[1, 2]
    return np.clip(np.stack([u, w], 1), -4 * W, 5 * W)


def draw(packed, layers, pose, cam):
    """packed (2, 6, 128, 256) uint8 -> new packed frame with the layers drawn (both views)."""
    import cv2
    if not layers:
        return packed
    from jevdrive import op_interp as I
    if isinstance(layers[0][0], str) and layers[0][0] == "SKY":
        return draw_sky(packed, layers[0][1], layers[0][2], pose)
    out = np.empty_like(packed)
    for k, view in enumerate(("road", "wide")):
        Y, U, V = (z.astype(np.float32) for z in I.unpack(packed[k]))
        K = OP_K[view]
        for polys, rgb, alpha, noise in layers:
            m = np.zeros((H * SS, W * SS), np.uint8)
            pts = [project(p, pose, cam, K) for p in polys]
            pts = [np.round((q + 0.5) * SS - 0.5).astype(np.int32) for q in pts if q is not None]
            if not pts:
                continue
            cv2.fillPoly(m, pts, 255)
            a = cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA) * alpha
            if not a.any():
                continue
            cy = ycc(rgb)
            Yc = cy[0] + noise * _NOISE if noise else cy[0]
            Y = Y * (1 - a) + Yc * a
            ah = a.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
            U = U * (1 - ah) + cy[1] * ah
            V = V * (1 - ah) + cy[2] * ah
        q = lambda z: np.clip(np.rint(z), 0, 255).astype(np.uint8)  # noqa: E731
        out[k] = I.pack(q(Y), q(U), q(V))
    return out


def draw_sky(packed, shape, d0, pose):
    """The sky arrow / disc (screen-fixed) on both views; distance of this frame = d0 + its distance behind the t0 pose."""
    import cv2
    from jevdrive import op_interp as I
    d = 60.0 if d0 is None or not np.isfinite(d0) else float(d0) + max(0.0, -float(pose[0]))
    sc = sky_scale(d)
    out = np.empty_like(packed)
    for k, view in enumerate(("road", "wide")):
        Y, U, V = (z.astype(np.float32) for z in I.unpack(packed[k]))
        top, h = SKY_BOX[view]
        hh = h * sc
        pts = np.array([[W / 2 + x * hh, top + (h - hh) / 2 + (1 - y) * hh] for x, y in SKY_SHAPES[shape]])
        q = np.round((pts + 0.5) * SS - 0.5).astype(np.int32)
        for rgb, thick in ((BLACK, 2.0), (MAGENTA, 0)):
            m = np.zeros((H * SS, W * SS), np.uint8)
            cv2.fillPoly(m, [q], 255)
            if thick:
                cv2.polylines(m, [q], True, 255, int(thick * SS))
            a = cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)
            cy = ycc(rgb)
            Y = Y * (1 - a) + cy[0] * a
            ah = a.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
            U = U * (1 - ah) + cy[1] * ah
            V = V * (1 - ah) + cy[2] * ah
        r = lambda z: np.clip(np.rint(z), 0, 255).astype(np.uint8)  # noqa: E731
        out[k] = I.pack(r(Y), r(U), r(V))
    return out
