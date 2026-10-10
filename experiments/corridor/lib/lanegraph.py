"""Lane graph of a nuPlan map as plain arrays + the lane-sequence matching CORR0 uses (experiments/corridor, decision 240).

Built straight from the vector layers (lanes, lane connectors, baseline paths, scaled connector polygons), so matching thousands of
poses does not go through the per-object map API. Everything here reads the map and the logged future: privileged, analysis only.

  MapG(loc)                 nodes = lanes + lane connectors: polygon, dense baseline (STEP m), successors, predecessors, group
                            (roadblock / roadblock connector), lateral neighbours (same group)
  g.candidates(xy, yaw)     nodes containing the point with |heading difference| < H_MAX (none: nearest within NEAR m), with their cost
  g.viterbi(P)              lowest-cost node per pose; transitions: same node, a successor up to HOPS hops (skips filled in), a lateral
                            neighbour of any of those (lane change, + LC_PEN)
  g.sequence(nodes)         the matched node list with skipped successors filled in -> (unique sequence, index of the last lane change)
  g.centreline(seq, xy)     dense centreline of the final connected run of a sequence, extended upstream to cover xy and downstream
                            along the straightest successor to `ahead` m -> Line
  g.paths_ahead(node, s, d) every downstream node path from station s of `node` that reaches d m (or a dead end)
"""
import math
import os

import numpy as np

STEP = 0.25                 # m, baseline resampling
H_MAX = math.radians(60)    # candidate heading gate
NEAR = 3.0                  # m, nearest-node fallback when no polygon contains the pose
HOPS = 3                    # successor hops per 0.5 s step (two nodes may be skipped)
LC_PEN = 0.5                # added cost of a lateral (lane-change) transition
HALF_W = 1.1485             # m, half width of the nuPlan ego (Pacifica)
_G = {}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Line:
    """Dense polyline with arc length, tangent heading and (optionally) the half lane width at each sample."""

    def __init__(self, xy, hw=None):
        xy = np.asarray(xy, np.float64)
        seg = np.hypot(*np.diff(xy, axis=0).T)
        keep = np.r_[True, seg > 1e-6]
        self.xy = xy[keep]
        self.hw = None if hw is None else np.asarray(hw, np.float64)[keep]
        d = np.diff(self.xy, axis=0)
        self.S = np.r_[0, np.cumsum(np.hypot(*d.T))]
        h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.h = np.r_[h, h[-1]] if len(h) else np.zeros(1)
        self.L = float(self.S[-1])

    def at(self, s):
        """(x, y, yaw) at arc lengths s; straight beyond either end."""
        s = np.atleast_1d(np.asarray(s, np.float64))
        si = np.clip(s, 0, self.L)
        out = np.stack([np.interp(si, self.S, self.xy[:, 0]), np.interp(si, self.S, self.xy[:, 1]), np.interp(si, self.S, self.h)], 1)
        for m, h in ((s > self.L, self.h[-1]), (s < 0, self.h[0])):
            if m.any():
                out[m, 0] += (s[m] - si[m]) * np.cos(h)
                out[m, 1] += (s[m] - si[m]) * np.sin(h)
        return out

    def frenet(self, pts):
        """Station and signed lateral offset (left positive) of points (m, 2): nearest point on the segments."""
        pts = np.atleast_2d(pts)
        a, ab = self.xy[:-1], np.diff(self.xy, axis=0)
        ap = pts[:, None] - a[None]
        t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        d = ap - t[..., None] * ab
        j = np.hypot(d[..., 0], d[..., 1]).argmin(1)
        r = np.arange(len(pts))
        cr = ab[j, 0] * ap[r, j, 1] - ab[j, 1] * ap[r, j, 0]
        return self.S[j] + t[r, j] * np.hypot(*ab[j].T), np.sign(cr) * np.hypot(d[r, j, 0], d[r, j, 1])


def _resample(c):
    seg = np.hypot(*np.diff(c, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    u = np.linspace(0, s[-1], max(2, int(math.ceil(s[-1] / STEP)) + 1))
    return np.stack([np.interp(u, s, c[:, 0]), np.interp(u, s, c[:, 1])], 1)


class MapG:
    def __init__(self, loc):
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
        from shapely.strtree import STRtree
        m = NuPlanMapFactory(get_maps_db(os.environ["NUPLAN_MAPS_ROOT"], os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"))).build_map_from_name(loc)
        ln, lc, bp = (m._get_vector_map_layer(x) for x in (L.LANE, L.LANE_CONNECTOR, L.BASELINE_PATHS))
        cp = m._load_vector_map_layer("gen_lane_connectors_scaled_width_polygons")
        cpoly = {str(int(k)): g for k, g in zip(cp.lane_connector_fid, cp.geometry)}
        base = {}
        for lf, cf, g in zip(bp.lane_fid, bp.lane_connector_fid, bp.geometry):
            base[str(int(lf if lf == lf else cf))] = _resample(np.asarray(g.coords)[:, :2])
        self.loc, ids, self.kind, self.poly, self.group, self.lidx = loc, [], [], [], [], []
        for fid, g, grp, li in zip(ln.fid, ln.geometry, ln.lane_group_fid, ln.lane_index):
            if str(fid) in base:
                ids.append(str(fid)); self.kind.append(0); self.poly.append(g); self.group.append(str(int(grp))); self.lidx.append(int(li))
        for fid, grp in zip(lc.fid, lc.lane_group_connector_fid):
            if str(fid) in base and str(fid) in cpoly:
                ids.append(str(fid)); self.kind.append(1); self.poly.append(cpoly[str(fid)]); self.group.append(str(int(grp))); self.lidx.append(-1)
        self.ids, self.ix = ids, {k: i for i, k in enumerate(ids)}
        n = len(ids)
        self.line = [Line(base[k]) for k in ids]
        self.succ, self.pred = [[] for _ in range(n)], [[] for _ in range(n)]
        for fid, ex, en in zip(lc.fid, lc.exit_lane_fid, lc.entry_lane_fid):       # exit_lane = the lane the connector leaves
            c, a, b = self.ix.get(str(fid)), self.ix.get(str(int(ex))), self.ix.get(str(int(en)))
            if c is None:
                continue
            if a is not None:
                self.succ[a].append(c); self.pred[c].append(a)
            if b is not None:
                self.succ[c].append(b); self.pred[b].append(c)
        by = {}
        for i, gk in enumerate(self.group):
            by.setdefault((self.kind[i], gk), []).append(i)
        self.nbr = [[j for j in by[(self.kind[i], self.group[i])] if j != i and (self.kind[i] == 1 or abs(self.lidx[j] - self.lidx[i]) == 1)]
                    for i in range(n)]
        self.tree = STRtree(self.poly)
        self._reach = {}

    # ------------------------------------------------------------ matching
    def candidates(self, xy, yaw):
        """[(node, cost)]: cost = |heading difference| (rad) + lateral distance to the baseline (m)."""
        from shapely.geometry import Point
        p = Point(float(xy[0]), float(xy[1]))
        idx = [int(i) for i in self.tree.query(p, predicate="intersects")]
        if not idx:
            idx = [int(i) for i in self.tree.query(p.buffer(NEAR), predicate="intersects")]
        out = []
        for i in idx:
            ln = self.line[i]
            s, d = ln.frenet(np.asarray(xy)[None])
            dh = abs(float(wrap(np.interp(s[0], ln.S, ln.h) - yaw)))
            if dh < H_MAX:
                out.append((i, dh + abs(float(d[0]))))
        return out

    def reach(self, i):
        """node -> (hops, lateral) for every allowed transition target of node i."""
        if i not in self._reach:
            r, front = {i: (0, False)}, [i]
            for h in range(1, HOPS + 1):
                front = [k for j in front for k in self.succ[j]]
                for k in front:
                    r.setdefault(k, (h, False))
            for j, (h, _) in list(r.items()):
                for k in self.nbr[j]:
                    r.setdefault(k, (h, True))
            self._reach[i] = r
        return self._reach[i]

    def viterbi(self, cands):
        """cands: per pose [(node, cost)] -> node per pose (lowest total cost over allowed transitions), or None when no connected
        assignment exists."""
        if not cands or any(not c for c in cands):
            return None
        best = {i: (c, None) for i, c in cands[0]}
        back = []
        for cs in cands[1:]:
            new = {}
            for j, cj in cs:
                bv, bi = None, None
                for i, (ci, _) in best.items():
                    r = self.reach(i).get(j)
                    if r is None:
                        continue
                    v = ci + (LC_PEN if r[1] else 0.0)
                    if bv is None or v < bv:
                        bv, bi = v, i
                if bv is not None:
                    new[j] = (bv + cj, bi)
            if not new:
                return None
            back.append({j: v[1] for j, v in new.items()})
            best = new
        j = min(best, key=lambda k: best[k][0])
        path = [j]
        for b in reversed(back):
            j = b[j]
            path.append(j)
        return path[::-1]

    def _fill(self, a, b):
        """Nodes strictly between a and b on a successor chain (b at most HOPS hops after a), [] if adjacent / not found."""
        front = [[a]]
        for _ in range(HOPS):
            front = [p + [k] for p in front for k in self.succ[p[-1]]]
            for p in front:
                if p[-1] == b:
                    return p[1:-1]
        return []

    def sequence(self, nodes):
        """Matched node per pose -> (sequence with skipped successors filled in, index in it of the first node after the last lane change)."""
        seq, lc = [nodes[0]], 0
        for b in nodes[1:]:
            a = seq[-1]
            if b == a:
                continue
            hops, lat = self.reach(a)[b]
            if lat:                                  # lateral move: the target is a neighbour of a node `hops` after a
                tgt = [k for k in self.reach(a) if self.reach(a)[k] == (hops, False) and b in self.nbr[k]]
                seq += (self._fill(a, tgt[0]) if tgt and tgt[0] != a else [])
                lc = len(seq)
            else:
                seq += self._fill(a, b)
            seq.append(b)
        return seq, lc

    def straightest(self, i):
        if not self.succ[i]:
            return None
        h = self.line[i].h[-1]
        return min(self.succ[i], key=lambda k: abs(float(wrap(self.line[k].h[-1] - h))))

    def _chain(self, seq):
        pts, own = [], []
        for n, k in enumerate(seq):
            p = self.line[k].xy if n == 0 else self.line[k].xy[1:]
            pts.append(p)
            own += [k] * len(p)
        return np.vstack(pts), np.array(own)

    def centreline(self, seq, xy, ahead=90.0):
        """Centreline of a connected node sequence from the projection of xy onto the whole chain, extended
        downstream along the straightest successors until `ahead` m, and upstream through the predecessor nearest to xy while xy
        projects onto the very start. Returns (samples in map coordinates starting at the projection, nodes used, node per sample)."""
        seq, xy = list(seq), np.asarray(xy, np.float64)
        for _ in range(5):
            pts, own = self._chain(seq)
            ln = Line(pts)
            s0 = float(ln.frenet(xy[None])[0][0])
            if s0 > 1e-3 or not self.pred[seq[0]]:
                break
            seq.insert(0, min(self.pred[seq[0]], key=lambda k: float(np.hypot(*(self.line[k].xy - xy).T).min())))
        tot = ln.L - s0
        while tot < ahead:
            k = self.straightest(seq[-1])
            if k is None or k in seq:
                break
            seq.append(k)
            tot += self.line[k].L
        pts, own = self._chain(seq)
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        st = s0 + np.arange(0, max(min(ahead, seg[-1] - s0), STEP) + 1e-9, STEP)
        xy2 = np.stack([np.interp(st, seg, pts[:, 0]), np.interp(st, seg, pts[:, 1])], 1)
        return xy2, seq, own[np.clip(np.searchsorted(seg, st), 0, len(own) - 1)]

    def runs(self, nodes):
        """Matched node per pose -> [(index of the run's first pose, node sequence of the run)]; a new run starts at each lateral move."""
        out = [(0, [nodes[0]])]
        for i, b in enumerate(nodes[1:], 1):
            a = out[-1][1][-1]
            if b == a:
                continue
            if self.reach(a)[b][1]:
                out.append((i, [b]))
            else:
                out[-1][1].extend(self._fill(a, b) + [b])
        return out

    def half_width(self, xy, own):
        """Distance from each centreline sample to the boundary of its own node polygon."""
        import shapely
        return shapely.distance(shapely.boundary(np.array([self.poly[k] for k in own], dtype=object)), shapely.points(xy))

    def paths_ahead(self, node, s, dist):
        """Downstream node paths from station s of `node` out to `dist` m: [(nodes, reached_length)]."""
        out, stack = [], [([node], self.line[node].L - s)]
        while stack:
            p, cum = stack.pop()
            if cum >= dist or not self.succ[p[-1]] or len(p) > 12:
                out.append((p, cum))
                continue
            for k in self.succ[p[-1]]:
                if k not in p:
                    stack.append((p + [k], cum + self.line[k].L))
        return out

    def heading_at(self, nodes, s, dist):
        """Baseline heading `dist` m after station s of nodes[0] along the node path (its end if shorter)."""
        rem = dist + s
        for k in nodes:
            ln = self.line[k]
            if rem <= ln.L or k == nodes[-1]:
                return float(np.interp(min(rem, ln.L), ln.S, ln.h))
            rem -= ln.L


def get(loc):
    if loc not in _G:
        _G[loc] = MapG(loc)
    return _G[loc]
