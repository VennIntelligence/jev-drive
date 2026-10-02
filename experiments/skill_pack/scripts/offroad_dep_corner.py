"""Where exactly the car leaves the map: the departing corner and the nearest drivable-area boundary point, ego frame at t0 (Q3).

navsim2 env, CPU. For every stage-2 token that fails DAC for native or N4: simulate the cached pose file (official LQR + bicycle),
take the state of the first departure, the outside corner of largest distance, and the nearest point of the drivable-area union to
that corner. Output dep_corner.pkl: {token: {model: dict(first, corner (x, y), boundary (x, y), dist, side)}}.
"""
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offroad_lib as L  # noqa: E402
from offroad_replay_cf import corners_of, geoms  # noqa: E402

_G = {}


def _init():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    sim, *_ = L.setup_scoring()
    _G.update(sim=sim, cp=L.cache_paths(), P={"native": L.poses_by_token(L.NATIVE_POSES), "n4": L.poses_by_token(L.N4_POSES)})


def work(t):
    import shapely
    from shapely.ops import nearest_points
    mc = L.load_cache(_G["cp"][t])
    g = geoms(mc)
    out = {}
    for m in ("native", "n4"):
        st = L.simulate(_G["sim"], mc, _G["P"][m][t])
        cor = corners_of(mc, st)
        inside = g["am"].points_in_polygons(cor[None, :, :-1, :])[g["area"]].any(axis=0)[0]
        bad = ~inside.all(axis=1)
        if not bad.any():
            continue
        i = int(np.flatnonzero(bad)[0])
        pts = shapely.points(cor[i, :4])
        d = np.where(~inside[i], shapely.distance(g["area_u"], pts), 0.0)
        c = int(d.argmax())
        npnt = nearest_points(g["area_u"], pts[c])[0]
        e = L.to_ego(mc, np.array([[cor[i, c, 0], cor[i, c, 1], 0.0], [npnt.x, npnt.y, 0.0]]))
        out[m] = dict(first=i, corner_idx=c, corner=e[0, :2], boundary=e[1, :2], dist=float(d[c]),
                      side=float(-np.sin(st[i, 2]) * (cor[i, c, 0] - st[i, 0]) + np.cos(st[i, 2]) * (cor[i, c, 1] - st[i, 1])))
    return t, out


if __name__ == "__main__":
    tab = __import__("pandas").read_pickle(L.OUT / "stage2_table.pkl")
    toks = tab[(tab.native_dac == 0) | (tab.n4_dac == 0)].token.tolist()
    with mp.get_context("fork").Pool(24, initializer=_init) as pool:
        res = dict(pool.imap_unordered(work, toks, chunksize=8))
    pickle.dump(res, open(L.OUT / "dep_corner.pkl", "wb"))
    print(len(res))
