"""Check figure for drivable/nav.npz: 8 samples, sdf image + zero contour + logged future. Run on the box (reads tab.npz + nav.npz), then copy the png to the Mac."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jevdrive.common import data_dir

d = np.load(data_dir() / "runs/op_route_ft/drivable/nav.npz")
t = np.load(data_dir() / "runs/op_adapt_H/samples/nav/tab.npz", allow_pickle=True)
assert (d["id"] == t["id"]).all()
ok, ins, mn = d["ok"], d["logged_inside"], d["min_sdf_logged"]
rng = np.random.default_rng(0)
bad = np.flatnonzero(ok & ~ins)
pick = np.r_[rng.choice(np.flatnonzero(ok & ins), 5, replace=False), rng.choice(bad, 3, replace=False)]
x0, y0, r = float(d["x0"]), float(d["y0"]), float(d["res"])
ext = [y0, y0 + 160 * r, x0, x0 + 160 * r]                       # imshow: columns = y (left), rows = x (forward); flip y so left is on the left
fig, ax = plt.subplots(2, 4, figsize=(16, 9))
for a, k in zip(ax.ravel(), pick):
    s = d["sdf"][k].astype(np.float32)
    a.imshow(s, origin="lower", extent=[ext[1], ext[0], ext[2], ext[3]], cmap="RdBu", vmin=-8, vmax=8)
    yy, xx = np.meshgrid(y0 + (np.arange(160) + .5) * r, x0 + (np.arange(160) + .5) * r)
    a.contour(yy, xx, s, levels=[0], colors="k", linewidths=1)
    f = t["fut20"][k]
    a.plot(f[:, 1], f[:, 0], "g.-", ms=4, lw=1.5)
    a.plot(0, 0, "y^", ms=8, mec="k")
    a.set_xlim(ext[1], ext[0]); a.set_ylim(ext[2], ext[3])
    a.set_title(f"{str(d['id'][k])[:8]} min_sdf_logged {mn[k]:.2f}  {'inside' if ins[k] else 'OUTSIDE'}", fontsize=9)
    a.set_xlabel("y left (m)"); a.set_ylabel("x forward (m)")
fig.suptitle("Drivable SDF (blue = inside, red = outside, black = zero contour, green = logged future, yellow = ego rear axle). Top row + first of bottom: logged_inside; last 3: logged_inside False.")
fig.tight_layout()
fig.savefig("/root/autodl-tmp/ujs/runs/op_route_ft/drivable/drivable_sdf_check.png", dpi=90)
print("ok", ok.sum(), len(ok), "inside", ins.sum() / ok.sum(), "min_sdf pct", np.percentile(mn[ok], [1, 5, 10, 50]), "bad", len(bad))
